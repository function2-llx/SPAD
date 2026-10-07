"""Tests for filter-driven latent sample selection."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import threading
from types import SimpleNamespace

import msgpack
import pytest
from safetensors.torch import save_file
import torch

from pumit.ucpt import latent_codec
from pumit.ucpt.stream import latent_backend, latents
from pumit.ucpt.stream.latent_backend import (
    LatentEncodeResult,
    LatentEncodeWork,
    _load_unlabeled_samples,
)
from pumit.ucpt.stream.latents import _select_samples


def _write_shard(tmp_path, samples):
    path = tmp_path / 'shard_00000.msgpack'
    path.write_bytes(msgpack.packb({'batches': [{'samples': samples}]}))
    return path


def _sample(name: str, *, labeled: bool) -> dict:
    sample = {
        'img': name,
        'n_patches': 1,
        'da_enc': 0,
        'depth': 1,
        'params': [{'patch_size': [1, 64, 64]}],
        'labeled': labeled,
    }
    if labeled:
        sample['classes'] = [{'source': 'src', 'name': 'liver'}]
    return sample


def test_build_encoder_compiles_only_repeated_static_signatures(monkeypatch, tmp_path):
    class FakeEncoder(torch.nn.Module):
        def forward(self, x, *, da):
            assert da == 0
            return x + 1

    class FakeCodec:
        def __init__(self, *, grad_ckpt):
            assert grad_ckpt is False
            self.encoder = FakeEncoder()

        def load_state_dict(self, state_dict):
            assert state_dict == {}

    compile_args = {}
    compiled_calls = []
    monkeypatch.setattr(latent_codec, 'SPADFlux2AE', FakeCodec)
    monkeypatch.setattr(latent_codec.torch, 'load', lambda *args, **kwargs: {'model': {}})

    def compile_encoder(_encoder, *, dynamic, mode):
        compile_args.update(dynamic=dynamic, mode=mode)

        def compiled(x, *, da):
            compiled_calls.append((tuple(x.shape), da))
            return _encoder(x, da=da)

        return compiled

    monkeypatch.setattr(latent_codec.torch, 'compile', compile_encoder)

    actual = latent_codec.build_encoder(
        'flux2',
        tmp_path / 'codec.pt',
        torch.device('cpu'),
        compile_mode='max-autotune',
    )

    x = torch.zeros(1, 1)
    assert torch.equal(actual(x, da=0), x + 1)
    assert torch.equal(actual(x, da=0), x + 1)
    assert torch.equal(actual(x, da=0), x + 1)
    assert compile_args == {'dynamic': False, 'mode': 'max-autotune'}
    assert compiled_calls == [((1, 1), 0), ((1, 1), 0)]


def test_static_hot_encoder_keeps_new_signatures_eager_after_cap():
    eager_calls = []
    compiled_calls = []

    def eager(x, *, da):
        eager_calls.append((tuple(x.shape), da))
        return x

    def compiled(x, *, da):
        compiled_calls.append((tuple(x.shape), da))
        return x

    encoder = latent_codec._StaticHotEncoder(eager, compiled, max_signatures=2)
    for batch_size, da in ((1, 0), (1, 1), (2, 0)):
        x = torch.zeros(batch_size, 1)
        encoder(x, da=da)
        encoder(x, da=da)
    encoder(torch.zeros(1, 1), da=0)

    assert eager_calls == [((1, 1), 0), ((1, 1), 1), ((2, 1), 0), ((2, 1), 0)]
    assert compiled_calls == [((1, 1), 0), ((1, 1), 1), ((1, 1), 0)]


def test_prepare_encode_work_settles_replay_tasks_before_raising():
    class ReplayError(RuntimeError):
        pass

    slow_started = threading.Event()
    fail_started = threading.Event()
    release_slow = threading.Event()
    slow_finished = threading.Event()

    class Pipeline:
        def replay(self, sample, _params):
            if sample['img'] == 'fail':
                fail_started.set()
                assert slow_started.wait(timeout=2)
                raise ReplayError('sample replay failed')
            slow_started.set()
            assert release_slow.wait(timeout=2)
            slow_finished.set()
            return {'img': torch.zeros(1)}

    samples = [
        {'img': 'fail', 'params': [], 'da_enc': 0, 'n_patches': 1},
        {'img': 'slow', 'params': [], 'da_enc': 0, 'n_patches': 1},
    ]
    work = LatentEncodeWork(7, 0, (0, 1), 1, 1)
    with (
        ThreadPoolExecutor(max_workers=2) as replay_pool,
        ThreadPoolExecutor(max_workers=1) as caller_pool,
    ):
        result = caller_pool.submit(
            latent_backend._prepare_encode_work_from_samples,
            work,
            samples,
            Pipeline(),
            replay_pool,
        )
        assert fail_started.wait(timeout=2)
        assert slow_started.wait(timeout=2)
        assert not result.done()
        release_slow.set()
        with pytest.raises(ReplayError, match='sample replay failed'):
            result.result()

    assert slow_finished.is_set()


def test_prepared_loader_forks_replay_workers_and_returns_stacked_tensor(
    monkeypatch,
    tmp_path,
):
    class Pipeline:
        def replay(self, sample, _params):
            value = int(sample['img'].split('-')[-1])
            return {'img': torch.tensor([value], dtype=torch.float32)}

    samples = [_sample(f'image-{index}', labeled=False) for index in range(2)]
    _write_shard(tmp_path, samples)
    monkeypatch.setattr(latent_backend, '_build_pipeline', lambda _stream_dir: Pipeline())
    work = LatentEncodeWork(0, 17, (0, 1), 2, 2)
    loader = latent_backend._LatentPreparedLoader(tmp_path, replay_threads=2)
    try:
        loader.start()
        loader.submit(work)
        prepared = loader.get()
    finally:
        loader.close()

    assert prepared.work == work
    assert prepared.sample_rows == (1, 1)
    assert torch.equal(prepared.images, torch.tensor([[0.0], [1.0]]))


def test_prepared_scheduler_preserves_part_ids_across_out_of_order_replay():
    work = [
        LatentEncodeWork(7, part_id, (part_id,), 1, part_id + 1)
        for part_id in range(3)
    ]

    class Loader:
        capacity = 2

        def __init__(self):
            self.pending = []
            self.submitted = []

        def submit(self, item):
            self.pending.append(item)
            self.submitted.append(item)

        def get(self):
            item = self.pending.pop()
            return latent_backend._PreparedLatentEncode(
                work=item,
                sample_rows=(1,),
                images=torch.zeros(1, 1),
                da_enc=0,
                prepare_seconds=1,
                replay_seconds=2,
            )

    class Pool:
        def __init__(self):
            self.submitted = []

        def submit(self, callable_, prepared):
            assert callable_ is latent_backend._encode_prepared_work
            self.submitted.append(prepared.work)
            future = Future()
            future.set_result(
                LatentEncodeResult(
                    prepared.work,
                    1,
                    2,
                    3,
                    4,
                    5,
                    15,
                    device_id=prepared.work.part_id % 2,
                )
            )
            return future

    loader = Loader()
    pool = Pool()
    results = list(
        latent_backend._iter_prepared_encode_results(
            work,
            loader,
            pool,
            max_inflight=2,
        )
    )

    returned = [result.result.work for result in results]
    assert loader.submitted == work
    assert pool.submitted == [work[1], work[2], work[0]]
    assert set(returned) == set(work)


def test_dataloader_workers_start_before_encoding_dispatch(monkeypatch, tmp_path):
    events = []

    class PreparedLoader:
        capacity = 4

        def __init__(self, _stream_dir, *, replay_threads):
            assert replay_threads == 8

        def start(self):
            events.append('fork')

        def close(self):
            events.append('close-loader')

    def encode(_args, prepared_loader):
        assert isinstance(prepared_loader, PreparedLoader)
        events.append('dispatch')

    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1')
    monkeypatch.setattr(latents, '_LatentPreparedLoader', PreparedLoader)
    monkeypatch.setattr(latents, '_encode_filtered_latents', encode)
    args = SimpleNamespace(
        async_finalize=None,
        filter='all',
        num_workers=8,
        source_latent_dir=None,
        stream=tmp_path,
    )

    latents.cmd_encode_latents(args)

    assert events == ['fork', 'dispatch', 'close-loader']


def test_encode_shard_parts_resumes_gap_and_aggregates_device_timings(
    monkeypatch,
    tmp_path,
):
    work = [
        LatentEncodeWork(7, part_id, (part_id,), 1, part_id + 10)
        for part_id in range(6)
    ]
    unlabeled = [{'n_patches': 1} for _ in work]
    latent_backend._write_latent_part(
        tmp_path,
        work[2],
        torch.zeros(1, 32, dtype=torch.float16),
    )
    monkeypatch.setattr(latent_backend, '_build_encode_work', lambda *args, **kwargs: work)

    class Loader:
        capacity = 2

        def __init__(self):
            self.pending = []

        def submit(self, item):
            self.pending.append(item)

        def get(self):
            item = self.pending.pop()
            return latent_backend._PreparedLatentEncode(
                work=item,
                sample_rows=(1,),
                images=torch.zeros(1, 1),
                da_enc=0,
                prepare_seconds=1,
                replay_seconds=2,
            )

    class Pool:
        def submit(self, callable_, prepared):
            assert callable_ is latent_backend._encode_prepared_work
            future = Future()
            future.set_result(
                LatentEncodeResult(
                    prepared.work,
                    1,
                    2,
                    3,
                    4,
                    0.0,
                    10,
                    device_id=prepared.work.part_id % 2,
                    encoded=torch.zeros(1, 32, dtype=torch.float16),
                )
            )
            return future

    loader = Loader()
    encoded = latent_backend._encode_selected_latent_shard_parts(
        {'shard_id': 7, 'logical_latent_rows': 6},
        unlabeled,
        list(range(6)),
        stream_dir=tmp_path,
        source_latent_dir=None,
        parts_dir=tmp_path,
        output_dir=tmp_path,
        source_prefix_samples=0,
        gpu_pool=Pool(),
        memory_budget_gb=1,
        max_inflight=2,
        cache_archiver=None,
        prepared_loader=loader,
    )

    assert encoded.finalization.work == tuple(work)
    assert encoded.result['gpu_batches'] == 6
    scheduler = encoded.result['scheduler_summary']
    assert scheduler['pending_gpu_batches'] == 5
    assert scheduler['prepared_capacity'] == 2
    timings = encoded.result['timings']
    assert timings['prepare'] == 5
    assert timings['replay'] == 10
    assert timings['transfer'] == 15
    assert timings['encode'] == 20
    assert timings['write'] >= 0.0
    assert timings['total'] == pytest.approx(50 + timings['write'])
    assert timings['pipeline_wall'] == pytest.approx(0, abs=1)
    devices = scheduler['devices']
    assert devices[0]['gpu_batches'] == 2
    assert devices[0]['voxel_work'] == 24
    assert devices[1]['gpu_batches'] == 3
    assert devices[1]['voxel_work'] == 39
    assert devices[0]['write'] + devices[1]['write'] == pytest.approx(timings['write'])
    # The parent-side writer must have persisted every part before returning.
    for item in work:
        latent_backend._validate_latent_part(
            latent_backend._latent_part_path(tmp_path, item),
            item,
        )


def test_finalize_selected_latent_shard_preserves_commit_order(monkeypatch, tmp_path):
    calls = []

    class Archiver:
        def submit(self):
            calls.append('archive')

    monkeypatch.setattr(
        latent_backend,
        '_materialize_latent_parts',
        lambda *args, **kwargs: calls.append('materialize'),
    )
    monkeypatch.setattr(
        latent_backend,
        '_validate_materialized_latent',
        lambda *args, **kwargs: calls.append('validate'),
    )
    monkeypatch.setattr(
        latent_backend,
        '_remove_latent_parts',
        lambda *args, **kwargs: calls.append('remove'),
    )
    finalization = latent_backend._LatentShardFinalization(
        shard_id=7,
        expected_rows=1,
        source_path=None,
        unlabeled=[{'n_patches': 1}],
        source_prefix_samples=0,
        work=(LatentEncodeWork(7, 0, (0,), 1, 1),),
        parts_dir=tmp_path,
        output_path=tmp_path / 'shard_00007.safetensors',
        plan_sha256=None,
        cache_archiver=Archiver(),
    )

    result = latent_backend._finalize_selected_latent_shard(finalization)

    assert result.shard_id == 7
    assert result.total_seconds >= 0
    assert calls == ['materialize', 'validate', 'remove', 'archive']


def test_bounded_finalizer_drains_before_scheduling_next(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def finalize(shard_id):
        if shard_id == 0:
            first_started.set()
            assert release_first.wait(timeout=2)
        else:
            second_started.set()
        return latent_backend._LatentShardFinalizationResult(
            shard_id=shard_id,
            materialize_seconds=1,
            validate_seconds=2,
            cleanup_seconds=3,
            total_seconds=6,
        )

    monkeypatch.setattr(latent_backend, '_finalize_selected_latent_shard', finalize)
    finalizer = latent_backend._BoundedLatentShardFinalizer()
    assert finalizer.submit(0) is None
    assert first_started.wait(timeout=2)
    with ThreadPoolExecutor(max_workers=1) as caller_pool:
        second_submission = caller_pool.submit(finalizer.submit, 1)
        assert not second_submission.done()
        assert not second_started.is_set()
        release_first.set()
        assert second_submission.result().shard_id == 0

    assert second_started.wait(timeout=2)
    assert finalizer.close().shard_id == 1


def test_bounded_finalizer_propagates_same_exception(monkeypatch):
    error = RuntimeError('finalization failed')

    def finalize(_task):
        raise error

    monkeypatch.setattr(latent_backend, '_finalize_selected_latent_shard', finalize)
    finalizer = latent_backend._BoundedLatentShardFinalizer()
    assert finalizer.submit(object()) is None
    with pytest.raises(RuntimeError) as raised:
        finalizer.submit(object())
    assert raised.value is error
    finalizer.close()


def test_async_finalize_is_restricted_to_full_reencoding(monkeypatch):
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1')

    with pytest.raises(ValueError, match='only valid with --filter all'):
        latents.cmd_encode_latents(
            SimpleNamespace(
                filter='canonical-inplane',
                async_finalize=True,
            )
        )


@pytest.mark.parametrize(
    ('filter_name', 'requested', 'expected'),
    [
        ('all', None, True),
        ('suffix', None, False),
        ('canonical-inplane', None, False),
        ('all', False, False),
        ('all', True, True),
    ],
)
def test_async_finalize_default_depends_on_filter(filter_name, requested, expected):
    args = SimpleNamespace(filter=filter_name, async_finalize=requested)

    assert latents._async_finalize_enabled(args) is expected


def test_load_unlabeled_samples_preserves_stream_order(tmp_path):
    samples = [
        _sample('unlabeled-0', labeled=False),
        _sample('labeled', labeled=True),
        _sample('unlabeled-1', labeled=False),
    ]

    actual = _load_unlabeled_samples(_write_shard(tmp_path, samples))

    assert [sample['img'] for sample in actual] == ['unlabeled-0', 'unlabeled-1']


def test_full_and_suffix_filters_differ_only_by_selected_indices(tmp_path):
    shard = _write_shard(
        tmp_path,
        [_sample(f'unlabeled-{index}', labeled=False) for index in range(3)],
    )
    row = {
        'logical_latent_rows': 3,
        'used_old_unlabeled_samples': 2,
    }

    _, full, full_source_samples = _select_samples('all', shard, row)
    _, suffix, suffix_source_samples = _select_samples('suffix', shard, row)

    assert (full, full_source_samples) == ([0, 1, 2], 0)
    assert (suffix, suffix_source_samples) == ([2], 2)


def test_load_unlabeled_samples_rejects_invalid_label_schema(tmp_path):
    unlabeled_with_classes = _sample('invalid', labeled=False)
    unlabeled_with_classes['classes'] = [{'source': 'src', 'name': 'liver'}]

    with pytest.raises(ValueError, match='invalid labeled sample schema'):
        _load_unlabeled_samples(_write_shard(tmp_path, [unlabeled_with_classes]))


def test_load_unlabeled_samples_rejects_legacy_label_contract(tmp_path):
    sample = _sample('legacy', labeled=True)
    sample['label_classes'] = {'src': {'positive': ['liver'], 'negative': []}}

    with pytest.raises(ValueError, match='full label contract'):
        _load_unlabeled_samples(_write_shard(tmp_path, [sample]))


def _guard_stream(tmp_path, checkpoint_bytes=b'codec-A'):
    stream = tmp_path / 'stream'
    latent_dir = stream / 'latents'
    latent_dir.mkdir(parents=True)
    checkpoint = tmp_path / 'codec.pt'
    checkpoint.write_bytes(checkpoint_bytes)
    return stream, latent_dir, checkpoint


def test_full_plan_guard_writes_plan_and_accepts_matching_resume(tmp_path):
    stream, latent_dir, checkpoint = _guard_stream(tmp_path)

    latents._guard_full_latent_plan(
        stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
    )

    plan = json.loads((latent_dir / 'plan.json').read_text())
    assert plan['filter'] == 'all'
    assert plan['codec_model'] == 'flux2'
    assert plan['codec_checkpoint_sha256'] == hashlib.sha256(b'codec-A').hexdigest()
    latents._guard_full_latent_plan(
        stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
    )


def test_full_plan_guard_rejects_codec_change_on_resume(tmp_path):
    stream, latent_dir, checkpoint = _guard_stream(tmp_path)
    latents._guard_full_latent_plan(
        stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
    )

    checkpoint.write_bytes(b'codec-B')
    with pytest.raises(ValueError, match='existing full-latent plan differs'):
        latents._guard_full_latent_plan(
            stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
        )


def test_full_plan_guard_rejects_unattributed_partial_latents(tmp_path):
    stream, latent_dir, checkpoint = _guard_stream(tmp_path)
    (latent_dir / 'shard_00000.safetensors').write_bytes(b'orphan')

    with pytest.raises(ValueError, match='without plan.json'):
        latents._guard_full_latent_plan(
            stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
        )


def test_full_plan_guard_rejects_unattributed_parts(tmp_path):
    stream, latent_dir, checkpoint = _guard_stream(tmp_path)
    parts = stream / latent_backend._PARTS_DIR_NAME / 'shard_00000'
    parts.mkdir(parents=True)
    (parts / 'part_00000.safetensors').write_bytes(b'orphan')

    with pytest.raises(ValueError, match='without plan.json'):
        latents._guard_full_latent_plan(
            stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
        )


def test_full_plan_guard_skips_completed_stream(tmp_path):
    stream, latent_dir, checkpoint = _guard_stream(tmp_path)
    (latent_dir / 'shard_00000.safetensors').write_bytes(b'complete')
    (stream / 'latent-materialized.json').write_text('{}')

    latents._guard_full_latent_plan(
        stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
    )

    assert not (latent_dir / 'plan.json').exists()


def test_encode_filtered_latents_rejects_codec_change_on_resume(tmp_path, monkeypatch):
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    stream, latent_dir, checkpoint = _guard_stream(tmp_path)
    (stream / 'manifest.jsonl').write_text(
        json.dumps({'shard_id': 0, 'logical_latent_rows': 2, 'new_unlabeled_samples': 1}) + '\n'
    )
    latents._guard_full_latent_plan(
        stream, latent_dir, codec_model='flux2', codec_checkpoint=checkpoint, rank=0,
    )
    save_file(
        {'latents': torch.zeros(2, 32, dtype=torch.float16)},
        latent_dir / 'shard_00000.safetensors',
    )
    checkpoint.write_bytes(b'codec-B')
    args = SimpleNamespace(
        stream=stream,
        source_latent_dir=None,
        codec_checkpoint=checkpoint,
        codec_model='flux2',
        filter='all',
        shard_offset=0,
        shards=None,
        memory_budget_gb=1.0,
        async_finalize=None,
    )

    with pytest.raises(ValueError, match='existing full-latent plan differs'):
        latents._encode_filtered_latents(args, None)

    assert not (stream / 'latent-materialized.json').exists()
