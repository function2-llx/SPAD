"""Test stream generation produces valid msgpack shards."""
import itertools
import tempfile
from pathlib import Path

import msgpack
import numpy as np
import pytest

from pumit.codec.transforms import build_codec_pipeline
from pumit.codec.datamodule import TransformConf
from pumit.data.config import DepthTierConfig


def _make_sample_data(n=100):
    """Create fake sample metadata (no .npy files needed for generation)."""
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        depth = int(rng.choice([1, 20, 50, 100]))
        records.append({
            'shape': np.array([depth, 256, 256]),
            'spacing': np.array([2.0, 1.0, 1.0]) if depth > 1 else np.array([float('nan'), 1.0, 1.0]),
            'img': f'/fake/sample_{i}.npy',
            'weight': 1.0,
        })
    return records


def _generate_stream(output_dir: Path, records: list, total_batches: int, seed: int = 42):
    """Generate a small deterministic stream with the legacy codec bucketing path."""
    conf = TransformConf()
    depth_tiers = {
        0: DepthTierConfig(tiers=(64, 48, 32, 24, 16), batch_sizes=(2, 2, 3, 4, 6)),
        1: DepthTierConfig(tiers=(64, 48, 32, 24, 16), batch_sizes=(2, 2, 3, 4, 6)),
        2: DepthTierConfig(tiers=(32, 24, 16), batch_sizes=(3, 4, 6)),
        3: DepthTierConfig(tiers=(16,), batch_sizes=(6,)),
        4: DepthTierConfig(tiers=(16,), batch_sizes=(6,)),
        None: DepthTierConfig(tiers=(1,), batch_sizes=(16,)),
    }
    pipeline = build_codec_pipeline(conf, depth_tiers=depth_tiers, max_da=4, smooth_spad=True)

    rng = np.random.default_rng(seed)
    weights = np.array([r['weight'] for r in records])
    weights = weights / weights.sum()

    buckets: dict[tuple, list] = {}
    completed_batches = []

    while len(completed_batches) < total_batches:
        idx = int(rng.choice(len(records), p=weights))
        state = records[idx]
        params = pipeline.sample_params(state, rng)
        if params is None:
            continue

        sp = params[0]  # spatial params
        key = (sp['da_enc'], sp['da_dec'], sp['patch_size'][0])

        tier_key = sp['da_enc'] if sp['da_enc'] is not None else None
        if tier_key is not None:
            tier_key = min(tier_key, 4)
        tier_cfg = depth_tiers[tier_key]
        depth_idx = tier_cfg.tiers.index(sp['patch_size'][0])
        batch_size = tier_cfg.batch_sizes[depth_idx]

        sample_entry = {
            'img': state['img'],
            'spacing': [float(x) for x in state['spacing']],
            'params': params,
        }
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(sample_entry)

        if len(buckets[key]) >= batch_size:
            batch_samples = buckets.pop(key)
            completed_batches.append({
                'da_enc': sp['da_enc'],
                'da_dec': sp['da_dec'],
                'patch_size': sp['patch_size'],
                'samples': batch_samples,
            })

    # Write shard
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / 'shard_00000.msgpack'
    with open(shard_path, 'wb') as f:
        msgpack.pack({'batches': completed_batches}, f)

    return shard_path


def test_generation_produces_valid_shard():
    records = _make_sample_data(200)
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = _generate_stream(Path(tmp), records, total_batches=10, seed=42)
        assert shard_path.exists()

        with open(shard_path, 'rb') as f:
            shard = msgpack.unpack(f, raw=False)

        assert 'batches' in shard
        assert len(shard['batches']) == 10

        for batch in shard['batches']:
            assert 'da_enc' in batch
            assert 'da_dec' in batch
            assert 'patch_size' in batch
            assert 'samples' in batch
            assert len(batch['samples']) > 0
            for s in batch['samples']:
                assert 'img' in s
                assert 'spacing' in s
                assert 'params' in s
                assert isinstance(s['params'], list)
                assert len(s['params']) == 5  # 5 transforms in pipeline


def test_generation_deterministic():
    records = _make_sample_data(100)
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        _generate_stream(Path(tmp1), records, total_batches=5, seed=123)
        _generate_stream(Path(tmp2), records, total_batches=5, seed=123)

        with open(Path(tmp1) / 'shard_00000.msgpack', 'rb') as f:
            s1 = msgpack.unpack(f, raw=False)
        with open(Path(tmp2) / 'shard_00000.msgpack', 'rb') as f:
            s2 = msgpack.unpack(f, raw=False)

        assert s1 == s2


# --- stream dispatcher robustness ---

def _dispatch(tmp, monkeypatch, *, shard_fn, seed=42, shards=3, force=False,
              budget_ms=100.0, shard_offset=0, rank=None, world_size=None):
    """Run the stream dispatcher inline with the per-shard builder replaced."""
    from pumit.ucpt.stream import build

    cfg = tmp / 'gen.yaml'
    if not cfg.exists():
        cfg.write_text('seg_positive_queries: 1\nseg_negative_queries: 1\n')
    monkeypatch.setattr(build, 'build_one_shard', shard_fn)
    monkeypatch.setattr(
        build,
        '_load_worker_globals',
        lambda config_path: (object(), (([object()], np.ones(1)), ([object()], np.ones(1))), object()),
    )
    if rank is None:
        monkeypatch.delenv('RANK', raising=False)
        monkeypatch.delenv('WORLD_SIZE', raising=False)
    else:
        monkeypatch.setenv('RANK', str(rank))
        monkeypatch.setenv('WORLD_SIZE', str(world_size))
    build.build_stream(
        config_path=cfg, shards=shards, output_dir=tmp / 'stream', seed=seed,
        batches_per_shard=2, shard_workers=1, sample_workers=1, sample_prefetch_factor=1,
        budget_ms=budget_ms,
        shard_offset=shard_offset, label_budget_fraction=0.5, force=force,
        cost_model_path=None, use_default_cost_model=True,
        source_stream=None,
    )


def _write_ok_shard(request):
    from pumit.ucpt.mask_store import MASK_STORAGE, mask_shard_path
    from pumit.ucpt.stream.manifest import sha256_file, write_report
    from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT

    output_dir = Path(request.output_dir)
    shard_path = output_dir / f'shard_{request.shard_id:05d}.msgpack'
    mask_path = mask_shard_path(output_dir, request.shard_id)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(b'x')
    mask_path.write_bytes(b'')
    report = {
        'shard_id': request.shard_id,
        'build_fingerprint': request.build_fingerprint,
        'output_sha256': sha256_file(shard_path),
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'mask_sha256': sha256_file(mask_path),
        'mask_storage_bytes': 0,
    }
    write_report(output_dir, report)
    return report


def test_shard_failure_stops_inline_generation(tmp_path, monkeypatch):
    ran = []

    def flaky(request):
        ran.append(request.shard_id)
        if request.shard_id == 1:
            raise RuntimeError('boom')
        return _write_ok_shard(request)

    with pytest.raises(RuntimeError, match='boom'):
        _dispatch(tmp_path, monkeypatch, shard_fn=flaky)
    assert ran == [0, 1]


def test_default_resume_refuses_fingerprint_mismatch(tmp_path, monkeypatch):
    """An existing stream generated with different knobs is refused."""
    _dispatch(tmp_path, monkeypatch, shard_fn=_write_ok_shard, seed=1)
    with pytest.raises(RuntimeError, match='build plan mismatch'):
        _dispatch(tmp_path, monkeypatch, shard_fn=_write_ok_shard, seed=2)


def test_default_resume_same_fingerprint_skips_done_shards(tmp_path, monkeypatch):
    _dispatch(tmp_path, monkeypatch, shard_fn=_write_ok_shard, seed=1)
    (tmp_path / 'stream' / 'shard_00002.msgpack').unlink()
    from pumit.ucpt.mask_store import mask_shard_path
    mask_shard_path(tmp_path / 'stream', 2).unlink()
    (tmp_path / 'stream' / 'reports' / 'shard_00002.json').unlink()

    regenerated = []

    def tracking(request):
        regenerated.append(request.shard_id)
        return _write_ok_shard(request)

    _dispatch(tmp_path, monkeypatch, shard_fn=tracking, seed=1)
    assert regenerated == [2]


def test_default_resume_rejects_partial_mask_sidecar(tmp_path, monkeypatch):
    from pumit.ucpt.mask_store import mask_shard_path

    _dispatch(tmp_path, monkeypatch, shard_fn=_write_ok_shard, seed=1)
    mask_shard_path(tmp_path / 'stream', 1).unlink()

    with pytest.raises(RuntimeError, match='partial output'):
        _dispatch(tmp_path, monkeypatch, shard_fn=_write_ok_shard, seed=1)


def test_force_rebuilds_target_shards(tmp_path, monkeypatch):
    _dispatch(tmp_path, monkeypatch, shard_fn=_write_ok_shard, seed=1)
    regenerated = []

    def tracking(request):
        regenerated.append(request.shard_id)
        return _write_ok_shard(request)

    _dispatch(tmp_path, monkeypatch, shard_fn=tracking, seed=1, force=True)
    assert regenerated == [0, 1, 2]


def test_shard_offset_uses_absolute_ids(tmp_path, monkeypatch):
    _dispatch(
        tmp_path,
        monkeypatch,
        shard_fn=_write_ok_shard,
        shards=3,
        shard_offset=7,
    )
    assert sorted(path.name for path in (tmp_path / 'stream').glob('shard_*.msgpack')) == [
        'shard_00007.msgpack',
        'shard_00008.msgpack',
        'shard_00009.msgpack',
    ]


def test_multi_node_rank_owns_strided_global_shard_subset(tmp_path, monkeypatch):
    _dispatch(
        tmp_path,
        monkeypatch,
        shard_fn=_write_ok_shard,
        shards=8,
        shard_offset=10,
        rank=1,
        world_size=3,
    )
    assert sorted(path.name for path in (tmp_path / 'stream').glob('shard_*.msgpack')) == [
        'shard_00011.msgpack',
        'shard_00014.msgpack',
        'shard_00017.msgpack',
    ]


@pytest.mark.parametrize('normalization_location', ['composition', 'top'])
def test_repack_allows_current_data_pool_count_to_differ_from_source(
    tmp_path, monkeypatch, normalization_location,
):
    import yaml

    from pumit.ucpt.stream import build

    cfg = tmp_path / 'gen.yaml'
    cfg.write_text('size_xy_choices: [128]\nseg_positive_queries: 1\n')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'meta.yaml').write_text(
        'stream_complete: true\n'
        'n_shards: 1\n'
        'n_total_records: 999\n'
        'config:\n'
        '  size_xy_choices: [128]\n'
        '  seg_positive_queries: 1\n'
        'composition:\n'
        '  normalization:\n'
        '    contract: fixed-source-v1\n'
        '    source_stats_count: 123\n'
    )
    if normalization_location == 'top':
        source_meta = yaml.safe_load((source / 'meta.yaml').read_text())
        source_meta['normalization'] = source_meta.pop('composition')['normalization']
        (source / 'meta.yaml').write_text(yaml.safe_dump(source_meta))
    requests = []

    def capture(request):
        requests.append(request)
        return _write_ok_shard(request)

    monkeypatch.setattr(build, 'build_one_shard', capture)
    monkeypatch.setattr(
        build,
        '_load_worker_globals',
        lambda config_path: (object(), (([object()], np.ones(1)), ([object()], np.ones(1))), object()),
    )
    build.build_stream(
        config_path=cfg,
        shards=1,
        shard_offset=0,
        output_dir=tmp_path / 'stream',
        seed=42,
        batches_per_shard=1,
        shard_workers=1,
        sample_workers=1,
        sample_prefetch_factor=1,
        budget_ms=100,
        label_budget_fraction=0.5,
        cost_model_path=None,
        use_default_cost_model=True,
        source_stream=source,
        force=False,
    )

    assert requests[0].n_total_records == 1
    plan = yaml.safe_load((tmp_path / 'stream' / 'build.yaml').read_text())
    assert 'normalization' not in plan


def test_repack_discards_source_labeled_and_exactly_reuses_unlabeled(tmp_path, monkeypatch):
    from pumit.ucpt.mask_store import MASK_FRAME_KEY, MASK_REF_KEY
    from pumit.ucpt.stream import build

    source_dir = tmp_path / 'source'
    output_dir = tmp_path / 'output'
    source_dir.mkdir()
    output_dir.mkdir()
    source_unlabeled = [
        {
            'id': f'old-u{i}',
            'img': f'legacy/Test/images/case{i}.npy',
            'labeled': False,
            'n_patches': i + 1,
            'params': [{
                'crop_size': [1, 1, 1],
                'affine': np.eye(4).ravel().tolist(),
                'load_slice_start': [0, 0, 0],
                'load_slice_stop': [1, 1, 1],
            }, {'enabled': True, 'factors': [0.9]}, {'enabled': True, 'factor': 0.5},
                {'enabled': True, 'gammas': [1.4], 'invert': True}, {}],
        }
        for i in range(4)
    ]
    source_batches = [{
        'step_idx': 0,
        'samples': [
            {
                'id': 'old-labeled',
                'labeled': True,
                'n_patches': 1,
                'classes': [{'is_positive': False}],
            },
            *source_unlabeled,
        ],
    }]
    with (source_dir / 'shard_00000.msgpack').open('wb') as file:
        msgpack.pack({'batches': source_batches}, file)

    def fresh_samples(rng, config_path, labeled, **kwargs):
        assert labeled
        for index in itertools.count():
            yield {
                'id': f'fresh-l{index}-{rng.integers(1000000)}',
                'labeled': True,
                'n_patches': 1,
                'classes': [{'is_positive': False, 'target_voxels': 0}],
                'seg_cost_queries': 1,
                'da_enc': 0,
                MASK_FRAME_KEY: b'',
            }

    class UnitCost:
        @staticmethod
        def sample_cost(sample):
            return 1.0

    monkeypatch.setattr(build, '_sample_iter', fresh_samples)
    report = build.build_one_shard(build.ShardBuildRequest(
        shard_id=0,
        shard_rng=np.random.default_rng(42),
        config_path=str(tmp_path / 'gen.yaml'),
        cost_model=UnitCost(),
        budget_ms=4.0,
        label_budget_fraction=0.5,
        batches_per_shard=2,
        output_dir=str(output_dir),
        build_fingerprint='test',
        source_stream=str(source_dir),
        n_total_records=10,
        n_labeled_records=5,
    ))

    with (output_dir / 'shard_00000.msgpack').open('rb') as file:
        output = msgpack.unpack(file, raw=False)
    output_samples = [sample for batch in output['batches'] for sample in batch['samples']]
    output_unlabeled = [sample for sample in output_samples if not sample['labeled']]
    output_labeled = [sample for sample in output_samples if sample['labeled']]

    assert output_unlabeled == source_unlabeled
    assert 'source_latent_row_spans' not in report
    assert all(sample['id'].startswith('fresh-l') for sample in output_labeled)
    expected_rng = np.random.default_rng(42).spawn(2)[0]
    assert [sample['id'] for sample in output_labeled] == [
        f'fresh-l{i}-{expected_rng.integers(1000000)}' for i in range(len(output_labeled))
    ]
    assert all(MASK_FRAME_KEY not in sample and MASK_REF_KEY in sample for sample in output_labeled)
    assert report['old_labeled_samples'] == 1
    assert report['used_old_labeled_samples'] == 0
    assert report['old_unlabeled_samples'] == len(source_unlabeled)
    assert report['used_old_unlabeled_samples'] == 4
    assert report['dropped_old_unlabeled_samples'] == 0
    assert report['old_unlabeled_patches'] == 10
    assert report['used_old_unlabeled_patches'] == 10
    assert report['dropped_old_unlabeled_patches'] == 0
    assert report['logical_latent_rows'] == 10
    assert report['generated_unlabeled_samples'] == 0


def test_shard_threads_share_one_sample_process_pool(tmp_path, monkeypatch):
    from pumit.ucpt.stream import build

    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('WORLD_SIZE', raising=False)

    class _Executor:
        instances = []

        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            assert max_workers == 4
            assert initializer is build._init_sample_worker
            assert len(initargs) == 1
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(build, 'ProcessPoolExecutor', _Executor)
    monkeypatch.setattr(
        build,
        '_load_worker_globals',
        lambda config_path: (object(), (([object()], np.ones(1)), ([object()], np.ones(1))), object()),
    )
    sample_executors = []
    sample_prefetches = []

    def capture(request):
        sample_executors.append(request.sample_executor)
        sample_prefetches.append(request.sample_prefetch)

    monkeypatch.setattr(build, 'build_one_shard', capture)
    cfg = tmp_path / 'gen.yaml'
    cfg.write_text('seg_positive_queries: 1\nseg_negative_queries: 1\n')

    build.build_stream(
        config_path=cfg,
        shards=4,
        shard_offset=10,
        output_dir=tmp_path / 'stream',
        seed=42,
        batches_per_shard=2,
        shard_workers=8,
        sample_workers=4,
        sample_prefetch_factor=3,
        budget_ms=100,
        label_budget_fraction=0.5,
        cost_model_path=None,
        use_default_cost_model=True,
        source_stream=None,
        force=False,
    )

    assert len(_Executor.instances) == 1
    assert len(sample_executors) == 4
    assert all(executor is _Executor.instances[0] for executor in sample_executors)
    assert sample_prefetches == [3] * 4


def test_fingerprint_sensitivity():
    from pumit.ucpt.stream.build import generation_fingerprint

    base = dict(seed=1, batches_per_shard=10, budget_ms=2500.0,
                label_budget_fraction=0.5, cost_model={'coef': [1.0]},
                config={'seg_positive_queries': 16}, source_meta_sha256=None)
    fp = generation_fingerprint(**base)
    assert fp == generation_fingerprint(**base)  # stable
    for key, value in [('seed', 2), ('batches_per_shard', 20),
                       ('budget_ms', 2000.0), ('label_budget_fraction', 0.6),
                       ('cost_model', {'coef': [2.0]}),
                       ('config', {'seg_positive_queries': 8})]:
        assert generation_fingerprint(**{**base, key: value}) != fp, key


def test_build_stream_spawns_rngs_by_absolute_shard_id(tmp_path, monkeypatch):
    observed = {}

    def capture(request):
        observed[request.shard_id] = request.shard_rng.integers(0, 2**63, size=8)
        return _write_ok_shard(request)

    _dispatch(
        tmp_path,
        monkeypatch,
        shard_fn=capture,
        seed=123,
        shards=3,
        shard_offset=7,
    )

    stream_rng = np.random.default_rng(123).spawn(1)[0]
    expected = stream_rng.spawn(10)
    assert observed.keys() == {7, 8, 9}
    for shard_id, values in observed.items():
        assert np.array_equal(values, expected[shard_id].integers(0, 2**63, size=8))


def test_new_sample_rng_tree_does_not_replay_legacy_attempts():
    seed = 123
    shard_id = 7
    legacy = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(shard_id, 1, 0))
    )

    stream_rng = np.random.default_rng(seed).spawn(1)[0]
    shard_rng = stream_rng.spawn(shard_id + 1)[shard_id]
    _, unlabeled_rng = shard_rng.spawn(2)
    sample_rng = unlabeled_rng.spawn(1)[0]

    assert not np.array_equal(
        legacy.integers(0, 2**63, size=8),
        sample_rng.integers(0, 2**63, size=8),
    )


def test_distributed_context_requires_complete_valid_environment():
    from pumit.ucpt.stream.build import distributed_context

    assert distributed_context({}) == (0, 1)
    assert distributed_context({'RANK': '2', 'WORLD_SIZE': '4'}) == (2, 4)
    with pytest.raises(RuntimeError, match='both be set'):
        distributed_context({'RANK': '0'})
    with pytest.raises(ValueError, match='RANK must be'):
        distributed_context({'RANK': '4', 'WORLD_SIZE': '4'})


def test_rank_partitions_are_disjoint_and_cover_global_shards():
    from pumit.ucpt.stream.build import partition_shard_ids

    shard_ids = list(range(10, 23))
    partitions = [partition_shard_ids(shard_ids, rank, 4) for rank in range(4)]

    assert sorted(itertools.chain.from_iterable(partitions)) == shard_ids
    assert sum(len(partition) for partition in partitions) == len(set(itertools.chain.from_iterable(partitions)))
