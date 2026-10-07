import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.multiprocessing as mp
import yaml

from pumit.ucpt.mask_store import MASK_STORAGE, mask_shard_path
from pumit.ucpt.stream import build, latent_backend, latents, manifest, stats
from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT


def _sample(name: str, *, labeled: bool, legacy: bool = False) -> dict:
    sample = {
        "img": name,
        "n_patches": 1,
        "da_enc": 0,
        "params": [{
            "crop_size": [1, 1, 1],
            "affine": np.eye(4).ravel().tolist(),
            "load_slice_start": [0, 0, 0],
            "load_slice_stop": [1, 1, 1],
        }],
    }
    if legacy and labeled:
        sample["label_classes"] = {
            "src": {"positive": ["liver"], "negative": []},
        }
    elif not legacy:
        sample["labeled"] = labeled
    if labeled:
        if not legacy:
            sample["classes"] = [
                {
                    "source": "src",
                    "name": "liver",
                    "is_positive": True,
                    "target_voxels": 1,
                },
            ]
        sample["seg_cost_queries"] = 1
    return sample


class _UnitCostModel:
    def sample_cost(self, sample: dict) -> float:
        return 1.0


def _write_finalized_prefix(stream, n_shards: int = 2) -> None:
    stream.mkdir()
    (stream / 'reports').mkdir()
    (stream / 'latents').mkdir()
    (stream / 'masks').mkdir()
    plan = {'fingerprint': 'build-fingerprint'}
    (stream / 'build.yaml').write_text(yaml.safe_dump(plan))
    rows = []
    for shard_id in range(n_shards):
        shard_path = stream / f'shard_{shard_id:05d}.msgpack'
        mask_path = mask_shard_path(stream, shard_id)
        shard_path.write_bytes(f'shard-{shard_id}'.encode())
        mask_path.write_bytes(b'')
        row = {
            'shard_id': shard_id,
            'build_fingerprint': plan['fingerprint'],
            'output_sha256': manifest.sha256_file(shard_path),
            'mask_storage': MASK_STORAGE,
            'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
            'mask_sha256': manifest.sha256_file(mask_path),
            'mask_storage_bytes': 0,
        }
        rows.append(row)
        (stream / 'reports' / f'shard_{shard_id:05d}.json').write_text(
            json.dumps(row)
        )
        save_file(
            {'latents': torch.zeros(1, 32, dtype=torch.float16)},
            str(stream / 'latents' / f'shard_{shard_id:05d}.safetensors'),
        )
    manifest_path = stream / 'manifest.jsonl'
    manifest_path.write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
    )
    (stream / 'summary.json').write_text('{}\n')
    (stream / 'latent-materialized.json').write_text('{}\n')
    (stream / 'latent-stats.json').write_text('{}\n')
    save_file(
        {'count': torch.tensor(n_shards)},
        str(stream / 'latents' / 'stats.safetensors'),
    )
    (stream / 'READY.json').write_text('{}\n')
    meta = {
        'fingerprint': 'final-fingerprint',
        'build_fingerprint': plan['fingerprint'],
        'stream_complete': True,
        'n_shards': n_shards,
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'manifest_sha256': manifest.sha256_file(manifest_path),
    }
    (stream / 'meta.yaml').write_text(yaml.safe_dump(meta))


def test_msgpack_equal_treats_only_matching_nans_as_equal():
    left = {"spacing": [float("nan"), 1.0], "nested": [{"value": 2}]}
    right = {"spacing": [float("nan"), 1.0], "nested": [{"value": 2}]}

    assert build._msgpack_equal(left, right)
    right["nested"][0]["value"] = 3
    assert not build._msgpack_equal(left, right)


def test_prepare_stream_extension_preserves_prefix_and_reopens_aggregates(tmp_path):
    stream = tmp_path / 'stream'
    _write_finalized_prefix(stream)
    shard_inodes = {
        path.name: path.stat().st_ino
        for path in stream.glob('shard_*.msgpack')
    }
    latent_inodes = {
        path.name: path.stat().st_ino
        for path in (stream / 'latents').glob('shard_*.safetensors')
    }

    result = manifest.prepare_stream_extension(stream, expected_shards=4)

    assert result['status'] == 'reopened'
    backup = stream / '.finalized-prefixes' / '00002-final-fingerprint'
    receipt = json.loads((backup / 'extension.json').read_text())
    assert receipt['source_shards'] == 2
    assert receipt['target_shards'] == 4
    for relative in receipt['artifacts']:
        assert not (stream / relative).exists()
        assert (backup / relative).exists()
    assert {
        path.name: path.stat().st_ino
        for path in stream.glob('shard_*.msgpack')
    } == shard_inodes
    assert {
        path.name: path.stat().st_ino
        for path in (stream / 'latents').glob('shard_*.safetensors')
    } == latent_inodes
    assert (stream / 'build.yaml').exists()
    assert len(list((stream / 'reports').glob('shard_*.json'))) == 2


def test_prepare_stream_extension_is_noop_at_current_size_and_rejects_shrink(tmp_path):
    stream = tmp_path / 'stream'
    _write_finalized_prefix(stream)

    result = manifest.prepare_stream_extension(stream, expected_shards=2)

    assert result['status'] == 'already-finalized'
    assert (stream / 'meta.yaml').exists()
    with pytest.raises(ValueError, match='cannot shrink'):
        manifest.prepare_stream_extension(stream, expected_shards=1)


def test_prepare_stream_extension_resumes_after_receipt_before_unlink(tmp_path):
    stream = tmp_path / 'stream'
    _write_finalized_prefix(stream)
    meta = yaml.safe_load((stream / 'meta.yaml').read_text())
    backup = stream / '.finalized-prefixes' / '00002-final-fingerprint'
    artifacts = [
        'manifest.jsonl',
        'summary.json',
        'latent-materialized.json',
        'latent-stats.json',
        'latents/stats.safetensors',
        'READY.json',
        'meta.yaml',
    ]
    for relative in artifacts:
        source = stream / relative
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.hardlink_to(source)
    manifest.write_json(
        backup / 'extension.json',
        {
            'source_fingerprint': meta['fingerprint'],
            'source_shards': 2,
            'target_shards': 4,
            'artifacts': artifacts,
        },
    )
    (stream / 'manifest.jsonl').unlink()

    result = manifest.prepare_stream_extension(stream, expected_shards=4)

    assert result['status'] == 'reopened'
    assert not (stream / 'meta.yaml').exists()
    assert all(not (stream / relative).exists() for relative in artifacts)


def test_repack_shard_reuses_only_source_unlabeled_and_generates_all_labeled(monkeypatch, tmp_path):
    from pumit.ucpt.mask_store import MASK_FRAME_KEY, MASK_REF_KEY, encode_positive_masks

    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    source_path = source_dir / 'shard_00000.msgpack'
    source_batches = [
        {
            "step_idx": 0,
            "mask_seed": 1,
            "samples": [
                _sample("lab-old", labeled=True),
                _sample("unlab-old-0", labeled=False, legacy=True),
            ],
        },
        {
            "step_idx": 1,
            "mask_seed": 2,
            "samples": [
                _sample("unlab-old-1", labeled=False, legacy=True),
                _sample("unlab-old-2", labeled=False, legacy=True),
            ],
        },
    ]
    source_path.write_bytes(
        msgpack.packb({"batches": source_batches, "total_patches": 4})
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    pipeline = SimpleNamespace(transforms=[SimpleNamespace(fg_cache=None)])
    pools = (([object(), object()], np.ones(2)), ([object()], np.ones(1)))
    monkeypatch.setattr(
        build,
        "_load_worker_globals",
        lambda config_path: (pipeline, pools, object()),
    )
    counters = {True: 0, False: 0}

    def fake_generate_sample(pipeline, pools, class_sampler, rng, labeled):
        index = counters[labeled]
        counters[labeled] += 1
        sample = _sample(f"{'lab' if labeled else 'unlab'}-new-{index}", labeled=labeled)
        if labeled:
            sample[MASK_FRAME_KEY] = encode_positive_masks([
                np.ones((1, 1, 1), dtype=np.bool_),
            ])
        return sample

    monkeypatch.setattr(build, 'generate_sample', fake_generate_sample)
    row = build.build_one_shard(
        build.ShardBuildRequest(
            shard_id=0,
            shard_rng=np.random.default_rng(42).spawn(1)[0],
            config_path='config.yaml',
            cost_model=_UnitCostModel(),
            budget_ms=2.0,
            label_budget_fraction=0.5,
            batches_per_shard=3,
            output_dir=str(output_dir),
            build_fingerprint='test',
            source_stream=str(source_dir),
            n_total_records=2,
            n_labeled_records=1,
            sample_prefetch=1,
        )
    )

    shard = msgpack.unpackb(
        (output_dir / "shard_00000.msgpack").read_bytes(), raw=False
    )
    labeled = [
        sample
        for batch in shard["batches"]
        for sample in batch["samples"]
        if sample["labeled"]
    ]
    unlabeled = [
        sample
        for batch in shard["batches"]
        for sample in batch["samples"]
        if not sample["labeled"]
    ]
    assert [sample["img"] for sample in labeled] == [
        "lab-new-0",
        "lab-new-1",
        "lab-new-2",
    ]
    assert all(MASK_REF_KEY in sample and MASK_FRAME_KEY not in sample for sample in labeled)
    assert [sample["img"] for sample in unlabeled] == [
        "unlab-old-0",
        "unlab-old-1",
        "unlab-old-2",
    ]
    assert row["old_labeled_samples"] == 1
    assert row["used_old_labeled_samples"] == 0
    assert row["used_old_unlabeled_samples"] == 3
    assert row["generated_labeled_samples"] == 3
    assert row["generated_unlabeled_samples"] == 0
    assert row["logical_latent_rows"] == 3


def test_link_latents_uses_hardlinks_only_for_prefix_shards(tmp_path):
    source_latents = tmp_path / "source-latents"
    source_latents.mkdir()
    save_file(
        {"latents": torch.arange(14, dtype=torch.float16).reshape(7, 2).repeat(1, 16)},
        str(source_latents / "shard_00000.safetensors"),
    )
    save_file(
        {"latents": torch.arange(8, dtype=torch.float16).reshape(4, 2).repeat(1, 16)},
        str(source_latents / "shard_00001.safetensors"),
    )
    stream = tmp_path / "stream"
    stream.mkdir()
    rows = [
        {
            "shard_id": 0,
            "old_unlabeled_patches": 6,
            "generated_unlabeled_samples": 0,
        },
        {
            "shard_id": 1,
            "old_unlabeled_patches": 4,
            "generated_unlabeled_samples": 1,
        },
    ]
    (stream / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )

    latents.cmd_link_latents(
        SimpleNamespace(
            source_latent_dir=source_latents,
            stream=stream,
        )
    )

    source_stat = (source_latents / "shard_00000.safetensors").stat()
    linked_stat = (stream / "latents" / "shard_00000.safetensors").stat()
    assert (source_stat.st_dev, source_stat.st_ino) == (
        linked_stat.st_dev,
        linked_stat.st_ino,
    )
    assert not (stream / "latents" / "shard_00001.safetensors").exists()


def _write_filter_link_fixture(tmp_path):
    source_stream = tmp_path / 'source'
    source_latents = source_stream / 'latents'
    source_latents.mkdir(parents=True)
    stream = tmp_path / 'filtered'
    stream.mkdir()
    values = torch.arange(8 * 32, dtype=torch.float16).reshape(8, 32)
    for shard_id, n_rows in enumerate((7, 7, 8)):
        save_file({'latents': values[:n_rows]}, source_latents / f'shard_{shard_id:05d}.safetensors')
    save_file(
        {'count': torch.tensor(3), 'mean': torch.arange(32).float(), 'std': torch.ones(32)},
        source_latents / 'stats.safetensors',
    )
    (source_stream / 'latent-materialized.json').write_text(
        json.dumps({'codec_model': 'flux2', 'codec_checkpoint': '/codec/checkpoint.pt'}),
    )
    normalization = {
        'contract': 'fixed-source-v1',
        'source_stream': str(source_stream),
        'source_fingerprint': 'source-fixture',
        'source_manifest_sha256': 'a' * 64,
        'source_stats_sha256': manifest.sha256_file(source_latents / 'stats.safetensors'),
        'source_stats_count': 3,
    }
    (source_stream / 'meta.yaml').write_text(
        yaml.safe_dump({'composition': {'normalization': normalization}}),
    )
    (stream / 'meta.yaml').write_text(
        yaml.safe_dump(
            {
                'source_stream': str(source_stream),
                'unlabeled_filter': {'contract': 'unlabeled-filter-v1', 'exclude_datasets': ['ISLES22']},
                'normalization': normalization,
            },
        ),
    )
    rows = []
    for shard_id, spans in enumerate(([[0, 7]], [[0, 2], [4, 7]], [[0, 7]])):
        retained = sum(stop - start for start, stop in spans)
        rows.append(
            {
                'shard_id': shard_id,
                'source_latent_row_spans': spans,
                'old_unlabeled_patches': 7,
                'used_old_unlabeled_patches': retained,
                'dropped_old_unlabeled_patches': 7 - retained,
                'logical_latent_rows': retained,
                'generated_unlabeled_samples': 0,
                'generated_unlabeled_patches': 0,
            },
        )
    (stream / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return SimpleNamespace(stream=stream, source_latent_dir=source_latents, workers=2), rows, values


def test_link_filtered_latents_preserves_selected_rows_stats_and_source_on_resume(
    tmp_path, monkeypatch,
):
    args, rows, values = _write_filter_link_fixture(tmp_path)
    monkeypatch.setattr(latent_backend, 'STATS_CHUNK_ROWS', 2)
    source_hashes = {path.name: manifest.sha256_file(path) for path in args.source_latent_dir.iterdir()}
    latent_dir = args.stream / 'latents'
    latent_dir.mkdir()
    (latent_dir / 'shard_00001.safetensors.tmp').write_bytes(b'interrupted-copy')

    latent_backend.cmd_link_latents(args)
    output_hashes = {path.name: manifest.sha256_file(path) for path in latent_dir.iterdir()}
    latent_backend.cmd_link_latents(args)

    assert (latent_dir / 'shard_00000.safetensors').samefile(args.source_latent_dir / 'shard_00000.safetensors')
    for shard_id, expected in ((1, torch.cat([values[:2], values[4:7]])), (2, values[:7])):
        output = latent_dir / f'shard_{shard_id:05d}.safetensors'
        assert not output.samefile(args.source_latent_dir / output.name)
        with safe_open(output, framework='pt') as file:
            assert file.get_slice('latents').get_dtype() == 'F16'
            assert torch.equal(file.get_tensor('latents'), expected)
    assert {path.name: manifest.sha256_file(path) for path in args.source_latent_dir.iterdir()} == source_hashes
    assert {path.name: manifest.sha256_file(path) for path in latent_dir.iterdir()} == output_hashes
    assert output_hashes['stats.safetensors'] == source_hashes['stats.safetensors']
    receipt = json.loads((args.stream / 'latent-materialized.json').read_text())
    assert receipt == {
        'latent_contract': 'reused-latent-v1',
        'codec_model': 'flux2',
        'codec_checkpoint': '/codec/checkpoint.pt',
        'source_latent_dir': str(args.source_latent_dir),
        'source_latent_receipt_sha256': manifest.sha256_file(args.source_latent_dir.parent / 'latent-materialized.json'),
        'logical_latent_rows': sum(row['logical_latent_rows'] for row in rows),
        'materialized_shards': [1, 2],
    }


def test_link_filtered_latents_rejects_changed_completed_copy(tmp_path):
    args, _, _ = _write_filter_link_fixture(tmp_path)
    latent_backend.cmd_link_latents(args)
    output = args.stream / 'latents' / 'shard_00001.safetensors'
    output.unlink()
    save_file({'latents': torch.zeros(5, 32, dtype=torch.float16)}, output)

    with pytest.raises(ValueError, match='reused latent values differ'):
        latent_backend.cmd_link_latents(args)


@pytest.mark.parametrize('spans', [[[0, 2], [2, 5]], [[2, 5], [0, 2]], [[0, 8]], [[0, 0]], [[False, 5]]])
def test_reused_latents_reject_invalid_source_spans(tmp_path, spans):
    _, rows, _ = _write_filter_link_fixture(tmp_path)
    with pytest.raises(ValueError, match='source latent row span'):
        latent_backend._source_latent_row_spans({**rows[1], 'source_latent_row_spans': spans}, 7)


def test_prepare_materialization_rows_skips_complete_and_removes_partial(tmp_path):
    stream = tmp_path / 'stream'
    latent_dir = stream / 'latents'
    latent_dir.mkdir(parents=True)
    rows = [
        {'shard_id': 0, 'logical_latent_rows': 3},
        {'shard_id': 1, 'logical_latent_rows': 5},
    ]
    save_file(
        {'latents': torch.zeros(3, 32, dtype=torch.float16)},
        str(latent_dir / 'shard_00000.safetensors'),
    )
    tmp = latent_dir / 'shard_00001.safetensors.tmp'
    tmp.write_bytes(b'interrupted')

    pending = latent_backend._prepare_materialization_rows(stream, rows)

    assert pending == [rows[1]]
    assert not tmp.exists()


def test_prepare_materialization_rows_rejects_invalid_complete_output(tmp_path):
    stream = tmp_path / 'stream'
    latent_dir = stream / 'latents'
    latent_dir.mkdir(parents=True)
    path = latent_dir / 'shard_00000.safetensors'
    save_file({'latents': torch.zeros(2, 32)}, str(path))

    with pytest.raises(ValueError, match='expected latent shape'):
        latent_backend._prepare_materialization_rows(
            stream,
            [{'shard_id': 0, 'logical_latent_rows': 3}],
        )


def test_select_manifest_rows_supports_resumable_subsets():
    rows = [{'shard_id': shard_id} for shard_id in range(4)]

    selected, full_request = latent_backend._select_manifest_rows(
        rows,
        shard_offset=1,
        shards=2,
    )

    assert [row['shard_id'] for row in selected] == [1, 2]
    assert not full_request
    assert latent_backend._select_manifest_rows(rows, shard_offset=0, shards=None) == (
        rows,
        True,
    )


def test_select_manifest_rows_rejects_out_of_range():
    rows = [{'shard_id': shard_id} for shard_id in range(2)]

    with pytest.raises(ValueError, match='outside the manifest range'):
        latent_backend._select_manifest_rows(rows, shard_offset=1, shards=2)


def test_migration_filter_partitions_completed_shards_by_node(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / 'codec.pt'
    checkpoint.touch()
    source = tmp_path / 'source'
    source_latents = source / 'latents'
    source_latents.mkdir(parents=True)
    (source / 'latent-materialized.json').write_text(
        json.dumps(
            {
                'codec_model': 'flux2',
                'codec_checkpoint': str(checkpoint),
            }
        )
    )
    stream = tmp_path / 'stream'
    latent_dir = stream / 'latents'
    latent_dir.mkdir(parents=True)
    rows = [
        {
            'shard_id': shard_id,
            'migration_contract': 'requested-load-padding-v2',
            'logical_latent_rows': 1,
            'migration_affected_unlabeled_samples': 0,
            'migration_affected_unlabeled_patches': 0,
            'generated_unlabeled_samples': 0,
            'generated_unlabeled_patches': 0,
        }
        for shard_id in range(4)
    ]
    (stream / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )
    for shard_id in (1, 3):
        save_file(
            {'latents': torch.zeros(1, 32, dtype=torch.float16)},
            str(latent_dir / f'shard_{shard_id:05d}.safetensors'),
        )

    distributed = {'initialized': False, 'barriers': 0}
    events = []

    class PreparedLoader:
        def __init__(self, _stream_dir, *, replay_threads):
            assert replay_threads == 8

        def start(self):
            events.append('fork')

        def close(self):
            events.append('close')

    def init_process_group(*args, **kwargs):
        events.append('gloo')
        distributed['initialized'] = True

    def barrier():
        distributed['barriers'] += 1

    def destroy_process_group():
        distributed['initialized'] = False

    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '1')
    monkeypatch.setattr(latents, '_LatentPreparedLoader', PreparedLoader)
    monkeypatch.setattr(latents.dist, 'init_process_group', init_process_group)
    monkeypatch.setattr(
        latents.dist,
        'is_initialized',
        lambda: distributed['initialized'],
    )
    monkeypatch.setattr(latents.dist, 'barrier', barrier)
    monkeypatch.setattr(latents.dist, 'destroy_process_group', destroy_process_group)

    latents.cmd_encode_latents(
        SimpleNamespace(
            filter='migration',
            stream=stream,
            source_latent_dir=source_latents,
            codec_model='flux2',
            codec_checkpoint=checkpoint,
            cache_archive=None,
            shard_offset=0,
            shards=None,
            num_workers=8,
            memory_budget_gb=80,
            compile_mode='default',
        )
    )

    assert distributed == {'initialized': False, 'barriers': 2}
    assert events == ['fork', 'gloo', 'close']
    assert not (stream / 'latent-materialized.json').exists()


def test_latent_compile_cache_live_dir_is_namespaced(tmp_path, monkeypatch):
    root = tmp_path / 'live'
    monkeypatch.setenv('UCPT_LATENT_COMPILE_CACHE_ROOT', str(root))
    first = tmp_path / 'first.tar.zst'
    second = tmp_path / 'second.tar.zst'

    assert latent_backend._compile_cache_live_dir(first).parent == root
    assert latent_backend._compile_cache_live_dir(first) == latent_backend._compile_cache_live_dir(first)
    assert latent_backend._compile_cache_live_dir(first) != latent_backend._compile_cache_live_dir(second)


def test_extract_latent_compile_cache_clears_live_dir(tmp_path, monkeypatch):
    root = tmp_path / 'live'
    monkeypatch.setenv('UCPT_LATENT_COMPILE_CACHE_ROOT', str(root))
    archive = tmp_path / 'cache.tar.zst'
    cache_dir = latent_backend._compile_cache_live_dir(archive)
    cache_dir.mkdir(parents=True)
    stale = cache_dir / 'stale'
    stale.write_text('stale')

    def extract(actual_archive, actual_cache_dir, *, synchronize):
        assert actual_archive == archive
        assert actual_cache_dir == cache_dir
        assert synchronize is False
        assert not stale.exists()
        actual_cache_dir.mkdir()
        return actual_cache_dir, False

    monkeypatch.setattr(latent_backend, 'extract_compile_cache', extract)

    assert latent_backend._extract_latent_compile_cache(archive) == cache_dir


def test_per_shard_compile_cache_archiver_archives_every_submission(
    tmp_path,
    monkeypatch,
):
    calls = []

    def archive(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(latent_backend, 'archive_compile_cache', archive)
    archive_path = tmp_path / 'cache.tar.zst'
    cache_dir = tmp_path / 'cache'
    archiver = latent_backend._PerShardCompileCacheArchiver(archive_path, cache_dir)

    archiver.submit()
    archiver.submit()
    archiver.close()

    assert calls == [
        (
            (archive_path, cache_dir),
            {
                'best_effort': True,
                'zstd_threads': latent_backend._COMPILE_CACHE_ARCHIVE_ZSTD_THREADS,
            },
        ),
        (
            (archive_path, cache_dir),
            {
                'best_effort': True,
                'zstd_threads': latent_backend._COMPILE_CACHE_ARCHIVE_ZSTD_THREADS,
            },
        ),
    ]


def test_latent_gpu_pool_does_not_replace_body_error_with_archiver_error(
    tmp_path,
    monkeypatch,
):
    body_error = RuntimeError('GPU pipeline failed')
    cleanup_error = RuntimeError('archive cleanup failed')
    normal_close_error = RuntimeError('archive close failed')
    close_errors = iter((cleanup_error, normal_close_error))

    class Archiver:
        def __init__(self, *_args):
            pass

        def close(self):
            raise next(close_errors)

    class DeviceQueue:
        def put(self, _device_id):
            pass

    class SpawnContext:
        def Queue(self):
            return DeviceQueue()

    class Pool:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    monkeypatch.setattr(latent_backend.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(latent_backend, 'get_context', lambda _method: SpawnContext())
    monkeypatch.setattr(latent_backend, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(latent_backend, '_PerShardCompileCacheArchiver', Archiver)
    monkeypatch.setattr(
        latent_backend,
        '_extract_latent_compile_cache',
        lambda _archive: tmp_path / 'cache',
    )
    args = SimpleNamespace(
        cache_archive=tmp_path / 'cache.tar.zst',
        codec_checkpoint=tmp_path / 'codec.pt',
        codec_model='flux2',
        compile_mode='default',
        num_workers=1,
    )
    prepared_loader = SimpleNamespace(capacity=4)

    with pytest.raises(RuntimeError) as raised:
        with latent_backend._latent_gpu_pool(
            args,
            rank=0,
            prepared_loader=prepared_loader,
        ):
            raise body_error
    assert raised.value is body_error
    assert body_error.__notes__ == [
        'latent compile-cache archiver cleanup also failed: '
        'RuntimeError: archive cleanup failed'
    ]

    with pytest.raises(RuntimeError) as raised:
        with latent_backend._latent_gpu_pool(
            args,
            rank=0,
            prepared_loader=prepared_loader,
        ):
            pass
    assert raised.value is normal_close_error


def test_encode_suffix_with_no_generated_rows_does_not_require_cuda(tmp_path):
    stream = tmp_path / 'stream'
    stream.mkdir()
    rows = [
        {
            'shard_id': 0,
            'generated_unlabeled_samples': 0,
            'generated_unlabeled_patches': 0,
        }
    ]
    (stream / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )

    latents.cmd_encode_latents(
        SimpleNamespace(
            filter='suffix',
            stream=stream,
            source_latent_dir=tmp_path / 'source-latents',
            codec_model='flux2',
            codec_checkpoint=tmp_path / 'unused.ckpt',
            num_workers=1,
        )
    )

    receipt = json.loads((stream / 'latent-materialized.json').read_text())
    assert receipt['materialized_shards'] == []
    latents.cmd_encode_latents(
        SimpleNamespace(
            filter='suffix',
            stream=stream,
            source_latent_dir=tmp_path / 'source-latents',
            codec_model='flux2',
            codec_checkpoint=tmp_path / 'unused.ckpt',
            num_workers=1,
        )
    )


def test_encode_suffix_skips_valid_completed_rows_without_cuda(tmp_path):
    checkpoint = tmp_path / 'codec.pt'
    checkpoint.touch()
    source = tmp_path / 'source'
    source_latents = source / 'latents'
    source_latents.mkdir(parents=True)
    (source / 'latent-materialized.json').write_text(
        json.dumps(
            {
                'codec_model': 'flux2',
                'codec_checkpoint': str(checkpoint),
            }
        )
    )
    stream = tmp_path / 'stream'
    latent_dir = stream / 'latents'
    latent_dir.mkdir(parents=True)
    row = {
        'shard_id': 0,
        'generated_unlabeled_samples': 1,
        'generated_unlabeled_patches': 2,
        'logical_latent_rows': 5,
    }
    (stream / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
    save_file(
        {'latents': torch.zeros(5, 32, dtype=torch.float16)},
        str(latent_dir / 'shard_00000.safetensors'),
    )
    args = SimpleNamespace(
        filter='suffix',
        stream=stream,
        source_latent_dir=source_latents,
        codec_model='flux2',
        codec_checkpoint=checkpoint,
        memory_budget_gb=1,
        num_workers=1,
        compile_mode='default',
        cache_archive=None,
        shard_offset=0,
        shards=None,
    )

    latents.cmd_encode_latents(args)
    latents.cmd_encode_latents(args)

    receipt = json.loads((stream / 'latent-materialized.json').read_text())
    assert receipt['materialized_shards'] == [0]


def test_stats_reads_only_logical_prefix(tmp_path):
    stream = tmp_path / "stream"
    latent_dir = stream / "latents"
    latent_dir.mkdir(parents=True)
    first = torch.arange(6 * 32, dtype=torch.float16).reshape(6, 32)
    second = torch.arange(4 * 32, dtype=torch.float16).reshape(4, 32) + 1000
    save_file({"latents": first}, str(latent_dir / "shard_00000.safetensors"))
    save_file({"latents": second}, str(latent_dir / "shard_00001.safetensors"))
    rows = [
        {"shard_id": 0, "logical_latent_rows": 4},
        {"shard_id": 1, "logical_latent_rows": 3},
    ]
    (stream / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )

    stats.cmd_stats(SimpleNamespace(stream=stream, workers=2))

    expected = torch.cat((first[:4], second[:3])).float()
    with safe_open(str(latent_dir / "stats.safetensors"), framework="pt") as file:
        assert int(file.get_tensor("count")) == 7
        assert torch.allclose(file.get_tensor("mean"), expected.mean(dim=0))
        assert torch.allclose(file.get_tensor("std"), expected.std(dim=0, correction=0))


def _run_distributed_stats(rank: int, world_size: int, port: int, stream: str) -> None:
    os.environ.update(
        {
            'MASTER_ADDR': '127.0.0.1',
            'MASTER_PORT': str(port),
            'RANK': str(rank),
            'WORLD_SIZE': str(world_size),
        }
    )
    stats.cmd_stats(SimpleNamespace(stream=Path(stream), workers=1))


def test_stats_distributes_shards_across_nodes(tmp_path):
    stream = tmp_path / 'stream'
    latent_dir = stream / 'latents'
    latent_dir.mkdir(parents=True)
    tensors = []
    rows = []
    for shard_id in range(4):
        tensor = torch.arange(2 * 32, dtype=torch.float16).reshape(2, 32) + shard_id * 100
        tensors.append(tensor)
        save_file(
            {'latents': tensor},
            str(latent_dir / f'shard_{shard_id:05d}.safetensors'),
        )
        rows.append({'shard_id': shard_id, 'logical_latent_rows': 2})
    (stream / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]

    mp.spawn(
        _run_distributed_stats,
        args=(2, port, str(stream)),
        nprocs=2,
        join=True,
    )

    expected = torch.cat(tensors).float()
    with safe_open(str(latent_dir / 'stats.safetensors'), framework='pt') as file:
        assert int(file.get_tensor('count')) == 8
        assert torch.allclose(file.get_tensor('mean'), expected.mean(dim=0))
        assert torch.allclose(file.get_tensor('std'), expected.std(dim=0, correction=0))
