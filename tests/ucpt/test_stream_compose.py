import json
import os
from pathlib import Path

from safetensors.torch import save_file
import pytest
import torch
import yaml

from pumit.ucpt.mask_store import MASK_STORAGE, mask_shard_path
from pumit.ucpt.stream import compose as compose_module
from pumit.ucpt.stream.compose import (
    COMPOSITION_CONTRACT,
    JOURNAL_PATH,
    NORMALIZATION_CONTRACT,
    compose_stream,
    composition_fingerprint,
)
from pumit.ucpt.stream.manifest import finalize_stream, load_manifest, sha256_file, write_report
from pumit.ucpt.stream.verify import FULL_LATENT_CONTRACT
from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT


def _row(stream: Path, shard_id: int, fingerprint: str) -> dict:
    shard = stream / f'shard_{shard_id:05d}.msgpack'
    mask = mask_shard_path(stream, shard_id)
    row = {
        'shard_id': shard_id,
        'build_fingerprint': fingerprint,
        'output_sha256': sha256_file(shard),
        'mask_sha256': sha256_file(mask),
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'mask_storage_bytes': mask.stat().st_size,
        'n_total_records': 10,
        'n_labeled_records': 4,
        'old_labeled_samples': 0,
        'old_unlabeled_samples': 0,
        'used_old_labeled_samples': 0,
        'used_old_unlabeled_samples': 0,
        'dropped_old_unlabeled_samples': 0,
        'generated_labeled_samples': 1,
        'generated_unlabeled_samples': 1,
        'old_unlabeled_patches': 0,
        'used_old_unlabeled_patches': 0,
        'dropped_old_unlabeled_patches': 0,
        'generated_labeled_patches': 1,
        'generated_unlabeled_patches': 1,
        'logical_latent_rows': 1,
        'new_labeled_samples': 1,
        'new_unlabeled_samples': 1,
        'mask_labeled_samples': 1,
        'mask_positive_masks': 0,
        'batch_size_histogram': {'2': 1},
        'labeled_per_batch_histogram': {'1': 1},
        'unlabeled_per_batch_histogram': {'1': 1},
    }
    write_report(stream, row)
    return row


def _write_stream(path: Path, *, seed: int, n_shards: int, ready: bool, latents: bool) -> None:
    path.mkdir()
    (path / 'masks').mkdir()
    (path / 'reports').mkdir()
    fingerprint = f'build-{seed}'
    plan = {
        'fingerprint': fingerprint,
        'algorithm': 'ucpt-stream-v4',
        'packing_policy': 'nearest',
        'seed': seed,
        'batches_per_shard': 1,
        'budget_ms': 600.0,
        'label_budget_fraction': 0.75,
        'cost_model_source': 'fixture',
        'cost_model': {'unlab': {'coef': [1.0]}},
        'config': {'fixture': True},
        'source_stream': None,
        'source_meta_sha256': None,
    }
    (path / 'build.yaml').write_text(yaml.safe_dump(plan))
    if latents or ready:
        (path / 'latents').mkdir()
    for shard_id in range(n_shards):
        (path / f'shard_{shard_id:05d}.msgpack').write_bytes(
            f'{path.name}-shard-{shard_id}'.encode()
        )
        mask_shard_path(path, shard_id).write_bytes(b'')
        _row(path, shard_id, fingerprint)
        if latents or ready:
            (path / 'latents' / f'shard_{shard_id:05d}.safetensors').write_bytes(
                f'{path.name}-latent-{shard_id}'.encode()
            )
    finalize_stream(path, expected_shards=n_shards, workers=0)
    if not ready:
        return
    stats = path / 'latents' / 'stats.safetensors'
    save_file(
        {
            'count': torch.tensor(n_shards, dtype=torch.int64),
            'mean': torch.zeros(32),
            'std': torch.ones(32),
        },
        stats,
    )
    (path / 'latent-stats.json').write_text(
        json.dumps({'logical_rows': n_shards, 'mean': [0.0] * 32, 'std': [1.0] * 32})
    )
    # Full-stream shape: the encode-latents receipt carries no latent_contract key;
    # verify derives READY's 'full-v1' from that absence.
    latent_receipt = path / 'latent-materialized.json'
    latent_receipt.write_text(json.dumps({
        'codec_model': 'fixture-codec',
        'codec_checkpoint': '/fixture/checkpoint.pt',
    }))
    meta = yaml.safe_load((path / 'meta.yaml').read_text())
    (path / 'READY.json').write_text(
        json.dumps({
            'fingerprint': meta['fingerprint'],
            'manifest_sha256': meta['manifest_sha256'],
            'verified_shards': n_shards,
            'logical_latent_rows': n_shards,
            'latent_shard_sha256': [
                sha256_file(path / 'latents' / f'shard_{shard_id:05d}.safetensors')
                for shard_id in range(n_shards)
            ],
            'stats_sha256': sha256_file(stats),
            'latent_contract': FULL_LATENT_CONTRACT,
            'latent_receipt_sha256': sha256_file(latent_receipt),
        })
    )


def test_load_finalized_rejects_prefix_contract_mismatch(tmp_path):
    prefix = tmp_path / 'prefix'
    _write_stream(prefix, seed=42, n_shards=2, ready=True, latents=True)
    receipt_path = prefix / 'latent-materialized.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['latent_contract'] = 'canonical-inplane-latent-v1'
    receipt_path.write_text(json.dumps(receipt))
    ready_path = prefix / 'READY.json'
    ready = json.loads(ready_path.read_text())
    ready['latent_receipt_sha256'] = sha256_file(receipt_path)
    ready_path.write_text(json.dumps(ready))

    with pytest.raises(ValueError, match='READY latent contract differs'):
        compose_module._load_finalized(prefix, require_ready=True)


def test_compose_stream_hardlinks_prefix_and_consumes_suffix(tmp_path):
    prefix = tmp_path / 'prefix'
    suffix = tmp_path / 'suffix'
    target = tmp_path / 'combined'
    _write_stream(prefix, seed=42, n_shards=2, ready=True, latents=True)
    _write_stream(suffix, seed=43, n_shards=2, ready=False, latents=True)

    prefix_inodes = {
        'shard': (prefix / 'shard_00000.msgpack').stat().st_ino,
        'mask': mask_shard_path(prefix, 0).stat().st_ino,
        'latent': (prefix / 'latents' / 'shard_00000.safetensors').stat().st_ino,
        'stats': (prefix / 'latents' / 'stats.safetensors').stat().st_ino,
    }
    suffix_inodes = {
        'shard': (suffix / 'shard_00000.msgpack').stat().st_ino,
        'mask': mask_shard_path(suffix, 0).stat().st_ino,
        'latent': (suffix / 'latents' / 'shard_00000.safetensors').stat().st_ino,
    }

    result = compose_stream(
        prefix_stream=prefix,
        suffix_stream=suffix,
        target_stream=target,
        workers=0,
    )

    staging = Path(result['staging_stream'])
    assert result | {'summary': None} == {
        'status': 'staged',
        'staging_stream': str(staging),
        'target_stream': str(target),
        'prefix_shards': 2,
        'suffix_shards': 2,
        'summary': None,
    }
    assert not suffix.exists()
    assert not target.exists()
    assert staging.is_dir()
    assert [path.name for path in sorted(staging.glob('shard_*.msgpack'))] == [
        f'shard_{shard_id:05d}.msgpack' for shard_id in range(4)
    ]
    assert [path.name for path in sorted((staging / 'masks').glob('shard_*.bin'))] == [
        f'shard_{shard_id:05d}.bin' for shard_id in range(4)
    ]
    assert [
        path.name for path in sorted((staging / 'latents').glob('shard_*.safetensors'))
    ] == [f'shard_{shard_id:05d}.safetensors' for shard_id in range(4)]
    assert (staging / 'shard_00000.msgpack').stat().st_ino == prefix_inodes['shard']
    assert mask_shard_path(staging, 0).stat().st_ino == prefix_inodes['mask']
    assert (staging / 'latents' / 'shard_00000.safetensors').stat().st_ino == prefix_inodes['latent']
    assert (staging / 'latents' / 'stats.safetensors').stat().st_ino == prefix_inodes['stats']
    assert (staging / 'shard_00002.msgpack').stat().st_ino == suffix_inodes['shard']
    assert mask_shard_path(staging, 2).stat().st_ino == suffix_inodes['mask']
    assert (staging / 'latents' / 'shard_00002.safetensors').stat().st_ino == suffix_inodes['latent']

    archive = staging / '.components' / 'suffix'
    assert (archive / 'build.yaml').is_file()
    assert (archive / 'meta.yaml').is_file()
    assert (archive / 'manifest.jsonl').is_file()
    assert (archive / 'summary.json').is_file()
    assert len(list((archive / 'reports').glob('shard_*.json'))) == 2

    plan = yaml.safe_load((staging / 'build.yaml').read_text())
    assert plan['composition']['contract'] == COMPOSITION_CONTRACT
    assert plan['composition']['normalization']['contract'] == NORMALIZATION_CONTRACT
    assert plan['composition']['segments'][1]['target_shard_start'] == 2
    assert plan['composition']['segments'][0]['source_ready_sha256'] == sha256_file(
        prefix / 'READY.json'
    )
    assert plan['source_stream'] is None
    assert plan['seed'] == 42
    assert plan['fingerprint'] == composition_fingerprint(plan['composition'])
    meta = yaml.safe_load((staging / 'meta.yaml').read_text())
    assert meta['composition'] == plan['composition']
    rows = load_manifest(staging)
    assert [row['shard_id'] for row in rows] == [0, 1, 2, 3]
    assert {row['build_fingerprint'] for row in rows} == {plan['fingerprint']}
    stats_receipt = json.loads((staging / 'latent-stats.json').read_text())
    assert stats_receipt['logical_rows'] == 2
    assert stats_receipt['normalization'] == plan['composition']['normalization']
    journal = [json.loads(line) for line in (staging / JOURNAL_PATH).read_text().splitlines()]
    assert journal[0]['event'] == 'suffix-renamed'
    assert journal[-1]['event'] == 'composite-finalized'

    resumed = compose_stream(
        prefix_stream=prefix,
        suffix_stream=suffix,
        target_stream=target,
        workers=0,
    )
    assert resumed['status'] == 'staged'


def test_compose_stream_resumes_after_suffix_move_interruption(tmp_path, monkeypatch):
    prefix = tmp_path / 'prefix'
    suffix = tmp_path / 'suffix'
    target = tmp_path / 'combined'
    _write_stream(prefix, seed=42, n_shards=2, ready=True, latents=True)
    _write_stream(suffix, seed=43, n_shards=2, ready=False, latents=False)
    original_move = compose_module._move_by_link
    injected = False

    def fail_after_one_suffix_move(source, destination, **kwargs):
        nonlocal injected
        original_move(source, destination, **kwargs)
        if not injected and source.suffix == '.msgpack':
            injected = True
            raise RuntimeError('injected suffix move failure')

    monkeypatch.setattr(compose_module, '_move_by_link', fail_after_one_suffix_move)
    with pytest.raises(RuntimeError, match='injected suffix move failure'):
        compose_stream(
            prefix_stream=prefix,
            suffix_stream=suffix,
            target_stream=target,
            workers=0,
        )

    staging = target.with_name(f'{target.name}.staging')
    assert not suffix.exists()
    assert staging.is_dir()
    assert (staging / 'shard_00003.msgpack').is_file()
    monkeypatch.setattr(compose_module, '_move_by_link', original_move)

    result = compose_stream(
        prefix_stream=prefix,
        suffix_stream=suffix,
        target_stream=target,
        workers=0,
    )

    assert result['status'] == 'staged'
    assert len(load_manifest(staging)) == 4
    assert [
        path.name for path in sorted((staging / 'latents').glob('shard_*.safetensors'))
    ] == ['shard_00000.safetensors', 'shard_00001.safetensors']


def test_move_by_link_finishes_same_inode_interruption(tmp_path):
    source = tmp_path / 'source'
    destination = tmp_path / 'destination'
    source.write_bytes(b'artifact')
    os.link(source, destination)
    stat = source.stat()

    compose_module._move_by_link(
        source,
        destination,
        expected_inode=(stat.st_dev, stat.st_ino),
    )

    assert not source.exists()
    assert destination.read_bytes() == b'artifact'


def test_composition_fingerprint_ignores_source_paths():
    composition = {
        'contract': COMPOSITION_CONTRACT,
        'replay_seed': 42,
        'segments': [{
            'target_shard_start': 0,
            'target_shard_stop': 1,
            'source_stream': '/first/path',
            'source_fingerprint': 'stream',
            'source_manifest_sha256': 'manifest',
            'source_shard_start': 0,
            'source_shard_stop': 1,
            'generation_seed': 42,
        }],
        'normalization': {
            'contract': NORMALIZATION_CONTRACT,
            'source_stream': '/first/path',
            'source_fingerprint': 'stream',
            'source_manifest_sha256': 'manifest',
            'source_stats_sha256': 'stats',
            'source_stats_count': 1,
        },
    }
    relocated = json.loads(json.dumps(composition))
    relocated['segments'][0]['source_stream'] = '/other/path'
    relocated['normalization']['source_stream'] = '/other/path'

    assert composition_fingerprint(composition) == composition_fingerprint(relocated)
