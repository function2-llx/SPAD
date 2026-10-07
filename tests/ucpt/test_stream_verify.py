import json
import os
from copy import deepcopy
from pathlib import Path
import socket
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from safetensors.torch import save_file
import torch
import torch.multiprocessing as mp
import yaml

from pumit.ucpt.mask_store import MASK_STORAGE, encode_positive_masks, mask_shard_path
from pumit.ucpt.packing import CostModel
from pumit.ucpt.stream import latent_backend, manifest, verify
from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT, canonicalize_sample_metadata


class _UnitCostModel:
    def sample_cost(self, sample: dict) -> float:
        return 1.0


def test_canonical_json_normalizes_integer_object_keys():
    assert verify._canonical_json({2: 1, 10: 3}) == verify._canonical_json(
        {'2': 1, '10': 3}
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_source_provenance(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / 'codec.pt'
    checkpoint.touch()
    source_stream = tmp_path / 'source-stream'
    source_stream.mkdir()
    source_manifest = source_stream / manifest.MANIFEST_NAME
    source_manifest.write_text('{}\n')
    (source_stream / 'meta.yaml').write_text(
        yaml.safe_dump({'manifest_sha256': manifest.sha256_file(source_manifest)})
    )
    source_latent_dir = tmp_path / 'source-latents' / 'latents'
    source_latent_dir.mkdir(parents=True)
    _write_json(
        source_latent_dir.parent / 'latent-materialized.json',
        {
            'codec_model': 'flux2',
            'codec_checkpoint': str(checkpoint.resolve()),
        },
    )
    return source_stream.resolve(), source_latent_dir.resolve(), checkpoint.resolve()


def _canonical_receipt(
    source_stream: Path,
    source_latent_dir: Path,
    checkpoint: Path,
) -> dict:
    source_receipt = source_latent_dir.parent / 'latent-materialized.json'
    return {
        'latent_contract': verify.CANONICAL_INPLANE_LATENT_CONTRACT,
        'codec_model': 'flux2',
        'codec_checkpoint': str(checkpoint),
        'source_stream': str(source_stream),
        'source_manifest_sha256': manifest.sha256_file(
            source_stream / manifest.MANIFEST_NAME
        ),
        'source_latent_dir': str(source_latent_dir),
        'source_latent_receipt_sha256': manifest.sha256_file(source_receipt),
        'materialized_shards': [0, 1],
        'logical_latent_rows': 12,
        'affected_unlabeled_samples': 3,
        'affected_latent_rows': 7,
    }


def test_validate_canonical_inplane_latent_receipt_binds_sources_and_counts(tmp_path):
    source_stream, source_latent_dir, checkpoint = _write_source_provenance(tmp_path)
    stream = tmp_path / 'stream'
    stream.mkdir()
    rows = [
        {'shard_id': 0, 'logical_latent_rows': 5},
        {'shard_id': 1, 'logical_latent_rows': 7},
    ]
    receipt = _canonical_receipt(source_stream, source_latent_dir, checkpoint)
    _write_json(stream / 'latent-materialized.json', receipt)

    contract, receipt_path = verify._validate_latent_receipt(
        stream.resolve(),
        source_stream,
        source_latent_dir,
        rows,
        affected_unlabeled_samples=3,
        affected_latent_rows=7,
    )

    assert contract == verify.CANONICAL_INPLANE_LATENT_CONTRACT
    assert receipt_path == stream.resolve() / 'latent-materialized.json'

    receipt['affected_latent_rows'] = 6
    (stream / 'latent-materialized.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='affected_latent_rows'):
        verify._validate_latent_receipt(
            stream.resolve(),
            source_stream,
            source_latent_dir,
            rows,
            affected_unlabeled_samples=3,
            affected_latent_rows=7,
        )


def test_validate_canonical_inplane_latent_receipt_rejects_unknown_fields(tmp_path):
    source_stream, source_latent_dir, checkpoint = _write_source_provenance(tmp_path)
    stream = tmp_path / 'stream'
    stream.mkdir()
    receipt = _canonical_receipt(source_stream, source_latent_dir, checkpoint)
    receipt['unexpected'] = True
    _write_json(stream / 'latent-materialized.json', receipt)

    with pytest.raises(ValueError, match=r"unexpected=\['unexpected'\]"):
        verify._validate_latent_receipt(
            stream.resolve(),
            source_stream,
            source_latent_dir,
            [
                {'shard_id': 0, 'logical_latent_rows': 5},
                {'shard_id': 1, 'logical_latent_rows': 7},
            ],
            affected_unlabeled_samples=3,
            affected_latent_rows=7,
        )


def test_validate_suffix_and_migration_latent_receipts(tmp_path):
    source_stream, source_latent_dir, checkpoint = _write_source_provenance(tmp_path)
    rows = [
        {
            'shard_id': 0,
            'logical_latent_rows': 5,
            'generated_unlabeled_samples': 0,
            'generated_unlabeled_patches': 0,
        },
        {
            'shard_id': 1,
            'logical_latent_rows': 7,
            'generated_unlabeled_samples': 2,
            'generated_unlabeled_patches': 3,
        },
    ]
    suffix_stream = tmp_path / 'suffix-stream'
    suffix_stream.mkdir()
    _write_json(
        suffix_stream / 'latent-materialized.json',
        {
            'codec_model': 'flux2',
            'codec_checkpoint': str(checkpoint),
            'materialized_shards': [1],
            'generated_unlabeled_samples': 2,
            'generated_unlabeled_patches': 3,
        },
    )
    _write_json(
        suffix_stream / 'latent-links.json',
        {
            'source_latent_dir': str(source_latent_dir),
            'hardlinked_shards': [0],
            'materialize_shards': [1],
        },
    )
    assert verify._validate_latent_receipt(
        suffix_stream.resolve(),
        source_stream,
        source_latent_dir,
        rows,
        affected_unlabeled_samples=0,
        affected_latent_rows=0,
    )[0] == 'suffix-v1'
    with pytest.raises(ValueError, match='canonical in-plane metadata requires'):
        verify._validate_latent_receipt(
            suffix_stream.resolve(),
            source_stream,
            source_latent_dir,
            rows,
            affected_unlabeled_samples=1,
            affected_latent_rows=2,
        )

    migration_stream = tmp_path / 'migration-stream'
    migration_stream.mkdir()
    migration_rows = [
        {
            **row,
            'migration_contract': 'requested-load-padding-v3',
            'migration_affected_unlabeled_samples': shard_id + 1,
            'migration_affected_unlabeled_patches': shard_id + 2,
        }
        for shard_id, row in enumerate(rows)
    ]
    _write_json(
        migration_stream / 'latent-materialized.json',
        {
            'codec_model': 'flux2',
            'codec_checkpoint': str(checkpoint),
            'source_latent_dir': str(source_latent_dir),
            'migration_contract': 'requested-load-padding-v3',
            'materialized_shards': [0, 1],
            'affected_unlabeled_samples': 3,
            'affected_unlabeled_patches': 5,
            'generated_unlabeled_samples': 2,
            'generated_unlabeled_patches': 3,
        },
    )
    assert verify._validate_latent_receipt(
        migration_stream.resolve(),
        source_stream,
        source_latent_dir,
        migration_rows,
        affected_unlabeled_samples=0,
        affected_latent_rows=0,
    )[0] == 'requested-load-padding-v3'

    migration_receipt_path = migration_stream / 'latent-materialized.json'
    migration_receipt = json.loads(migration_receipt_path.read_text())
    migration_receipt['migration_contract'] = 'unknown-migration'
    migration_receipt_path.write_text(json.dumps(migration_receipt))
    unknown_rows = [
        {**row, 'migration_contract': 'unknown-migration'} for row in migration_rows
    ]
    with pytest.raises(ValueError, match='unsupported migration latent contract'):
        verify._validate_latent_receipt(
            migration_stream.resolve(),
            source_stream,
            source_latent_dir,
            unknown_rows,
            affected_unlabeled_samples=0,
            affected_latent_rows=0,
        )


def _strict_inplane_affine() -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _write_verify_shard(tmp_path: Path) -> tuple[tuple, Path, Path]:
    source_stream = tmp_path / 'source-stream'
    stream = tmp_path / 'stream'
    source_latent_dir = tmp_path / 'source-latents'
    source_stream.mkdir()
    stream.mkdir()
    source_latent_dir.mkdir()
    affine = np.asarray(_strict_inplane_affine()).reshape(4, 4)
    affine[0, 1] = 1e-16
    affine[2, 0] = -1e-16
    source_spatial = {
        'crop_size': [1, 1, 1],
        'affine': affine.ravel().tolist(),
        'load_slice_start': [0, 0, 0],
        'load_slice_stop': [1, 1, 1],
    }
    target_spatial = canonicalize_sample_metadata({'params': [source_spatial]})[0]['params'][0]
    source_labeled = {
        'img': 'old-labeled',
        'n_patches': 1,
        'label_classes': {'source': {'positive': ['class'], 'negative': []}},
    }
    source_unlabeled = {
        'img': 'unlabeled',
        'n_patches': 1,
        'da_enc': 0,
        'params': [source_spatial],
    }
    output_labeled = {
        'img': 'new-labeled',
        'n_patches': 1,
        'da_enc': 0,
        'seg_cost_queries': 1,
        'labeled': True,
        'classes': [{'source': 'source', 'name': 'class', 'is_positive': True}],
        'params': [target_spatial],
    }
    output_unlabeled = {
        **source_unlabeled,
        'params': [target_spatial],
        'labeled': False,
    }
    source_shard = {
        'batches': [{'step_idx': 0, 'samples': [source_labeled, source_unlabeled]}],
        'total_patches': 2,
    }
    output_shard = {
        'batches': [{'step_idx': 0, 'samples': [output_labeled, output_unlabeled]}],
        'total_patches': 2,
    }
    source_bytes = msgpack.packb(source_shard)
    output_bytes = msgpack.packb(output_shard)
    (source_stream / 'shard_00000.msgpack').write_bytes(source_bytes)
    (stream / 'shard_00000.msgpack').write_bytes(output_bytes)
    mask_path = mask_shard_path(stream, 0)
    mask_path.parent.mkdir()
    mask_path.write_bytes(b'mask-frame')
    row = {
        'shard_id': 0,
        'source_sha256': manifest.sha256_bytes(source_bytes),
        'output_sha256': manifest.sha256_bytes(output_bytes),
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'mask_sha256': manifest.sha256_file(mask_path),
        'mask_storage_bytes': mask_path.stat().st_size,
        'old_labeled_samples': 1,
        'used_old_labeled_samples': 0,
        'generated_labeled_samples': 1,
        'used_old_unlabeled_samples': 1,
        'generated_unlabeled_samples': 0,
        'dropped_old_unlabeled_samples': 0,
        'new_labeled_samples': 1,
        'new_unlabeled_samples': 1,
        'logical_latent_rows': 1,
        'old_unlabeled_patches': 1,
    }
    _write_json(stream / 'reports' / 'shard_00000.json', row)
    save_file(
        {'latents': torch.zeros(1, 32, dtype=torch.float16)},
        str(source_latent_dir / 'shard_00000.safetensors'),
    )
    output_latent = stream / 'latents' / 'shard_00000.safetensors'
    output_latent.parent.mkdir()
    save_file(
        {'latents': torch.ones(1, 32, dtype=torch.float16)},
        str(output_latent),
    )
    args = (
        str(source_stream),
        str(stream),
        str(source_latent_dir),
        row,
        _UnitCostModel(),
        2.0,
        1,
    )
    return args, mask_path, output_latent


def _write_filtered_verify_shard(tmp_path):
    from pumit.ucpt.stream.filtering import filter_unlabeled_batches

    args, _, output_latent = _write_verify_shard(tmp_path)
    source_stream, stream, source_latents = map(Path, args[:3])
    source_path = source_stream / 'shard_00000.msgpack'
    output = msgpack.unpackb((stream / 'shard_00000.msgpack').read_bytes(), raw=False)
    labeled, template = output['batches'][0]['samples']
    kept = {**deepcopy(template), 'img': '/preprocess/MRI/images/kept.npy', 'n_patches': 2}
    excluded = {**deepcopy(template), 'img': '/preprocess/ISLES22/images/excluded.npy'}
    display = {**deepcopy(template), 'img': '/preprocess/Display/images/case.npy'}
    for sample in (kept, excluded, display):
        sample['params'].append({'enabled': True, 'scale': 1.3})
    source = {}
    labeled_second = deepcopy(labeled)
    labeled_second['img'] = 'second-labeled'
    frame = encode_positive_masks([np.ones((1, 1, 1), dtype=np.bool_)])
    labeled['positive_mask_ref'] = [0, len(frame)]
    labeled_second['positive_mask_ref'] = [len(frame), len(frame)]
    source['batches'] = [
        {'step_idx': 0, 'samples': [labeled, kept, excluded]},
        {'step_idx': 1, 'samples': [labeled_second, display]},
    ]
    source_path.write_bytes(msgpack.packb(source))
    source_mask = mask_shard_path(source_stream, 0)
    source_mask.parent.mkdir()
    source_mask.write_bytes(frame * 2)
    output_mask = mask_shard_path(stream, 0)
    output_mask.unlink()
    os.link(source_mask, output_mask)
    batches, spans = filter_unlabeled_batches(source['batches'], ['ISLES22'])
    output_path = stream / 'shard_00000.msgpack'
    output_path.write_bytes(msgpack.packb({'batches': batches, 'total_patches': 5}))
    row = {
        **args[3],
        'source_sha256': manifest.sha256_file(source_path),
        'output_sha256': manifest.sha256_file(output_path),
        'old_labeled_samples': 2,
        'old_unlabeled_samples': 3,
        'used_old_unlabeled_samples': 2,
        'dropped_old_unlabeled_samples': 1,
        'old_unlabeled_patches': 4,
        'new_unlabeled_samples': 2,
        'generated_unlabeled_patches': 0,
        'used_old_labeled_samples': 2,
        'new_labeled_samples': 2,
        'generated_labeled_samples': 0,
        'generated_labeled_patches': 0,
        'used_old_unlabeled_patches': 3,
        'dropped_old_unlabeled_patches': 1,
        'logical_latent_rows': 3,
        'source_latent_row_spans': spans,
        'mask_storage_bytes': output_mask.stat().st_size,
        'mask_sha256': manifest.sha256_file(output_mask),
    }
    _write_json(stream / 'reports' / 'shard_00000.json', row)
    (stream / 'meta.yaml').write_text(
        yaml.safe_dump({'unlabeled_filter': {'contract': 'unlabeled-filter-v1', 'exclude_datasets': ['ISLES22']}}),
    )
    save_file(
        {'latents': torch.arange(4 * 32, dtype=torch.float16).reshape(4, 32)},
        source_latents / output_latent.name,
    )
    output_latent.unlink()
    latent_backend._reuse_latent_shard(source_latents, output_latent.parent, row)
    return (*args[:3], row, args[4], 10.0, 2), output_latent


def test_verify_unlabeled_filter_preserves_batches_masks_and_frozen_intensity(tmp_path):
    args, _ = _write_filtered_verify_shard(tmp_path)
    result = verify._verify_one_shard(args)
    assert result['logical_latent_rows'] == 3
    assert result['affected_unlabeled_samples'] == 0
    assert args[3]['source_latent_row_spans'] == [[0, 2], [3, 4]]


@pytest.mark.parametrize('mutation', ['labeled', 'intensity', 'batch', 'mask_reference'])
def test_verify_unlabeled_filter_rejects_changed_retained_samples(tmp_path, mutation):
    args, _ = _write_filtered_verify_shard(tmp_path)
    stream = Path(args[1])
    output_path = stream / 'shard_00000.msgpack'
    output = msgpack.unpackb(output_path.read_bytes(), raw=False)
    first_batch, second_batch = output['batches']
    if mutation == 'labeled':
        first_batch['samples'][0]['classes'][0]['name'] = 'changed-class'
    elif mutation == 'intensity':
        first_batch['samples'][1]['params'][1]['scale'] = 0.75
    elif mutation == 'batch':
        first_batch['samples'][1], second_batch['samples'][1] = second_batch['samples'][1], first_batch['samples'][1]
    else:
        first_batch['samples'][0]['positive_mask_ref'][0] = 1
    output_path.write_bytes(msgpack.packb(output))
    args[3]['output_sha256'] = manifest.sha256_file(output_path)
    _write_json(stream / 'reports' / 'shard_00000.json', args[3])
    with pytest.raises(ValueError, match='filtered batches differ from source'):
        verify._verify_one_shard(args)


def test_verify_unlabeled_filter_requires_the_source_mask_file(tmp_path):
    args, _ = _write_filtered_verify_shard(tmp_path)
    mask_path = mask_shard_path(Path(args[1]), 0)
    copied = mask_path.read_bytes()
    mask_path.unlink()
    mask_path.write_bytes(copied)
    with pytest.raises(ValueError, match='mask sidecar must be a source hardlink'):
        verify._verify_one_shard(args)


def _write_full_verify_shard(
    tmp_path: Path,
    *,
    cost_model=None,
) -> tuple[tuple, Path, Path]:
    source_args, mask_path, output_latent = _write_verify_shard(tmp_path)
    stream = Path(source_args[1])
    row = {
        **source_args[3],
        'source_sha256': None,
        'old_labeled_samples': 0,
        'old_unlabeled_samples': 0,
        'used_old_unlabeled_samples': 0,
        'dropped_old_unlabeled_samples': 0,
        'generated_unlabeled_samples': 1,
        'old_unlabeled_patches': 0,
        'used_old_unlabeled_patches': 0,
        'dropped_old_unlabeled_patches': 0,
        'generated_labeled_patches': 1,
        'generated_unlabeled_patches': 1,
    }
    cost_model = cost_model or _UnitCostModel()
    with (stream / 'shard_00000.msgpack').open('rb') as file:
        shard = msgpack.unpack(file, raw=False)
    row.update(verify._validate_batches(shard['batches'], cost_model, 1000.0, 0))
    (stream / 'reports' / 'shard_00000.json').write_text(json.dumps(row))
    return (
        None,
        str(stream),
        None,
        row,
        cost_model,
        1000.0,
        1,
    ), mask_path, output_latent


def test_verify_one_shard_accepts_distinct_strict_latent_and_binds_mask(tmp_path):
    args, mask_path, output_latent = _write_verify_shard(tmp_path)
    source_latent = tmp_path / 'source-latents' / output_latent.name
    assert source_latent.stat().st_ino != output_latent.stat().st_ino

    result = verify._verify_one_shard(args)

    assert result['affected_unlabeled_samples'] == 1
    assert result['affected_latent_rows'] == 1
    assert result['latent_sha256'] == manifest.sha256_file(output_latent)

    mask_path.write_bytes(b'mask-flame')
    with pytest.raises(ValueError, match='mask hash differs'):
        verify._verify_one_shard(args)

    mask_path.write_bytes(b'changed-mask-frame')
    with pytest.raises(ValueError, match='mask size differs'):
        verify._verify_one_shard(args)


def test_verify_one_shard_rejects_report_or_latent_schema_mismatch(tmp_path):
    args, _, output_latent = _write_verify_shard(tmp_path)
    stream = tmp_path / 'stream'
    report_path = stream / 'reports' / 'shard_00000.json'
    report = json.loads(report_path.read_text())
    report['logical_latent_rows'] = 2
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='manifest row differs'):
        verify._verify_one_shard(args)

    report_path.write_text(json.dumps(args[3]))
    output_latent.unlink()
    save_file(
        {
            'latents': torch.ones(1, 32, dtype=torch.float16),
            'unexpected': torch.ones(1),
        },
        str(output_latent),
    )
    with pytest.raises(ValueError, match='expected only a latents tensor'):
        verify._verify_one_shard(args)

    output_latent.unlink()
    save_file(
        {'latents': torch.ones(1, 32, dtype=torch.float32)},
        str(output_latent),
    )
    with pytest.raises(ValueError, match='expected F16 latents'):
        verify._verify_one_shard(args)

    output_latent.unlink()
    save_file(
        {'latents': torch.ones(2, 32, dtype=torch.float16)},
        str(output_latent),
    )
    with pytest.raises(ValueError, match=r'expected latent shape \[1, 32\]'):
        verify._verify_one_shard(args)


def test_verify_one_full_shard_validates_self_contained_counts_and_cost(tmp_path):
    args, _, output_latent = _write_full_verify_shard(tmp_path)

    result = verify._verify_one_shard(args)

    assert result['affected_unlabeled_samples'] == 0
    assert result['affected_latent_rows'] == 0
    assert result['latent_sha256'] == manifest.sha256_file(output_latent)

    row = args[3]
    row['generated_unlabeled_samples'] = 0
    (Path(args[1]) / 'reports' / 'shard_00000.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='used/generated unlabeled counts differ'):
        verify._verify_one_shard(args)


def test_verify_one_full_shard_rejects_negative_manifest_counts(tmp_path):
    args, _, _ = _write_full_verify_shard(tmp_path)
    row = args[3]
    row['used_old_unlabeled_samples'] = -1
    row['generated_unlabeled_samples'] = 2
    row['dropped_old_unlabeled_samples'] = 1
    (Path(args[1]) / 'reports' / 'shard_00000.json').write_text(json.dumps(row))

    with pytest.raises(ValueError, match='must be a non-negative integer'):
        verify._verify_one_shard(args)


def test_validate_full_latent_receipt_requires_every_target_row(tmp_path):
    stream = tmp_path / 'stream'
    stream.mkdir()
    rows = [
        {'shard_id': 0, 'new_unlabeled_samples': 2, 'logical_latent_rows': 5},
        {'shard_id': 1, 'new_unlabeled_samples': 3, 'logical_latent_rows': 7},
    ]
    receipt = {
        'codec_model': 'flux2',
        'codec_checkpoint': str(tmp_path / 'codec.pt'),
        'materialized_shards': [0, 1],
        'generated_unlabeled_samples': 5,
        'generated_unlabeled_patches': 12,
    }
    _write_json(stream / 'latent-materialized.json', receipt)

    contract, _ = verify._validate_latent_receipt(
        stream,
        None,
        None,
        rows,
        affected_unlabeled_samples=0,
        affected_latent_rows=0,
    )
    assert contract == verify.FULL_LATENT_CONTRACT

    receipt['materialized_shards'] = [1]
    (stream / 'latent-materialized.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='materialized_shards'):
        verify._validate_latent_receipt(
            stream,
            None,
            None,
            rows,
            affected_unlabeled_samples=0,
            affected_latent_rows=0,
        )


def test_validate_mask_inventory_requires_exact_manifest_shard_set(tmp_path):
    stream = tmp_path / 'stream'
    (stream / 'masks').mkdir(parents=True)
    (stream / 'reports').mkdir()
    rows = [{'shard_id': 0}, {'shard_id': 1}]
    for shard_id in range(2):
        mask_shard_path(stream, shard_id).touch()
        (stream / 'reports' / f'shard_{shard_id:05d}.json').touch()
    meta = {
        'n_shards': 2,
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
    }
    verify._validate_mask_inventory(stream, rows, meta)

    mask_shard_path(stream, 1).unlink()
    with pytest.raises(ValueError, match='mask shards are incomplete'):
        verify._validate_mask_inventory(stream, rows, meta)


def _write_verify_command_fixture(tmp_path: Path) -> SimpleNamespace:
    worker_args, _, _ = _write_verify_shard(tmp_path)
    source_stream = Path(worker_args[0]).resolve()
    stream = Path(worker_args[1]).resolve()
    source_latent_dir = Path(worker_args[2]).resolve()
    row = worker_args[3]

    source_manifest = source_stream / manifest.MANIFEST_NAME
    source_manifest.write_text('{}\n')
    (source_stream / 'meta.yaml').write_text(
        yaml.safe_dump({'manifest_sha256': manifest.sha256_file(source_manifest)})
    )
    stream_manifest = stream / manifest.MANIFEST_NAME
    stream_manifest.write_text(json.dumps(row, sort_keys=True) + '\n')
    (stream / 'meta.yaml').write_text(
        yaml.safe_dump(
            {
                'stream_complete': True,
                'manifest_sha256': manifest.sha256_file(stream_manifest),
                'n_shards': 1,
                'budget_ms': 1000.0,
                'batches_per_shard': 1,
                'fingerprint': 'fixture-stream',
                'source_stream': str(source_stream),
                'mask_storage': MASK_STORAGE,
                'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
            }
        )
    )
    (stream / 'summary.json').write_text('{}\n')

    checkpoint = tmp_path / 'codec.pt'
    checkpoint.touch()
    source_receipt_path = source_latent_dir.parent / 'latent-materialized.json'
    _write_json(
        source_receipt_path,
        {'codec_model': 'flux2', 'codec_checkpoint': str(checkpoint.resolve())},
    )
    _write_json(
        stream / 'latent-materialized.json',
        {
            'latent_contract': verify.CANONICAL_INPLANE_LATENT_CONTRACT,
            'codec_model': 'flux2',
            'codec_checkpoint': str(checkpoint.resolve()),
            'source_stream': str(source_stream),
            'source_manifest_sha256': manifest.sha256_file(source_manifest),
            'source_latent_dir': str(source_latent_dir),
            'source_latent_receipt_sha256': manifest.sha256_file(source_receipt_path),
            'materialized_shards': [0],
            'logical_latent_rows': 1,
            'affected_unlabeled_samples': 1,
            'affected_latent_rows': 1,
        },
    )
    save_file(
        {
            'count': torch.tensor(1, dtype=torch.int64),
            'mean': torch.zeros(32),
            'std': torch.ones(32),
        },
        str(stream / 'latents' / 'stats.safetensors'),
    )
    cost_model = tmp_path / 'cost-model.json'
    cost_model.write_text(json.dumps(CostModel.default().as_fit_result()))
    return SimpleNamespace(
        source_stream=source_stream,
        stream=stream,
        source_latent_dir=source_latent_dir,
        cost_model=cost_model,
        expected_summary=None,
        workers=1,
    )


def _write_full_verify_command_fixture(
    tmp_path: Path,
    *,
    fixed_source_stats: bool = False,
) -> SimpleNamespace:
    cost_model_value = CostModel.default()
    worker_args, _, _ = _write_full_verify_shard(
        tmp_path,
        cost_model=cost_model_value,
    )
    stream = Path(worker_args[1]).resolve()
    row = worker_args[3]
    rows = [row]
    if fixed_source_stats:
        second_row = {**row, 'shard_id': 1}
        (stream / 'shard_00001.msgpack').write_bytes(
            (stream / 'shard_00000.msgpack').read_bytes()
        )
        mask_shard_path(stream, 1).write_bytes(mask_shard_path(stream, 0).read_bytes())
        (stream / 'latents' / 'shard_00001.safetensors').write_bytes(
            (stream / 'latents' / 'shard_00000.safetensors').read_bytes()
        )
        (stream / 'reports' / 'shard_00001.json').write_text(json.dumps(second_row))
        rows.append(second_row)
    stream_manifest = stream / manifest.MANIFEST_NAME
    stream_manifest.write_text(
        ''.join(json.dumps(item, sort_keys=True) + '\n' for item in rows)
    )

    stats_count = (
        row['logical_latent_rows']
        if fixed_source_stats
        else sum(item['logical_latent_rows'] for item in rows)
    )
    stats_path = stream / 'latents' / 'stats.safetensors'
    save_file(
        {
            'count': torch.tensor(stats_count, dtype=torch.int64),
            'mean': torch.zeros(32),
            'std': torch.ones(32),
        },
        str(stats_path),
    )
    meta = {
        'algorithm': 'ucpt-stream-v4',
        'stream_complete': True,
        'manifest_sha256': manifest.sha256_file(stream_manifest),
        'n_shards': len(rows),
        'budget_ms': 1000.0,
        'batches_per_shard': 1,
        'fingerprint': 'fixture-full-stream',
        'seed': 42,
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
    }
    codec_checkpoint = (tmp_path / 'codec.pt').resolve()
    if fixed_source_stats:
        source_stream = (tmp_path / 'prefix-stream').resolve()
        source_stream.mkdir()
        source_fingerprint = 'fixture-prefix'
        source_manifest_sha256 = 'a' * 64
        source_receipt_path = source_stream / 'latent-materialized.json'
        _write_json(
            source_receipt_path,
            {
                'codec_model': 'flux2',
                'codec_checkpoint': str(codec_checkpoint),
            },
        )
        source_ready_path = source_stream / 'READY.json'
        _write_json(
            source_ready_path,
            {
                'fingerprint': source_fingerprint,
                'manifest_sha256': source_manifest_sha256,
                'verified_shards': 1,
                'latent_receipt_sha256': manifest.sha256_file(source_receipt_path),
                'latent_shard_sha256': [
                    manifest.sha256_file(
                        stream / 'latents' / 'shard_00000.safetensors'
                    )
                ],
            },
        )
        meta['composition'] = {
            'contract': verify.CONCATENATED_STREAM_CONTRACT,
            'replay_seed': 42,
            'segments': [
                {
                    'target_shard_start': 0,
                    'target_shard_stop': 1,
                    'source_stream': str(source_stream),
                    'source_fingerprint': source_fingerprint,
                    'source_manifest_sha256': source_manifest_sha256,
                    'source_ready_sha256': manifest.sha256_file(source_ready_path),
                    'source_shard_start': 0,
                    'source_shard_stop': 1,
                    'generation_seed': 42,
                },
                {
                    'target_shard_start': 1,
                    'target_shard_stop': 2,
                    'source_stream': str((tmp_path / 'suffix-stream').resolve()),
                    'source_fingerprint': 'fixture-suffix',
                    'source_manifest_sha256': 'c' * 64,
                    'source_shard_start': 0,
                    'source_shard_stop': 1,
                    'generation_seed': 43,
                },
            ],
            'normalization': {
                'contract': verify.FIXED_SOURCE_STATS_CONTRACT,
                'source_stream': str(source_stream),
                'source_fingerprint': source_fingerprint,
                'source_manifest_sha256': source_manifest_sha256,
                'source_stats_sha256': manifest.sha256_file(stats_path),
                'source_stats_count': stats_count,
            },
        }
    (stream / 'meta.yaml').write_text(yaml.safe_dump(meta))
    (stream / 'summary.json').write_text('{}\n')
    _write_json(
        stream / 'latent-materialized.json',
        {
            'codec_model': 'flux2',
            'codec_checkpoint': str(codec_checkpoint),
            'materialized_shards': [item['shard_id'] for item in rows],
            'generated_unlabeled_samples': sum(
                item['new_unlabeled_samples'] for item in rows
            ),
            'generated_unlabeled_patches': sum(
                item['logical_latent_rows'] for item in rows
            ),
        },
    )
    cost_model = tmp_path / 'cost-model.json'
    cost_model.write_text(json.dumps(cost_model_value.as_fit_result()))
    return SimpleNamespace(
        source_stream=None,
        stream=stream,
        source_latent_dir=None,
        cost_model=cost_model,
        expected_summary=None,
        workers=1,
    )


@pytest.mark.parametrize('top_level_normalization', [False, True])
def test_reused_latent_receipt_and_stats_inherit_fixed_source(
    tmp_path, top_level_normalization,
):
    source_args = _write_full_verify_command_fixture(tmp_path, fixed_source_stats=True)
    source_stream = source_args.stream
    source_meta = yaml.safe_load((source_stream / 'meta.yaml').read_text())
    normalization = source_meta['composition']['normalization']
    if top_level_normalization:
        source_meta['normalization'] = source_meta.pop('composition')['normalization']
        (source_stream / 'meta.yaml').write_text(yaml.safe_dump(source_meta))
    stream = tmp_path / 'filtered'
    stream.mkdir()
    meta = {
        'source_stream': str(source_stream),
        'unlabeled_filter': {'contract': 'unlabeled-filter-v1', 'exclude_datasets': ['ISLES22']},
        'normalization': normalization,
    }
    (stream / 'meta.yaml').write_text(yaml.safe_dump(meta))
    rows = [
        {
            'shard_id': shard_id,
            'source_latent_row_spans': [[0, 1]],
            'logical_latent_rows': 1,
            'old_unlabeled_patches': 1,
            'used_old_unlabeled_patches': 1,
            'dropped_old_unlabeled_patches': 0,
            'generated_unlabeled_samples': 0,
            'generated_unlabeled_patches': 0,
        }
        for shard_id in range(2)
    ]
    (stream / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    latent_backend.cmd_link_latents(
        SimpleNamespace(stream=stream, source_latent_dir=source_stream / 'latents', workers=1),
    )
    contract, receipt_path = verify._validate_latent_receipt(
        stream,
        source_stream,
        source_stream / 'latents',
        rows,
        affected_unlabeled_samples=0,
        affected_latent_rows=0,
    )
    assert contract == verify.REUSED_LATENT_CONTRACT
    _, stats_ready = verify._validate_stats(stream, meta, 2, strict_schema=True)
    assert stats_ready == {
        'stats_contract': 'fixed-source-v1',
        'stats_source_fingerprint': 'fixture-prefix',
        'stats_source_count': 1,
    }

    receipt = json.loads(receipt_path.read_text())
    receipt['source_latent_receipt_sha256'] = 'f' * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='source receipt hash differs'):
        verify._validate_latent_receipt(
            stream,
            source_stream,
            source_stream / 'latents',
            rows,
            affected_unlabeled_samples=0,
            affected_latent_rows=0,
        )


def _run_distributed_verify(
    rank: int,
    world_size: int,
    port: int,
    args: SimpleNamespace,
) -> None:
    os.environ.update(
        {
            'MASTER_ADDR': '127.0.0.1',
            'MASTER_PORT': str(port),
            'RANK': str(rank),
            'WORLD_SIZE': str(world_size),
        }
    )
    verify.cmd_verify(args)


def test_verify_distributes_shards_and_publishes_one_ready_proof(tmp_path):
    args = _write_verify_command_fixture(tmp_path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]

    mp.spawn(
        _run_distributed_verify,
        args=(2, port, args),
        nprocs=2,
        join=True,
    )

    ready = json.loads((args.stream / 'READY.json').read_text())
    assert ready['verified_shards'] == 1
    assert ready['latent_contract'] == verify.CANONICAL_INPLANE_LATENT_CONTRACT
    assert ready['latent_shard_sha256'] == [
        manifest.sha256_file(args.stream / 'latents' / 'shard_00000.safetensors')
    ]


@pytest.mark.parametrize('fixed_source_stats', [False, True])
def test_verify_full_stream_publishes_ready_contract(tmp_path, fixed_source_stats):
    args = _write_full_verify_command_fixture(
        tmp_path,
        fixed_source_stats=fixed_source_stats,
    )

    verify.cmd_verify(args)

    ready = json.loads((args.stream / 'READY.json').read_text())
    assert ready['latent_contract'] == verify.FULL_LATENT_CONTRACT
    if fixed_source_stats:
        assert ready['stats_contract'] == verify.FIXED_SOURCE_STATS_CONTRACT
        assert ready['stats_source_fingerprint'] == 'fixture-prefix'
        assert ready['stats_source_count'] == 1
    else:
        assert 'stats_contract' not in ready


def test_verify_fixed_source_stats_rejects_wrong_count_and_hash(tmp_path):
    args = _write_full_verify_command_fixture(tmp_path, fixed_source_stats=True)
    meta_path = args.stream / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['composition']['normalization']['source_stats_count'] = 2
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match='differs from source count'):
        verify.cmd_verify(args)

    meta['composition']['normalization']['source_stats_count'] = 1
    meta['composition']['normalization']['source_stats_sha256'] = 'b' * 64
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match='stats hash differs'):
        verify.cmd_verify(args)


@pytest.mark.parametrize(
    ('fault', 'match'),
    [
        ('codec', 'codec_model differs'),
        ('prefix-latent', 'composite prefix latent hash differs'),
        ('source-ready', 'source READY hash differs'),
    ],
)
def test_verify_composite_binds_prefix_latent_provenance(tmp_path, fault, match):
    args = _write_full_verify_command_fixture(tmp_path, fixed_source_stats=True)
    if fault == 'codec':
        receipt_path = args.stream / 'latent-materialized.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['codec_model'] = 'different-codec'
        receipt_path.write_text(json.dumps(receipt))
    elif fault == 'prefix-latent':
        latent_path = args.stream / 'latents' / 'shard_00000.safetensors'
        latent_path.unlink()
        save_file(
            {'latents': torch.full((1, 32), 9, dtype=torch.float16)},
            latent_path,
        )
    else:
        meta_path = args.stream / 'meta.yaml'
        meta = yaml.safe_load(meta_path.read_text())
        meta['composition']['segments'][0]['source_ready_sha256'] = 'c' * 64
        meta_path.write_text(yaml.safe_dump(meta))

    with pytest.raises(ValueError, match=match):
        verify.cmd_verify(args)


def test_verify_rejects_source_arguments_that_disagree_with_meta_mode(tmp_path):
    args = _write_full_verify_command_fixture(tmp_path)
    args.source_stream = (tmp_path / 'unexpected-source').resolve()
    args.source_latent_dir = (tmp_path / 'unexpected-latents').resolve()
    with pytest.raises(ValueError, match='must not use source verification arguments'):
        verify.cmd_verify(args)

    args.source_stream = None
    args.source_latent_dir = None
    meta_path = args.stream / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['source_stream'] = str((tmp_path / 'required-source').resolve())
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match='require source verification arguments'):
        verify.cmd_verify(args)
