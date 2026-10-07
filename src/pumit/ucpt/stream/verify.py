"""Verify a completed UCPT stream and its reused latent prefixes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import json
from pathlib import Path

import msgpack
from safetensors import safe_open
import torch
import torch.distributed as dist
from tqdm import tqdm
import yaml

from pumit.ucpt.mask_store import MASK_STORAGE, mask_shard_path
from pumit.ucpt.packing import CostModel

from .build import (
    _flatten_samples,
    _is_source_labeled,
    _msgpack_equal,
    _upgrade_source_unlabeled,
    _validate_batches,
    distributed_context,
)
from .latent_backend import (
    CANONICAL_INPLANE_LATENT_CONTRACT,
    REUSED_LATENT_CONTRACT,
    _latent_shape,
    _source_latent_row_spans,
    _validate_materialized_latent,
    _validate_reused_latent,
    _validate_reused_latent_operation,
)
from .manifest import (
    MANIFEST_NAME,
    _SUM_KEYS,
    load_manifest,
    report_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .migration import AFFINE_REPLAY_CONTRACT, PREVIOUS_AFFINE_REPLAY_CONTRACT
from .metadata import (
    SAMPLE_METADATA_CONTRACT,
    canonicalize_sample_metadata,
    metadata_changes_replay,
)


_SUFFIX_RECEIPT_KEYS = {
    'codec_model',
    'codec_checkpoint',
    'materialized_shards',
    'generated_unlabeled_samples',
    'generated_unlabeled_patches',
}
_MIGRATION_RECEIPT_KEYS = {
    'codec_model',
    'codec_checkpoint',
    'source_latent_dir',
    'migration_contract',
    'materialized_shards',
    'affected_unlabeled_samples',
    'affected_unlabeled_patches',
    'generated_unlabeled_samples',
    'generated_unlabeled_patches',
}
_CANONICAL_INPLANE_RECEIPT_KEYS = {
    'latent_contract',
    'codec_model',
    'codec_checkpoint',
    'source_stream',
    'source_manifest_sha256',
    'source_latent_dir',
    'source_latent_receipt_sha256',
    'materialized_shards',
    'logical_latent_rows',
    'affected_unlabeled_samples',
    'affected_latent_rows',
}
_LATENT_LINK_RECEIPT_KEYS = {
    'source_latent_dir',
    'hardlinked_shards',
    'materialize_shards',
}
_REUSED_LATENT_RECEIPT_KEYS = {
    'latent_contract',
    'codec_model',
    'codec_checkpoint',
    'source_latent_dir',
    'source_latent_receipt_sha256',
    'logical_latent_rows',
    'materialized_shards',
}
FULL_LATENT_CONTRACT = 'full-v1'
CONCATENATED_STREAM_CONTRACT = 'concatenated-stream-v1'
FIXED_SOURCE_STATS_CONTRACT = 'fixed-source-v1'


def _load_json_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def _require_exact_keys(value: dict, expected: set[str], path: Path) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f'{path}: receipt keys differ: missing={sorted(expected - actual)} '
            f'unexpected={sorted(actual - expected)}'
        )


def _canonical_json(value: object) -> str:
    normalized = json.loads(json.dumps(value))
    return json.dumps(normalized, sort_keys=True, separators=(',', ':'))


def _validate_mask_inventory(stream: Path, rows: list[dict], meta: dict) -> None:
    if meta.get('mask_storage') != MASK_STORAGE:
        raise ValueError(
            f"stream meta mask_storage={meta.get('mask_storage')!r}, expected {MASK_STORAGE!r}"
        )
    if meta.get('n_shards') != len(rows):
        raise ValueError('stream meta shard count differs from manifest')
    if meta.get('sample_metadata_contract') != SAMPLE_METADATA_CONTRACT:
        raise ValueError(
            f"stream meta sample_metadata_contract={meta.get('sample_metadata_contract')!r}, "
            f'expected {SAMPLE_METADATA_CONTRACT!r}'
        )
    expected_mask_names = [f"shard_{row['shard_id']:05d}.bin" for row in rows]
    actual_mask_names = [path.name for path in sorted((stream / 'masks').glob('shard_*.bin'))]
    if actual_mask_names != expected_mask_names:
        raise ValueError('mask shards are incomplete, non-contiguous, or exceed the manifest')
    expected_report_names = [f"shard_{row['shard_id']:05d}.json" for row in rows]
    actual_report_names = [
        path.name for path in sorted((stream / 'reports').glob('shard_*.json'))
    ]
    if actual_report_names != expected_report_names:
        raise ValueError('shard reports are incomplete, non-contiguous, or exceed the manifest')


def _validate_codec_identity(receipt: dict, source_latent_dir: Path) -> Path:
    codec_model = receipt.get('codec_model')
    codec_checkpoint = receipt.get('codec_checkpoint')
    if not isinstance(codec_model, str) or not codec_model:
        raise ValueError('latent receipt has an invalid codec_model')
    if not isinstance(codec_checkpoint, str) or not codec_checkpoint:
        raise ValueError('latent receipt has an invalid codec_checkpoint')

    source_receipt_path = source_latent_dir.parent / 'latent-materialized.json'
    source_receipt = _load_json_object(source_receipt_path)
    if source_receipt.get('codec_model') != codec_model:
        raise ValueError('latent receipt codec_model differs from source latent provenance')
    source_checkpoint = source_receipt.get('codec_checkpoint')
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError(f'{source_receipt_path}: missing codec_checkpoint identity')
    if Path(source_checkpoint).resolve() != Path(codec_checkpoint).resolve():
        raise ValueError('latent receipt codec_checkpoint differs from source latent provenance')
    return source_receipt_path


def _validate_latent_receipt(
    stream: Path,
    source_stream: Path | None,
    source_latent_dir: Path | None,
    rows: list[dict],
    *,
    affected_unlabeled_samples: int,
    affected_latent_rows: int,
) -> tuple[str, Path]:
    receipt_path = stream / 'latent-materialized.json'
    receipt = _load_json_object(receipt_path)
    shard_ids = [row['shard_id'] for row in rows]
    logical_rows = sum(row['logical_latent_rows'] for row in rows)

    if (source_stream is None) != (source_latent_dir is None):
        raise ValueError('source stream and source latent directory must be provided together')
    if source_stream is None:
        _require_exact_keys(receipt, _SUFFIX_RECEIPT_KEYS, receipt_path)
        if not isinstance(receipt.get('codec_model'), str) or not receipt['codec_model']:
            raise ValueError('latent receipt has an invalid codec_model')
        if not isinstance(receipt.get('codec_checkpoint'), str) or not receipt['codec_checkpoint']:
            raise ValueError('latent receipt has an invalid codec_checkpoint')
        expected_values = {
            'materialized_shards': shard_ids,
            'generated_unlabeled_samples': sum(
                row['new_unlabeled_samples'] for row in rows
            ),
            'generated_unlabeled_patches': logical_rows,
        }
        for key, expected_value in expected_values.items():
            if receipt[key] != expected_value:
                raise ValueError(f'full latent receipt {key} differs from manifest')
        return FULL_LATENT_CONTRACT, receipt_path

    assert source_latent_dir is not None
    source_receipt_path = _validate_codec_identity(receipt, source_latent_dir)
    meta_path = stream / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    if (
        (meta.get('input_migration') is not None or meta.get('unlabeled_filter') is not None)
        and receipt.get('latent_contract') != REUSED_LATENT_CONTRACT
    ):
        raise ValueError('input migration or unlabeled filtering requires reused-latent-v1 provenance')

    if receipt.get('latent_contract') == REUSED_LATENT_CONTRACT:
        _require_exact_keys(receipt, _REUSED_LATENT_RECEIPT_KEYS, receipt_path)
        _validate_reused_latent_operation(meta)
        if affected_unlabeled_samples != 0 or affected_latent_rows != 0:
            raise ValueError('latent reuse cannot change source geometry')
        if (
            not isinstance(receipt['source_latent_dir'], str)
            or Path(receipt['source_latent_dir']).resolve() != source_latent_dir
        ):
            raise ValueError('reused latent receipt source_latent_dir differs')
        if receipt['source_latent_receipt_sha256'] != sha256_file(source_receipt_path):
            raise ValueError('reused latent receipt source receipt hash differs')
        source_meta = yaml.safe_load((source_stream / 'meta.yaml').read_text())
        normalization = _fixed_source_normalization(meta)
        if normalization is None or normalization != _fixed_source_normalization(source_meta):
            raise ValueError('latent reuse normalization differs from source')
        materialized_shards = []
        for row in rows:
            source_rows, _ = _latent_shape(source_latent_dir / f"shard_{row['shard_id']:05d}.safetensors")
            spans = _source_latent_row_spans(row, source_rows)
            if spans != [[0, source_rows]]:
                materialized_shards.append(row['shard_id'])
        if receipt['logical_latent_rows'] != logical_rows:
            raise ValueError('reused latent receipt logical_latent_rows differs from manifest')
        if receipt['materialized_shards'] != materialized_shards:
            raise ValueError('reused latent receipt materialized_shards differs from manifest')
        links_path = stream / 'latent-links.json'
        links = _load_json_object(links_path)
        _require_exact_keys(links, _LATENT_LINK_RECEIPT_KEYS, links_path)
        if links != {
            'source_latent_dir': str(source_latent_dir),
            'hardlinked_shards': [shard_id for shard_id in shard_ids if shard_id not in materialized_shards],
            'materialize_shards': materialized_shards,
        }:
            raise ValueError('reused latent link receipt differs from manifest')
        return REUSED_LATENT_CONTRACT, receipt_path

    if (
        (affected_unlabeled_samples > 0 or affected_latent_rows > 0)
        and receipt.get('latent_contract') != CANONICAL_INPLANE_LATENT_CONTRACT
    ):
        raise ValueError(
            'canonical in-plane metadata requires canonical-inplane-latent-v1; '
            'legacy or suffix latents cannot be verified'
        )

    if 'latent_contract' in receipt:
        _require_exact_keys(receipt, _CANONICAL_INPLANE_RECEIPT_KEYS, receipt_path)
        if receipt['latent_contract'] != CANONICAL_INPLANE_LATENT_CONTRACT:
            raise ValueError(f"unsupported latent contract {receipt['latent_contract']!r}")
        source_manifest_path = source_stream / MANIFEST_NAME
        source_meta = yaml.safe_load((source_stream / 'meta.yaml').read_text())
        source_manifest_sha256 = sha256_file(source_manifest_path)
        if source_meta.get('manifest_sha256') != source_manifest_sha256:
            raise ValueError('source stream manifest hash differs from source meta')
        expected = {
            'source_stream': source_stream,
            'source_latent_dir': source_latent_dir,
        }
        for key, expected_path in expected.items():
            value = receipt[key]
            if not isinstance(value, str) or Path(value).resolve() != expected_path:
                raise ValueError(f'canonical in-plane latent receipt {key} differs')
        if receipt['source_manifest_sha256'] != source_manifest_sha256:
            raise ValueError('canonical in-plane latent receipt source manifest hash differs')
        if receipt['source_latent_receipt_sha256'] != sha256_file(source_receipt_path):
            raise ValueError('canonical in-plane latent receipt source receipt hash differs')
        expected_values = {
            'materialized_shards': shard_ids,
            'logical_latent_rows': logical_rows,
            'affected_unlabeled_samples': affected_unlabeled_samples,
            'affected_latent_rows': affected_latent_rows,
        }
        for key, expected_value in expected_values.items():
            if receipt[key] != expected_value:
                raise ValueError(
                    f'canonical in-plane latent receipt {key}={receipt[key]!r}, '
                    f'expected {expected_value!r}'
                )
        return CANONICAL_INPLANE_LATENT_CONTRACT, receipt_path

    if 'migration_contract' in receipt:
        _require_exact_keys(receipt, _MIGRATION_RECEIPT_KEYS, receipt_path)
        if receipt['migration_contract'] not in {
            PREVIOUS_AFFINE_REPLAY_CONTRACT,
            AFFINE_REPLAY_CONTRACT,
        }:
            raise ValueError(
                f"unsupported migration latent contract {receipt['migration_contract']!r}"
            )
        contracts = {row.get('migration_contract') for row in rows}
        if len(contracts) != 1 or None in contracts:
            raise ValueError('manifest has inconsistent migration contracts')
        expected_values = {
            'source_latent_dir': str(source_latent_dir),
            'migration_contract': contracts.pop(),
            'materialized_shards': shard_ids,
            'affected_unlabeled_samples': sum(
                row['migration_affected_unlabeled_samples'] for row in rows
            ),
            'affected_unlabeled_patches': sum(
                row['migration_affected_unlabeled_patches'] for row in rows
            ),
            'generated_unlabeled_samples': sum(
                row['generated_unlabeled_samples'] for row in rows
            ),
            'generated_unlabeled_patches': sum(
                row['generated_unlabeled_patches'] for row in rows
            ),
        }
        for key, expected_value in expected_values.items():
            actual = receipt[key]
            if key == 'source_latent_dir':
                if not isinstance(actual, str) or Path(actual).resolve() != source_latent_dir:
                    raise ValueError('migration latent receipt source_latent_dir differs')
            elif actual != expected_value:
                raise ValueError(f'migration latent receipt {key} differs from manifest')
        return str(receipt['migration_contract']), receipt_path

    _require_exact_keys(receipt, _SUFFIX_RECEIPT_KEYS, receipt_path)
    materialized_shards = [
        row['shard_id'] for row in rows if row['generated_unlabeled_samples'] > 0
    ]
    expected_values = {
        'materialized_shards': materialized_shards,
        'generated_unlabeled_samples': sum(
            row['generated_unlabeled_samples'] for row in rows
        ),
        'generated_unlabeled_patches': sum(
            row['generated_unlabeled_patches'] for row in rows
        ),
    }
    for key, expected_value in expected_values.items():
        if receipt[key] != expected_value:
            raise ValueError(f'suffix latent receipt {key} differs from manifest')

    links_path = stream / 'latent-links.json'
    links = _load_json_object(links_path)
    _require_exact_keys(links, _LATENT_LINK_RECEIPT_KEYS, links_path)
    hardlinked_shards = [row['shard_id'] for row in rows if row['shard_id'] not in materialized_shards]
    if (
        not isinstance(links['source_latent_dir'], str)
        or Path(links['source_latent_dir']).resolve() != source_latent_dir
        or links['hardlinked_shards'] != hardlinked_shards
        or links['materialize_shards'] != materialized_shards
    ):
        raise ValueError('latent link receipt differs from manifest or source latent directory')
    return 'suffix-v1', receipt_path


def _verify_one_shard(args: tuple) -> dict:
    (
        source_stream_raw,
        stream_raw,
        source_latent_dir_raw,
        row,
        cost_model,
        budget_ms,
        batches_per_shard,
    ) = args
    shard_id = row["shard_id"]
    stream = Path(stream_raw)
    meta_path = stream / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    unlabeled_filter = meta.get('unlabeled_filter')
    source_stream = Path(source_stream_raw) if source_stream_raw is not None else None
    source_path = (
        source_stream / f"shard_{shard_id:05d}.msgpack"
        if source_stream is not None
        else None
    )
    if unlabeled_filter is not None:
        _validate_reused_latent_operation(meta)
        if source_path is None:
            raise ValueError('unlabeled filtering requires source verification')
    output_path = stream / f"shard_{shard_id:05d}.msgpack"
    sidecar_path = report_path(stream, shard_id)
    report = _load_json_object(sidecar_path)
    if _canonical_json(report) != _canonical_json(row):
        raise ValueError(f'shard {shard_id}: manifest row differs from shard report')
    if report.get('mask_storage') != MASK_STORAGE:
        raise ValueError(f'shard {shard_id}: unsupported mask storage')
    if report.get('sample_metadata_contract') != SAMPLE_METADATA_CONTRACT:
        raise ValueError(f'shard {shard_id}: unsupported sample metadata contract')
    mask_path = mask_shard_path(stream, shard_id)
    if mask_path.stat().st_size != report.get('mask_storage_bytes'):
        raise ValueError(f'shard {shard_id}: mask size differs from shard report')
    if sha256_file(mask_path) != report.get('mask_sha256'):
        raise ValueError(f'shard {shard_id}: mask hash differs from shard report')
    output_bytes = output_path.read_bytes()
    if sha256_bytes(output_bytes) != row["output_sha256"]:
        raise ValueError(f"shard {shard_id}: output hash differs from manifest")
    output_shard = msgpack.unpackb(output_bytes, raw=False)
    if len(output_shard["batches"]) != batches_per_shard:
        raise ValueError(f"shard {shard_id}: batch count mismatch")
    if [batch["step_idx"] for batch in output_shard["batches"]] != list(
        range(batches_per_shard)
    ):
        raise ValueError(f"shard {shard_id}: step_idx sequence mismatch")

    output_samples = _flatten_samples(output_shard)
    output_labeled = [sample for sample in output_samples if _is_source_labeled(sample)]
    output_unlabeled = [sample for sample in output_samples if not _is_source_labeled(sample)]
    if any("label_classes" in sample for sample in output_samples):
        raise ValueError(f"shard {shard_id}: full label contract leaked into output")
    if any(not isinstance(sample.get("labeled"), bool) for sample in output_samples):
        raise ValueError(f"shard {shard_id}: output sample lacks a boolean labeled flag")
    if any(bool(sample.get("classes")) != sample["labeled"] for sample in output_samples):
        raise ValueError(f"shard {shard_id}: invalid labeled sample schema")
    for sample in output_samples:
        canonical, changes = canonicalize_sample_metadata(sample)
        if changes or canonical is not sample:
            raise ValueError(f'shard {shard_id}: output sample metadata is not canonical')
    if source_path is None:
        count_keys = (
            'old_labeled_samples',
            'used_old_labeled_samples',
            'generated_labeled_samples',
            'old_unlabeled_samples',
            'used_old_unlabeled_samples',
            'dropped_old_unlabeled_samples',
            'generated_unlabeled_samples',
            'new_labeled_samples',
            'new_unlabeled_samples',
            'old_unlabeled_patches',
            'used_old_unlabeled_patches',
            'dropped_old_unlabeled_patches',
            'generated_labeled_patches',
            'generated_unlabeled_patches',
            'logical_latent_rows',
        )
        for key in count_keys:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f'shard {shard_id}: {key} must be a non-negative integer'
                )
    if unlabeled_filter is not None:
        if row['used_old_labeled_samples'] != len(output_labeled) or row['generated_labeled_samples'] != 0:
            raise ValueError(f'shard {shard_id}: unlabeled filtering must reuse every labeled sample')
    else:
        if row['used_old_labeled_samples'] != 0:
            raise ValueError(f'shard {shard_id}: manifest claims reuse of old labeled samples')
        if row['generated_labeled_samples'] != len(output_labeled):
            raise ValueError(f'shard {shard_id}: generated labeled count differs from output')
    if (
        len(output_labeled) != row["new_labeled_samples"]
        or len(output_unlabeled) != row["new_unlabeled_samples"]
    ):
        raise ValueError(f"shard {shard_id}: output sample counts differ from manifest")
    total_patches = sum(sample["n_patches"] for sample in output_samples)
    if output_shard["total_patches"] != total_patches:
        raise ValueError(f"shard {shard_id}: total_patches header mismatch")
    if source_path is None:
        batch_stats = _validate_batches(
            output_shard['batches'], cost_model, budget_ms, shard_id,
        )
        max_cost = batch_stats['batch_cost_max_ms']
    else:
        batch_stats = None
        max_cost = 0.0
        for batch_index, batch in enumerate(output_shard['batches']):
            n_labeled = sum(sample['labeled'] for sample in batch['samples'])
            if n_labeled == 0 or n_labeled == len(batch['samples']):
                raise ValueError(
                    f'shard {shard_id} batch {batch_index}: sample-kind floor violation'
                )
            cost = sum(cost_model.sample_cost(sample) for sample in batch['samples'])
            labeled_costs = [
                cost_model.sample_cost(sample)
                for sample in batch['samples']
                if sample['labeled']
            ]
            unlabeled_costs = [
                cost_model.sample_cost(sample)
                for sample in batch['samples']
                if not sample['labeled']
            ]
            overshoot_bound = budget_ms + max(labeled_costs) + max(unlabeled_costs)
            if cost > overshoot_bound + 1e-6:
                raise ValueError(
                    f'shard {shard_id} batch {batch_index}: cost {cost} exceeds nearest '
                    f'rounding bound {overshoot_bound}'
                )
            max_cost = max(max_cost, cost)

    affected_indices = []
    if source_path is not None:
        source_bytes = source_path.read_bytes()
        if sha256_bytes(source_bytes) != row['source_sha256']:
            raise ValueError(f'shard {shard_id}: source hash differs from manifest')
        source_shard = msgpack.unpackb(source_bytes, raw=False)
        source_samples = _flatten_samples(source_shard)
        source_labeled = [
            sample for sample in source_samples if _is_source_labeled(sample)
        ]
        source_unlabeled = [
            sample for sample in source_samples if not _is_source_labeled(sample)
        ]
        if len(source_labeled) != row['old_labeled_samples']:
            raise ValueError(f'shard {shard_id}: source labeled count differs from manifest')
        used_unlabeled = row['used_old_unlabeled_samples']
        expected_unlabeled = []
        if unlabeled_filter is not None:
            from .filtering import filter_unlabeled_batches

            expected_batches, spans = filter_unlabeled_batches(
                source_shard['batches'], unlabeled_filter['exclude_datasets'],
            )
            if not _msgpack_equal(output_shard['batches'], expected_batches):
                raise ValueError(f'shard {shard_id}: filtered batches differ from source')
            expected_unlabeled = [
                sample
                for batch in expected_batches
                for sample in batch['samples']
                if not sample['labeled']
            ]
            if not mask_shard_path(source_stream, shard_id).samefile(mask_path):
                raise ValueError(f'shard {shard_id}: filtered mask sidecar must be a source hardlink')
        else:
            for index, sample in enumerate(source_unlabeled):
                migrated, changes = canonicalize_sample_metadata(
                    _upgrade_source_unlabeled(sample)
                )
                expected_unlabeled.append(migrated)
                if metadata_changes_replay(changes):
                    affected_indices.append(index)
        if unlabeled_filter is not None:
            old_patches = sum(sample['n_patches'] for sample in source_unlabeled)
            used_patches = sum(sample['n_patches'] for sample in expected_unlabeled)
            expected_counts = {
                'old_unlabeled_samples': len(source_unlabeled),
                'used_old_unlabeled_samples': len(expected_unlabeled),
                'dropped_old_unlabeled_samples': len(source_unlabeled) - len(expected_unlabeled),
                'generated_unlabeled_samples': 0,
                'old_unlabeled_patches': old_patches,
                'used_old_unlabeled_patches': used_patches,
                'dropped_old_unlabeled_patches': old_patches - used_patches,
                'generated_unlabeled_patches': 0,
                'generated_labeled_patches': (
                    0 if unlabeled_filter is not None else sum(sample['n_patches'] for sample in output_labeled)
                ),
                'source_latent_row_spans': spans,
            }
            for key, expected in expected_counts.items():
                if row.get(key) != expected:
                    raise ValueError(f'shard {shard_id}: {key} differs from retained source')
        elif (
            used_unlabeled != len(source_unlabeled)
            or row['generated_unlabeled_samples'] != 0
            or row['dropped_old_unlabeled_samples'] != 0
            or len(output_unlabeled) != len(source_unlabeled)
        ):
            raise ValueError(
                f'shard {shard_id}: repack did not preserve the complete unlabeled sequence'
            )
        if not _msgpack_equal(output_unlabeled, expected_unlabeled):
            raise ValueError(f'shard {shard_id}: unlabeled sequence differs from source')
    else:
        used_unlabeled = row['used_old_unlabeled_samples']
        generated_unlabeled = row['generated_unlabeled_samples']
        if used_unlabeled > len(output_unlabeled):
            raise ValueError(
                f'shard {shard_id}: used_old_unlabeled_samples exceeds output'
            )
        if used_unlabeled + generated_unlabeled != len(output_unlabeled):
            raise ValueError(
                f'shard {shard_id}: used/generated unlabeled counts differ from output'
            )
        if (
            row['old_unlabeled_samples'] - used_unlabeled
            != row['dropped_old_unlabeled_samples']
        ):
            raise ValueError(f'shard {shard_id}: dropped unlabeled count is inconsistent')
        used_unlabeled_patches = sum(
            sample['n_patches'] for sample in output_unlabeled[:used_unlabeled]
        )
        generated_unlabeled_patches = sum(
            sample['n_patches'] for sample in output_unlabeled[used_unlabeled:]
        )
        expected_patch_counts = {
            'used_old_unlabeled_patches': used_unlabeled_patches,
            'generated_unlabeled_patches': generated_unlabeled_patches,
            'generated_labeled_patches': sum(
                sample['n_patches'] for sample in output_labeled
            ),
        }
        for key, expected in expected_patch_counts.items():
            if row[key] != expected:
                raise ValueError(f'shard {shard_id}: {key} differs from output')
        if (
            row['old_unlabeled_patches'] - used_unlabeled_patches
            != row['dropped_old_unlabeled_patches']
        ):
            raise ValueError(f'shard {shard_id}: dropped unlabeled patches are inconsistent')
        assert batch_stats is not None
        for key, actual in batch_stats.items():
            if _canonical_json(row.get(key)) != _canonical_json(actual):
                raise ValueError(f'shard {shard_id}: {key} differs from output batches')

    logical_rows = sum(sample['n_patches'] for sample in output_unlabeled)
    if logical_rows != row["logical_latent_rows"]:
        raise ValueError(f"shard {shard_id}: logical latent rows differ from manifest")
    output_latent = stream / 'latents' / f'shard_{shard_id:05d}.safetensors'
    if source_latent_dir_raw is not None:
        source_latent = (
            Path(source_latent_dir_raw) / f'shard_{shard_id:05d}.safetensors'
        )
        source_rows, _ = _latent_shape(source_latent)
        if source_rows < row['old_unlabeled_patches']:
            raise ValueError(f'shard {shard_id}: source latent row contract failed')
    _validate_materialized_latent(output_latent, logical_rows)
    if unlabeled_filter is not None:
        if source_latent_dir_raw is None:
            raise ValueError('latent reuse requires source latent verification')
        _validate_reused_latent(source_latent, output_latent, row)
    affected = [output_unlabeled[index] for index in affected_indices]
    return {
        "shard_id": shard_id,
        "logical_latent_rows": logical_rows,
        "max_batch_cost_ms": max_cost,
        'affected_unlabeled_samples': len(affected),
        'affected_latent_rows': sum(sample['n_patches'] for sample in affected),
        'latent_sha256': sha256_file(output_latent),
    }


def _require_sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _validate_composition_structure(meta: dict) -> tuple[dict, list[dict]]:
    composition = meta.get('composition')
    if not isinstance(composition, dict):
        raise ValueError('stream composition must be an object')
    if composition.get('contract') != CONCATENATED_STREAM_CONTRACT:
        raise ValueError(
            f"unsupported stream composition contract {composition.get('contract')!r}"
        )
    replay_seed = composition.get('replay_seed')
    if (
        isinstance(replay_seed, bool)
        or not isinstance(replay_seed, int)
        or replay_seed != meta.get('seed')
    ):
        raise ValueError('concatenated stream replay_seed differs from stream seed')
    n_shards = meta.get('n_shards')
    if isinstance(n_shards, bool) or not isinstance(n_shards, int) or n_shards < 2:
        raise ValueError('concatenated stream has an invalid shard count')
    segments = composition.get('segments')
    if not isinstance(segments, list) or len(segments) != 2:
        raise ValueError('concatenated stream must contain exactly two segments')

    target_cursor = 0
    range_keys = (
        'target_shard_start',
        'target_shard_stop',
        'source_shard_start',
        'source_shard_stop',
    )
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f'concatenated stream segment {index} must be an object')
        ranges = {}
        for key in range_keys:
            value = segment.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'concatenated stream segment {index} has invalid {key}')
            ranges[key] = value
        target_start = ranges['target_shard_start']
        target_stop = ranges['target_shard_stop']
        source_start = ranges['source_shard_start']
        source_stop = ranges['source_shard_stop']
        if (
            target_start != target_cursor
            or target_stop <= target_start
            or source_stop <= source_start
            or target_stop - target_start != source_stop - source_start
        ):
            raise ValueError(f'concatenated stream segment {index} has an invalid mapping')
        source_stream = segment.get('source_stream')
        source_fingerprint = segment.get('source_fingerprint')
        generation_seed = segment.get('generation_seed')
        if not isinstance(source_stream, str) or not source_stream:
            raise ValueError(f'concatenated stream segment {index} has invalid source_stream')
        if not isinstance(source_fingerprint, str) or not source_fingerprint:
            raise ValueError(
                f'concatenated stream segment {index} has invalid source_fingerprint'
            )
        _require_sha256(
            segment.get('source_manifest_sha256'),
            f'concatenated stream segment {index} source_manifest_sha256',
        )
        if (
            isinstance(generation_seed, bool)
            or not isinstance(generation_seed, int)
            or generation_seed < 0
        ):
            raise ValueError(
                f'concatenated stream segment {index} has invalid generation_seed'
            )
        target_cursor = target_stop
    if target_cursor != n_shards:
        raise ValueError('concatenated stream segments do not cover every shard')
    _require_sha256(
        segments[0].get('source_ready_sha256'),
        'concatenated stream prefix source_ready_sha256',
    )
    return composition, segments


def _fixed_source_normalization(meta: dict) -> dict | None:
    composition = meta.get('composition')
    normalization = meta.get('normalization')
    if composition is None and normalization is None:
        return None
    if normalization is not None:
        segments = None
    else:
        composition, segments = _validate_composition_structure(meta)
        normalization = composition.get('normalization')
    if not isinstance(normalization, dict):
        raise ValueError('concatenated stream lacks normalization provenance')
    if normalization.get('contract') != FIXED_SOURCE_STATS_CONTRACT:
        raise ValueError(
            'concatenated stream must use fixed-source-v1 normalization'
        )

    source_stream = normalization.get('source_stream')
    source_fingerprint = normalization.get('source_fingerprint')
    source_count = normalization.get('source_stats_count')
    if not isinstance(source_stream, str) or not source_stream:
        raise ValueError('fixed-source normalization has an invalid source_stream')
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError('fixed-source normalization has an invalid source_fingerprint')
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 1:
        raise ValueError('fixed-source normalization has an invalid source_stats_count')
    _require_sha256(
        normalization.get('source_manifest_sha256'),
        'fixed-source normalization source_manifest_sha256',
    )
    _require_sha256(
        normalization.get('source_stats_sha256'),
        'fixed-source normalization source_stats_sha256',
    )

    if segments is None:
        return normalization
    prefix = segments[0]
    identity = {
        'source_stream': source_stream,
        'source_fingerprint': source_fingerprint,
        'source_manifest_sha256': normalization['source_manifest_sha256'],
    }
    for key, expected in identity.items():
        actual = prefix.get(key)
        if key == 'source_stream':
            if not isinstance(actual, str) or Path(actual).resolve() != Path(expected).resolve():
                raise ValueError(
                    'fixed-source normalization source_stream differs from prefix segment'
                )
        elif actual != expected:
            raise ValueError(
                f'fixed-source normalization {key} differs from prefix segment'
            )
    return normalization


def _validate_composite_prefix_latents(
    meta: dict,
    verified: list[dict],
    latent_receipt_path: Path,
) -> None:
    composition = meta.get('composition')
    if composition is None:
        return
    _, segments = _validate_composition_structure(meta)
    prefix = segments[0]
    target_start = prefix['target_shard_start']
    target_stop = prefix['target_shard_stop']
    source_start = prefix['source_shard_start']
    source_stop = prefix['source_shard_stop']
    if target_stop > len(verified):
        raise ValueError('concatenated stream prefix exceeds verified shards')

    source_stream_raw = prefix.get('source_stream')
    if not isinstance(source_stream_raw, str) or not source_stream_raw:
        raise ValueError('concatenated stream prefix has an invalid source_stream')
    source_stream = Path(source_stream_raw).resolve()
    source_ready_path = source_stream / 'READY.json'
    expected_ready_sha256 = _require_sha256(
        prefix.get('source_ready_sha256'),
        'concatenated stream prefix source_ready_sha256',
    )
    if sha256_file(source_ready_path) != expected_ready_sha256:
        raise ValueError('concatenated stream prefix source READY hash differs')
    source_ready = _load_json_object(source_ready_path)
    if source_ready.get('fingerprint') != prefix.get('source_fingerprint'):
        raise ValueError('concatenated stream prefix source READY fingerprint differs')
    if source_ready.get('manifest_sha256') != prefix.get('source_manifest_sha256'):
        raise ValueError('concatenated stream prefix source READY manifest hash differs')
    source_verified_shards = source_ready.get('verified_shards')
    if (
        isinstance(source_verified_shards, bool)
        or not isinstance(source_verified_shards, int)
        or source_start != 0
        or source_verified_shards != source_stop
    ):
        raise ValueError('concatenated stream prefix source READY shard count differs')
    source_latent_hashes = source_ready.get('latent_shard_sha256')
    if (
        not isinstance(source_latent_hashes, list)
        or len(source_latent_hashes) != source_verified_shards
    ):
        raise ValueError('concatenated stream prefix source READY latent inventory differs')

    source_receipt_path = source_stream / 'latent-materialized.json'
    source_receipt_sha256 = _require_sha256(
        source_ready.get('latent_receipt_sha256'),
        'concatenated stream prefix source READY latent_receipt_sha256',
    )
    if sha256_file(source_receipt_path) != source_receipt_sha256:
        raise ValueError('concatenated stream prefix source latent receipt hash differs')
    source_receipt = _load_json_object(source_receipt_path)
    target_receipt = _load_json_object(latent_receipt_path)
    if target_receipt.get('codec_model') != source_receipt.get('codec_model'):
        raise ValueError('full latent receipt codec_model differs from prefix source receipt')
    source_checkpoint = source_receipt.get('codec_checkpoint')
    target_checkpoint = target_receipt.get('codec_checkpoint')
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError('prefix source latent receipt has an invalid codec_checkpoint')
    if (
        not isinstance(target_checkpoint, str)
        or not target_checkpoint
        or Path(target_checkpoint).resolve() != Path(source_checkpoint).resolve()
    ):
        raise ValueError(
            'full latent receipt codec_checkpoint differs from prefix source receipt'
        )

    for target_shard, source_shard in zip(
        range(target_start, target_stop),
        range(source_start, source_stop),
        strict=True,
    ):
        source_hash = _require_sha256(
            source_latent_hashes[source_shard],
            f'prefix source latent hash for shard {source_shard}',
        )
        if verified[target_shard]['latent_sha256'] != source_hash:
            raise ValueError(
                f'composite prefix latent hash differs for target shard {target_shard}'
            )


def _validate_stats(
    stream: Path,
    meta: dict,
    expected_rows: int,
    *,
    strict_schema: bool,
) -> tuple[Path, dict]:
    stats_path = stream / 'latents' / 'stats.safetensors'
    stats_sha256 = sha256_file(stats_path)
    with safe_open(str(stats_path), framework='pt') as file:
        if strict_schema and set(file.keys()) != {'count', 'mean', 'std'}:
            raise ValueError(f'latent stats have an invalid tensor schema: {stats_path}')
        stats_count = int(file.get_tensor('count'))
        mean = file.get_tensor('mean')
        std = file.get_tensor('std')
    if strict_schema and (mean.shape != (32,) or std.shape != (32,)):
        raise ValueError('latent stats must contain 32-channel mean/std')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError('latent stats contain non-finite values')
    if not (std > 0).all():
        raise ValueError('latent stats contain non-positive standard deviations')

    normalization = _fixed_source_normalization(meta)
    if normalization is None:
        if stats_count != expected_rows:
            raise ValueError(
                f'latent stats count {stats_count} differs from manifest rows {expected_rows}'
            )
        return stats_path, {}

    source_count = normalization['source_stats_count']
    if stats_count != source_count:
        raise ValueError(
            f'fixed-source latent stats count {stats_count} differs from source count '
            f'{source_count}'
        )
    if stats_sha256 != normalization['source_stats_sha256']:
        raise ValueError('fixed-source latent stats hash differs from composition provenance')
    return stats_path, {
        'stats_contract': FIXED_SOURCE_STATS_CONTRACT,
        'stats_source_fingerprint': normalization['source_fingerprint'],
        'stats_source_count': source_count,
    }


def cmd_verify(args) -> None:
    stream = args.stream.resolve()
    rows = load_manifest(stream)
    meta = yaml.safe_load((stream / "meta.yaml").read_text())
    source_stream_arg = getattr(args, 'source_stream', None)
    source_latent_dir_arg = getattr(args, 'source_latent_dir', None)
    if (source_stream_arg is None) != (source_latent_dir_arg is None):
        raise ValueError(
            '--source-stream and --source-latent-dir must be provided together'
        )
    source_stream = source_stream_arg.resolve() if source_stream_arg is not None else None
    source_latent_dir = (
        source_latent_dir_arg.resolve() if source_latent_dir_arg is not None else None
    )
    meta_source_stream = meta.get('source_stream')
    if meta_source_stream is not None and (
        not isinstance(meta_source_stream, str) or not meta_source_stream
    ):
        raise ValueError('stream meta has an invalid source_stream')
    if meta_source_stream is not None and source_stream is None:
        raise ValueError('source-backed streams require source verification arguments')
    if meta_source_stream is None and source_stream is not None:
        raise ValueError('self-contained streams must not use source verification arguments')
    if not meta.get("stream_complete"):
        raise ValueError("stream meta does not declare stream_complete")
    manifest_sha256 = sha256_file(stream / MANIFEST_NAME)
    if manifest_sha256 != meta['manifest_sha256']:
        raise ValueError("manifest hash differs from stream meta")
    _validate_mask_inventory(stream, rows, meta)
    cost_model = CostModel.from_file(args.cost_model)
    rank, world_size = distributed_context()
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        initialized_here = True

    try:
        worker_args = [
            (
                str(source_stream) if source_stream is not None else None,
                str(stream),
                str(source_latent_dir) if source_latent_dir is not None else None,
                row,
                cost_model,
                meta["budget_ms"],
                meta["batches_per_shard"],
            )
            for row in rows[rank::world_size]
        ]
        verified_local = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_verify_one_shard, worker_arg): worker_arg[3]["shard_id"]
                for worker_arg in worker_args
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f'Verify stream rank {rank}',
                unit="shard",
            ):
                verified_local.append(future.result())

        if dist.is_initialized():
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(verified_local, gathered, dst=0)
        else:
            gathered = [verified_local]

        if rank == 0:
            assert gathered is not None
            verified = [item for rank_items in gathered for item in rank_items]
            verified.sort(key=lambda item: item["shard_id"])
            if [item['shard_id'] for item in verified] != [row['shard_id'] for row in rows]:
                raise ValueError('distributed verification did not cover every manifest shard')

            latent_contract, latent_receipt_path = _validate_latent_receipt(
                stream,
                source_stream,
                source_latent_dir,
                rows,
                affected_unlabeled_samples=sum(
                    item['affected_unlabeled_samples'] for item in verified
                ),
                affected_latent_rows=sum(
                    item['affected_latent_rows'] for item in verified
                ),
            )
            if source_stream is None:
                _validate_composite_prefix_latents(
                    meta,
                    verified,
                    latent_receipt_path,
                )

            expected_rows = sum(row['logical_latent_rows'] for row in rows)
            stats_path, stats_ready = _validate_stats(
                stream,
                meta,
                expected_rows,
                strict_schema=source_stream is None,
            )

            summary = json.loads((stream / 'summary.json').read_text())
            if args.expected_summary is not None:
                expected = json.loads(args.expected_summary.read_text())
                for key in _SUM_KEYS:
                    if key == "logical_latent_rows":
                        continue
                    if summary["totals"][key] != expected["totals"][key]:
                        raise ValueError(
                            f"production total {key}={summary['totals'][key]} "
                            f"differs from dry-run {expected['totals'][key]}"
                        )
                if (
                    summary["shard_ids_requiring_generated_unlabeled"]
                    != expected["shard_ids_requiring_generated_unlabeled"]
                ):
                    raise ValueError("generated-unlabeled shard IDs differ from dry-run")
            ready = {
                "fingerprint": meta["fingerprint"],
                "verified_shards": len(verified),
                "logical_latent_rows": expected_rows,
                "max_batch_cost_ms": max(
                    item["max_batch_cost_ms"] for item in verified
                ),
                'manifest_sha256': manifest_sha256,
                'mask_storage': MASK_STORAGE,
                'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
                'latent_contract': latent_contract,
                'latent_receipt_sha256': sha256_file(latent_receipt_path),
                'latent_shard_sha256': [item['latent_sha256'] for item in verified],
                "stats_sha256": sha256_file(stats_path),
                **stats_ready,
                "verified_at": datetime.datetime.now().isoformat(),
            }
            write_json(stream / 'READY.json', ready)
            print(json.dumps(ready, indent=2, sort_keys=True))
        if dist.is_initialized():
            dist.barrier()
    finally:
        if initialized_here:
            dist.destroy_process_group()
