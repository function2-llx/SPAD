"""Build plans, shard reports, and final stream metadata."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from ..mask_store import MASK_STORAGE, mask_shard_path
from .metadata import SAMPLE_METADATA_CONTRACT


ALGORITHM = 'ucpt-stream-v4'
STATS_CHUNK_ROWS = 1_000_000
BUILD_PLAN_NAME = 'build.yaml'
MANIFEST_NAME = 'manifest.jsonl'
SUMMARY_NAME = 'summary.json'
FINALIZED_PREFIX_DIR = '.finalized-prefixes'

_SUM_KEYS = (
    'old_labeled_samples',
    'old_unlabeled_samples',
    'used_old_labeled_samples',
    'used_old_unlabeled_samples',
    'dropped_old_unlabeled_samples',
    'generated_labeled_samples',
    'generated_unlabeled_samples',
    'old_unlabeled_patches',
    'used_old_unlabeled_patches',
    'dropped_old_unlabeled_patches',
    'generated_labeled_patches',
    'generated_unlabeled_patches',
    'logical_latent_rows',
    'new_labeled_samples',
    'new_unlabeled_samples',
    'mask_labeled_samples',
    'mask_positive_masks',
    'mask_storage_bytes',
)
_HISTOGRAM_KEYS = (
    'batch_size_histogram',
    'labeled_per_batch_histogram',
    'unlabeled_per_batch_histogram',
)


def require_absent(path: Path) -> None:
    """Reject an existing output path."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'refusing to overwrite existing path: {path}')


def sha256_bytes(data: bytes) -> str:
    """Hash bytes with SHA-256."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file with SHA-256."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish bytes atomically without replacing an existing file."""
    require_absent(path)
    tmp = path.parent / f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp'
    with tmp.open('xb') as file:
        file.write(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def write_json(path: Path, value) -> None:
    """Write indented JSON without replacing an existing file."""
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    _atomic_write_bytes(path, data)


def ensure_build_plan(output_dir: Path, plan: dict) -> None:
    """Create or validate the immutable plan shared by partitioned builders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'meta.yaml').exists():
        raise RuntimeError(f'stream is already finalized: {output_dir}')

    path = output_dir / BUILD_PLAN_NAME
    if not path.exists():
        if (
            any(output_dir.glob('shard_*.msgpack'))
            or (output_dir / 'masks').exists()
            or (output_dir / 'reports').exists()
        ):
            raise RuntimeError(f'{output_dir} contains build outputs but no {BUILD_PLAN_NAME}')
        data = yaml.safe_dump(plan, sort_keys=True).encode()
        try:
            _atomic_write_bytes(path, data)
        except FileExistsError:
            pass

    existing = yaml.safe_load(path.read_text())
    if existing != plan:
        raise RuntimeError(
            f"build plan mismatch: existing={existing.get('fingerprint')} current={plan['fingerprint']}"
        )


def report_path(stream_dir: Path, shard_id: int) -> Path:
    """Return the report sidecar path for one shard."""
    return stream_dir / 'reports' / f'shard_{shard_id:05d}.json'


def write_report(stream_dir: Path, report: dict) -> None:
    """Write one shard report atomically."""
    path = report_path(stream_dir, report['shard_id'])
    path.parent.mkdir(exist_ok=True)
    write_json(path, report)


def load_manifest(stream_dir: Path) -> list[dict]:
    """Load and validate a finalized stream manifest."""
    rows = [
        json.loads(line)
        for line in (stream_dir / MANIFEST_NAME).read_text().splitlines()
    ]
    if [row['shard_id'] for row in rows] != list(range(len(rows))):
        raise ValueError('manifest shard IDs are not contiguous from zero')
    return rows


def _link_finalized_artifact(source: Path, destination: Path) -> None:
    """Preserve one finalized artifact before reopening the stream."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            raise FileExistsError(f'finalized-prefix backup differs from source: {destination}')
        return
    os.link(source, destination)


def _unlink_finalized_artifacts(
    stream_dir: Path,
    backup_dir: Path,
    artifacts: list[str],
) -> None:
    for relative in artifacts:
        source = stream_dir / relative
        destination = backup_dir / relative
        if not destination.exists():
            raise FileNotFoundError(f'missing finalized-prefix backup: {destination}')
        if source.exists():
            source_stat = source.stat()
            destination_stat = destination.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                raise ValueError(f'finalized-prefix backup inode differs: {destination}')
            source.unlink()


def prepare_stream_extension(stream_dir: Path, expected_shards: int) -> dict:
    """Reopen a validated finalized prefix while preserving its aggregate artifacts."""
    stream_dir = stream_dir.resolve()
    if expected_shards < 1:
        raise ValueError(f'expected_shards must be positive, got {expected_shards}')
    meta_path = stream_dir / 'meta.yaml'
    if not meta_path.exists():
        raise FileNotFoundError(f'stream is not finalized: {stream_dir}')

    meta = yaml.safe_load(meta_path.read_text())
    if not meta.get('stream_complete'):
        raise ValueError(f'stream meta is not complete: {meta_path}')
    current_shards = int(meta['n_shards'])
    if expected_shards < current_shards:
        raise ValueError(
            f'cannot shrink finalized stream from {current_shards} to {expected_shards} shards'
        )
    if expected_shards == current_shards:
        return {
            'status': 'already-finalized',
            'current_shards': current_shards,
            'expected_shards': expected_shards,
        }

    backup_dir = (
        stream_dir
        / FINALIZED_PREFIX_DIR
        / f'{current_shards:05d}-{meta["fingerprint"]}'
    )
    receipt_path = backup_dir / 'extension.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        expected_identity = {
            'source_fingerprint': meta['fingerprint'],
            'source_shards': current_shards,
            'target_shards': expected_shards,
        }
        if any(receipt.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f'existing extension receipt differs: {receipt_path}')
        _unlink_finalized_artifacts(stream_dir, backup_dir, receipt['artifacts'])
        return {
            'status': 'reopened',
            'current_shards': current_shards,
            'expected_shards': expected_shards,
            'backup_dir': str(backup_dir),
        }

    plan = yaml.safe_load((stream_dir / BUILD_PLAN_NAME).read_text())
    if meta['build_fingerprint'] != plan['fingerprint']:
        raise ValueError('finalized meta and build plan fingerprints differ')
    manifest_path = stream_dir / MANIFEST_NAME
    if sha256_file(manifest_path) != meta['manifest_sha256']:
        raise ValueError('finalized manifest hash differs from stream meta')
    rows = load_manifest(stream_dir)
    if len(rows) != current_shards:
        raise ValueError(
            f'finalized manifest has {len(rows)} rows, expected {current_shards}'
        )
    for row in rows:
        validated = _validate_report(
            (
                stream_dir / f"shard_{row['shard_id']:05d}.msgpack",
                mask_shard_path(stream_dir, row['shard_id']),
                report_path(stream_dir, row['shard_id']),
                plan['fingerprint'],
                row['shard_id'],
            )
        )
        if validated != row:
            raise ValueError(
                f"shard {row['shard_id']}: manifest row differs from validated report"
            )

    optional_artifacts = (
        'latent-links.json',
        'latent-materialized.json',
        'latent-stats.json',
        'latents/stats.safetensors',
        'READY.json',
    )
    artifacts = [
        MANIFEST_NAME,
        SUMMARY_NAME,
        *(name for name in optional_artifacts if (stream_dir / name).exists()),
        'meta.yaml',
    ]
    receipt = {
        'source_fingerprint': meta['fingerprint'],
        'source_shards': current_shards,
        'target_shards': expected_shards,
        'artifacts': artifacts,
    }
    for relative in artifacts:
        source = stream_dir / relative
        if not source.exists():
            raise FileNotFoundError(f'missing finalized artifact: {source}')
        _link_finalized_artifact(source, backup_dir / relative)
    write_json(receipt_path, receipt)
    _unlink_finalized_artifacts(stream_dir, backup_dir, artifacts)

    return {
        'status': 'reopened',
        'current_shards': current_shards,
        'expected_shards': expected_shards,
        'backup_dir': str(backup_dir),
    }


def _histogram_distribution(histogram: Counter) -> dict:
    values = np.repeat(
        np.asarray(list(histogram), dtype=np.int32),
        np.asarray(list(histogram.values()), dtype=np.int64),
    )
    return {
        'mean': float(values.mean()),
        **{
            f'p{percentile}': float(np.percentile(values, percentile))
            for percentile in (0, 5, 25, 50, 75, 95, 100)
        },
    }


def summarize(rows: list[dict], plan: dict) -> dict:
    """Aggregate per-shard reports."""
    totals = {key: sum(row[key] for row in rows) for key in _SUM_KEYS}
    histograms = {}
    distributions = {}
    for key in _HISTOGRAM_KEYS:
        histogram = Counter()
        for row in rows:
            histogram.update({int(value): count for value, count in row[key].items()})
        histograms[key] = dict(sorted(histogram.items()))
        distributions[key.removesuffix('_histogram')] = _histogram_distribution(histogram)

    old_samples = totals['old_unlabeled_samples']
    old_patches = totals['old_unlabeled_patches']
    summary = {
        'algorithm': plan['algorithm'],
        'packing_policy': plan['packing_policy'],
        'budget_ms': plan['budget_ms'],
        'label_budget_fraction': plan['label_budget_fraction'],
        'batches_per_shard': plan['batches_per_shard'],
        'shards': len(rows),
        'totals': totals,
        'new_labeled_sample_fraction': (
            totals['new_labeled_samples']
            / (totals['new_labeled_samples'] + totals['new_unlabeled_samples'])
        ),
        'old_unlabeled_sample_reuse_fraction': (
            totals['used_old_unlabeled_samples'] / old_samples if old_samples else None
        ),
        'old_unlabeled_patch_reuse_fraction': (
            totals['used_old_unlabeled_patches'] / old_patches if old_patches else None
        ),
        'shards_requiring_generated_unlabeled': sum(
            row['generated_unlabeled_samples'] > 0 for row in rows
        ),
        'shard_ids_requiring_generated_unlabeled': [
            row['shard_id'] for row in rows if row['generated_unlabeled_samples'] > 0
        ],
        'distributions': distributions,
        'histograms': histograms,
    }
    return summary


def _validate_report(args: tuple[Path, Path, Path, str, int]) -> dict:
    shard_path, mask_path, sidecar_path, fingerprint, shard_id = args
    report = json.loads(sidecar_path.read_text())
    if report['shard_id'] != shard_id:
        raise ValueError(f'{sidecar_path}: shard ID mismatch')
    if report['build_fingerprint'] != fingerprint:
        raise ValueError(f'{sidecar_path}: build fingerprint mismatch')
    if sha256_file(shard_path) != report['output_sha256']:
        raise ValueError(f'{shard_path}: hash differs from report')
    if sha256_file(mask_path) != report['mask_sha256']:
        raise ValueError(f'{mask_path}: hash differs from report')
    if mask_path.stat().st_size != report['mask_storage_bytes']:
        raise ValueError(f'{mask_path}: size differs from report')
    if report['mask_storage'] != MASK_STORAGE:
        raise ValueError(f'{sidecar_path}: unsupported mask storage {report["mask_storage"]!r}')
    if report.get('sample_metadata_contract') != SAMPLE_METADATA_CONTRACT:
        raise ValueError(
            f'{sidecar_path}: unsupported sample metadata contract '
            f'{report.get("sample_metadata_contract")!r}'
        )
    return report


def finalize_stream(stream_dir: Path, expected_shards: int, workers: int) -> dict:
    """Validate all shard outputs and publish the production stream metadata."""
    stream_dir = stream_dir.resolve()
    if expected_shards < 1:
        raise ValueError(f'expected_shards must be positive, got {expected_shards}')
    require_absent(stream_dir / 'meta.yaml')
    plan = yaml.safe_load((stream_dir / BUILD_PLAN_NAME).read_text())

    expected_ids = list(range(expected_shards))
    expected_shard_names = [f'shard_{shard_id:05d}.msgpack' for shard_id in expected_ids]
    expected_mask_names = [f'shard_{shard_id:05d}.bin' for shard_id in expected_ids]
    expected_report_names = [f'shard_{shard_id:05d}.json' for shard_id in expected_ids]
    if [path.name for path in sorted(stream_dir.glob('shard_*.msgpack'))] != expected_shard_names:
        raise ValueError('stream shards are incomplete, non-contiguous, or exceed the expected range')
    reports_dir = stream_dir / 'reports'
    masks_dir = stream_dir / 'masks'
    if [path.name for path in sorted(masks_dir.glob('shard_*.bin'))] != expected_mask_names:
        raise ValueError('mask shards are incomplete, non-contiguous, or exceed the expected range')
    if [path.name for path in sorted(reports_dir.glob('shard_*.json'))] != expected_report_names:
        raise ValueError('shard reports are incomplete, non-contiguous, or exceed the expected range')

    tasks = [
        (
            stream_dir / expected_shard_names[shard_id],
            masks_dir / expected_mask_names[shard_id],
            reports_dir / expected_report_names[shard_id],
            plan['fingerprint'],
            shard_id,
        )
        for shard_id in expected_ids
    ]
    if workers <= 0:
        rows = [_validate_report(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_validate_report, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: row['shard_id'])

    pool_counts = {
        (row['n_total_records'], row['n_labeled_records'])
        for row in rows
    }
    if len(pool_counts) != 1:
        raise ValueError(f'workers observed inconsistent data pool counts: {pool_counts}')
    n_total_records, n_labeled_records = pool_counts.pop()
    manifest_path = stream_dir / MANIFEST_NAME
    manifest_data = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode()
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_data:
            raise ValueError('existing manifest differs from validated shard reports')
    else:
        _atomic_write_bytes(manifest_path, manifest_data)
    summary = summarize(rows, plan)
    summary_path = stream_dir / SUMMARY_NAME
    if summary_path.exists():
        summary_data = (json.dumps(summary, indent=2, sort_keys=True) + '\n').encode()
        if summary_path.read_bytes() != summary_data:
            raise ValueError('existing summary differs from validated shard reports')
    else:
        write_json(summary_path, summary)

    manifest_sha256 = sha256_file(manifest_path)
    fingerprint = sha256_bytes(f"{plan['fingerprint']}:{manifest_sha256}".encode())[:16]
    meta = {
        'fingerprint': fingerprint,
        'build_fingerprint': plan['fingerprint'],
        'algorithm': plan['algorithm'],
        'packing_policy': plan['packing_policy'],
        'stream_complete': True,
        'seed': plan['seed'],
        'n_shards': expected_shards,
        'total_batches': expected_shards * plan['batches_per_shard'],
        'batches_per_shard': plan['batches_per_shard'],
        'budget_ms': plan['budget_ms'],
        'label_budget_fraction': plan['label_budget_fraction'],
        'cost_model_source': plan['cost_model_source'],
        'cost_model': plan['cost_model'],
        'n_labeled_records': n_labeled_records,
        'n_total_records': n_total_records,
        'config': plan['config'],
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'manifest_sha256': manifest_sha256,
        'date': datetime.datetime.now().isoformat(),
    }
    if plan['source_stream'] is not None:
        meta.update({
            'source_stream': plan['source_stream'],
            'source_meta_sha256': plan['source_meta_sha256'],
        })
    if 'composition' in plan:
        meta['composition'] = plan['composition']
    for key in ('input_migration', 'unlabeled_filter', 'normalization'):
        if key in plan:
            meta[key] = plan[key]
    _atomic_write_bytes(
        stream_dir / 'meta.yaml',
        yaml.safe_dump(meta, default_flow_style=False, sort_keys=False).encode(),
    )
    return summary
