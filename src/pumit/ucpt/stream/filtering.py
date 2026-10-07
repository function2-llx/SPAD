"""Filter unlabeled datasets within frozen batches while reusing labeled targets."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import get_context
import os
from pathlib import Path

import msgpack
import orjson
import yaml

from pumit.ucpt.mask_store import mask_shard_path
from pumit.ucpt.packing import CostModel, write_shard

from .build import _completed_shard, _validate_batches
from .manifest import ALGORITHM, ensure_build_plan, load_manifest, sha256_bytes, sha256_file, write_report


UNLABELED_FILTER_CONTRACT = 'unlabeled-filter-v1'


def filter_unlabeled_batches(
    batches: list[dict],
    exclude_datasets: list[str],
) -> tuple[list[dict], list[list[int]]]:
    """Retain each surviving sample unchanged and return its merged source latent row spans."""
    excluded = set(exclude_datasets)
    filtered = []
    spans = []
    offset = 0
    for batch in batches:
        kept = []
        unlabeled_count = 0
        for sample in batch['samples']:
            if sample['labeled']:
                kept.append(sample)
                continue
            start = offset
            offset += sample['n_patches']
            dataset = sample.get('dataset', Path(sample['img']).parent.parent.name)
            if dataset in excluded:
                continue
            kept.append(sample)
            unlabeled_count += 1
            if spans and spans[-1][1] == start:
                spans[-1][1] = offset
            else:
                spans.append([start, offset])
        if not unlabeled_count:
            raise ValueError(f"batch {batch['step_idx']}: no unlabeled samples remain after filtering")
        filtered.append({**batch, 'samples': kept})
    return filtered, spans


def _filter_one_shard(
    row: dict,
    *,
    source_stream: Path,
    output_dir: Path,
    exclude_datasets: list[str],
    batches_per_shard: int,
    cost_model: CostModel,
    budget_ms: float,
    fingerprint: str,
) -> dict:
    shard_id = row['shard_id']
    source_path = source_stream / f'shard_{shard_id:05d}.msgpack'
    source_bytes = source_path.read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    if source_sha256 != row['output_sha256']:
        raise ValueError(f'{source_path}: hash differs from the source manifest')
    source = msgpack.unpackb(source_bytes, raw=False)
    batches = source['batches']
    if len(batches) != batches_per_shard:
        raise ValueError(f'{source_path}: batch count differs from source metadata')
    filtered, spans = filter_unlabeled_batches(batches, exclude_datasets)
    batch_stats = _validate_batches(filtered, cost_model, budget_ms, shard_id)
    old_unlabeled = [sample for batch in batches for sample in batch['samples'] if not sample['labeled']]
    unlabeled = [sample for batch in filtered for sample in batch['samples'] if not sample['labeled']]
    labeled_count = sum(sample['labeled'] for batch in batches for sample in batch['samples'])
    old_rows = sum(sample['n_patches'] for sample in old_unlabeled)
    used_rows = sum(sample['n_patches'] for sample in unlabeled)

    mask_path = mask_shard_path(output_dir, shard_id)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    os.link(mask_shard_path(source_stream, shard_id), mask_path)
    output_path = output_dir / source_path.name
    write_shard(filtered, output_path)
    report = {
        'build_fingerprint': fingerprint,
        'shard_id': shard_id,
        'source_sha256': source_sha256,
        'output_sha256': sha256_file(output_path),
        'old_labeled_samples': labeled_count,
        'used_old_labeled_samples': labeled_count,
        'generated_labeled_samples': 0,
        'generated_labeled_patches': 0,
        'old_unlabeled_samples': len(old_unlabeled),
        'used_old_unlabeled_samples': len(unlabeled),
        'dropped_old_unlabeled_samples': len(old_unlabeled) - len(unlabeled),
        'generated_unlabeled_samples': 0,
        'generated_unlabeled_patches': 0,
        'old_unlabeled_patches': old_rows,
        'used_old_unlabeled_patches': used_rows,
        'dropped_old_unlabeled_patches': old_rows - used_rows,
        'logical_latent_rows': used_rows,
        'source_latent_row_spans': spans,
        'new_labeled_samples': labeled_count,
        'new_unlabeled_samples': len(unlabeled),
        'new_labeled_sample_fraction': labeled_count / (labeled_count + len(unlabeled)),
        **{
            key: row[key]
            for key in (
                'n_total_records', 'n_labeled_records', 'sample_metadata_contract', 'mask_storage',
                'mask_labeled_samples', 'mask_positive_masks', 'mask_storage_bytes', 'mask_sha256',
            )
        },
        **batch_stats,
    }
    write_report(output_dir, report)
    return report


def filter_stream(
    *,
    source_stream: Path,
    output_dir: Path,
    exclude_datasets: list[str],
    shards: int | None = None,
    workers: int = 4,
) -> None:
    """Filter a prefix of a READY source stream without changing batch membership of survivors."""
    if workers < 1:
        raise ValueError('workers must be positive')
    source_stream = Path(source_stream).resolve()
    output_dir = Path(output_dir).resolve()
    source_meta_bytes = (source_stream / 'meta.yaml').read_bytes()
    source_meta = yaml.safe_load(source_meta_bytes)
    if source_meta.get('stream_complete') is not True or not (source_stream / 'READY.json').is_file():
        raise RuntimeError(f'source stream is not READY: {source_stream}')
    if sha256_file(source_stream / 'manifest.jsonl') != source_meta['manifest_sha256']:
        raise ValueError('source manifest hash differs from source metadata')
    rows = load_manifest(source_stream)
    if len(rows) != source_meta['n_shards']:
        raise ValueError('source manifest shard count differs from source metadata')
    shard_count = len(rows) if shards is None else shards
    if not 1 <= shard_count <= len(rows):
        raise ValueError(f'shards must be in [1, {len(rows)}], got {shard_count}')
    normalization = source_meta.get('normalization')
    if normalization is None:
        normalization = source_meta.get('composition', {}).get('normalization')
    if normalization is None:
        raise ValueError('source lacks fixed normalization provenance')
    source_meta_sha256 = sha256_bytes(source_meta_bytes)
    unlabeled_filter = {
        'contract': UNLABELED_FILTER_CONTRACT,
        'exclude_datasets': list(exclude_datasets),
    }
    fingerprint = sha256_bytes(
        orjson.dumps(
            {
                'algorithm': ALGORITHM,
                'source_stream': str(source_stream),
                'source_meta_sha256': source_meta_sha256,
                'unlabeled_filter': unlabeled_filter,
            },
            option=orjson.OPT_SORT_KEYS,
        ),
    )[:16]
    plan = {
        'fingerprint': fingerprint,
        'algorithm': ALGORITHM,
        **{
            key: source_meta[key]
            for key in (
                'packing_policy', 'seed', 'batches_per_shard', 'budget_ms', 'label_budget_fraction',
                'cost_model_source', 'cost_model', 'config',
            )
        },
        'source_stream': str(source_stream),
        'source_meta_sha256': source_meta_sha256,
        'normalization': normalization,
        'unlabeled_filter': unlabeled_filter,
    }
    ensure_build_plan(output_dir, plan)
    pending = [row for row in rows[:shard_count] if not _completed_shard(output_dir, row['shard_id'], fingerprint)]
    process = partial(
        _filter_one_shard,
        source_stream=source_stream,
        output_dir=output_dir,
        exclude_datasets=exclude_datasets,
        batches_per_shard=source_meta['batches_per_shard'],
        cost_model=CostModel.from_fit_result(source_meta['cost_model']),
        budget_ms=source_meta['budget_ms'],
        fingerprint=fingerprint,
    )
    if workers == 1 or len(pending) <= 1:
        for row in pending:
            process(row)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as executor:
            futures = [executor.submit(process, row) for row in pending]
            for future in as_completed(futures):
                future.result()
