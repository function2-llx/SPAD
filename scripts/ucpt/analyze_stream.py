"""Analyze a pre-generated UCPT batch stream: pre-launch sanity statistics.

Reports stream shape (shards/batches/samples), the realized label fraction
(vs the meta.yaml target), and the augmentation / labeled-pool composition.
The UCPT batch format is per-sample (da_enc/n_patches/depth live on each sample,
not the batch), and labeled samples carry the final raw-class queries selected during generation.

Exits 1 if any batch has zero labeled samples (DDP-fatal); this is the pre-launch gate.

Usage:
    pixi run -e default python scripts/ucpt/analyze_stream.py precompute/ucpt/stream_seed42_40k/
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import msgpack
import numpy as np
import yaml

from pumit.ucpt.packing import CostModel


def _percentiles(values, ps=(0, 5, 25, 50, 75, 95, 100)) -> str:
    arr = np.asarray(values)
    return '  '.join(f'p{p}={np.percentile(arr, p):.0f}' for p in ps)


def _analyze_shard(shard_path: Path, cost_model: CostModel) -> dict:
    """Per-shard partial statistics (runs in a worker process).

    Returns Counters + numpy arrays so the parent can merge cheaply. Percentile
    inputs come back as arrays (concatenated in the parent) to keep percentiles
    exact rather than approximating across shards.
    """
    with open(shard_path, 'rb') as f:
        shard = msgpack.unpack(f, raw=False)

    batch_sizes: Counter[int] = Counter()
    da_counts: Counter[int | None] = Counter()
    depth_counts: Counter[int] = Counter()
    dataset_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    pos_class_counts: Counter[tuple[str, str]] = Counter()
    n_patches: list[int] = []
    labeled_per_batch: list[int] = []
    batch_costs: list[float] = []
    pos_per_sample: list[int] = []
    neg_per_sample: list[int] = []
    positive_target_voxels: list[int] = []
    total_batches = 0
    total_samples = 0
    n_labeled = 0
    labeled_cost = 0.0

    for batch in shard['batches']:
        samples = batch['samples']
        for s in samples:
            if 'label_classes' in s:
                raise ValueError(f'{shard_path}: stream embeds a full label contract')
            if not isinstance(s.get('labeled'), bool):
                raise ValueError(f'{shard_path}: sample lacks a boolean labeled flag')
            if bool(s.get('classes')) != s['labeled']:
                raise ValueError(f'{shard_path}: invalid labeled sample schema')
        labeled_per_batch.append(sum(sample['labeled'] for sample in samples))
        bc = sum(cost_model.sample_cost(s) for s in samples)
        batch_costs.append(bc)
        labeled_cost += sum(cost_model.sample_cost(sample) for sample in samples if sample['labeled'])
        batch_sizes[len(samples)] += 1
        total_batches += 1
        total_samples += len(samples)
        for s in samples:
            da_counts[s['da_enc']] += 1
            depth_counts[s['depth']] += 1
            n_patches.append(s['n_patches'])
            if s['labeled']:
                n_labeled += 1
                dataset_counts[s['dataset']] += 1
                modality_counts[s.get('modality', '?')] += 1
                positive = [item for item in s['classes'] if item['is_positive']]
                negative = [item for item in s['classes'] if not item['is_positive']]
                pos_per_sample.append(len(positive))
                neg_per_sample.append(len(negative))
                for item in positive:
                    pos_class_counts[(item['source'], item['name'])] += 1
                    positive_target_voxels.append(item['target_voxels'])

    return {
        'batch_sizes': batch_sizes,
        'da_counts': da_counts,
        'depth_counts': depth_counts,
        'dataset_counts': dataset_counts,
        'modality_counts': modality_counts,
        'pos_class_counts': pos_class_counts,
        'n_patches': np.asarray(n_patches, dtype=np.int32),
        'labeled_per_batch': np.asarray(labeled_per_batch, dtype=np.int32),
        'batch_costs': np.asarray(batch_costs, dtype=np.float64),
        'pos_per_sample': np.asarray(pos_per_sample, dtype=np.int32),
        'neg_per_sample': np.asarray(neg_per_sample, dtype=np.int32),
        'positive_target_voxels': np.asarray(positive_target_voxels, dtype=np.int64),
        'total_batches': total_batches,
        'total_samples': total_samples,
        'n_labeled': n_labeled,
        'labeled_cost': labeled_cost,
    }


def main():
    ap = argparse.ArgumentParser(description='Analyze a UCPT batch stream')
    ap.add_argument('stream_dir', type=Path)
    ap.add_argument('--top-classes', type=int, default=30,
                    help='How many most-frequent positive classes to list')
    ap.add_argument('--workers', type=int, default=64,
                    help='Parallel shard-reader processes')
    cost_model_group = ap.add_mutually_exclusive_group()
    cost_model_group.add_argument('--cost-model', type=Path,
                                  help='Cost model for a legacy stream whose meta.yaml does not record coefficients')
    cost_model_group.add_argument('--use-default-cost-model', action='store_true',
                                  help='Explicitly use the embedded cost model for a legacy stream')
    args = ap.parse_args()

    shards = sorted(args.stream_dir.glob('shard_*.msgpack'))
    if not shards:
        raise SystemExit(f'No shards found in {args.stream_dir}')

    meta_path = args.stream_dir / 'meta.yaml'
    if not meta_path.exists():
        raise SystemExit(f'No meta.yaml found in {args.stream_dir}')
    meta = yaml.safe_load(meta_path.read_text())
    if args.cost_model is not None:
        cost_model = CostModel.from_file(args.cost_model)
    elif args.use_default_cost_model:
        cost_model = CostModel.default()
    elif 'cost_model' in meta:
        cost_model = CostModel.from_fit_result(meta['cost_model'])
    else:
        raise SystemExit(
            f'{meta_path} does not record cost-model coefficients; pass --cost-model or '
            f'--use-default-cost-model explicitly'
        )

    batch_sizes: Counter[int] = Counter()
    da_counts: Counter[int | None] = Counter()
    depth_counts: Counter[int] = Counter()
    dataset_counts: Counter[str] = Counter()       # labeled samples per dataset
    modality_counts: Counter[str] = Counter()       # labeled samples per modality
    pos_class_counts: Counter[tuple[str, str]] = Counter()  # (source, name) positives
    n_patches_parts: list[np.ndarray] = []
    labeled_per_batch_parts: list[np.ndarray] = []
    batch_costs_parts: list[np.ndarray] = []
    pos_per_sample_parts: list[np.ndarray] = []
    neg_per_sample_parts: list[np.ndarray] = []
    positive_target_voxels_parts: list[np.ndarray] = []
    total_batches = 0
    total_samples = 0
    n_labeled = 0
    labeled_cost = 0.0

    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_analyze_shard, p, cost_model): p for p in shards}
        for fut in tqdm(as_completed(futures), total=len(shards),
                        desc='Analyzing shards', unit='shard'):
            r = fut.result()
            batch_sizes.update(r['batch_sizes'])
            da_counts.update(r['da_counts'])
            depth_counts.update(r['depth_counts'])
            dataset_counts.update(r['dataset_counts'])
            modality_counts.update(r['modality_counts'])
            pos_class_counts.update(r['pos_class_counts'])
            n_patches_parts.append(r['n_patches'])
            labeled_per_batch_parts.append(r['labeled_per_batch'])
            batch_costs_parts.append(r['batch_costs'])
            pos_per_sample_parts.append(r['pos_per_sample'])
            neg_per_sample_parts.append(r['neg_per_sample'])
            positive_target_voxels_parts.append(r['positive_target_voxels'])
            total_batches += r['total_batches']
            total_samples += r['total_samples']
            n_labeled += r['n_labeled']
            labeled_cost += r['labeled_cost']

    n_patches = np.concatenate(n_patches_parts)
    labeled_per_batch = np.concatenate(labeled_per_batch_parts)
    batch_costs = np.concatenate(batch_costs_parts)
    pos_per_sample = np.concatenate(pos_per_sample_parts)
    neg_per_sample = np.concatenate(neg_per_sample_parts)
    positive_target_voxels = np.concatenate(positive_target_voxels_parts)

    lf_realized = n_labeled / total_samples
    lf_target = meta.get('label_fraction')

    print(f'Stream: {args.stream_dir}')
    print(f'Shards: {len(shards)}  Batches: {total_batches}  Samples: {total_samples}')
    print(f'Avg samples/batch: {total_samples / total_batches:.1f}')
    if meta:
        print(f"meta: seed={meta.get('seed')} budget_ms={meta.get('budget_ms')} "
              f"n_total_records={meta.get('n_total_records')} "
              f"n_labeled_records={meta.get('n_labeled_records')}")
    print()

    print('Label fraction (per-sample):')
    print(f'  realized: {n_labeled}/{total_samples} = {lf_realized:.4f}'
          + (f'   target: {lf_target}' if lf_target is not None else ''))
    print()

    min_lab = min(labeled_per_batch)
    print('Labeled per batch:')
    print(f'  {_percentiles(labeled_per_batch)}  min={min_lab}')
    print()

    total_cost = sum(batch_costs)
    f_target = meta.get('label_budget_fraction')
    print('Labeled compute fraction:')
    print(f'  realized: {labeled_cost / total_cost:.4f}'
          + (f'   target (label_budget_fraction): {f_target}' if f_target is not None else ''))
    budget = meta.get('budget_ms')
    if budget is not None:
        overweight = sum(1 for c in batch_costs if c > budget)
        print(f'  overweight batches (cost > budget_ms): {overweight}')
    print()

    print('Batch size distribution:')
    for size, count in sorted(batch_sizes.items()):
        print(f'  batch_size={size:>3d}: {count:>6d} ({count / total_batches * 100:5.1f}%)')
    print()

    print('n_patches per sample:')
    print(f'  {_percentiles(n_patches)}  mean={np.mean(n_patches):.0f}')
    print()

    print('da_enc distribution (samples):')
    for key, count in sorted(da_counts.items(), key=lambda x: (x[0] is None, x[0])):
        print(f'  da_enc={key}: {count:>7d} ({count / total_samples * 100:5.1f}%)')
    print()

    print('depth distribution (samples):')
    for key, count in sorted(depth_counts.items()):
        print(f'  depth={key:>3d}: {count:>7d} ({count / total_samples * 100:5.1f}%)')
    print()

    print('--- Labeled pool composition (labeled samples only) ---')
    if n_labeled:
        print(f'positives/sample: {_percentiles(pos_per_sample)}  mean={np.mean(pos_per_sample):.2f}')
        print(f'negatives/sample: {_percentiles(neg_per_sample)}  mean={np.mean(neg_per_sample):.2f}')
        if positive_target_voxels.size:
            print(
                f'positive target voxels: {_percentiles(positive_target_voxels)}  '
                f'mean={np.mean(positive_target_voxels):.1f}'
            )
            for threshold in (2, 4, 8, 16, 32, 64, 128, 256):
                retained = np.mean(positive_target_voxels >= threshold)
                print(f'  retained at tau={threshold:>3d}: {retained:.4f}')
        print()
        print('Labeled samples per modality:')
        for mod, count in modality_counts.most_common():
            print(f'  {mod:>10s}: {count:>6d} ({count / n_labeled * 100:5.1f}%)')
        print()
        print(f'Labeled samples per dataset (top 20 of {len(dataset_counts)}):')
        for ds, count in dataset_counts.most_common(20):
            print(f'  {ds:>28s}: {count:>6d} ({count / n_labeled * 100:5.1f}%)')
        print()
        print(f'Most-frequent positive classes (top {args.top_classes} of {len(pos_class_counts)}):')
        for (src, name), count in pos_class_counts.most_common(args.top_classes):
            print(f'  {src + "/" + name:>40s}: {count:>6d}')
    else:
        print('(no labeled samples)')

    if min_lab == 0:
        n_zero = sum(1 for n in labeled_per_batch if n == 0)
        raise SystemExit(
            f'VERDICT: FAIL. {n_zero} zero-labeled batch(es); DDP-fatal under '
            f'find_unused_parameters=False. Regenerate with the two-stream packer.')
    print('VERDICT: PASS (min labeled per batch >= 1)')


if __name__ == '__main__':
    main()
