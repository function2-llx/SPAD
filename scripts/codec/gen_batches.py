"""Generate pre-computed batch stream for codec training.

Usage:
    python scripts/codec/gen_batches.py \
        --config configs/codec/flux2_lr4e4_v2.yaml \
        --batches 40000 \
        --output-dir precompute/codec/stream_seed42/ \
        --seed 42 \
        --batches-per-shard 1000 \
        --workers 8 \
        --chunk-size 1000
"""

from __future__ import annotations

import argparse
import datetime
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import msgpack
import numpy as np
import yaml

from pumit.codec.config import MAX_DA
from pumit.codec.datamodule import TransformConf
from pumit.codec.transforms import build_codec_pipeline
from pumit.data import build_training_data
from pumit.data.config import DepthTierConfig
from pumit.utils import deterministic_seed


@cache
def _load_state(config_path: str):
    """Loaded once per worker process, reused across chunks. Read-only."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    depth_tiers = {}
    for key, val in raw['depth_tiers'].items():
        parsed_key = None if key is None or str(key).lower() == 'null' else int(key)
        depth_tiers[parsed_key] = DepthTierConfig(
            tiers=tuple(val['tiers']),
            batch_sizes=tuple(val['batch_sizes']),
        )

    conf = TransformConf()
    smooth_spad = raw.get('smooth_spad', True)
    pipeline = build_codec_pipeline(conf, depth_tiers=depth_tiers, max_da=MAX_DA, smooth_spad=smooth_spad)

    train_data, _, _ = build_training_data(depth_tiers=depth_tiers, verbose=False, max_da=MAX_DA)
    records = train_data.to_dict('records')
    weights = np.array([r['weight'] for r in records], dtype=np.float64)
    weights = weights / weights.sum()

    return pipeline, records, weights, depth_tiers


def generate_chunk(args: tuple) -> list[dict]:
    """Produce one chunk of samples. Pure, deterministic, fixed-attempt."""
    chunk_id, chunk_size, seed, config_path = args
    pipeline, records, weights, _ = _load_state(config_path)

    rng = np.random.default_rng(deterministic_seed(seed, chunk_id))
    results = []
    for _ in range(chunk_size):
        idx = int(rng.choice(len(records), p=weights))
        state = records[idx]
        params = pipeline.sample_params(state, rng)
        if params is None:
            continue
        sp = params[0]
        results.append({
            'img': state['img'],
            'spacing': [float(x) for x in state['spacing']],
            'params': params,
            'key': (sp['da_enc'], sp['da_dec'], sp['patch_size'][0]),
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Generate codec batch stream')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batches', type=int, required=True,
                        help='Total micro-batches to generate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batches-per-shard', type=int, default=1_000_000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.batches % args.batches_per_shard == 0, (
        f"--batches ({args.batches}) must be divisible by --batches-per-shard ({args.batches_per_shard})"
    )
    config_path = str(Path(args.config).resolve())

    # Load state in main process for metadata/printing
    pipeline, records, weights, depth_tiers = _load_state(config_path)
    print(f'[gen_batches] Generating {args.batches} batches from {len(records)} samples')
    print(f'[gen_batches] Workers: {args.workers}, Seed: {args.seed}, Chunk size: {args.chunk_size}')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write meta.yaml upfront
    meta = {
        'seed': args.seed,
        'total_batches': args.batches,
        'batches_per_shard': args.batches_per_shard,
        'workers': args.workers,
        'chunk_size': args.chunk_size,
        'depth_tiers': {
            str(k): {'tiers': list(v.tiers), 'batch_sizes': list(v.batch_sizes)}
            for k, v in depth_tiers.items()
        },
        'date': datetime.datetime.now().isoformat(),
    }
    with open(args.output_dir / 'meta.yaml', 'w') as f:
        yaml.dump(meta, f, default_flow_style=False)

    # Bucketing state (main thread only)
    buckets: dict[tuple, list] = {}
    completed_batches: list[dict] = []
    total_emitted = 0
    shard_count = 0
    target = args.batches
    t_start = time.time()

    from tqdm import tqdm
    pbar = tqdm(total=target, desc='Generating', unit='batch')

    def bucket_sample(sample: dict):
        nonlocal total_emitted, shard_count
        key = sample.pop('key')

        tier_key = key[0] if key[0] is not None else None
        if tier_key is not None:
            tier_key = min(tier_key, MAX_DA)
        tier_cfg = depth_tiers[tier_key]
        depth_idx = tier_cfg.tiers.index(key[2])
        batch_size = tier_cfg.batch_sizes[depth_idx]

        if key not in buckets:
            buckets[key] = []
        buckets[key].append(sample)

        if len(buckets[key]) >= batch_size:
            batch_samples = buckets.pop(key)
            completed_batches.append({
                'da_enc': key[0],
                'da_dec': key[1],
                'patch_size': batch_samples[0]['params'][0]['patch_size'],
                'samples': batch_samples,
            })
            total_emitted += 1
            pbar.update(1)

            if len(completed_batches) >= args.batches_per_shard:
                shard_path = args.output_dir / f'shard_{shard_count:05d}.msgpack'
                with open(shard_path, 'wb') as f:
                    msgpack.pack({'batches': completed_batches}, f)
                completed_batches.clear()
                shard_count += 1

    # Bounded producer-consumer (sliding window)
    max_inflight = args.workers * 2
    chunk_id = 0
    pending: deque = deque()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Seed the window
        for _ in range(max_inflight):
            pending.append(pool.submit(generate_chunk,
                (chunk_id, args.chunk_size, args.seed, config_path)))
            chunk_id += 1

        # Consumer loop
        while total_emitted < target:
            future = pending.popleft()
            for sample in future.result():
                bucket_sample(sample)
                if total_emitted >= target:
                    break
            if total_emitted < target:
                pending.append(pool.submit(generate_chunk,
                    (chunk_id, args.chunk_size, args.seed, config_path)))
                chunk_id += 1

        # Cancel remaining queued futures
        for f in pending:
            f.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    pbar.close()

    # Flush remaining completed batches
    if completed_batches:
        shard_path = args.output_dir / f'shard_{shard_count:05d}.msgpack'
        with open(shard_path, 'wb') as f:
            msgpack.pack({'batches': completed_batches}, f)
        shard_count += 1

    total_elapsed = time.time() - t_start
    print(f'[gen_batches] Done: {total_emitted} batches in {shard_count} shards, {chunk_id} chunks ({total_elapsed:.1f}s)')


if __name__ == '__main__':
    main()
