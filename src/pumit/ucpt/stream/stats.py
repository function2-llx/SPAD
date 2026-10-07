"""Compute normalization statistics over logical latent rows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.distributed as dist
from tqdm import tqdm

from .build import distributed_context
from .manifest import STATS_CHUNK_ROWS, load_manifest, require_absent, write_json

def _combine_moments(
    n_a: int,
    mean_a: torch.Tensor,
    m2_a: torch.Tensor,
    n_b: int,
    mean_b: torch.Tensor,
    m2_b: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    if n_a == 0:
        return n_b, mean_b, m2_b
    if n_b == 0:
        return n_a, mean_a, m2_a
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + delta.square() * (n_a * n_b / n)
    return n, mean, m2


def _read_logical_moments(
    path: Path, logical_rows: int
) -> tuple[int, torch.Tensor, torch.Tensor]:
    n_total = 0
    mean_total = torch.zeros(32, dtype=torch.float64)
    m2_total = torch.zeros(32, dtype=torch.float64)
    with safe_open(str(path), framework="pt") as file:
        latent_slice = file.get_slice("latents")
        stored_rows, latent_dim = latent_slice.get_shape()
        if latent_dim != 32 or logical_rows > stored_rows:
            raise ValueError(
                f"{path}: logical rows/dim ({logical_rows}, 32) incompatible with {stored_rows, latent_dim}"
            )
        for start in range(0, logical_rows, STATS_CHUNK_ROWS):
            chunk = latent_slice[
                start : min(start + STATS_CHUNK_ROWS, logical_rows)
            ].to(torch.float64)
            n_chunk = chunk.shape[0]
            mean_chunk = chunk.mean(dim=0)
            m2_chunk = (chunk - mean_chunk).square().sum(dim=0)
            n_total, mean_total, m2_total = _combine_moments(
                n_total,
                mean_total,
                m2_total,
                n_chunk,
                mean_chunk,
                m2_chunk,
            )
    return n_total, mean_total, m2_total


def cmd_stats(args) -> None:
    stream_dir = args.stream.resolve()
    latent_dir = stream_dir / "latents"
    rows = load_manifest(stream_dir)
    stats_path = latent_dir / "stats.safetensors"
    tmp_path = latent_dir / "stats.safetensors.tmp"
    receipt_path = stream_dir / 'latent-stats.json'
    require_absent(stats_path)
    require_absent(receipt_path)

    rank, world_size = distributed_context()
    initialized_here = False
    if world_size > 1:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        initialized_here = True
    if rank == 0:
        tmp_path.unlink(missing_ok=True)
    if dist.is_initialized():
        dist.barrier()

    try:
        owned_rows = rows[rank::world_size]
        n_local = 0
        mean_local = torch.zeros(32, dtype=torch.float64)
        m2_local = torch.zeros(32, dtype=torch.float64)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _read_logical_moments,
                    latent_dir / f"shard_{row['shard_id']:05d}.safetensors",
                    row["logical_latent_rows"],
                ): row["shard_id"]
                for row in owned_rows
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Logical latent stats rank {rank}",
                unit="shard",
            ):
                n_shard, mean_shard, m2_shard = future.result()
                n_local, mean_local, m2_local = _combine_moments(
                    n_local,
                    mean_local,
                    m2_local,
                    n_shard,
                    mean_shard,
                    m2_shard,
                )

        local = (n_local, mean_local.tolist(), m2_local.tolist())
        if dist.is_initialized():
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(local, gathered, dst=0)
        else:
            gathered = [local]

        if rank == 0:
            assert gathered is not None
            n_global = 0
            mean_global = torch.zeros(32, dtype=torch.float64)
            m2_global = torch.zeros(32, dtype=torch.float64)
            for n_rank, mean_rank, m2_rank in gathered:
                n_global, mean_global, m2_global = _combine_moments(
                    n_global,
                    mean_global,
                    m2_global,
                    n_rank,
                    torch.tensor(mean_rank, dtype=torch.float64),
                    torch.tensor(m2_rank, dtype=torch.float64),
                )
            expected_rows = sum(row["logical_latent_rows"] for row in rows)
            if n_global != expected_rows:
                raise ValueError(f"stats counted {n_global} rows, expected {expected_rows}")
            mean = mean_global.float()
            std = torch.sqrt(m2_global / n_global).float()
            save_file(
                {
                    "mean": mean,
                    "std": std,
                    "count": torch.tensor(n_global, dtype=torch.int64),
                },
                str(tmp_path),
            )
            tmp_path.rename(stats_path)
            write_json(
                receipt_path,
                {
                    "logical_rows": n_global,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                },
            )
            print(f"computed stats from {n_global:,} logical latent rows")
        if dist.is_initialized():
            dist.barrier()
    finally:
        if initialized_here:
            dist.destroy_process_group()
