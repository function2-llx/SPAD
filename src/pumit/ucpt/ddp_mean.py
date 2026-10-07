"""DDP global-mean loss normalization.

Under variable-shape / variable-K packing each rank reduces a different number of items, so a rank-local mean
followed by DDP's cross-rank gradient average optimizes ``mean_rank(local_mean_r)`` rather than the global
per-item mean. Scaling a local *sum* by ``W / global_count`` fixes this: DDP's own 1/W averaging then collapses
to ``grad(global_sum / global_count)``, the true global mean, because ``global_count`` is a detached constant.

Counts are reduced asynchronously so the collective overlaps the model body. Detached loss sums are reduced
separately only on logging steps; skipping those metric collectives does not change the training gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


def _f64_scalar(x: Tensor | int, device: torch.device) -> Tensor:
    if torch.is_tensor(x):
        return x.to(torch.float64)
    return torch.tensor(float(x), device=device, dtype=torch.float64)


@dataclass
class PendingGlobalCounts:
    """An asynchronous count reduction started before the model body."""

    local: Tensor
    reduced: Tensor
    work: Any | None
    world: int


def start_global_count_reduce(
    counts: list[Tensor | int],
    device: torch.device,
) -> PendingGlobalCounts:
    """Launch an asynchronous all-reduce for loss denominators."""
    local = torch.stack([_f64_scalar(c, device) for c in counts])
    reduced = local.clone()
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    work = dist.all_reduce(reduced, op=dist.ReduceOp.SUM, async_op=True) if distributed else None
    world = dist.get_world_size() if distributed else 1
    return PendingGlobalCounts(local=local, reduced=reduced, work=work, world=world)


def finish_global_count_reduce(pending: PendingGlobalCounts) -> tuple[Tensor, Tensor, int]:
    """Wait for global denominators and return clamped local/global counts."""
    if pending.work is not None:
        pending.work.wait()
    return pending.local.clamp(min=1), pending.reduced.clamp(min=1), pending.world


def reduce_detached_sums(detached_sums: list[Tensor]) -> Tensor:
    """Synchronously reduce detached metric numerators."""
    payload = torch.stack([s.detach().to(torch.float64) for s in detached_sums])
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return payload
