"""2-rank gradient-equivalence test for the global-mean loss normalization.

Proves the core P1-1 identity: scaling a local sum by ``W / global_count`` and letting DDP average gradients
by 1/W yields exactly the single-process global-mean gradient over the concatenated batch. Uses Gloo (CPU) so
it runs anywhere. Covers imbalanced counts, a local-zero-count rank, a global-zero-count term, and the
balanced regression case.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from pumit.ucpt.ddp_mean import finish_global_count_reduce, start_global_count_reduce


def _linear_sum_and_count(w: nn.Parameter, x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """A toy elementwise loss: sum of (w * x)^2 over all elements, with its element count."""
    y = (w * x) ** 2
    return y.float().sum(), y.numel()


def _global_mean_grad(w_init: torch.Tensor, per_rank_x: list[torch.Tensor]) -> torch.Tensor:
    """Single-process reference: grad of (Σ_r Σ elems / Σ_r count) w.r.t. w."""
    w = nn.Parameter(w_init.clone())
    total_sum = torch.zeros([], dtype=torch.float32)
    total_count = 0
    for x in per_rank_x:
        s, n = _linear_sum_and_count(w, x)
        total_sum = total_sum + s
        total_count += n
    loss = total_sum / max(total_count, 1)
    loss.backward()
    return w.grad.clone()


class _ScaledSumModule(nn.Module):
    """forward(x) -> scaled local loss = local_sum * (W / global_count). Running the whole computation
    inside forward arms DDP's reducer so backward triggers the 1/W gradient average."""

    def __init__(self, w_init: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(w_init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_sum, local_count = _linear_sum_and_count(self.weight, x)
        pending = start_global_count_reduce([local_count], torch.device('cpu'))
        _, n_global, world = finish_global_count_reduce(pending)
        return local_sum * (world / n_global[0]).to(torch.float32)


def _worker(
    rank: int,
    world: int,
    per_rank_x: list[torch.Tensor],
    w_init: torch.Tensor,
    init_method: str,
    out_q,
):
    dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=world)
    try:
        # find_unused_parameters and a zero-element input: an empty-count rank still produces a
        # (differentiable) zero loss touching weight, so the param participates every step.
        ddp = DDP(_ScaledSumModule(w_init))
        loss = ddp(per_rank_x[rank])
        loss.backward()
        if rank == 0:
            # Send plain values rather than a Tensor backed by a shared FD. The worker exits immediately after
            # put(), so Tensor storage transfer can race the parent's deserialization.
            out_q.put(ddp.module.weight.grad.detach().tolist())
    finally:
        dist.destroy_process_group()


def _run_2rank(per_rank_x: list[torch.Tensor], w_init: torch.Tensor) -> torch.Tensor:
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_method = f'file://{tmp_dir}/dist-init'
        procs = [ctx.Process(target=_worker, args=(r, 2, per_rank_x, w_init, init_method, q)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f'worker exited {p.exitcode}'
        grad = torch.tensor(q.get(timeout=5), dtype=w_init.dtype)
    q.close()
    q.join_thread()
    return grad


@pytest.mark.parametrize('n0,n1', [(2, 5), (3, 3)])
def test_2rank_grad_equals_global_mean(n0, n1):
    """Imbalanced (2 vs 5) and balanced (3 vs 3) counts: post-DDP grad == single-process global-mean grad."""
    torch.manual_seed(0)
    w_init = torch.tensor([[0.7]])
    per_rank_x = [torch.randn(n0, 1), torch.randn(n1, 1)]
    ddp_grad = _run_2rank(per_rank_x, w_init)
    ref_grad = _global_mean_grad(w_init, per_rank_x)
    assert torch.allclose(ddp_grad, ref_grad, atol=1e-6), \
        f'DDP global-mean grad {ddp_grad} != reference {ref_grad}'


def test_2rank_local_zero_count():
    """One rank has zero elements (count=0 locally): the global count still sums the other rank, and the
    grad matches the single-process mean over the non-empty rank alone."""
    torch.manual_seed(1)
    w_init = torch.tensor([[0.5]])
    per_rank_x = [torch.randn(4, 1), torch.zeros(0, 1)]
    ddp_grad = _run_2rank(per_rank_x, w_init)
    ref_grad = _global_mean_grad(w_init, per_rank_x)  # rank1 contributes 0 sum, 0 count
    assert torch.allclose(ddp_grad, ref_grad, atol=1e-6)


def test_2rank_global_zero_count():
    """Both ranks empty (global count 0): clamp(min=1) with zero sums gives finite zero grad, no hang."""
    w_init = torch.tensor([[0.9]])
    per_rank_x = [torch.zeros(0, 1), torch.zeros(0, 1)]
    ddp_grad = _run_2rank(per_rank_x, w_init)
    assert torch.isfinite(ddp_grad).all()
    assert torch.allclose(ddp_grad, torch.zeros_like(ddp_grad), atol=1e-8)
