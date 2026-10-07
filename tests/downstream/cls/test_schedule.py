"""LR-schedule tests: 5% linear warmup into cosine, per-group base LRs preserved.

Runs peaked at epoch 1 then declined under bare cosine -- the classic no-warmup signature
(full LR applied while the randomly initialized head is still untrained). Warmup fixes the
start; these tests pin the shape so a future edit cannot silently drop it.
"""
import pytest
import torch
from torch import nn

from pumit.downstream.cls.optim import build_warmup_cosine


def _opt(lrs=(1e-5, 1e-3)):
    params = [nn.Linear(4, 4).parameters(), nn.Linear(4, 2).parameters()]
    return torch.optim.AdamW([{'params': list(p), 'lr': lr} for p, lr in zip(params, lrs)])


def _trace(sched, opt, steps):
    out = []
    for _ in range(steps):
        out.append([g['lr'] for g in opt.param_groups])
        opt.step()
        sched.step()
    return out


def test_warmup_ramps_from_near_zero_to_base():
    opt = _opt()
    steps, warmup = 100, 5
    sched = build_warmup_cosine(opt, total_steps=steps, warmup_steps=warmup)
    trace = _trace(sched, opt, steps)
    # first step is a small fraction of base, not the full LR
    assert trace[0][0] < 1e-5 * 0.5, 'warmup must start well below the base LR'
    # by the end of warmup each group is at (approximately) its own base LR
    assert trace[warmup][0] == pytest.approx(1e-5, rel=0.02)
    assert trace[warmup][1] == pytest.approx(1e-3, rel=0.02)
    # strictly increasing through warmup
    assert all(trace[i][0] < trace[i + 1][0] for i in range(warmup - 1))


def test_cosine_decays_to_near_zero_after_warmup():
    opt = _opt()
    steps, warmup = 100, 5
    sched = build_warmup_cosine(opt, total_steps=steps, warmup_steps=warmup)
    trace = _trace(sched, opt, steps)
    peak = trace[warmup][0]
    assert trace[-1][0] < peak * 0.02, 'cosine must anneal to ~0 by the final step'
    # monotone decreasing after warmup
    post = [t[0] for t in trace[warmup:]]
    assert all(a >= b for a, b in zip(post, post[1:])), 'post-warmup LR must not increase'


def test_per_group_ratio_is_preserved_throughout():
    """The encoder/head LR ratio (and any LLRD per-layer ratios) must hold at every step."""
    opt = _opt((1e-5, 1e-3))
    sched = build_warmup_cosine(opt, total_steps=60, warmup_steps=3)
    for encoder_lr, head_lr in _trace(sched, opt, 60):
        if encoder_lr > 0:
            assert head_lr / encoder_lr == pytest.approx(100.0, rel=1e-6)


def test_warmup_fraction_of_five_percent():
    """total=330 (10 epochs x 33 steps at bs=32) -> 16 warmup steps at 5%."""
    from pumit.downstream.cls.optim import warmup_steps_for
    assert warmup_steps_for(330, 0.05) == 16
    assert warmup_steps_for(100, 0.05) == 5
    assert warmup_steps_for(10, 0.05) == 1, 'always at least one warmup step'
    assert warmup_steps_for(330, 0.0) == 0, 'zero fraction disables warmup'


def test_zero_warmup_is_plain_cosine():
    opt = _opt()
    sched = build_warmup_cosine(opt, total_steps=50, warmup_steps=0)
    trace = _trace(sched, opt, 50)
    assert trace[0][0] == pytest.approx(1e-5), 'no warmup: step 0 is already at base LR'
    assert trace[-1][0] < 1e-5 * 0.02
