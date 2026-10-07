"""Stochastic-depth behavior for packed ViT sequences."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from pumit.model.vit import Block, ViT, ViTConfig


class _ConstantResidual(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, *args):
        return torch.full_like(x, self.value)


def _vit(depth=2, drop_path=0.5, grad_ckpt=False):
    config = ViTConfig(
        hidden_size=32, num_hidden_layers=depth, num_attention_heads=2,
        intermediate_size=64, patch_size=2, num_register_tokens=1,
        drop_path_rate=drop_path, grad_ckpt=grad_ckpt,
    )
    return ViT(config)


def _constant_residuals(vit):
    vit.norm = nn.Identity()
    for block in vit.layer:
        block.attention = _ConstantResidual(1)
        block.mlp = _ConstantResidual(10)


def _inputs(lengths):
    n = sum(lengths)
    x = torch.randn(1, n, 32)
    rope = torch.zeros(1, n, 2, 16)
    rope[..., 0, :] = 1
    bias = BlockDiagonalMask.from_seqlens(lengths, device=torch.device('cpu'))
    return x, rope, bias


def _sdpa_attention(q, k, v, attn_bias=None):
    mask = None
    if attn_bias is not None:
        indices = torch.arange(q.shape[1], device=q.device)
        sequence_ids = torch.bucketize(indices, attn_bias.q_seqinfo.seqstart[1:], right=True)
        mask = sequence_ids[:, None] == sequence_ids[None, :]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
    ).transpose(1, 2)


def test_drop_path_depth_ramp():
    vit = _vit(depth=24, drop_path=0.1)
    probabilities = [getattr(block.drop_path, 'drop_prob', 0) for block in vit.layer]
    assert probabilities == pytest.approx([0.1 * i / 23 for i in range(24)])


def test_packed_drop_path_shares_tokens_and_samples_branches_independently():
    vit = _vit()
    _constant_residuals(vit)
    lengths = [2, 3, 4, 5] * 16
    x, rope, bias = _inputs(lengths)
    torch.manual_seed(7)
    out = vit(torch.zeros_like(x), rope, bias)
    values = []
    for sequence in out[0].split(lengths):
        torch.testing.assert_close(sequence, sequence[0, 0].expand_as(sequence))
        values.append(sequence[0, 0].item())
    # Block 0 always adds 11; block 1 independently keeps attention (+2) and MLP (+20).
    assert set(values) == {11, 13, 31, 33}


def test_unpacked_drop_path_keeps_timm_batch_semantics():
    block = Block(dim=32, num_heads=2, intermediate_size=64, drop_path=0.5)
    block.attention = _ConstantResidual(1)
    block.mlp = _ConstantResidual(10)
    torch.manual_seed(7)
    out = block(torch.zeros(64, 3, 32), torch.empty(0))
    torch.testing.assert_close(out, out[:, :1, :1].expand_as(out))
    assert set(out[:, 0, 0].tolist()) == {0, 2, 20, 22}


@pytest.mark.parametrize('training,drop_path', [(False, 0.5), (True, 0)])
def test_eval_or_zero_drop_path_does_not_build_masks(training, drop_path, monkeypatch):
    vit = _vit(drop_path=drop_path).train(training)
    _constant_residuals(vit)
    x, rope, bias = _inputs([3, 5, 4])

    def unexpected_bucketize(*args, **kwargs):
        raise AssertionError('drop path must not build token indices')

    monkeypatch.setattr(torch, 'bucketize', unexpected_bucketize)
    before = torch.random.get_rng_state()
    out = vit(torch.zeros_like(x), rope, bias)
    torch.testing.assert_close(out, torch.full_like(out, 22))
    assert torch.equal(before, torch.random.get_rng_state())


def test_packed_checkpoint_preserves_forward_and_backward_rng(monkeypatch):
    monkeypatch.setattr('pumit.model.vit.xops.memory_efficient_attention', _sdpa_attention)
    plain = _vit(depth=3, drop_path=0.1)
    checkpointed = deepcopy(plain)
    checkpointed._grad_ckpt = True
    x, rope, bias = _inputs([3, 5, 4])
    results = []
    for vit in (plain, checkpointed):
        leaf = x.detach().clone().requires_grad_()
        torch.manual_seed(17)
        out = vit(leaf, rope, bias)
        out.sin().sum().backward()
        results.append((out, leaf.grad, [p.grad for p in vit.layer.parameters()], torch.random.get_rng_state()))
    for a, b in zip(results[0][:3], results[1][:3]):
        torch.testing.assert_close(a, b)
    assert torch.equal(results[0][3], results[1][3])


def test_compiled_packed_drop_path_reads_updated_boundaries():
    vit = _vit()
    _constant_residuals(vit)
    compiled = torch.compile(vit, backend='aot_eager', fullgraph=True, dynamic=True)
    for lengths in ([3, 5, 4], [4, 3, 5], [5, 6, 3, 2]):
        x, rope, bias = _inputs(lengths)
        x.zero_().requires_grad_()
        torch.manual_seed(23)
        expected = vit(x, rope, bias)
        torch.manual_seed(23)
        actual = compiled(x, rope, bias)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
