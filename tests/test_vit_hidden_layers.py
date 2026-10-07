import pytest
import torch
from pumit.model.vit import ViT, ViTConfig
from xformers.ops.fmha.attn_bias import BlockDiagonalMask


def _make_vit(depth=6):
    cfg = ViTConfig(
        hidden_size=128, num_hidden_layers=depth, num_attention_heads=4,
        intermediate_size=256, patch_size=16, num_register_tokens=2,
        grad_ckpt=True,
    )
    return ViT(cfg, skip_embed=True).cuda().eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
def test_hidden_layers_returns_only_requested():
    vit = _make_vit(depth=6)
    n_prefix = vit.n_prefix
    x = torch.randn(1, n_prefix + 10, 128, device='cuda')
    rope = torch.zeros(1, n_prefix + 10, 2, 32, device='cuda')
    rope[..., 0, :] = 1.0  # identity
    bias = BlockDiagonalMask.from_seqlens([n_prefix + 10])

    with torch.no_grad(), torch.amp.autocast('cuda'):
        out, hiddens = vit(x, rope, bias, hidden_layers={2, 4})
    assert len(hiddens) == 2, f'expected 2 hidden states, got {len(hiddens)}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
def test_hidden_layers_returns_correct_layers():
    """Cross-check: hidden_layers={2,4} matches all_h[1] and all_h[3] (1-indexed)."""
    vit = _make_vit(depth=6)
    n_prefix = vit.n_prefix
    x = torch.randn(1, n_prefix + 10, 128, device='cuda')
    rope = torch.zeros(1, n_prefix + 10, 2, 32, device='cuda')
    rope[..., 0, :] = 1.0
    bias = BlockDiagonalMask.from_seqlens([n_prefix + 10])

    with torch.no_grad(), torch.amp.autocast('cuda'):
        _, hiddens = vit(x, rope, bias, hidden_layers={2, 4})
        _, all_h = vit(x, rope, bias, return_hidden=True)
    torch.testing.assert_close(hiddens[0], all_h[1])
    torch.testing.assert_close(hiddens[1], all_h[3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
def test_hidden_layers_default_false_returns_none():
    vit = _make_vit(depth=6)
    n_prefix = vit.n_prefix
    x = torch.randn(1, n_prefix + 10, 128, device='cuda')
    rope = torch.zeros(1, n_prefix + 10, 2, 32, device='cuda')
    rope[..., 0, :] = 1.0
    bias = BlockDiagonalMask.from_seqlens([n_prefix + 10])

    with torch.no_grad(), torch.amp.autocast('cuda'):
        result = vit(x, rope, bias)
    assert isinstance(result, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
def test_hidden_layers_carries_gradient():
    """hidden_layers path must carry gradient (seg loss flows through aux)."""
    vit = _make_vit(depth=6)
    vit.train()
    n_prefix = vit.n_prefix
    x = torch.randn(1, n_prefix + 10, 128, device='cuda', requires_grad=True)
    rope = torch.zeros(1, n_prefix + 10, 2, 32, device='cuda')
    rope[..., 0, :] = 1.0
    bias = BlockDiagonalMask.from_seqlens([n_prefix + 10])

    with torch.amp.autocast('cuda'):
        out, hiddens = vit(x, rope, bias, hidden_layers={3})
    hiddens[0].float().pow(2).sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
def test_hidden_layers_and_return_hidden_mutually_exclusive():
    vit = _make_vit(depth=6)
    n_prefix = vit.n_prefix
    x = torch.randn(1, n_prefix + 10, 128, device='cuda')
    rope = torch.zeros(1, n_prefix + 10, 2, 32, device='cuda')
    rope[..., 0, :] = 1.0
    bias = BlockDiagonalMask.from_seqlens([n_prefix + 10])

    with pytest.raises(AssertionError):
        vit(x, rope, bias, return_hidden=True, hidden_layers={1})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA for xformers BDM')
def test_hidden_layers_out_of_range_raises():
    vit = _make_vit(depth=6)
    n_prefix = vit.n_prefix
    x = torch.randn(1, n_prefix + 10, 128, device='cuda')
    rope = torch.zeros(1, n_prefix + 10, 2, 32, device='cuda')
    rope[..., 0, :] = 1.0
    bias = BlockDiagonalMask.from_seqlens([n_prefix + 10])

    with pytest.raises(AssertionError):
        vit(x, rope, bias, hidden_layers={30})
