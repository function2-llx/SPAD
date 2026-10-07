"""FixedGridPatchEmbed3d unit coverage (hub-independent, tiny random-init sources)."""
import pytest
import timm
import torch
from torch import nn

from pumit.downstream.cls.backbones._timm_patch_embed import FixedGridPatchEmbed3d


def _source(patch: int = 4, in_chans: int = 3, embed_dim: int = 8, img_size: int = 16,
            norm_layer=None) -> timm.layers.PatchEmbed:
    src = timm.layers.PatchEmbed(img_size=img_size, patch_size=patch, in_chans=in_chans,
                                 embed_dim=embed_dim, norm_layer=norm_layer)
    with torch.no_grad():
        src.proj.weight.copy_(torch.randn_like(src.proj.weight))
        src.proj.bias.copy_(torch.randn_like(src.proj.bias))
    return src


def test_from_2d_derives_configuration_from_source():
    src = _source()
    pe3d = FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)
    assert pe3d.proj.in_channels == src.proj.in_channels
    assert pe3d.proj.out_channels == src.proj.out_channels
    assert pe3d.proj.kernel_size == (4, 4, 4)
    assert pe3d.proj.stride == (4, 4, 4)
    assert pe3d.proj.weight.dtype == src.proj.weight.dtype
    assert pe3d.proj.weight.device == src.proj.weight.device
    assert pe3d.grid_size == (4, 4, 4)
    assert pe3d.num_patches == 64


def test_from_2d_uniform_inflates_weights_and_copies_bias():
    src = _source()
    pe3d = FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)
    p = src.proj.kernel_size[0]
    w3d = pe3d.proj.weight
    # every depth slice carries an equal share, and the depth-sum recovers the 2D kernel
    for d in range(p):
        torch.testing.assert_close(w3d[:, :, d], src.proj.weight / p)
    torch.testing.assert_close(w3d.sum(dim=2), src.proj.weight)
    assert torch.equal(pe3d.proj.bias, src.proj.bias)
    assert pe3d.proj.bias is not src.proj.bias  # copied, not aliased


def test_from_2d_applies_2d_filter_to_patch_depth_mean():
    src = _source()
    pe3d = FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)
    x = torch.randn(2, 3, 16, 16, 16)
    out = pe3d(x)                       # (B, 64, 8), NLC, d-major
    out_grid = out.transpose(1, 2).reshape(2, 8, 4, 4, 4)
    p = 4
    for d in range(4):
        depth_mean = x[:, :, d * p:(d + 1) * p].mean(dim=2)
        expected = torch.conv2d(depth_mean, src.proj.weight, src.proj.bias, stride=p)
        torch.testing.assert_close(out_grid[:, :, d], expected)


def test_from_2d_preserves_source_norm():
    src = _source(norm_layer=nn.LayerNorm)
    pe3d = FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)
    x = torch.randn(2, 3, 16, 16, 16)
    out = pe3d(x)
    raw = pe3d.proj(x).flatten(2).transpose(1, 2)
    torch.testing.assert_close(out, src.norm(raw))


def test_from_2d_output_is_nlc():
    src = _source()
    pe3d = FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)
    out = pe3d(torch.randn(2, 3, 16, 16, 16))
    assert out.shape == (2, 64, 8)


def test_from_2d_rejects_non_divisible_image_size():
    src = _source()
    with pytest.raises(ValueError, match='divisible'):
        FixedGridPatchEmbed3d.from_2d(src, image_size_3d=18)


def test_from_2d_rejects_overlapping_stride():
    src = _source()
    src.proj.stride = (2, 2)
    with pytest.raises(ValueError, match='stride'):
        FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)


def test_strict_state_dict_roundtrip():
    src = _source()
    a = FixedGridPatchEmbed3d.from_2d(src, image_size_3d=16)
    b = FixedGridPatchEmbed3d.from_2d(_source(), image_size_3d=16)
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(2, 3, 16, 16, 16)
    assert torch.equal(a(x), b(x))


def test_patch_embed_does_not_gate_on_timm_version(monkeypatch):
    monkeypatch.setattr(timm, '__version__', '0.0.0')
    embedding = FixedGridPatchEmbed3d.from_2d(_source(), image_size_3d=16)
    assert embedding(torch.randn(1, 3, 16, 16, 16)).shape == (1, 64, 8)
