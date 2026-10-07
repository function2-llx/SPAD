"""Tests for SPAD conv promote mode (da=1 keeps full kernel)."""

import torch
import pytest

from pumit.spadop.conv import (
    Conv3d,
    SPADAvgPool,
    SPADConv3d_K3S1,
    SPADConv3d_K3S2,
    SPADConvTranspose3d_K2S2,
    SPADConvTranspose3d_K3S2,
)


@pytest.fixture
def conv():
    """Create a K3S1 conv with deterministic weights."""
    torch.manual_seed(42)
    c = SPADConv3d_K3S1(4, 8, kernel_size=3, padding=1)
    return c


@pytest.fixture
def x():
    """Input tensor: (B=1, C=4, D=6, H=8, W=8)."""
    torch.manual_seed(0)
    return torch.randn(1, 4, 6, 8, 8)


def test_da0_full_kernel(conv: SPADConv3d_K3S1, x: torch.Tensor):
    """da=0 uses full [3,3,3] kernel, output shape preserves spatial dims."""
    out = conv(x, da=0)
    assert out.shape == (1, 8, 6, 8, 8), f"Expected (1,8,6,8,8), got {out.shape}"


def test_da1_full_kernel_same_as_da0(conv: SPADConv3d_K3S1, x: torch.Tensor):
    """da=1 (promote) uses full kernel, output identical to da=0."""
    out0 = conv(x, da=0)
    out1 = conv(x, da=1)
    assert out0.shape == out1.shape
    assert torch.allclose(out0, out1), "da=1 should produce identical output to da=0"


def test_da2_collapsed(conv: SPADConv3d_K3S1, x: torch.Tensor):
    """da>=2 uses collapsed kernel, output differs from da=1."""
    out1 = conv(x, da=1)
    out2 = conv(x, da=2)
    # Collapsed kernel reduces depth padding to 0, so depth shrinks
    assert out2.shape == (1, 8, 6, 8, 8), f"Expected (1,8,6,8,8), got {out2.shape}"
    # Output values must differ (collapsed kernel != full kernel in general)
    assert not torch.allclose(out1, out2), "da>=2 should produce different output from da=1"


def test_da_none_collapsed(conv: SPADConv3d_K3S1, x: torch.Tensor):
    """da=None behaves same as da>=2 (collapsed kernel)."""
    out2 = conv(x, da=2)
    out_none = conv(x, da=None)
    assert out_none.shape == out2.shape
    assert torch.allclose(out_none, out2), "da=None should produce identical output to da=2"


def test_lkr_is_disabled_by_default_without_state_dict_parameters():
    conv = SPADConv3d_K3S1(4, 8, kernel_size=3, padding=1)

    assert conv.learnable_kernel_reduction is False
    assert conv.kernel_reduction_delta is None
    assert 'kernel_reduction_delta' not in conv.state_dict()


def test_lkr_zero_initialization_exactly_matches_sum_reduction():
    fixed = SPADConv3d_K3S1(4, 8, kernel_size=3, padding=1)
    learned = SPADConv3d_K3S1(
        4,
        8,
        kernel_size=3,
        padding=1,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        learned.weight.copy_(fixed.weight)
        learned.bias.copy_(fixed.bias)
    x = torch.randn(1, 4, 6, 8, 8)

    assert learned.learnable_kernel_reduction is True
    assert learned.kernel_reduction_delta.shape == (8, 3, 3, 3)
    assert torch.count_nonzero(learned.kernel_reduction_delta) == 0
    torch.testing.assert_close(learned(x, da=2), fixed(x, da=2))


def test_lkr_uses_independent_depth_weights_at_each_kernel_position():
    conv = SPADConv3d_K3S1(
        2,
        3,
        kernel_size=3,
        padding=1,
        bias=False,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel()).reshape_as(conv.weight))
        conv.kernel_reduction_delta.copy_(
            torch.linspace(-0.5, 0.5, conv.kernel_reduction_delta.numel()).reshape_as(
                conv.kernel_reduction_delta
            )
        )

    expected = (
        conv.weight
        * (1 + conv.kernel_reduction_delta).unsqueeze(1)
    ).sum(dim=2, keepdim=True)

    torch.testing.assert_close(conv._reduce_kernel_depth(), expected)


def test_lkr_only_changes_collapsed_routes_and_receives_gradient():
    fixed = SPADConv3d_K3S1(2, 3, kernel_size=3, padding=1, bias=False)
    learned = SPADConv3d_K3S1(
        2,
        3,
        kernel_size=3,
        padding=1,
        bias=False,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        learned.weight.copy_(fixed.weight)
        learned.kernel_reduction_delta[0, 0, 0, 0] = 0.5
    x = torch.randn(1, 2, 5, 7, 9)

    torch.testing.assert_close(learned(x, da=0), fixed(x, da=0))
    torch.testing.assert_close(learned(x, da=1), fixed(x, da=1))
    assert not torch.allclose(learned(x, da=2), fixed(x, da=2))

    learned(x, da=2).sum().backward()
    assert learned.kernel_reduction_delta.grad is not None
    assert torch.count_nonzero(learned.kernel_reduction_delta.grad) > 0


def test_lkr_covers_k3s2_size_clamped_reduction():
    conv = SPADConv3d_K3S2(
        2,
        3,
        kernel_size=3,
        stride=2,
        bias=False,
        learnable_kernel_reduction=True,
    )
    x = torch.randn(1, 2, 5, 7, 9)

    conv(x, da=2, stride_override=(1, 1, 1)).sum().backward()

    assert conv.kernel_reduction_delta.grad is not None
    assert torch.count_nonzero(conv.kernel_reduction_delta.grad) > 0


def test_lkr_covers_k3s2_depth_reduction_without_changing_full_routes():
    fixed = SPADConv3d_K3S2(
        2,
        3,
        kernel_size=3,
        stride=2,
        bias=False,
    )
    learned = SPADConv3d_K3S2(
        2,
        3,
        kernel_size=3,
        stride=2,
        bias=False,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        learned.weight.copy_(fixed.weight)
        learned.kernel_reduction_delta[0, 0, 0, 0] = 0.5
    x = torch.randn(1, 2, 7, 9, 11)

    torch.testing.assert_close(learned(x, da=0), fixed(x, da=0))
    torch.testing.assert_close(learned(x, da=1), fixed(x, da=1))
    assert not torch.allclose(learned(x, da=2), fixed(x, da=2))

    learned(x, da=2).sum().backward()
    assert learned.kernel_reduction_delta.grad is not None
    assert torch.count_nonzero(learned.kernel_reduction_delta.grad) > 0


def test_lkr_transposed_conv_broadcasts_over_input_channels():
    conv = SPADConvTranspose3d_K2S2(
        4,
        3,
        kernel_size=2,
        stride=2,
        bias=False,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        conv.kernel_reduction_delta.copy_(
            torch.linspace(-0.25, 0.25, conv.kernel_reduction_delta.numel()).reshape_as(
                conv.kernel_reduction_delta
            )
        )

    expected = (
        conv.weight
        * (1 + conv.kernel_reduction_delta).unsqueeze(0)
    ).sum(dim=2, keepdim=True)

    assert conv.kernel_reduction_delta.shape == (3, 2, 2, 2)
    torch.testing.assert_close(conv._reduce_kernel_depth(), expected)


def test_lkr_transposed_conv_depth_reduction_matches_functional_reference():
    conv = SPADConvTranspose3d_K2S2(
        4,
        3,
        kernel_size=2,
        stride=2,
        bias=True,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        conv.kernel_reduction_delta.copy_(
            torch.linspace(-0.25, 0.25, conv.kernel_reduction_delta.numel()).reshape_as(
                conv.kernel_reduction_delta
            )
        )
    x = torch.randn(1, 4, 5, 7, 9)

    actual = conv(x, da=1)
    expected_weight = (
        conv.weight
        * (1 + conv.kernel_reduction_delta).unsqueeze(0)
    ).sum(dim=2, keepdim=True)
    expected = torch.nn.functional.conv_transpose3d(
        x,
        expected_weight,
        conv.bias,
        stride=(1, 2, 2),
    )

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert conv.kernel_reduction_delta.grad is not None
    assert torch.count_nonzero(conv.kernel_reduction_delta.grad) > 0


@pytest.mark.parametrize('da', [0, 1, 2, None])
def test_k2_transposed_conv_size_clamp_reuses_spatially_reduced_kernel(da):
    conv = SPADConvTranspose3d_K2S2(
        4,
        3,
        kernel_size=2,
        stride=2,
        bias=True,
    )
    x = torch.randn(1, 4, 5, 7, 9)

    actual = conv(x, da=da, stride_override=(1, 1, 1))
    expected = torch.nn.functional.conv_transpose3d(
        x,
        conv.weight.sum(dim=(2, 3, 4), keepdim=True),
        conv.bias,
        stride=1,
    )

    assert actual.shape == (1, 3, 5, 7, 9)
    torch.testing.assert_close(actual, expected)


def test_k2_transposed_conv_size_clamp_uses_lkr_over_every_kernel_axis():
    conv = SPADConvTranspose3d_K2S2(
        4,
        3,
        kernel_size=2,
        stride=2,
        bias=False,
        learnable_kernel_reduction=True,
    )
    with torch.no_grad():
        conv.kernel_reduction_delta.copy_(
            torch.linspace(-0.25, 0.25, conv.kernel_reduction_delta.numel()).reshape_as(
                conv.kernel_reduction_delta
            )
        )
    x = torch.randn(1, 4, 5, 7, 9)

    conv(x, da=0, stride_override=(1, 1, 1)).sum().backward()
    expected = (
        conv.weight
        * (1 + conv.kernel_reduction_delta).unsqueeze(0)
    ).sum(dim=(2, 3, 4), keepdim=True)

    torch.testing.assert_close(conv._reduce_kernel_spatial(), expected)
    assert conv.kernel_reduction_delta.grad is not None
    assert torch.count_nonzero(conv.kernel_reduction_delta.grad) > 0


@pytest.mark.parametrize(
    ('da', 'stride'),
    [
        (0, (2, 2, 2)),
        (1, (1, 2, 2)),
        (2, (1, 2, 2)),
        (None, (1, 2, 2)),
    ],
)
def test_k2_transposed_conv_explicit_runtime_stride_matches_default(da, stride):
    conv = SPADConvTranspose3d_K2S2(4, 3)
    x = torch.randn(1, 4, 5, 7, 9)

    torch.testing.assert_close(
        conv(x, da=da, stride_override=stride),
        conv(x, da=da),
    )


@pytest.mark.parametrize(
    ('da', 'stride'),
    [(0, (1, 2, 2)), (1, (2, 2, 2))],
)
def test_k2_transposed_conv_runtime_stride_must_match_da_mode(da, stride):
    conv = SPADConvTranspose3d_K2S2(4, 3)

    with pytest.raises(ValueError, match='requires runtime stride'):
        conv(
            torch.randn(1, 4, 5, 7, 9),
            da=da,
            stride_override=stride,
        )


def test_lkr_flag_requires_boolean_value():
    with pytest.raises(TypeError, match='must be boolean'):
        SPADConv3d_K3S1(
            2,
            3,
            kernel_size=3,
            learnable_kernel_reduction=1,
        )


def test_spad_k3_convs_reject_unsupported_string_padding_and_dilation():
    with pytest.raises(ValueError, match='does not support string padding'):
        SPADConv3d_K3S1(2, 3, kernel_size=3, padding='same')
    with pytest.raises(ValueError, match='requires dilation=1'):
        SPADConv3d_K3S2(2, 3, kernel_size=3, stride=2, dilation=2)


def test_conv3d_factory_rejects_unknown_1x1_options():
    conv = Conv3d(
        2,
        3,
        kernel_size=1,
        learnable_kernel_reduction=True,
    )

    assert isinstance(conv, torch.nn.Conv3d)
    assert not hasattr(conv, 'kernel_reduction_delta')
    with pytest.raises(TypeError, match='unexpected Conv3d options'):
        Conv3d(2, 3, kernel_size=1, typo=True)


class TestSPADAvgPool:
    def test_da0_pools_all_axes(self):
        pool = SPADAvgPool(kernel_size=2)
        x = torch.randn(1, 32, 8, 16, 16)
        y = pool(x, da=0)
        assert y.shape == (1, 32, 4, 8, 8)

    def test_da1_preserves_depth(self):
        pool = SPADAvgPool(kernel_size=2)
        x = torch.randn(1, 32, 8, 16, 16)
        y = pool(x, da=1)
        assert y.shape == (1, 32, 8, 8, 8)

    def test_da2_preserves_depth(self):
        pool = SPADAvgPool(kernel_size=2)
        x = torch.randn(1, 32, 8, 16, 16)
        y = pool(x, da=2)
        assert y.shape == (1, 32, 8, 8, 8)

    def test_da_none_preserves_depth(self):
        pool = SPADAvgPool(kernel_size=2)
        x = torch.randn(1, 32, 8, 16, 16)
        y = pool(x, da=None)
        assert y.shape == (1, 32, 8, 8, 8)


class TestSPADConvTranspose3dK3S2:
    def test_da0_upsamples_all(self):
        """da=0: full upsample [2,2,2]."""
        conv_t = SPADConvTranspose3d_K3S2(32, 16, kernel_size=3, stride=2)
        x = torch.randn(1, 32, 4, 8, 8)
        y = conv_t(x, da=0)
        assert y.shape == (1, 16, 8, 16, 16)

    def test_da1_preserves_depth(self):
        """da=1 (promote): upsample [1,2,2], full kernel."""
        conv_t = SPADConvTranspose3d_K3S2(32, 16, kernel_size=3, stride=2)
        x = torch.randn(1, 32, 4, 8, 8)
        y = conv_t(x, da=1)
        assert y.shape == (1, 16, 4, 16, 16)

    def test_da2_preserves_depth_collapsed(self):
        """da>=2: collapsed kernel, upsample [1,2,2]."""
        conv_t = SPADConvTranspose3d_K3S2(32, 16, kernel_size=3, stride=2)
        x = torch.randn(1, 32, 4, 8, 8)
        y = conv_t(x, da=2)
        assert y.shape == (1, 16, 4, 16, 16)

    def test_da_none_same_as_da2(self):
        """da=None should behave like da>=2."""
        torch.manual_seed(42)
        conv_t = SPADConvTranspose3d_K3S2(32, 16, kernel_size=3, stride=2)
        x = torch.randn(1, 32, 4, 8, 8)
        y_none = conv_t(x, da=None)
        y2 = conv_t(x, da=2)
        assert torch.allclose(y_none, y2)

    def test_da1_differs_from_da2(self):
        """da=1 (full kernel) should produce different output from da=2 (collapsed)."""
        torch.manual_seed(42)
        conv_t = SPADConvTranspose3d_K3S2(32, 16, kernel_size=3, stride=2)
        x = torch.randn(1, 32, 4, 8, 8)
        y1 = conv_t(x, da=1)
        y2 = conv_t(x, da=2)
        assert not torch.allclose(y1, y2)
