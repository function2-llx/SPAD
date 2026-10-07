from collections.abc import Callable, Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf


def _ensure_tuple_rep(val: int | Sequence[int], dim: int) -> tuple[int, ...]:
    if isinstance(val, int):
        return (val,) * dim
    val = tuple(val)
    if len(val) != dim:
        raise ValueError(f'Sequence must have length {dim}, got length {len(val)}.')
    return val


def _get_padding(kernel_size: int | Sequence[int], stride: int | Sequence[int]) -> tuple[int, ...] | int:
    ks = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    st = (stride,) if isinstance(stride, int) else tuple(stride)
    padding = tuple((k - s + 1) // 2 for k, s in zip(ks, st))
    return padding[0] if len(padding) == 1 else padding


def _get_output_padding(
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
) -> tuple[int, ...] | int:
    ks = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    st = (stride,) if isinstance(stride, int) else tuple(stride)
    pad = (padding,) if isinstance(padding, int) else tuple(padding)
    out_padding = tuple(2 * p + s - k for k, s, p in zip(ks, st, pad))
    return out_padding[0] if len(out_padding) == 1 else out_padding

from pumit.types import param3_t, tuple3_t
from .resample import resample

__all__ = [
    'Conv3d',
    'Inflator',
    'INFLATORS',
    'gaussian_inflator',
    'uniform_inflator',
    'center_inflator',
    'noise_inflator',
    'SPADWeightInflationMixin',
    'SPADConv3d_K3S1',
    'SPADConv3d_K3S2',
    'SPADConvTranspose3d_K3S2',
    'SPADConvTranspose3d_K2S2',
    'TransposedConv3d',
    'ConvTranspose3d',
    'MaxPool',
    'SPADAvgPool',
]

# ---------------------------------------------------------------------------
# Inflator: callable that turns a 2D weight into a 3D weight along depth.
# Invariant: weight_3d.sum(dim=0) == weight_2d
# ---------------------------------------------------------------------------

type Inflator = Callable[[torch.Tensor, int], torch.Tensor]


def gaussian_inflator(weight_2d: torch.Tensor, d: int) -> torch.Tensor:
    """Inflate 2D weight to 3D using a Gaussian window along depth.

    Returns shape (d, *weight_2d.shape) with weight_3d.sum(dim=0) == weight_2d.
    """
    if d == 1:
        return weight_2d.unsqueeze(0)
    window = torch.signal.windows.gaussian(d, std=d / 4, device=weight_2d.device, dtype=weight_2d.dtype)
    window = window / window.sum()
    # broadcast: (d, 1, 1, ...) * (1, co, ci, ...)
    return window.reshape(d, *([1] * weight_2d.ndim)) * weight_2d.unsqueeze(0)


def uniform_inflator(weight_2d: torch.Tensor, d: int) -> torch.Tensor:
    """Inflate 2D weight to 3D using uniform distribution along depth.

    Returns shape (d, *weight_2d.shape) with weight_3d.sum(dim=0) == weight_2d.
    """
    return einops.repeat(weight_2d / d, '... -> d ...', d=d)


def center_inflator(weight_2d: torch.Tensor, d: int) -> torch.Tensor:
    """Inflate 2D weight to 3D: weight concentrated at center depth slice(s).

    For odd d, all weight goes to the single center slice.
    For even d, weight is split equally between the two center slices.
    Returns shape (d, *weight_2d.shape) with weight_3d.sum(dim=0) == weight_2d.
    """
    if d == 1:
        return weight_2d.unsqueeze(0)
    out = torch.zeros(d, *weight_2d.shape, device=weight_2d.device, dtype=weight_2d.dtype)
    if d % 2 == 1:
        out[d // 2] = weight_2d
    else:
        out[d // 2 - 1] = weight_2d / 2
        out[d // 2] = weight_2d / 2
    return out


def noise_inflator(weight_2d: torch.Tensor, d: int) -> torch.Tensor:
    """Inflate 2D weight to 3D using random distribution along depth.

    Each element gets a random share of the 2D weight across depth slices.
    Returns shape (d, *weight_2d.shape) with weight_3d.sum(dim=0) == weight_2d.
    """
    if d == 1:
        return weight_2d.unsqueeze(0)
    logits = torch.randn(d, *weight_2d.shape, device=weight_2d.device, dtype=weight_2d.dtype)
    shares = logits.softmax(dim=0)
    return shares * weight_2d.unsqueeze(0)


INFLATORS: dict[str, Inflator] = {
    'gaussian': gaussian_inflator,
    'uniform': uniform_inflator,
    'center': center_inflator,
    'noise': noise_inflator,
}


# ---------------------------------------------------------------------------
# SPADWeightInflationMixin: handles _load_from_state_dict for 2D -> 3D
# ---------------------------------------------------------------------------

class SPADWeightInflationMixin:
    """Mixin for SPAD Conv3d subclasses that inflates 2D pretrained weights on load."""

    inflator: Inflator

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        weight_key = f'{prefix}weight'
        if (weight := state_dict.get(weight_key)) is not None and weight.ndim + 1 == self.weight.ndim:
            # handle 2D pretrained weight
            if weight.shape[2:] != self.kernel_size[1:]:
                weight = resample(weight, self.kernel_size[1:], scale=True)
            d = self.kernel_size[0]
            # inflator returns (d, co, ci, ...) with sum(dim=0) == weight_2d
            weight_3d = self.inflator(weight, d)
            state_dict[weight_key] = einops.rearrange(weight_3d, 'd co ci ... -> co ci d ...')
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LearnableKernelReductionMixin:
    """Optionally learn per-output weights for shared-kernel reduction."""

    learnable_kernel_reduction: bool
    kernel_reduction_delta: nn.Parameter | None
    _kernel_reduction_output_channel_axis: int

    def _init_kernel_reduction(
        self,
        learnable_kernel_reduction: bool,
        *,
        output_channel_axis: int,
    ) -> None:
        if not isinstance(learnable_kernel_reduction, bool):
            raise TypeError('learnable_kernel_reduction must be boolean')
        if output_channel_axis not in (0, 1):
            raise ValueError('output_channel_axis must be 0 or 1')
        if self.weight.shape[output_channel_axis] != self.out_channels:
            raise ValueError(
                'learnable kernel reduction requires an explicit output-channel '
                'axis in the convolution weight'
            )
        self.learnable_kernel_reduction = learnable_kernel_reduction
        self._kernel_reduction_output_channel_axis = output_channel_axis
        if learnable_kernel_reduction:
            self.kernel_reduction_delta = nn.Parameter(
                self.weight.new_zeros((self.out_channels, *self.kernel_size))
            )
        else:
            self.kernel_reduction_delta = None

    def _reduce_kernel_depth(self) -> torch.Tensor:
        return self._weighted_kernel().sum(dim=2, keepdim=True)

    def _reduce_kernel_spatial(self) -> torch.Tensor:
        return self._weighted_kernel().sum(dim=(2, 3, 4), keepdim=True)

    def _weighted_kernel(self) -> torch.Tensor:
        if self.kernel_reduction_delta is None:
            return self.weight
        alpha = 1 + self.kernel_reduction_delta
        if self._kernel_reduction_output_channel_axis == 0:
            alpha = alpha.unsqueeze(1)
        else:
            alpha = alpha.unsqueeze(0)
        return self.weight * alpha


# ---------------------------------------------------------------------------
# SPADConv3d_K3S1: kernel=3, stride=1
# ---------------------------------------------------------------------------

class SPADConv3d_K3S1(
    SPADWeightInflationMixin,
    LearnableKernelReductionMixin,
    nn.Conv3d,
):
    """SPAD Conv3d with kernel_size=3, stride=1.

    forward(x, da):
        da==0: full [3,3,3] kernel, original stride/padding.
        da==1: PROMOTE mode. Full [3,3,3] kernel, same as da==0.
            Decouples feature extraction (kernel) from resolution (stride).
        else (da>=2 or None): collapsed kernel via fixed sum or configured LKR,
            stride stays 1, padding[0]=0.
    """

    kernel_size: tuple3_t[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 1,
        padding: param3_t[int] = 0,
        dilation: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        *,
        inflator: Inflator = center_inflator,
        learnable_kernel_reduction: bool = False,
    ):
        if isinstance(padding, str):
            raise ValueError('SPADConv3d_K3S1 does not support string padding')
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype,
        )
        assert self.kernel_size == (3, 3, 3), f'K3S1 requires kernel_size=(3,3,3), got {self.kernel_size}'
        assert self.stride == (1, 1, 1), f'K3S1 requires stride=(1,1,1), got {self.stride}'
        assert self.padding_mode == 'zeros'
        self.inflator = inflator
        self._init_kernel_reduction(
            learnable_kernel_reduction,
            output_channel_axis=0,
        )

    def forward(self, x: torch.Tensor, da: int | None = 0) -> torch.Tensor:
        if da is None or da >= 2:
            weight = self._reduce_kernel_depth()
            padding = (0, self.padding[1], self.padding[2])
        else:
            weight = self.weight
            padding = self.padding
        return nnf.conv3d(x, weight, self.bias, self.stride, padding, self.dilation, self.groups)


# ---------------------------------------------------------------------------
# SPADConv3d_K3S2: kernel=3, stride=2
# ---------------------------------------------------------------------------

class SPADConv3d_K3S2(
    SPADWeightInflationMixin,
    LearnableKernelReductionMixin,
    nn.Conv3d,
):
    """SPAD Conv3d with a downsampling-capable shared kernel.

    ``stride_override=(1, 1, 1)`` executes the same parameter tensor as K3S1
    when a size-clamped U-Net stage must preserve its feature-map shape.

    ``forward(x, da)``:
        da==0: full [3,3,3] kernel, stride [2,2,2], padding (0,0,0).
            Caller pads externally.
        da==1: PROMOTE mode. Full [3,3,3] kernel, stride [1,2,2],
            internal padding (1,0,0). Transition stage.
        else (da>1 or None): collapsed kernel via fixed sum or configured LKR,
            stride [1,2,2], padding (0,0,0).
    """

    kernel_size: tuple3_t[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 2,
        padding: str | param3_t[int] = 0,
        dilation: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        *,
        inflator: Inflator = center_inflator,
        learnable_kernel_reduction: bool = False,
    ):
        dilation = _ensure_tuple_rep(dilation, 3)
        if dilation != (1, 1, 1):
            raise ValueError('SPADConv3d_K3S2 requires dilation=1')
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype,
        )
        assert self.kernel_size == (3, 3, 3), f'K3S2 requires kernel_size=(3,3,3), got {self.kernel_size}'
        assert self.stride == (2, 2, 2), f'K3S2 requires stride=(2,2,2), got {self.stride}'
        assert self.padding_mode == 'zeros'
        self.inflator = inflator
        self._init_kernel_reduction(
            learnable_kernel_reduction,
            output_channel_axis=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        da: int | None = 0,
        stride_override: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if stride_override == (1, 1, 1):
            if da is None or da >= 2:
                weight = self._reduce_kernel_depth()
                padding = (0, 1, 1)
            else:
                weight = self.weight
                padding = (1, 1, 1)
            return nnf.conv3d(
                x,
                weight,
                self.bias,
                stride_override,
                padding,
                self.dilation,
                self.groups,
            )
        expected_stride = (2, 2, 2) if da == 0 else (1, 2, 2)
        if stride_override is not None and stride_override != expected_stride:
            raise ValueError(
                f'da={da!r} requires runtime stride {expected_stride}, '
                f'got {stride_override}'
            )
        if da == 0:
            # full 3D: [3,3,3] kernel, stride [2,2,2], no padding (caller pads)
            return nnf.conv3d(
                x, self.weight, self.bias,
                self.stride, (0, 0, 0), self.dilation, self.groups,
            )
        elif da == 1:
            # PROMOTE: full [3,3,3] kernel, stride [1,2,2], internal padding (1,0,0)
            return nnf.conv3d(
                x, self.weight, self.bias,
                (1, self.stride[1], self.stride[2]),
                (1, 0, 0),
                self.dilation, self.groups,
            )
        else:
            # collapsed: sum depth dim, stride [1,2,2], no padding
            weight = self._reduce_kernel_depth()
            return nnf.conv3d(
                x, weight, self.bias,
                (1, self.stride[1], self.stride[2]),
                (0, 0, 0),
                self.dilation, self.groups,
            )


# ---------------------------------------------------------------------------
# SPADConvTranspose3d_K3S2: kernel=3, stride=2 (upsampling)
# ---------------------------------------------------------------------------

class SPADConvTranspose3d_K3S2(SPADWeightInflationMixin, nn.ConvTranspose3d):
    """SPAD ConvTranspose3d with kernel_size=3, stride=2.

    forward(x, da):
        da==0: full [3,3,3] kernel, stride [2,2,2]. Upsamples all axes by 2x.
        da==1: PROMOTE mode. Full [3,3,3] kernel, stride [1,2,2].
            Upsamples H,W only; depth preserved.
        else (da>=2 or None): collapsed kernel via weight.sum(dim=2, keepdim=True),
            stride [1,2,2]. No depth upsample, 2D-like operation.
    """

    kernel_size: tuple3_t[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 2,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        inflator: Inflator = center_inflator,
    ):
        # kernel=3, stride=2: padding=1, output_padding=1 gives exact 2x upsample
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding=1, output_padding=1,
            groups=groups, bias=bias, device=device, dtype=dtype,
        )
        self.kernel_size = _ensure_tuple_rep(self.kernel_size, 3)
        self.stride = _ensure_tuple_rep(self.stride, 3)
        assert self.kernel_size == (3, 3, 3), f'requires kernel_size=(3,3,3), got {self.kernel_size}'
        assert self.stride == (2, 2, 2), f'requires stride=(2,2,2), got {self.stride}'
        self.inflator = inflator

    def forward(self, x: torch.Tensor, da: int | None = 0) -> torch.Tensor:
        if da == 0:
            # Full 3D: upsample all axes by 2x
            return nnf.conv_transpose3d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation,
            )
        elif da == 1:
            # PROMOTE: full kernel, stride [1,2,2] (no depth upsample)
            # depth: stride=1, kernel=3, padding=1, output_padding=0 -> preserves size
            return nnf.conv_transpose3d(
                x, self.weight, self.bias,
                (1, self.stride[1], self.stride[2]),
                (1, self.padding[1], self.padding[2]),
                (0, self.output_padding[1], self.output_padding[2]),
                self.groups, self.dilation,
            )
        else:
            # Collapsed: sum depth kernel dim, stride [1,2,2]
            # depth: stride=1, kernel=1, padding=0, output_padding=0 -> preserves size
            weight = self.weight.sum(dim=2, keepdim=True)
            return nnf.conv_transpose3d(
                x, weight, self.bias,
                (1, self.stride[1], self.stride[2]),
                (0, self.padding[1], self.padding[2]),
                (0, self.output_padding[1], self.output_padding[2]),
                self.groups, self.dilation,
            )


# ---------------------------------------------------------------------------
# SPADConvTranspose3d_K2S2: kernel=2, stride=2 (matches vanilla nnU-Net decoder)
# ---------------------------------------------------------------------------

class SPADConvTranspose3d_K2S2(
    SPADWeightInflationMixin,
    LearnableKernelReductionMixin,
    nn.ConvTranspose3d,
):
    """SPAD ConvTranspose3d with kernel_size=2, stride=2. Matches vanilla nnU-Net decoder.

    forward(x, da, stride_override):
        da==0: full [2,2,2] kernel, stride [2,2,2]. Upsamples all axes by 2x.
        else (da>=1 or None): collapsed kernel via fixed sum or configured LKR,
            stride [1,2,2]. No depth upsample.
        stride_override==(1,1,1): reduce the shared kernel over every spatial
            axis to [1,1,1]. Preserve shape while projecting channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int] = 2,
        stride: param3_t[int] = 2,
        bias: bool = False,
        device=None,
        dtype=None,
        *,
        inflator: Inflator = center_inflator,
        learnable_kernel_reduction: bool = False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding=0, output_padding=0, bias=bias, device=device, dtype=dtype,
        )
        self.kernel_size = _ensure_tuple_rep(self.kernel_size, 3)
        self.stride = _ensure_tuple_rep(self.stride, 3)
        assert self.kernel_size == (2, 2, 2), f'requires kernel_size=(2,2,2), got {self.kernel_size}'
        assert self.stride == (2, 2, 2), f'requires stride=(2,2,2), got {self.stride}'
        self.inflator = inflator
        self._init_kernel_reduction(
            learnable_kernel_reduction,
            output_channel_axis=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        da: int | None = 0,
        stride_override: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if stride_override == (1, 1, 1):
            weight = self._reduce_kernel_spatial()
            return nnf.conv_transpose3d(
                x,
                weight,
                self.bias,
                stride_override,
                (0, 0, 0),
                (0, 0, 0),
                self.groups,
                self.dilation,
            )
        expected_stride = (2, 2, 2) if da == 0 else (1, 2, 2)
        if stride_override is not None and stride_override != expected_stride:
            raise ValueError(
                f'da={da!r} requires runtime stride {expected_stride}, '
                f'got {stride_override}'
            )
        if da is None or da >= 1:
            weight = self._reduce_kernel_depth()
            return nnf.conv_transpose3d(
                x, weight, self.bias,
                (1, self.stride[1], self.stride[2]),
                (0, 0, 0), (0, 0, 0),
                self.groups, self.dilation,
            )
        else:
            return nnf.conv_transpose3d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation,
            )


# ---------------------------------------------------------------------------
# Conv3d factory: returns the right subclass based on kernel/stride
# ---------------------------------------------------------------------------

def Conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: param3_t[int],
    stride: param3_t[int] = 1,
    padding: str | param3_t[int] = 0,
    dilation: param3_t[int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
    **kwargs,
) -> nn.Conv3d:
    """Factory that returns the appropriate SPAD Conv3d subclass.

    - kernel=3, stride=1 -> SPADConv3d_K3S1
    - kernel=3, stride=2 -> SPADConv3d_K3S2
    - kernel=1           -> plain nn.Conv3d
    - else               -> NotImplementedError
    """
    ks = _ensure_tuple_rep(kernel_size, 3)
    st = _ensure_tuple_rep(stride, 3)

    if ks == (3, 3, 3) and st == (1, 1, 1):
        return SPADConv3d_K3S1(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype, **kwargs,
        )
    elif ks == (3, 3, 3) and st == (2, 2, 2):
        return SPADConv3d_K3S2(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype, **kwargs,
        )
    elif ks == (1, 1, 1):
        kwargs.pop('inflator', None)
        learnable_kernel_reduction = kwargs.pop(
            'learnable_kernel_reduction',
            False,
        )
        if not isinstance(learnable_kernel_reduction, bool):
            raise TypeError('learnable_kernel_reduction must be boolean')
        if kwargs:
            raise TypeError(
                f'unexpected Conv3d options for kernel_size=1: {sorted(kwargs)}'
            )
        return nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype,
        )
    else:
        raise NotImplementedError(
            f'Conv3d factory does not support kernel_size={ks}, stride={st}. '
            f'Supported: (3,1), (3,2), (1,*)'
        )


class ConvTranspose3d(nn.ConvTranspose3d):
    @staticmethod
    def _check_depth_adaptable(kernel_size: int, stride: int):
        assert stride & stride - 1 == 0, 'only power of 2 is supported'
        assert kernel_size == stride or (kernel_size, stride) in {(3, 2), (4, 2)}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        stride: param3_t[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: param3_t[int] = 1,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        adaptive: bool = True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, 0,
            groups, bias, dilation, padding_mode, device, dtype,
        )
        self.adaptive = adaptive
        assert self.stride[0] == self.stride[1] == self.stride[2], 'only isotropic stride is supported'
        self._check_depth_adaptable(self.kernel_size[0], self.stride[0])
        self.num_upsamples = self.stride[1].bit_length() - 1
        self.padding = _get_padding(self.kernel_size, self.stride)
        self.output_padding = _get_output_padding(self.kernel_size, self.stride, self.padding)

    def forward(self, x: torch.Tensor, adapt_level: int = 0, output_size=None) -> torch.Tensor:
        if self.adaptive:
            assert output_size is None
            stride = list(self.stride)
            padding = list(self.padding)
            output_padding = list(self.output_padding)
            if adapt_level == 0:
                weight = self.weight
            elif self.kernel_size[0] == self.stride[0]:
                stride[0] = 1 << adapt_level
                weight = einops.reduce(
                    self.weight,
                    'co ci (dr dc) ... -> co ci dr ...',
                    'sum',
                    dr=stride[0],
                )
            else:
                stride[0] = 1
                weight = self.weight.sum(dim=2, keepdim=True)
                padding[0] = output_padding[0]
            return nnf.conv_transpose3d(
                x, weight, self.bias, stride, padding, output_padding,
                self.groups, self.dilation,
            )
        else:
            return super().forward(x)

# TransposedConv3d is deprecated
TransposedConv3d = ConvTranspose3d

class AdaptiveTransposedConvUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.transposed_conv = TransposedConv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.conv = nn.Sequential(
            Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, adapt_level: int = 0):
        x = self.transposed_conv(x, adapt_level)
        return self.conv(x)

class MaxPool(nn.MaxPool3d):
    def __init__(
        self,
        kernel_size: param3_t,
        stride: param3_t | None = None,
        padding: param3_t = 0,
        dilation: param3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.kernel_size = _ensure_tuple_rep(self.kernel_size, 3)
        assert self.kernel_size == (2, 2, 2)
        assert stride is None
        self.padding = _ensure_tuple_rep(self.padding, 3)
        self.dilation = _ensure_tuple_rep(self.dilation, 3)

    def forward(self, x: torch.Tensor, adapt_level: int = 0):
        kernel_size = list(self.kernel_size)
        kernel_size[0] = max(self.kernel_size[0] >> adapt_level, 1)
        return nnf.max_pool3d(
            x, kernel_size, kernel_size,
            self.padding, self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices,
        )


class SPADAvgPool(nn.AvgPool3d):
    def __init__(self, kernel_size: param3_t, stride: param3_t | None = None, padding: param3_t = 0):
        super().__init__(kernel_size, stride, padding)
        self.kernel_size = _ensure_tuple_rep(self.kernel_size, 3)
        assert self.kernel_size == (2, 2, 2)
        assert stride is None
        self.padding = _ensure_tuple_rep(self.padding, 3)

    def forward(self, x: torch.Tensor, da: int | None = 0) -> torch.Tensor:
        kernel_size = list(self.kernel_size)
        if da is None or da >= 1:
            kernel_size[0] = 1
        return nnf.avg_pool3d(x, kernel_size, kernel_size, self.padding)
