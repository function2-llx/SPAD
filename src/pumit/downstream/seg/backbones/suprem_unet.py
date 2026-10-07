"""SuPreM 3D U-Net encoder adapter for dense segmentation.

Vendors the encoder half of the SuPreM release's UNet3D (Models Genesis lineage): four DownTransition
stages exposing skips with channels 64/128/256/512. The network is consumed at its native depth through a
four-stage plan; the parameter-free inter-stage pools follow the plan's strides (isotropic 2x in the
release, re-strided nnU-Net style for anisotropic plans), and no scratch stages extend the hierarchy.

ContBatchNorm3d reproduces the release's semantics: it normalizes with current-batch statistics in every
mode. Running buffers are updated but never read, so a frozen trunk still computes checkpoint-faithful
features while its buffers drift harmlessly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_NATIVE_CHANNELS = (64, 128, 256, 512)
_INPUT_WEIGHT_KEY = 'down_tr64.ops.0.conv1.weight'
_CHECKPOINT_PREFIX = 'module.backbone.'
# Non-encoder scopes inside the released checkpoint: the UNet3D decoder half plus the
# CLIP-driven Universal Model head that produced the supervision during pretraining.
_ALLOWED_UNUSED_PREFIXES = ('up_tr', 'out_tr', 'module.')
CONFIG_KEYS = frozenset()


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    """BatchNorm3d that always uses current-batch statistics, as released."""

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 5:
            raise ValueError(f'expected 5D input, got {input.dim()}D')

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )


class LUConv(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn1(self.conv1(x)))


def _make_nconv(in_channel: int, depth: int) -> nn.Sequential:
    return nn.Sequential(
        LUConv(in_channel, 32 * 2**depth),
        LUConv(32 * 2**depth, 32 * 2**depth * 2),
    )


class DownTransition(nn.Module):
    """Two LUConvs then a max-pool over the plan's stage stride; the deepest stage (depth 3) does not pool.

    The release pools 2x2x2 everywhere; the pool is parameter-free, so re-striding it to an anisotropic
    plan stride (nnU-Net style) changes no pretrained weights.
    """

    def __init__(self, in_channel: int, depth: int, pool_stride: tuple[int, int, int] | None):
        super().__init__()
        self.ops = _make_nconv(in_channel, depth)
        self.maxpool = None if pool_stride is None else nn.MaxPool3d(pool_stride, stride=pool_stride)
        self.current_depth = depth

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.ops(x)
        if self.maxpool is None:
            return skip, skip
        return self.maxpool(skip), skip


class SupremUNetPyramidBackbone(nn.Module):
    """Expose the four pretrained UNet3D encoder stages at the plan's cumulative strides."""

    def __init__(self, input_channels: int, pool_strides: tuple[tuple[int, int, int], ...]):
        super().__init__()
        if len(pool_strides) != 3:
            raise ValueError(f'expected three inter-stage pool strides, got {len(pool_strides)}')
        self.down_tr64 = DownTransition(input_channels, 0, pool_strides[0])
        self.down_tr128 = DownTransition(64, 1, pool_strides[1])
        self.down_tr256 = DownTransition(128, 2, pool_strides[2])
        self.down_tr512 = DownTransition(256, 3, None)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        out, skip64 = self.down_tr64(x)
        out, skip128 = self.down_tr128(out)
        out, skip256 = self.down_tr256(out)
        out512, _ = self.down_tr512(out)
        return skip64, skip128, skip256, out512

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the four convolutional stages from shallow to deep."""
        return tuple(
            tuple(stage.parameters())
            for stage in (self.down_tr64, self.down_tr128, self.down_tr256, self.down_tr512)
        )


class SupremUNetEncoder3D(nn.Module):
    """Serve the pretrained four-stage hierarchy as a native-depth four-stage plan encoder."""

    def __init__(self, plan: EncoderPlan, *, input_channels: int):
        super().__init__()
        if tuple(plan.output_channels) != _NATIVE_CHANNELS:
            raise ValueError(
                f'SuPreM U-Net requires plan channels {_NATIVE_CHANNELS}, '
                f'got {tuple(plan.output_channels)}'
            )
        strides = tuple(tuple(int(v) for v in s) for s in plan.strides)
        if len(strides) != 4 or strides[0] != (1, 1, 1):
            raise ValueError(
                f'SuPreM U-Net requires a four-stage plan starting at stride (1, 1, 1), '
                f'got {strides}'
            )
        self.backbone = SupremUNetPyramidBackbone(input_channels, strides[1:])
        self.input_channels = input_channels
        adopt_plan_contract(self, plan)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )

        features = list(self.backbone(x))
        target_shapes = compute_stage_shapes(tuple(x.shape[2:]), self.cumulative_strides)
        for stage, (feature, channels, shape) in enumerate(
            zip(features, self.output_channels, target_shapes)
        ):
            expected = (x.shape[0], channels, *shape)
            if tuple(feature.shape) != expected:
                raise RuntimeError(
                    f'encoder stage {stage} shape {tuple(feature.shape)} does not match {expected}'
                )
        return features


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate options for the pinned SuPreM U-Net encoder."""
    if weights is None:
        raise ValueError('SuPreM requires the released supervised_suprem_unet_2100.pth checkpoint')
    if checkpoint_format is not None:
        raise ValueError('SuPreM U-Net does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('SuPreM U-Net does not currently expose gradient checkpointing')
    return {}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> SupremUNetEncoder3D:
    """Construct the plan-aligned SuPreM U-Net encoder without reading its weights."""
    return SupremUNetEncoder3D(plan, input_channels=input_channels)


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the UNet3D encoder stages; the decoder half and Universal Model head are dropped."""
    if weights is None:
        raise ValueError('SuPreM requires the released supervised_suprem_unet_2100.pth checkpoint')
    if not isinstance(encoder, SupremUNetEncoder3D):
        raise TypeError(f'expected SupremUNetEncoder3D, got {type(encoder).__name__}')

    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    state_dict = checkpoint['net']
    encoder_state: dict[str, Tensor] = {}
    dropped: list[str] = []
    for key, value in state_dict.items():
        stripped = key.removeprefix(_CHECKPOINT_PREFIX)
        if stripped.startswith('down_tr'):
            encoder_state[stripped] = value
        else:
            dropped.append(key if stripped == key else stripped)
    bad_dropped = [key for key in dropped if not key.startswith(_ALLOWED_UNUSED_PREFIXES)]
    if bad_dropped:
        raise RuntimeError(f'SuPreM checkpoint has unexpected non-encoder keys: {bad_dropped}')

    input_channels = encoder.backbone.down_tr64.ops[0].conv1.in_channels
    if input_channels != 1:
        encoder_state[_INPUT_WEIGHT_KEY] = repeat_single_channel_weight(
            encoder_state[_INPUT_WEIGHT_KEY],
            input_channels,
        )
    encoder.backbone.load_state_dict(encoder_state, strict=True)
