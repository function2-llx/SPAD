"""Plan-aligned SimpleFPN adaptation for multi-level 3D feature backbones.

Both encoders complete the backbone-derived P2-P5 pyramid with a raw-image P0-P1 stem for the full U-Net
decoder; without the stem (``with_high_resolution_stem=False``) they expose only P2-P5, the contract of the
U-Net readout without its refiner.

``pyramid_branch`` selects how a backbone map reaches a level: ``deconv`` is the ViTDet SimpleFPN branch,
``resample`` the projection-first branch whose size depends only on the trunk width and the plan schedule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from torch import Tensor, nn
from torch.nn import functional as F

from dynamic_network_architectures.architectures.unet import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)

from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_HIGH_RESOLUTION_STAGES = 2
_PYRAMID_START_STAGE = 2
PYRAMID_BRANCHES = ('deconv', 'resample')


def pyramid_options(config: Mapping[str, object]) -> dict[str, object]:
    """The optional stem and branch keys of a backbone config as encoder keyword arguments."""
    return {
        'with_high_resolution_stem': bool(config.get('with_high_resolution_stem', True)),
        'pyramid_branch': str(config.get('pyramid_branch', 'deconv')),
    }


def _high_resolution_stem(plan: EncoderPlan, input_channels: int) -> ResidualEncoder:
    """Raw-image P0-P1 stages that complete a P2-P5 backbone pyramid for the full U-Net decoder."""
    return ResidualEncoder(
        input_channels=input_channels,
        n_stages=_HIGH_RESOLUTION_STAGES,
        features_per_stage=plan.output_channels[:_HIGH_RESOLUTION_STAGES],
        conv_op=plan.conv_op,
        kernel_sizes=plan.kernel_sizes[:_HIGH_RESOLUTION_STAGES],
        strides=plan.strides[:_HIGH_RESOLUTION_STAGES],
        n_blocks_per_stage=plan.n_blocks_per_stage[:_HIGH_RESOLUTION_STAGES],
        conv_bias=plan.conv_bias,
        norm_op=plan.norm_op,
        norm_op_kwargs=plan.norm_op_kwargs,
        dropout_op=plan.dropout_op,
        dropout_op_kwargs=plan.dropout_op_kwargs,
        nonlin=plan.nonlin,
        nonlin_kwargs=plan.nonlin_kwargs,
        return_skips=True,
    )


def _initialize_scratch_modules(stem: ResidualEncoder | None, simple_fpn: nn.ModuleList) -> None:
    # Applied after every module is constructed so the seeded init draws keep their order.
    initializer = InitWeights_He(1e-2)
    if stem is not None:
        stem.apply(initializer)
        stem.apply(init_last_bn_before_add_to_0)
    simple_fpn.apply(initializer)


def _power_of_two_steps(ratio: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    remaining = tuple(int(value) for value in ratio)
    if any(value <= 0 or value & (value - 1) for value in remaining):
        raise ValueError(f'feature-scale adaptation requires power-of-two ratios, got {remaining}')

    steps = []
    while any(value > 1 for value in remaining):
        step = tuple(2 if value > 1 else 1 for value in remaining)
        steps.append(step)
        remaining = tuple(value // divisor for value, divisor in zip(remaining, step))
    return tuple(steps)


class SimpleFPNBranch3D(nn.Module):
    """Map one fixed-stride feature map onto one spatial pyramid level."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        source_stride: Sequence[int],
        target_stride: Sequence[int],
        kernel_size: Sequence[int],
        *,
        conv_bias: bool,
    ):
        super().__init__()
        source_stride = tuple(int(value) for value in source_stride)
        target_stride = tuple(int(value) for value in target_stride)
        up_ratio = tuple(
            source // target if source >= target else 1
            for source, target in zip(source_stride, target_stride)
        )
        down_ratio = tuple(
            target // source if target >= source else 1
            for source, target in zip(source_stride, target_stride)
        )
        for source, target in zip(source_stride, target_stride):
            larger, smaller = max(source, target), min(source, target)
            if larger % smaller:
                raise ValueError(f'cannot align source stride {source_stride} to target stride {target_stride}')

        scale_layers: list[nn.Module] = []
        current_channels = in_channels
        up_steps = _power_of_two_steps(up_ratio)
        for step_idx, step in enumerate(up_steps):
            next_channels = max(out_channels, current_channels // 2)
            scale_layers.append(
                nn.ConvTranspose3d(
                    current_channels,
                    next_channels,
                    kernel_size=step,
                    stride=step,
                    bias=conv_bias,
                )
            )
            current_channels = next_channels
            if step_idx + 1 < len(up_steps):
                scale_layers.append(nn.GELU())

        for step in _power_of_two_steps(down_ratio):
            scale_layers.append(nn.MaxPool3d(kernel_size=step, stride=step))

        kernel_size = tuple(int(value) for value in kernel_size)
        if len(kernel_size) != 3 or any(value <= 0 or value % 2 == 0 for value in kernel_size):
            raise ValueError(f'refinement kernels must be positive and odd, got {kernel_size}')

        self.scale_layers = nn.Sequential(*scale_layers)
        self.proj1 = nn.Conv3d(current_channels, out_channels, kernel_size=1, bias=conv_bias)
        self.proj2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=tuple(value // 2 for value in kernel_size),
            bias=conv_bias,
        )

    def forward(self, x: Tensor, stage_shapes: Sequence[Sequence[int]] | None = None) -> Tensor:
        # The target grid follows from the fixed stride ratios; stage_shapes is the shared branch signature.
        return self.proj2(self.proj1(self.scale_layers(x)))


def nominal_stage(feature_stride: Sequence[int], cumulative_strides: Sequence[Sequence[int]]) -> int:
    """Plan stage whose grid a feature map stands in for: the stage with the closest stride volume.

    A stride-16 ViT is stage 4 of a six-stage plan whether its patch is 16x16x16 or 8x16x16 on an anisotropic
    plan, and a native pyramid level is its own stage.
    """
    log_volume = math.log2(math.prod(int(value) for value in feature_stride))
    return min(
        range(len(cumulative_strides)),
        key=lambda stage: abs(math.log2(math.prod(cumulative_strides[stage])) - log_volume),
    )


def _resize(x: Tensor, shape: Sequence[int]) -> Tensor:
    """Parameter-free resize to a plan stage grid: trilinear when finer, max-pool when coarser."""
    current = tuple(int(value) for value in x.shape[2:])
    shape = tuple(int(value) for value in shape)
    if current == shape:
        return x
    if all(target >= size for size, target in zip(current, shape)):
        return F.interpolate(x, size=shape, mode='trilinear', align_corners=False)
    if all(target <= size and size % target == 0 for size, target in zip(current, shape)):
        kernel = tuple(size // target for size, target in zip(current, shape))
        return F.max_pool3d(x, kernel_size=kernel, stride=kernel)
    raise ValueError(f'cannot resize grid {current} to plan stage grid {shape}')


class ResampleFPNBranch3D(nn.Module):
    """Map one feature map onto one pyramid level with parameter-free resampling.

    A 1x1x1 conv block to the level's width at the source grid, then one 3x3x3 conv block per plan stage crossed on
    the way down from the source's stage to the level, each after a resize to that stage's grid. A source at or above
    the level's stage gets the single block at the level's grid, so the block count never depends on the patch shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        source_stage: int,
        target_stage: int,
        plan: EncoderPlan,
    ):
        super().__init__()
        if target_stage < source_stage:
            block_stages = tuple(range(source_stage - 1, target_stage - 1, -1))
        else:
            block_stages = (target_stage,)
        self.block_stages = block_stages

        def conv_block(input_channels: int, kernel_size: Sequence[int]) -> ConvDropoutNormReLU:
            return ConvDropoutNormReLU(
                plan.conv_op,
                input_channels,
                out_channels,
                tuple(int(value) for value in kernel_size),
                1,
                plan.conv_bias,
                plan.norm_op,
                plan.norm_op_kwargs,
                plan.dropout_op,
                plan.dropout_op_kwargs,
                plan.nonlin,
                plan.nonlin_kwargs,
            )

        self.proj = conv_block(in_channels, (1, 1, 1))
        self.blocks = nn.ModuleList(
            conv_block(out_channels, plan.kernel_sizes[stage]) for stage in block_stages
        )

    def forward(self, x: Tensor, stage_shapes: Sequence[Sequence[int]]) -> Tensor:
        x = self.proj(x)
        for stage, block in zip(self.block_stages, self.blocks):
            x = block(_resize(x, stage_shapes[stage]))
        return x


def _make_branch(
    pyramid_branch: str,
    in_channels: int,
    out_channels: int,
    feature_stride: Sequence[int],
    target_stage: int,
    plan: EncoderPlan,
) -> nn.Module:
    if pyramid_branch == 'deconv':
        return SimpleFPNBranch3D(
            in_channels,
            out_channels,
            feature_stride,
            plan.cumulative_strides[target_stage],
            plan.kernel_sizes[target_stage],
            conv_bias=plan.conv_bias,
        )
    if pyramid_branch == 'resample':
        return ResampleFPNBranch3D(
            in_channels,
            out_channels,
            nominal_stage(feature_stride, plan.cumulative_strides),
            target_stage,
            plan,
        )
    raise ValueError(f'unsupported pyramid_branch {pyramid_branch!r}, expected one of {PYRAMID_BRANCHES}')


def adopt_plan_contract(encoder: nn.Module, plan: EncoderPlan, *, output_start: int = 0) -> None:
    """Copy the plan's stage geometry and layer types onto an encoder in the form nnU-Net's ``UNetDecoder`` reads.

    ``output_start`` drops the shallowest stages from the exposed levels; ``cumulative_strides`` always keeps the full
    plan list because forward passes derive every stage grid from it.
    """
    encoder.output_channels = plan.output_channels[output_start:]
    encoder.kernel_sizes = plan.kernel_sizes[output_start:]
    encoder.strides = plan.strides[output_start:]
    encoder.cumulative_strides = plan.cumulative_strides
    encoder.conv_op = plan.conv_op
    encoder.conv_bias = plan.conv_bias
    encoder.norm_op = plan.norm_op
    encoder.norm_op_kwargs = plan.norm_op_kwargs
    encoder.dropout_op = plan.dropout_op
    encoder.dropout_op_kwargs = plan.dropout_op_kwargs
    encoder.nonlin = plan.nonlin
    encoder.nonlin_kwargs = plan.nonlin_kwargs


def _check_feature_stride(feature_stride: Sequence[int], label: str) -> tuple[int, int, int]:
    stride = tuple(int(value) for value in feature_stride)
    if len(stride) != 3 or any(value <= 0 for value in stride):
        raise ValueError(f'{label} must contain three positive values, got {stride}')
    return stride


def _check_feature_map(
    feature: Tensor,
    input_shape: Sequence[int],
    feature_stride: Sequence[int],
    label: str,
) -> None:
    if any(size % step for size, step in zip(input_shape, feature_stride)):
        raise ValueError(f'input shape {tuple(input_shape)} is not divisible by {label} stride {feature_stride}')
    expected = tuple(size // step for size, step in zip(input_shape, feature_stride))
    if feature.ndim != 5:
        raise RuntimeError(f'{label} must be a 5D feature map, got {tuple(feature.shape)}')
    if tuple(feature.shape[2:]) != expected:
        raise RuntimeError(f'{label} shape {tuple(feature.shape[2:])} does not match feature-stride contract {expected}')


class _PyramidEncoder3D(nn.Module):
    """Optional raw-image P0-P1 stem plus one branch per backbone level onto plan stages P2 onward.

    Subclasses build ``self.simple_fpn`` and implement ``_pyramid`` (backbone call plus per-level checks); the module
    attribute names are part of every stored checkpoint.
    """

    def __init__(self, backbone: nn.Module, plan: EncoderPlan, *, input_channels: int, with_high_resolution_stem: bool):
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels
        self.with_high_resolution_stem = with_high_resolution_stem
        adopt_plan_contract(self, plan, output_start=0 if with_high_resolution_stem else _PYRAMID_START_STAGE)
        self.high_resolution_stem = (
            _high_resolution_stem(plan, input_channels) if with_high_resolution_stem else None
        )

    def _pyramid(self, x: Tensor, input_shape: tuple[int, ...], stage_shapes: Sequence[Sequence[int]]) -> list[Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )
        input_shape = tuple(int(value) for value in x.shape[2:])
        stage_shapes = compute_stage_shapes(input_shape, self.cumulative_strides)
        skips = [
            *(self.high_resolution_stem(x) if self.with_high_resolution_stem else ()),
            *self._pyramid(x, input_shape, stage_shapes),
        ]
        target_shapes = stage_shapes if self.with_high_resolution_stem else stage_shapes[_PYRAMID_START_STAGE:]
        for stage, (skip, target_shape) in enumerate(zip(skips, target_shapes, strict=True)):
            if skip.shape[2:] != target_shape:
                raise RuntimeError(
                    f'encoder level {stage} shape {tuple(skip.shape[2:])} does not match '
                    f'nnU-Net target shape {target_shape}'
                )
        return skips


class SimpleFPNEncoder3D(_PyramidEncoder3D):
    """Pyramid from one fixed-stride backbone feature map."""

    def __init__(
        self,
        backbone: nn.Module,
        plan: EncoderPlan,
        *,
        input_channels: int,
        with_high_resolution_stem: bool = True,
        pyramid_branch: str = 'deconv',
    ):
        super().__init__(backbone, plan, input_channels=input_channels, with_high_resolution_stem=with_high_resolution_stem)
        feature_channels = int(backbone.feature_channels)
        self.feature_stride = _check_feature_stride(backbone.feature_stride, 'backbone feature_stride')
        self.simple_fpn = nn.ModuleList(
            _make_branch(pyramid_branch, feature_channels, out_channels, self.feature_stride, stage, plan)
            for stage, out_channels in enumerate(
                plan.output_channels[_PYRAMID_START_STAGE:], start=_PYRAMID_START_STAGE
            )
        )
        _initialize_scratch_modules(self.high_resolution_stem, self.simple_fpn)

    def _pyramid(self, x: Tensor, input_shape: tuple[int, ...], stage_shapes: Sequence[Sequence[int]]) -> list[Tensor]:
        feature = self.backbone(x)
        _check_feature_map(feature, input_shape, self.feature_stride, 'backbone feature')
        return [branch(feature, stage_shapes) for branch in self.simple_fpn]


class PlanAlignedPyramidEncoder3D(_PyramidEncoder3D):
    """Pyramid from one backbone feature map per plan stage P2 onward."""

    def __init__(
        self,
        backbone: nn.Module,
        plan: EncoderPlan,
        *,
        input_channels: int,
        with_high_resolution_stem: bool = True,
        pyramid_branch: str = 'deconv',
    ):
        super().__init__(backbone, plan, input_channels=input_channels, with_high_resolution_stem=with_high_resolution_stem)
        feature_channels = tuple(int(value) for value in backbone.feature_channels)
        feature_strides = tuple(
            _check_feature_stride(stride, 'backbone feature strides') for stride in backbone.feature_strides
        )
        n_pyramid_levels = len(plan.output_channels) - _PYRAMID_START_STAGE
        if len(feature_channels) != n_pyramid_levels or len(feature_strides) != n_pyramid_levels:
            raise ValueError(
                f'backbone must provide {n_pyramid_levels} feature maps for plan stages '
                f'{_PYRAMID_START_STAGE} onward, got channels={feature_channels}, '
                f'strides={feature_strides}'
            )
        self.feature_strides = feature_strides
        self.simple_fpn = nn.ModuleList(
            _make_branch(pyramid_branch, in_channels, out_channels, feature_stride, stage, plan)
            for stage, (in_channels, feature_stride, out_channels) in enumerate(
                zip(feature_channels, feature_strides, plan.output_channels[_PYRAMID_START_STAGE:]),
                start=_PYRAMID_START_STAGE,
            )
        )
        _initialize_scratch_modules(self.high_resolution_stem, self.simple_fpn)

    def _pyramid(self, x: Tensor, input_shape: tuple[int, ...], stage_shapes: Sequence[Sequence[int]]) -> list[Tensor]:
        features = tuple(self.backbone(x))
        if len(features) != len(self.simple_fpn):
            raise RuntimeError(
                f'backbone returned {len(features)} feature maps, expected {len(self.simple_fpn)}'
            )
        for level, (feature, feature_stride) in enumerate(zip(features, self.feature_strides)):
            _check_feature_map(feature, input_shape, feature_stride, f'backbone level {level}')
        return [branch(feature, stage_shapes) for branch, feature in zip(self.simple_fpn, features)]
