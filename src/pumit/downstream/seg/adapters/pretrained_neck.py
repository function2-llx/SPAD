"""Plan-aligned encoder built from the pretrained UCPT segmentation neck.

The neck already produces the 1/4, 1/8, and 1/16 levels that nnU-Net's P2-P4 stages expect, so only their
channel counts need adaptation. The coarsest stage is max-pooled from the ViT patch grid at its pretrained
width, so it reaches the decoder without a projection, normalization, or activation.
"""

from __future__ import annotations

from collections.abc import Sequence

import einops
from torch import Tensor, nn

from dynamic_network_architectures.architectures.unet import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)

from pumit.downstream.seg.adapters.fixed_patch_vit import (
    fixed_patch_size,
    prepare_fixed_patch_inputs,
)
from pumit.downstream.seg.adapters.vit import make_parameter_layers
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes
from pumit.model.vit import ViT

_HIGH_RESOLUTION_STAGES = 2
_PYRAMID_START_STAGE = 2
# Fine-to-coarse, aligning with the plan's P2, P3, and P4 stages.
_NECK_LEVELS = ('1/4', '1/8', '1/16')
# Upsampling factor of each neck level relative to the ViT patch grid, in _NECK_LEVELS order.
_NECK_UPSAMPLE_FACTORS = (4, 2, 1)


class FixedGridPretrainedNeck(nn.Module):
    """Plain 3D-convolution form of the UCPT segmentation neck at fixed DA=0 geometry."""

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.up40_0 = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.act40 = nn.GELU()
        self.up40_1 = nn.ConvTranspose3d(
            in_channels // 2,
            hidden_size,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.proj40_1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
        self.proj40_2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.up20_0 = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.proj20_1 = nn.Conv3d(in_channels // 2, hidden_size, kernel_size=1)
        self.proj20_2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.proj10_1 = nn.Conv3d(in_channels, hidden_size, kernel_size=1)
        self.proj10_2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        l16 = self.proj10_2(self.proj10_1(x))
        h = self.up20_0(x)
        l8 = self.proj20_2(self.proj20_1(h))
        h = self.act40(self.up40_0(x))
        h = self.up40_1(h)
        l4 = self.proj40_2(self.proj40_1(h))
        return {'1/4': l4, '1/8': l8, '1/16': l16}


def _neck_feature_strides(patch_size: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Resolve each neck level's input stride from the ViT patch stride."""
    patch_size = tuple(int(value) for value in patch_size)
    finest_factor = max(_NECK_UPSAMPLE_FACTORS)
    if any(stride % finest_factor for stride in patch_size):
        raise ValueError(
            f'the fixed-grid neck upsamples every axis by {finest_factor}, so ViT patch '
            f'stride {patch_size} must be divisible by {finest_factor}'
        )
    return tuple(
        tuple(stride // factor for stride in patch_size)
        for factor in _NECK_UPSAMPLE_FACTORS
    )


def _plan_derived_pool(
    source_stride: Sequence[int],
    target_stride: Sequence[int],
) -> nn.Module:
    """Resolve the coarsest stage's pooling from the plan's stride relative to the ViT patch grid."""
    source_stride = tuple(int(value) for value in source_stride)
    target_stride = tuple(int(value) for value in target_stride)
    if any(
        target < source or target % source
        for source, target in zip(source_stride, target_stride, strict=True)
    ):
        raise ValueError(
            f'the coarsest plan stride {target_stride} must be an integer multiple of the ViT patch '
            f'stride {source_stride}'
        )
    pool_stride = tuple(
        target // source
        for source, target in zip(source_stride, target_stride, strict=True)
    )
    if pool_stride == (1, 1, 1):
        return nn.Identity()
    return nn.MaxPool3d(kernel_size=pool_stride, stride=pool_stride)


class PretrainedNeckBackbone(nn.Module):
    """Run a fixed-patch ViT and its pretrained UCPT segmentation neck as one transfer unit.

    Both submodules come from the same EMA checkpoint pair, so ``freeze_backbone`` covers them together.
    Forward returns the neck's 1/4, 1/8, and 1/16 levels followed by the normalized final ViT patch grid,
    which is also the tensor the neck itself consumes.
    """

    def __init__(
        self,
        vit: ViT,
        input_channels: int,
        *,
        neck_hidden_size: int,
    ):
        """
        Args:
            vit: ViT whose patch embedding has already been replaced by a fixed 3D convolution.
            input_channels: Plan input channels, either one or the ViT's native width.
            neck_hidden_size: Pretrained pyramid width recorded by the UCPT training config.
        """
        super().__init__()
        if input_channels not in (1, vit.config.in_channels):
            raise ValueError(
                f'pretrained-neck ViT accepts one input channel or its native '
                f'{vit.config.in_channels} channels, got {input_channels}'
            )
        if neck_hidden_size <= 0:
            raise ValueError(f'neck_hidden_size must be positive, got {neck_hidden_size}')
        self.vit = vit
        self.neck = FixedGridPretrainedNeck(
            in_channels=vit.embed_dim,
            hidden_size=neck_hidden_size,
        )
        self.input_channels = input_channels
        patch_stride = fixed_patch_size(vit)
        self.feature_channels = (
            *(neck_hidden_size,) * len(_NECK_LEVELS),
            vit.embed_dim,
        )
        self.feature_strides = (*_neck_feature_strides(patch_stride), patch_stride)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if self.input_channels == 1 and self.vit.config.in_channels != 1:
            x = x.expand(-1, self.vit.config.in_channels, -1, -1, -1)
        tokens, rope, patch_shape = prepare_fixed_patch_inputs(self.vit, x)
        patch_tokens = self.vit(tokens, rope)[:, self.vit.n_prefix:]
        feature = einops.rearrange(
            patch_tokens,
            'b (d h w) c -> b c d h w',
            d=patch_shape[0],
            h=patch_shape[1],
            w=patch_shape[2],
        )
        levels = self.neck(feature)
        return (*(levels[level] for level in _NECK_LEVELS), feature)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the ViT parameters shallow-to-deep, with the neck grouped at the deepest layer."""
        return make_parameter_layers(
            self.vit.embeddings.parameters(),
            self.vit.layer,
            (*self.vit.norm.parameters(), *self.neck.parameters()),
        )


class PretrainedNeckEncoder3D(nn.Module):
    """Combine a raw-image P0-P1 stem with the pretrained UCPT segmentation pyramid.

    The three neck levels must already sit on the plan's P2-P4 cumulative strides; construction fails
    otherwise. The coarsest stage is pooled from the ViT patch grid and keeps its pretrained width, so it
    carries no projection, normalization, or activation.
    """

    def __init__(
        self,
        backbone: PretrainedNeckBackbone,
        plan: EncoderPlan,
        *,
        input_channels: int,
    ):
        super().__init__()
        if len(plan.output_channels) != 6:
            raise ValueError(
                f'the pretrained segmentation neck spans six nnU-Net encoder stages, got '
                f'{len(plan.output_channels)}'
            )
        cumulative_strides = plan.cumulative_strides
        pyramid_channels = plan.output_channels[_PYRAMID_START_STAGE:]
        pyramid_strides = cumulative_strides[_PYRAMID_START_STAGE:]
        for level, (feature_stride, target_stride) in enumerate(
            zip(backbone.feature_strides[:-1], pyramid_strides[:-1], strict=True)
        ):
            if feature_stride != target_stride:
                raise ValueError(
                    f'pretrained neck level {_NECK_LEVELS[level]} runs at stride {feature_stride}, but '
                    f'plan stage {level + _PYRAMID_START_STAGE} expects {target_stride}'
                )
        if backbone.feature_channels[-1] != pyramid_channels[-1]:
            raise ValueError(
                f'the pooled ViT stage keeps its pretrained width {backbone.feature_channels[-1]}, but '
                f'the plan expects {pyramid_channels[-1]} channels'
            )

        self.backbone = backbone
        self.input_channels = input_channels
        adopt_plan_contract(self, plan)
        self.high_resolution_stem = ResidualEncoder(
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
        self.neck_projections = nn.ModuleList(
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=plan.conv_bias)
            for in_channels, out_channels in zip(
                backbone.feature_channels[:-1],
                pyramid_channels[:-1],
                strict=True,
            )
        )
        self.coarse_pool = _plan_derived_pool(
            backbone.feature_strides[-1],
            pyramid_strides[-1],
        )


        initializer = InitWeights_He(1e-2)
        self.high_resolution_stem.apply(initializer)
        self.high_resolution_stem.apply(init_last_bn_before_add_to_0)
        self.neck_projections.apply(initializer)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )

        stage_shapes = compute_stage_shapes(x.shape[2:], self.cumulative_strides)
        high_resolution_skips = self.high_resolution_stem(x)
        *neck_features, coarse_feature = self.backbone(x)
        skips = [
            *high_resolution_skips,
            *(
                projection(feature)
                for projection, feature in zip(
                    self.neck_projections,
                    neck_features,
                    strict=True,
                )
            ),
            self.coarse_pool(coarse_feature),
        ]
        for stage, (skip, target_shape) in enumerate(zip(skips, stage_shapes, strict=True)):
            if tuple(skip.shape[2:]) != target_shape:
                raise RuntimeError(
                    f'encoder stage {stage} shape {tuple(skip.shape[2:])} does not match '
                    f'nnU-Net target shape {target_shape}'
                )
        return skips
