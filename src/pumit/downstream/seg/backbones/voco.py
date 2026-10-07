"""VoCo-v2 SwinUNETR encoder adapter for dense segmentation.

Consumes the released encoder-only checkpoints (Luffy503/VoCo, Apache-2.0): the SwinUNETR-v2
SwinViT plus its residual conv skip encoders, self-supervised on PreCT-160K (images only; KiTS21
and AMOS22 images are inside that corpus). The six pretrained levels arrive at fixed isotropic
strides 1/2/4/8/16/32 with channels F/F/2F/4F/8F/16F for feature size F (B: 48, L: 96); Swin patch
merging cannot be re-strided, so each level is trilinearly resampled onto the plan's stage shapes
instead. The stride-16 level is the raw SwinViT stage-3 output, exactly as SwinUNETR consumes it.

This module binds the B variant; ``voco_l`` binds the same classes at feature size 96.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.native_pyramid import build_native_pyramid_encoder, native_encoder
from pumit.downstream.seg.backbones._voco_swin import SwinTransformer, UnetrBasicBlock
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_INPUT_WEIGHT_KEYS = (
    'swinViT.patch_embed.proj.weight',
    'encoder1.layer.conv1.conv.weight',
    'encoder1.layer.conv3.conv.weight',
)
CONFIG_KEYS = frozenset()
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch'})


def native_channels(feature_size: int) -> tuple[int, ...]:
    """The six pretrained level widths for one SwinUNETR feature size."""
    return (feature_size, feature_size, *(feature_size * 2**k for k in range(1, 5)))


class VoCoSwinPyramidBackbone(nn.Module):
    """Expose the six pretrained VoCo encoder levels at their native isotropic strides."""

    def __init__(self, input_channels: int, feature_size: int):
        super().__init__()
        self.swinViT = SwinTransformer(in_chans=input_channels, embed_dim=feature_size)
        self.encoder1 = UnetrBasicBlock(input_channels, feature_size)
        self.encoder2 = UnetrBasicBlock(feature_size, feature_size)
        self.encoder3 = UnetrBasicBlock(2 * feature_size, 2 * feature_size)
        self.encoder4 = UnetrBasicBlock(4 * feature_size, 4 * feature_size)
        self.encoder10 = UnetrBasicBlock(16 * feature_size, 16 * feature_size)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        hidden = self.swinViT(x)
        return (
            self.encoder1(x),
            self.encoder2(hidden[0]),
            self.encoder3(hidden[1]),
            self.encoder4(hidden[2]),
            hidden[3],
            self.encoder10(hidden[4]),
        )

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the pretrained groups from shallow to deep."""
        vit = self.swinViT
        return (
            tuple(self.encoder1.parameters()),
            tuple(vit.patch_embed.parameters()),
            (*vit.layers1c.parameters(), *vit.layers1.parameters(), *self.encoder2.parameters()),
            (*vit.layers2c.parameters(), *vit.layers2.parameters(), *self.encoder3.parameters()),
            (*vit.layers3c.parameters(), *vit.layers3.parameters(), *self.encoder4.parameters()),
            (*vit.layers4c.parameters(), *vit.layers4.parameters(), *self.encoder10.parameters()),
        )


class VoCoEncoder3D(nn.Module):
    """Resample the six native VoCo levels onto the plan's stage grid."""

    def __init__(self, plan: EncoderPlan, *, input_channels: int, feature_size: int):
        super().__init__()
        channels = native_channels(feature_size)
        n_stages = len(plan.output_channels)
        if not len(channels) - 1 <= n_stages <= len(channels):
            raise ValueError(f'VoCo requires a five- or six-stage plan, got {n_stages} stages')
        if tuple(plan.output_channels) != channels[:n_stages]:
            raise ValueError(
                f'VoCo at feature size {feature_size} requires plan channels {channels[:n_stages]}, '
                f'got {tuple(plan.output_channels)}'
            )
        strides = tuple(tuple(int(v) for v in s) for s in plan.strides)
        if len(strides) != n_stages or strides[0] != (1, 1, 1):
            raise ValueError(
                f'VoCo requires a plan starting at stride (1, 1, 1), got {strides}'
            )
        # A five-stage plan consumes the shallowest five pretrained levels; the full backbone still
        # loads and runs, and the unused deepest level is discarded (frozen, negligible compute).
        self.backbone = VoCoSwinPyramidBackbone(input_channels, feature_size)
        self.input_channels = input_channels
        adopt_plan_contract(self, plan)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )

        native = self.backbone(x)[:len(self.output_channels)]
        target_shapes = compute_stage_shapes(tuple(x.shape[2:]), self.cumulative_strides)
        features = [
            feature
            if tuple(feature.shape[2:]) == shape
            else F.interpolate(feature, size=shape, mode='trilinear', align_corners=False)
            for feature, shape in zip(native, target_shapes)
        ]
        for stage, (feature, channels, shape) in enumerate(
            zip(features, self.output_channels, target_shapes)
        ):
            expected = (x.shape[0], channels, *shape)
            if tuple(feature.shape) != expected:
                raise RuntimeError(
                    f'encoder stage {stage} shape {tuple(feature.shape)} does not match {expected}'
                )
        return features


def make_module_functions(feature_size: int, checkpoint_label: str):
    """Bind the module-level backbone functions for one released VoCo variant."""

    def prepare_config(
        weights: Path | None,
        checkpoint_format: str | None,
        gradient_checkpointing: bool,
    ) -> dict[str, object]:
        if weights is None:
            raise ValueError(f'VoCo requires the released {checkpoint_label} checkpoint')
        if checkpoint_format is not None:
            raise ValueError('VoCo does not accept checkpoint_format')
        if gradient_checkpointing:
            raise ValueError('VoCo does not currently expose gradient checkpointing')
        return {}

    def build_encoder(
        plan: EncoderPlan,
        input_channels: int,
        config: Mapping[str, object],
    ) -> nn.Module:
        # Without the stem, P2-P5 of the native encoder are projected onto the plan schedule.
        return build_native_pyramid_encoder(
            lambda encoder_plan: VoCoEncoder3D(encoder_plan, input_channels=input_channels, feature_size=feature_size),
            plan,
            native_channels(feature_size),
            input_channels=input_channels,
            config=config,
        )

    def load_pretrained(
        encoder: nn.Module,
        weights: Path | None,
    ) -> None:
        """Load the released encoder-only state dict (bare keys, no wrapper) into the backbone."""
        if weights is None:
            raise ValueError(f'VoCo requires the released {checkpoint_label} checkpoint')
        encoder = native_encoder(encoder)
        if not isinstance(encoder, VoCoEncoder3D):
            raise TypeError(f'expected VoCoEncoder3D, got {type(encoder).__name__}')

        state_dict = dict(torch.load(weights, map_location='cpu', weights_only=True))
        input_channels = encoder.backbone.swinViT.patch_embed.proj.in_channels
        if input_channels != 1:
            for key in _INPUT_WEIGHT_KEYS:
                state_dict[key] = repeat_single_channel_weight(state_dict[key], input_channels)
        encoder.backbone.load_state_dict(state_dict, strict=True)

    return prepare_config, build_encoder, load_pretrained


prepare_config, build_encoder, load_pretrained = make_module_functions(48, 'VoCo_B_SSL_head.pt')
