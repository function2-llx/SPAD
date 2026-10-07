"""SAM-Med3D planned-grid image-encoder adapter for dense segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

from pumit.downstream.cls.backbones.sam_med3d import (
    OUT_CHANS,
    PATCH,
    _load_state_dict,
    adapt_state_dict,
    build_encoder as build_sam_med3d,
)
from pumit.downstream.seg.adapters.pyramid import PlanAlignedPyramidEncoder3D, SimpleFPNEncoder3D, pyramid_options
from pumit.downstream.seg.adapters.vit import (
    make_parameter_layers,
    pinned_feature_layers,
    select_evenly_spaced_layers,
)
from pumit.downstream.seg.adapters.checkpoint import (
    adapt_patch_embed_weight,
    repeat_single_channel_weight,
)
from pumit.downstream.seg.adapters.fixed_patch_vit import validate_fixed_patch_size
from pumit.downstream.seg.plan import EncoderPlan

_INPUT_WEIGHT_KEY = 'patch_embed.proj.weight'
_NECK_PREFIX = 'neck.'
_DEPTH = 12
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
CONFIG_KEYS = frozenset({'vit_patch_size'})
# Plans without ``feature_layers`` predate the multi-depth readout and build the final neck-output SimpleFPN.
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch', 'feature_layers'})


class SAMMed3DFeatureBackbone(nn.Module):
    """Expose the SAM-Med3D image encoder at one fixed dataset-aligned patch size.

    Without ``feature_layers`` the output is the neck's image embedding (384-d); with them it is the raw output of
    each selected block (768-d, the trunk has no final norm) and the neck, SAM's embedding head, is dropped.
    """

    def __init__(self, input_channels: int, patch_size: Sequence[int], feature_layers: tuple[int, ...] | None = None):
        super().__init__()
        self.patch_size = validate_fixed_patch_size(
            patch_size,
            in_plane_patch_size=PATCH,
        )
        self.feature_stride = self.patch_size
        self.model = build_sam_med3d(input_channels)
        self.feature_layers = feature_layers
        if feature_layers is None:
            self.feature_channels = OUT_CHANS
        else:
            del self.model.neck
            self.feature_channels = (self.model.patch_embed.proj.out_channels,) * len(feature_layers)
            self.feature_strides = (self.patch_size,) * len(feature_layers)
        source = self.model.patch_embed.proj
        if tuple(source.kernel_size) != self.patch_size:
            self.model.patch_embed.proj = nn.Conv3d(
                source.in_channels,
                source.out_channels,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=source.bias is not None,
            )
        if len(self.model.blocks) != _DEPTH:
            raise RuntimeError(f'expected {_DEPTH} SAM-Med3D blocks, got {len(self.model.blocks)}')

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        features = self.model.patch_embed(x)
        if self.model.pos_embed is not None:
            position = self.model.pos_embed
            if position.shape[1:4] != features.shape[1:4]:
                position = F.interpolate(
                    position.permute(0, 4, 1, 2, 3),
                    size=features.shape[1:4],
                    mode='trilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 4, 1)
            features = features + position
        wanted = set(self.feature_layers or ())
        hidden = []
        for depth, block in enumerate(self.model.blocks, start=1):
            features = block(features)
            if depth in wanted:
                hidden.append(features.permute(0, 4, 1, 2, 3))
        if self.feature_layers is None:
            return self.model.neck(features.permute(0, 4, 1, 2, 3))
        return tuple(hidden)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        embedding_parameters = list(self.model.patch_embed.parameters())
        if self.model.pos_embed is not None:
            embedding_parameters.append(self.model.pos_embed)
        return make_parameter_layers(
            embedding_parameters,
            self.model.blocks,
            self.model.neck.parameters() if self.feature_layers is None else (),
        )


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate options for the pinned SAM-Med3D ``vit_b_ori`` image encoder."""
    if checkpoint_format is not None:
        raise ValueError('SAM-Med3D does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('SAM-Med3D does not currently expose gradient checkpointing')
    return {'feature_layers': list(_FEATURE_LAYERS)}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D:
    """Construct a plan-complete SAM-Med3D encoder without reading pretrained weights."""
    if input_channels <= 0:
        raise ValueError(f'SAM-Med3D input_channels must be positive, got {input_channels}')
    feature_layers = pinned_feature_layers(config, _FEATURE_LAYERS, consumer='SAM-Med3D')
    backbone = SAMMed3DFeatureBackbone(input_channels, config['vit_patch_size'], feature_layers)
    encoder_class = SimpleFPNEncoder3D if feature_layers is None else PlanAlignedPyramidEncoder3D
    return encoder_class(backbone, plan, input_channels=input_channels, **pyramid_options(config))


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the SAM-Med3D image encoder, with no prompt or mask-decoder weights."""
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(f'expected a pyramid encoder, got {type(encoder).__name__}')

    state_dict = adapt_state_dict(
        dict(_load_state_dict(weights)),
        encoder.backbone.model,
    )
    if encoder.backbone.feature_layers is not None:
        for key in [key for key in state_dict if key.startswith(_NECK_PREFIX)]:
            del state_dict[key]
    input_channels = encoder.backbone.model.patch_embed.proj.in_channels
    if input_channels != 1:
        state_dict[_INPUT_WEIGHT_KEY] = repeat_single_channel_weight(
            state_dict[_INPUT_WEIGHT_KEY],
            input_channels,
        )
    target_patch_size = encoder.backbone.patch_size
    if tuple(state_dict[_INPUT_WEIGHT_KEY].shape[2:]) != target_patch_size:
        state_dict[_INPUT_WEIGHT_KEY] = adapt_patch_embed_weight(
            state_dict[_INPUT_WEIGHT_KEY],
            target_patch_size,
        )
    encoder.backbone.model.load_state_dict(state_dict, strict=True)
