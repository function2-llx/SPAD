"""3DINO encoder adapter for dense segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import einops
from torch import Tensor, nn

from pumit.downstream.cls.backbones.three_dino import (
    PATCH,
    _HEAD_PREFIXES,
    _load_state_dict,
    build_encoder as build_three_dino,
)
from pumit.downstream.seg.adapters.pyramid import PlanAlignedPyramidEncoder3D, SimpleFPNEncoder3D, pyramid_options
from pumit.downstream.seg.adapters.vit import (
    make_parameter_layers,
    pinned_feature_layers,
    select_evenly_spaced_layers,
)
from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.plan import EncoderPlan

_INPUT_WEIGHT_KEY = 'patch_embed.proj.weight'
_DEPTH = 24
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
CONFIG_KEYS = frozenset()
# Plans without ``feature_layers`` predate the multi-depth readout and build the final-map SimpleFPN.
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch', 'feature_layers'})


def _transformer_blocks(model: nn.Module) -> tuple[nn.Module, ...]:
    if model.chunked_blocks:
        return tuple(
            block
            for chunk in model.blocks
            for block in chunk
            if not isinstance(block, nn.Identity)
        )
    return tuple(model.blocks)


class ThreeDinoFeatureBackbone(nn.Module):
    """Expose 3DINO's patch grids in nnU-Net spatial order: the final one, or one per evenly spaced depth."""

    feature_stride = (PATCH,) * 3

    def __init__(self, input_channels: int, feature_layers: tuple[int, ...] | None = None):
        super().__init__()
        self.model = build_three_dino(input_channels)
        if self.model.n_blocks != _DEPTH:
            raise RuntimeError(f'expected {_DEPTH} 3DINO blocks, got {self.model.n_blocks}')
        self.model.mask_token.requires_grad_(False)
        self.feature_layers = feature_layers
        if feature_layers is None:
            self.feature_channels = self.model.embed_dim
        else:
            self.feature_channels = (self.model.embed_dim,) * len(feature_layers)
            self.feature_strides = (self.feature_stride,) * len(feature_layers)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        # nnU-Net tensors use [D, H, W], while native 3DINO puts the two in-plane axes first
        # and the cross-slice axis last: [H, W, D].
        native_input = einops.rearrange(x, 'b c d h w -> b c h w d')
        native_feature_shape = tuple(
            size // stride
            for size, stride in zip(native_input.shape[2:], self.feature_stride)
        )

        def to_grid(tokens: Tensor) -> Tensor:
            return einops.rearrange(
                tokens,
                'b (h w d) c -> b c d h w',
                h=native_feature_shape[0],
                w=native_feature_shape[1],
                d=native_feature_shape[2],
            )

        if self.feature_layers is None:
            return to_grid(self.model.forward_features(native_input)['x_norm_patchtokens'])
        # Every depth passes through the final norm, as in the flat-ViT path and DINOv2's intermediate layers.
        outputs = self.model.get_intermediate_layers(
            native_input,
            n=[layer - 1 for layer in self.feature_layers],
            norm=True,
        )
        return tuple(to_grid(tokens) for tokens in outputs)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return make_parameter_layers(
            (
                *self.model.patch_embed.parameters(),
                self.model.cls_token,
                self.model.pos_embed,
                self.model.mask_token,
            ),
            _transformer_blocks(self.model),
            self.model.norm.parameters(),
        )


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate 3DINO options; the pinned architecture serializes only its readout depths."""
    if checkpoint_format is not None:
        raise ValueError('3DINO does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('3DINO does not currently expose gradient checkpointing')
    return {'feature_layers': list(_FEATURE_LAYERS)}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D:
    """Construct a plan-complete 3DINO encoder with random weights."""
    if input_channels <= 0:
        raise ValueError(f'3DINO input_channels must be positive, got {input_channels}')
    feature_layers = pinned_feature_layers(config, _FEATURE_LAYERS, consumer='3DINO')
    backbone = ThreeDinoFeatureBackbone(input_channels, feature_layers)
    encoder_class = SimpleFPNEncoder3D if feature_layers is None else PlanAlignedPyramidEncoder3D
    return encoder_class(backbone, plan, input_channels=input_channels, **pyramid_options(config))


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load the pinned 3DINO teacher after nnU-Net initializes the full network."""
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(f'expected a pyramid encoder, got {type(encoder).__name__}')
    state_dict = _load_state_dict(weights)
    input_channels = encoder.backbone.model.patch_embed.proj.in_channels
    if input_channels != 1:
        state_dict = dict(state_dict)
        state_dict[_INPUT_WEIGHT_KEY] = repeat_single_channel_weight(
            state_dict[_INPUT_WEIGHT_KEY],
            input_channels,
        )
    missing, unexpected = encoder.backbone.model.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing:
        raise RuntimeError(f'3DINO checkpoint is missing encoder keys: {missing}')
    bad_unexpected = [key for key in unexpected if not key.startswith(_HEAD_PREFIXES)]
    if bad_unexpected:
        raise RuntimeError(
            f'3DINO checkpoint has unexpected non-head keys: {bad_unexpected}'
        )
