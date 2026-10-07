"""SAM-Med3D-initialized 3D ViT-Adapter for downstream segmentation.

Routes the SAM-Med3D image-encoder trunk (flat 3D ViT-B: 768d/12L, windowed attention with four global
blocks) through the shared ViT-Adapter, dropping the pretrained SAM neck so the trunk is consumed like the
other flat-ViT baselines. The trunk has no final norm; block outputs feed the adapter unnormalized, matching
how the native neck consumed them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import checkpoint as torch_checkpoint

from pumit.downstream.cls.backbones.sam_med3d import (
    PATCH,
    _load_state_dict,
    adapt_state_dict,
    build_encoder as build_sam_med3d,
)
from pumit.downstream.seg.adapters.checkpoint import (
    adapt_patch_embed_weight,
    repeat_single_channel_weight,
)
from pumit.downstream.seg.adapters.fixed_patch_vit import validate_fixed_patch_size
from pumit.downstream.seg.adapters.vit import make_parameter_layers, select_evenly_spaced_layers
from pumit.downstream.seg.adapters.vit_adapter import (
    InteractiveViTBackbone,
    VIT_ADAPTER_CONFIG_KEYS,
    VIT_ADAPTER_OPTIONAL_CONFIG_KEYS,
    ViTAdapterEncoder3D,
    build_vit_adapter_encoder,
    vit_adapter_config,
)
from pumit.downstream.seg.plan import EncoderPlan

_DEPTH = 12
_EMBED_DIM = 768
_INPUT_WEIGHT_KEY = 'patch_embed.proj.weight'
_NECK_PREFIX = 'neck.'
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
CONFIG_KEYS = frozenset({
    'feature_layers',
    'gradient_checkpointing',
    'vit_patch_size',
    *VIT_ADAPTER_CONFIG_KEYS,
})
OPTIONAL_CONFIG_KEYS = VIT_ADAPTER_OPTIONAL_CONFIG_KEYS


class SAMMed3DInteractiveBackbone(InteractiveViTBackbone):
    """Expose the SAM-Med3D trunk, without its neck, at four interaction depths."""

    def __init__(
        self,
        input_channels: int,
        patch_size: Sequence[int],
        *,
        gradient_checkpointing: bool,
    ):
        model = build_sam_med3d(input_channels)
        # Trunk-only transfer: the SAM image-embedding neck stays out of the module so the frozen
        # scope and the checkpoint load cover exactly the twelve blocks and their embedding.
        del model.neck
        if len(model.blocks) != _DEPTH:
            raise RuntimeError(f'expected {_DEPTH} SAM-Med3D blocks, got {len(model.blocks)}')
        super().__init__(
            embed_dim=_EMBED_DIM,
            feature_layers=_FEATURE_LAYERS,
            num_prefix_tokens=0,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.model = model
        self.input_channels = int(input_channels)
        self.patch_size = tuple(int(value) for value in patch_size)
        source = model.patch_embed.proj
        if tuple(source.kernel_size) != self.patch_size:
            model.patch_embed.proj = nn.Conv3d(
                source.in_channels,
                source.out_channels,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=source.bias is not None,
            )

    def prepare_tokens(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int], tuple[int, int, int]]:
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
        grid = tuple(int(value) for value in features.shape[1:4])
        return features.flatten(1, 3), grid, grid

    def run_blocks(self, tokens: Tensor, start: int, end: int, context: Any) -> Tensor:
        # SAM blocks window-partition internally and need the spatial grid layout back.
        features = tokens.view(tokens.shape[0], *context, tokens.shape[-1])
        for block in self.model.blocks[start:end]:
            if self.training and self.gradient_checkpointing:
                features = torch_checkpoint.checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        return features.flatten(1, 3)

    def normalize_patch_tokens(self, tokens: Tensor) -> Tensor:
        # The trunk has no final norm; its native consumer (the neck) read raw block output.
        return tokens

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        embedding_parameters = list(self.model.patch_embed.parameters())
        if self.model.pos_embed is not None:
            embedding_parameters.append(self.model.pos_embed)
        return make_parameter_layers(embedding_parameters, self.model.blocks, ())


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    if weights is None:
        raise ValueError('SAM-Med3D ViT-Adapter requires sam_med3d_turbo.pth')
    if checkpoint_format is not None:
        raise ValueError('SAM-Med3D ViT-Adapter does not accept checkpoint_format')
    return {
        'feature_layers': list(_FEATURE_LAYERS),
        'gradient_checkpointing': gradient_checkpointing,
        **vit_adapter_config(_EMBED_DIM),
    }


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> ViTAdapterEncoder3D:
    if tuple(int(layer) for layer in config['feature_layers']) != _FEATURE_LAYERS:
        raise ValueError(f'SAM-Med3D feature_layers must be {_FEATURE_LAYERS}')
    patch_size = validate_fixed_patch_size(
        config['vit_patch_size'],
        in_plane_patch_size=PATCH,
    )
    backbone = SAMMed3DInteractiveBackbone(
        input_channels,
        patch_size,
        gradient_checkpointing=bool(config['gradient_checkpointing']),
    )
    return build_vit_adapter_encoder(
        backbone,
        plan,
        input_channels=input_channels,
        config=config,
    )


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the SAM-Med3D trunk; the neck and the prompt/mask decoders are dropped."""
    if weights is None:
        raise ValueError('SAM-Med3D ViT-Adapter requires sam_med3d_turbo.pth')
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    if not isinstance(encoder.backbone, SAMMed3DInteractiveBackbone):
        raise TypeError(
            f'expected SAMMed3DInteractiveBackbone, got {type(encoder.backbone).__name__}'
        )

    model = encoder.backbone.model
    state_dict = adapt_state_dict(dict(_load_state_dict(weights)), model)
    for key in [key for key in state_dict if key.startswith(_NECK_PREFIX)]:
        del state_dict[key]
    input_channels = model.patch_embed.proj.in_channels
    if input_channels != state_dict[_INPUT_WEIGHT_KEY].shape[1]:
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
    model.load_state_dict(state_dict, strict=True)
