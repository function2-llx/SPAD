"""DINOv3-L initialized 3D ViT-Adapter for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import nn

from pumit.downstream.seg.backbones import dinov3
from pumit.downstream.seg.adapters.fixed_patch_vit import FixedPatchViTPyramidBackbone
from pumit.downstream.seg.adapters.vit_adapter import (
    VIT_ADAPTER_CONFIG_KEYS,
    VIT_ADAPTER_OPTIONAL_CONFIG_KEYS,
    ViTAdapterEncoder3D,
    build_vit_adapter_encoder,
    vit_adapter_config,
)
from pumit.downstream.seg.plan import EncoderPlan

CONFIG_KEYS = frozenset({
    'architecture',
    'feature_layers',
    'vit_patch_size',
    *VIT_ADAPTER_CONFIG_KEYS,
})
OPTIONAL_CONFIG_KEYS = VIT_ADAPTER_OPTIONAL_CONFIG_KEYS


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Return the pinned DINOv3-L plus 3D ViT-Adapter architecture."""
    backbone_config = dinov3.prepare_config(
        weights,
        checkpoint_format,
        gradient_checkpointing,
    )
    architecture = backbone_config['architecture']
    if not isinstance(architecture, Mapping):
        raise TypeError('DINOv3 architecture must be a mapping')
    return {
        **backbone_config,
        **vit_adapter_config(int(architecture['hidden_size'])),
    }


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> ViTAdapterEncoder3D:
    """Construct the plan-aligned adapter without reading pretrained weights."""
    architecture = config['architecture']
    if not isinstance(architecture, Mapping):
        raise TypeError('DINOv3 ViT-Adapter architecture must be a mapping')
    feature_layers = tuple(int(layer) for layer in config['feature_layers'])
    if feature_layers != dinov3.FEATURE_LAYERS:
        raise ValueError(
            f'DINOv3 ViT-Adapter feature_layers must be {dinov3.FEATURE_LAYERS}, '
            f'got {feature_layers}'
        )

    patch_size = dinov3.fixed_patch_size(config)
    vit = dinov3.build_fixed_vit(architecture, patch_size)
    backbone = FixedPatchViTPyramidBackbone(vit, input_channels, feature_layers)
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
    """Load the official DINOv3 checkpoint into only the pretrained ViT trunk."""
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    dinov3.load_vit_pretrained(encoder.backbone.vit, weights)
