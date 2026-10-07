"""Randomly initialized 3D ViT-Adapter: the trunk-contribution floor for the frozen-probe roster.

Same pinned ViT-L plus ViT-Adapter architecture as the DINOv3 and PUMIT arms, but no pretrained weights
are ever loaded: the trunk keeps the seeded random draw from `initialize`, which the saved initialization
checkpoint then pins for training. Under the frozen contract only the adapter and decoder train, so this
arm measures what the trainable interface achieves without any pretrained trunk.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from pumit.downstream.seg.adapters.vit_adapter import ViTAdapterEncoder3D, vit_adapter_config
from pumit.downstream.seg.backbones import dinov3, dinov3_vit_adapter

CONFIG_KEYS = dinov3_vit_adapter.CONFIG_KEYS
OPTIONAL_CONFIG_KEYS = dinov3_vit_adapter.OPTIONAL_CONFIG_KEYS


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Return the pinned ViT-L ViT-Adapter architecture; this arm takes no weights."""
    if weights is not None or checkpoint_format is not None:
        raise ValueError('random ViT-Adapter takes neither weights nor checkpoint_format')
    backbone_config = dinov3.pinned_config(gradient_checkpointing)
    return {
        **backbone_config,
        **vit_adapter_config(int(backbone_config['architecture']['hidden_size'])),
    }


# Same plan-aligned adapter as the DINOv3 arm.
build_encoder = dinov3_vit_adapter.build_encoder


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Leave the seeded random trunk untouched; weights are an error, not an option."""
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    if weights is not None:
        raise ValueError(f'random ViT-Adapter takes no weights, got {weights}')
