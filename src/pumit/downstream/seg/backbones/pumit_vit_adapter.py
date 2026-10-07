"""PUMIT EMA ViT initialized 3D ViT-Adapter for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import nn

from pumit.downstream.seg.adapters.vit_adapter import ViTAdapterEncoder3D
from pumit.downstream.seg.backbones._pumit import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    load_vit,
    vit_architecture,
)
from pumit.downstream.seg.backbones import dinov3_vit_adapter
from pumit.downstream.seg.plan import EncoderPlan

CONFIG_KEYS = dinov3_vit_adapter.CONFIG_KEYS
OPTIONAL_CONFIG_KEYS = dinov3_vit_adapter.OPTIONAL_CONFIG_KEYS


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate that the UCPT EMA ViT matches the pinned DINOv3 ViT-Adapter architecture."""
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(
            f'PUMIT ViT-Adapter requires checkpoint_format {CHECKPOINT_FORMAT!r}, '
            f'got {checkpoint_format!r}'
        )
    checkpoint = load_checkpoint(weights, consumer='PUMIT ViT-Adapter')
    source_architecture = vit_architecture(
        checkpoint,
        gradient_checkpointing=gradient_checkpointing,
    )
    adapter_config = dinov3_vit_adapter.prepare_config(
        weights,
        checkpoint_format=None,
        gradient_checkpointing=gradient_checkpointing,
    )
    if source_architecture != adapter_config['architecture']:
        raise ValueError(
            'PUMIT EMA ViT architecture does not match the pinned DINOv3 ViT-Adapter '
            f'architecture: source={source_architecture}, expected={adapter_config["architecture"]}'
        )
    return adapter_config


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> ViTAdapterEncoder3D:
    """Build the same ViT-Adapter architecture used by the DINOv3 initialization condition."""
    return dinov3_vit_adapter.build_encoder(plan, input_channels, config)


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the UCPT EMA ViT trunk, leaving the ViT-Adapter randomly initialized."""
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    load_vit(
        encoder.backbone.vit,
        load_checkpoint(weights, consumer='PUMIT ViT-Adapter'),
        consumer='PUMIT ViT-Adapter',
    )
