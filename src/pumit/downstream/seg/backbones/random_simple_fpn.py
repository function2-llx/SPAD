"""Randomly initialized fixed-patch ViT behind the scratch SimpleFPN: the trunk-contribution floor for the probe.

Same pinned ViT-L architecture and pyramid as the DINOv3 arm, but no pretrained weights are ever loaded: the
trunk keeps the seeded random draw from `initialize`, which the saved initialization checkpoint then pins.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from pumit.downstream.seg.adapters.pyramid import PlanAlignedPyramidEncoder3D, SimpleFPNEncoder3D
from pumit.downstream.seg.backbones import dinov3

CONFIG_KEYS = dinov3.CONFIG_KEYS
OPTIONAL_CONFIG_KEYS = dinov3.OPTIONAL_CONFIG_KEYS


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Return the pinned ViT-L architecture; this arm takes no weights."""
    if weights is not None or checkpoint_format is not None:
        raise ValueError('random SimpleFPN takes neither weights nor checkpoint_format')
    return dinov3.pinned_config(gradient_checkpointing)


# Same pyramid as the DINOv3 arm.
build_encoder = dinov3.build_encoder


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Leave the seeded random trunk untouched; weights are an error, not an option."""
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(
            f'expected PlanAlignedPyramidEncoder3D or SimpleFPNEncoder3D, got {type(encoder).__name__}'
        )
    if weights is not None:
        raise ValueError(f'random SimpleFPN takes no weights, got {weights}')
