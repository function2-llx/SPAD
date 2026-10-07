"""PUMIT EMA ViT and pretrained UCPT segmentation neck for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch import nn

from pumit.downstream.seg.adapters.fixed_patch_vit import (
    replace_with_fixed_patch_embedding,
    validate_fixed_patch_size,
    vit_config_from_mapping,
)
from pumit.downstream.seg.adapters.pretrained_neck import (
    PretrainedNeckBackbone,
    PretrainedNeckEncoder3D,
)
from pumit.downstream.seg.backbones._pumit import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    load_neck,
    load_vit,
    neck_hidden_size,
    vit_architecture,
)
from pumit.downstream.seg.plan import EncoderPlan
from pumit.model.vit import ViT

CONFIG_KEYS = frozenset({'architecture', 'neck_hidden_size', 'vit_patch_size'})
_CONSUMER = 'PUMIT pretrained neck'


def _fixed_patch_size(config: Mapping[str, object]) -> tuple[int, int, int]:
    value = config.get('vit_patch_size')
    if not isinstance(value, Sequence):
        raise TypeError(f'{_CONSUMER} vit_patch_size must be a sequence, got {value!r}')
    return validate_fixed_patch_size(value, in_plane_patch_size=16)


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Extract the fixed-grid PUMIT ViT architecture together with its pretrained neck width."""
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(
            f'{_CONSUMER} requires checkpoint_format {CHECKPOINT_FORMAT!r}, '
            f'got {checkpoint_format!r}'
        )
    checkpoint = load_checkpoint(weights, consumer=_CONSUMER)
    return {
        'architecture': vit_architecture(
            checkpoint,
            gradient_checkpointing=gradient_checkpointing,
        ),
        'neck_hidden_size': neck_hidden_size(checkpoint),
    }


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PretrainedNeckEncoder3D:
    """Construct the PUMIT ViT and segmentation neck behind a plan-aligned pyramid."""
    architecture = config['architecture']
    if not isinstance(architecture, Mapping):
        raise TypeError(f'{_CONSUMER} architecture must be a mapping')
    vit = ViT(vit_config_from_mapping(architecture))
    replace_with_fixed_patch_embedding(vit, _fixed_patch_size(config))
    return PretrainedNeckEncoder3D(
        PretrainedNeckBackbone(
            vit,
            input_channels,
            neck_hidden_size=int(config['neck_hidden_size']),
        ),
        plan,
        input_channels=input_channels,
    )


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load the matched UCPT EMA ViT trunk and segmentation neck, leaving the rest at scratch."""
    if not isinstance(encoder, PretrainedNeckEncoder3D):
        raise TypeError(f'expected PretrainedNeckEncoder3D, got {type(encoder).__name__}')
    checkpoint = load_checkpoint(weights, consumer=_CONSUMER)
    load_vit(encoder.backbone.vit, checkpoint, consumer=_CONSUMER)
    load_neck(encoder.backbone.neck, checkpoint, consumer=_CONSUMER)
