"""PUMIT EMA ViT initialized SimpleFPN encoder for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch import nn

from pumit.downstream.seg.adapters.fixed_patch_vit import (
    FixedPatchViTBackbone,
    FixedPatchViTPyramidBackbone,
    replace_with_fixed_patch_embedding,
    validate_fixed_patch_size,
    vit_config_from_mapping,
)
from pumit.downstream.seg.adapters.pyramid import (
    PlanAlignedPyramidEncoder3D,
    SimpleFPNEncoder3D,
    pyramid_options,
)
from pumit.downstream.seg.adapters.vit import select_evenly_spaced_layers
from pumit.downstream.seg.backbones._pumit import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    load_vit,
    vit_architecture,
)
from pumit.downstream.seg.plan import EncoderPlan
from pumit.model.vit import ViT

CONFIG_KEYS = frozenset({'architecture', 'feature_layers', 'vit_patch_size'})
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch'})


def _fixed_patch_size(config: Mapping[str, object]) -> tuple[int, int, int]:
    value = config.get('vit_patch_size')
    if not isinstance(value, Sequence):
        raise TypeError(f'PUMIT SimpleFPN vit_patch_size must be a sequence, got {value!r}')
    return validate_fixed_patch_size(value, in_plane_patch_size=16)


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Extract the fixed-grid PUMIT ViT architecture without its pretrained segmentation neck."""
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(
            f'PUMIT SimpleFPN requires checkpoint_format {CHECKPOINT_FORMAT!r}, '
            f'got {checkpoint_format!r}'
        )
    checkpoint = load_checkpoint(weights, consumer='PUMIT SimpleFPN')
    architecture = vit_architecture(
        checkpoint,
        gradient_checkpointing=gradient_checkpointing,
    )
    return {
        'architecture': architecture,
        'feature_layers': list(
            select_evenly_spaced_layers(int(architecture['num_hidden_layers']))
        ),
    }


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D:
    """Construct a plan-aligned scratch SimpleFPN around the PUMIT ViT trunk.

    Uniformly spaced depths supply one feature map per plan stage from P2 onward.
    The final-depth-only configuration supplies every level through SimpleFPN.
    """
    architecture = config['architecture']
    if not isinstance(architecture, Mapping):
        raise TypeError('PUMIT SimpleFPN architecture must be a mapping')
    vit = ViT(vit_config_from_mapping(architecture))
    feature_layers = tuple(int(layer) for layer in config['feature_layers'])
    replace_with_fixed_patch_embedding(vit, _fixed_patch_size(config))
    options = pyramid_options(config)
    if feature_layers == (vit.config.num_hidden_layers,):
        return SimpleFPNEncoder3D(
            FixedPatchViTBackbone(vit, input_channels),
            plan,
            input_channels=input_channels,
            **options,
        )
    expected_feature_layers = select_evenly_spaced_layers(
        vit.config.num_hidden_layers, count=len(plan.output_channels) - 2,
    )
    if feature_layers != expected_feature_layers:
        raise ValueError(
            f'PUMIT SimpleFPN feature_layers must be {expected_feature_layers} or '
            f'({vit.config.num_hidden_layers},), got {feature_layers}'
        )
    return PlanAlignedPyramidEncoder3D(
        FixedPatchViTPyramidBackbone(vit, input_channels, feature_layers),
        plan,
        input_channels=input_channels,
        **options,
    )


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the UCPT EMA ViT trunk, leaving SimpleFPN and U-Net randomly initialized."""
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(
            f'expected PlanAlignedPyramidEncoder3D or SimpleFPNEncoder3D, got {type(encoder).__name__}'
        )
    load_vit(
        encoder.backbone.vit,
        load_checkpoint(weights, consumer='PUMIT SimpleFPN'),
        consumer='PUMIT SimpleFPN',
    )
