"""DINOv3-initialized fixed-patch ViT adapter for dense segmentation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path

import safetensors.torch as st
from torch import nn

from pumit.downstream.seg.adapters.checkpoint import adapt_patch_embed_weight
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
from pumit.downstream.seg.plan import EncoderPlan
from pumit.model.vit import ViT, ViTConfig

_ALLOWED_UNEXPECTED = {'embeddings.mask_token', 'rope_embeddings.inv_freq'}
_VIT_L_CONFIG = ViTConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    num_register_tokens=4,
    patch_size=16,
    in_channels=3,
    pos_embed_rescale=None,
)
_FEATURE_LAYERS = select_evenly_spaced_layers(_VIT_L_CONFIG.num_hidden_layers)
FEATURE_LAYERS = _FEATURE_LAYERS
PATCH_WEIGHT_KEY = 'embeddings.patch_embeddings.weight'
CONFIG_KEYS = frozenset({'architecture', 'feature_layers', 'vit_patch_size'})
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch'})


def fixed_patch_size(config: Mapping[str, object]) -> tuple[int, int, int]:
    """Read and validate the explicit downstream DINOv3 patch size."""
    value = config.get('vit_patch_size')
    if not isinstance(value, Sequence):
        raise TypeError(f'DINOv3 vit_patch_size must be a sequence, got {value!r}')
    return validate_fixed_patch_size(
        value,
        in_plane_patch_size=_VIT_L_CONFIG.patch_size,
    )


def build_fixed_vit(
    architecture: Mapping[str, object],
    patch_size: Sequence[int],
) -> ViT:
    """Build a ViT whose downstream patch projection has one fixed 3D shape."""
    vit = ViT(vit_config_from_mapping(architecture))
    replace_with_fixed_patch_embedding(vit, patch_size)
    return vit


def pinned_config(gradient_checkpointing: bool) -> dict[str, object]:
    """Return the pinned DINOv3 ViT-L architecture and feature layers shared by every ViT-L arm."""
    return {
        'architecture': dataclasses.asdict(
            dataclasses.replace(_VIT_L_CONFIG, grad_ckpt=gradient_checkpointing)
        ),
        'feature_layers': list(_FEATURE_LAYERS),
    }


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Return the pinned DINOv3 ViT-L architecture used by this baseline."""
    if weights is None:
        raise ValueError('DINOv3 requires a safetensors checkpoint path')
    if checkpoint_format is not None:
        raise ValueError('DINOv3 does not accept checkpoint_format')
    return pinned_config(gradient_checkpointing)


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D:
    """Construct the DINOv3-initialized architecture without reading its weights.

    Uniformly spaced depths supply one feature map per plan stage from P2 onward.
    The final-depth-only configuration supplies every level through SimpleFPN.
    """
    architecture = config['architecture']
    if not isinstance(architecture, Mapping):
        raise TypeError('DINOv3 architecture must be a mapping')
    feature_layers = tuple(int(layer) for layer in config['feature_layers'])
    patch_size = fixed_patch_size(config)
    vit = build_fixed_vit(architecture, patch_size)
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
            f'DINOv3 feature_layers must be {expected_feature_layers} or ({vit.config.num_hidden_layers},), '
            f'got {feature_layers}'
        )
    return PlanAlignedPyramidEncoder3D(
        FixedPatchViTPyramidBackbone(vit, input_channels, feature_layers),
        plan,
        input_channels=input_channels,
        **options,
    )


def load_vit_pretrained(vit: ViT, weights: Path | None) -> None:
    """Adapt and load a 2D or full-depth DINOv3 checkpoint into one fixed 3D ViT."""
    if weights is None:
        raise ValueError('DINOv3 requires a safetensors checkpoint path')
    patch_embedding = vit.embeddings.patch_embeddings
    if not isinstance(patch_embedding, nn.Conv3d):
        raise TypeError(
            f'downstream DINOv3 requires a fixed Conv3d patch embedding, got '
            f'{type(patch_embedding).__name__}'
        )
    state_dict = st.load_file(str(weights))
    if PATCH_WEIGHT_KEY in state_dict:
        state_dict[PATCH_WEIGHT_KEY] = adapt_patch_embed_weight(
            state_dict[PATCH_WEIGHT_KEY],
            patch_embedding.kernel_size,
        )
    missing, unexpected = vit.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing:
        raise RuntimeError(f'DINOv3 checkpoint is missing encoder keys: {missing}')
    bad_unexpected = set(unexpected) - _ALLOWED_UNEXPECTED
    if bad_unexpected:
        raise RuntimeError(f'DINOv3 checkpoint has unexpected keys: {sorted(bad_unexpected)}')


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Adapt and load a DINOv3 checkpoint into the fixed 3D ViT."""
    if weights is None:
        raise ValueError('DINOv3 requires a safetensors checkpoint path')
    if not isinstance(encoder, PlanAlignedPyramidEncoder3D | SimpleFPNEncoder3D):
        raise TypeError(
            f'expected PlanAlignedPyramidEncoder3D or SimpleFPNEncoder3D, got {type(encoder).__name__}'
        )
    load_vit_pretrained(encoder.backbone.vit, weights)
