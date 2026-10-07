"""BiomedCLIP-initialized 3D ViT-Adapter for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch import nn

from pumit.downstream.seg.adapters.fixed_patch_vit import validate_fixed_patch_size
from pumit.downstream.seg.adapters.vit import select_evenly_spaced_layers
from pumit.downstream.seg.adapters.vit_adapter import (
    TimmAbsPosInteractiveBackbone,
    VIT_ADAPTER_CONFIG_KEYS,
    VIT_ADAPTER_OPTIONAL_CONFIG_KEYS,
    ViTAdapterEncoder3D,
    build_vit_adapter_encoder,
    vit_adapter_config,
)
from pumit.downstream.seg.backbones.biomedclip import (
    _build_visual_trunk,
    _load_pretrained_trunk,
)
from pumit.downstream.seg.plan import EncoderPlan

_DEPTH = 12
_EMBED_DIM = 768
_PATCH_SIZE = 16
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
CONFIG_KEYS = frozenset({
    'feature_layers',
    'gradient_checkpointing',
    'vit_patch_size',
    *VIT_ADAPTER_CONFIG_KEYS,
})
# Existing plans omit architecture and retain zero drop path.
OPTIONAL_CONFIG_KEYS = VIT_ADAPTER_OPTIONAL_CONFIG_KEYS | {'architecture'}


class BiomedCLIPInteractiveBackbone(TimmAbsPosInteractiveBackbone):
    """Expose the BiomedCLIP visual ViT at four interaction depths."""

    def __init__(
        self,
        input_channels: int,
        patch_size: Sequence[int],
        *,
        gradient_checkpointing: bool,
        drop_path_rate: float = 0.0,
    ):
        trunk = _build_visual_trunk(
            input_channels,
            patch_size,
            gradient_checkpointing=gradient_checkpointing,
            drop_path_rate=drop_path_rate,
        )
        super().__init__(trunk, _FEATURE_LAYERS)
        self.trunk = trunk

    @property
    def _timm_model(self) -> nn.Module:
        return self.trunk


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    if weights is None:
        raise ValueError('BiomedCLIP ViT-Adapter requires open_clip_pytorch_model.bin')
    if checkpoint_format is not None:
        raise ValueError('BiomedCLIP ViT-Adapter does not accept checkpoint_format')
    return {
        'architecture': {'drop_path_rate': 0.0},
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
        raise ValueError(f'BiomedCLIP feature_layers must be {_FEATURE_LAYERS}')
    patch_size = validate_fixed_patch_size(
        config['vit_patch_size'],
        in_plane_patch_size=_PATCH_SIZE,
    )
    backbone = BiomedCLIPInteractiveBackbone(
        input_channels,
        patch_size,
        gradient_checkpointing=bool(config['gradient_checkpointing']),
        drop_path_rate=float(config.get('architecture', {'drop_path_rate': 0.0})['drop_path_rate']),
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
    if weights is None:
        raise ValueError('BiomedCLIP ViT-Adapter requires open_clip_pytorch_model.bin')
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    if not isinstance(encoder.backbone, BiomedCLIPInteractiveBackbone):
        raise TypeError(
            f'expected BiomedCLIPInteractiveBackbone, got '
            f'{type(encoder.backbone).__name__}'
        )
    _load_pretrained_trunk(
        encoder.backbone.trunk,
        weights,
        input_channels=encoder.input_channels,
    )
