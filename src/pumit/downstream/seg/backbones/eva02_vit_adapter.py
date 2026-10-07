"""EVA-02-L-initialized 3D ViT-Adapter for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch import Tensor, nn

from pumit.downstream.cls.backbones.eva02 import (
    EVA_DEPTH_ROPE_BASE,
    _extend_eva_rope_3d,
)
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
from pumit.downstream.seg.backbones.eva02 import (
    _adapted_pretrained_state,
    _build_eva_model,
)
from pumit.downstream.seg.plan import EncoderPlan

_DEPTH = 24
_EMBED_DIM = 1024
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


class Eva02LargeInteractiveBackbone(TimmAbsPosInteractiveBackbone):
    """Expose EVA-02-L blocks with its lifted absolute positions and 3D RoPE."""

    def __init__(
        self,
        input_channels: int,
        patch_size: Sequence[int],
        *,
        gradient_checkpointing: bool,
        drop_path_rate: float = 0.0,
    ):
        model = _build_eva_model(
            input_channels,
            patch_size,
            gradient_checkpointing=gradient_checkpointing,
            drop_path_rate=drop_path_rate,
        )
        super().__init__(model, _FEATURE_LAYERS)
        self.model = model

    @property
    def _timm_model(self) -> nn.Module:
        return self.model

    def _prepare_block_kwargs(
        self,
        tokens: Tensor,
        patch_shape: tuple[int, int, int],
    ) -> dict[str, Tensor]:
        rope_2d = self.model.rope._get_pos_embed_values(
            patch_shape[1:],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        rope_3d = _extend_eva_rope_3d(
            rope_2d,
            patch_shape[0],
            EVA_DEPTH_ROPE_BASE,
        )
        return {'rope': rope_3d}


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    if weights is None or weights.suffix != '.safetensors':
        raise ValueError('EVA-02-L ViT-Adapter requires the released model.safetensors checkpoint')
    if checkpoint_format is not None:
        raise ValueError('EVA-02-L ViT-Adapter does not accept checkpoint_format')
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
        raise ValueError(f'EVA-02-L feature_layers must be {_FEATURE_LAYERS}')
    patch_size = validate_fixed_patch_size(
        config['vit_patch_size'],
        in_plane_patch_size=_PATCH_SIZE,
    )
    backbone = Eva02LargeInteractiveBackbone(
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
        raise ValueError('EVA-02-L ViT-Adapter requires the released model.safetensors checkpoint')
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    if not isinstance(encoder.backbone, Eva02LargeInteractiveBackbone):
        raise TypeError(
            f'expected Eva02LargeInteractiveBackbone, got '
            f'{type(encoder.backbone).__name__}'
        )
    state = _adapted_pretrained_state(
        weights,
        input_channels=encoder.backbone.model.patch_embed.proj.in_channels,
        patch_size=encoder.backbone.patch_size,
    )
    encoder.backbone.model.load_state_dict(state, strict=True)
