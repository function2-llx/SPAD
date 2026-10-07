"""SegVol-initialized 3D ViT-Adapter for downstream segmentation.

Routes the SegVol image encoder (flat 3D ViT-B, SimMIM-pretrained on 96K CT then supervised on the
25 M3D-Seg datasets — a set that includes KiTS19/23, KiPA22, and AMOS22) through the shared
ViT-Adapter, dropping the prompt encoder, mask decoder, and CLIP text tower. Both released formats
load: the canonical full model (``pytorch_model.bin``, post-SFT ``model.image_encoder.*``) and the
SSL-only trunk (``vit_pretrain.ckpt``, ``model.encoder.*``), selected by file content. The trunk
keeps its native (4, 16, 16) patch; the learned absolute position embedding is trilinearly
resampled from its (8, 16, 16) training grid onto the runtime token grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils import checkpoint as torch_checkpoint

from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.vit import make_parameter_layers, select_evenly_spaced_layers
from pumit.downstream.seg.adapters.vit_adapter import (
    InteractiveViTBackbone,
    VIT_ADAPTER_CONFIG_KEYS,
    VIT_ADAPTER_OPTIONAL_CONFIG_KEYS,
    ViTAdapterEncoder3D,
    build_vit_adapter_encoder,
    vit_adapter_config,
)
from pumit.downstream.seg.backbones._segvol_encoder import (
    DEPTH,
    EMBED_DIM,
    NATIVE_GRID,
    NATIVE_PATCH,
    SegVolViT,
    convert_patch_embedding_weight,
)
from pumit.downstream.seg.plan import EncoderPlan

_FEATURE_LAYERS = select_evenly_spaced_layers(DEPTH)
_INPUT_WEIGHT_KEY = 'patch_embed.weight'
CONFIG_KEYS = frozenset({
    'feature_layers',
    'gradient_checkpointing',
    'vit_patch_size',
    *VIT_ADAPTER_CONFIG_KEYS,
})
OPTIONAL_CONFIG_KEYS = VIT_ADAPTER_OPTIONAL_CONFIG_KEYS


class SegVolInteractiveBackbone(InteractiveViTBackbone):
    """Expose the SegVol trunk at four interaction depths with its final norm as readout."""

    def __init__(
        self,
        input_channels: int,
        patch_size: Sequence[int],
        *,
        gradient_checkpointing: bool,
    ):
        patch = tuple(int(value) for value in patch_size)
        if patch != NATIVE_PATCH:
            raise ValueError(
                f'SegVol keeps its native patch {NATIVE_PATCH}; got vit_patch_size {patch}'
            )
        super().__init__(
            embed_dim=EMBED_DIM,
            feature_layers=_FEATURE_LAYERS,
            num_prefix_tokens=0,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.model = SegVolViT(input_channels)
        self.input_channels = int(input_channels)
        self.patch_size = patch

    def prepare_tokens(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int], tuple[int, int, int]]:
        tokens, grid = self.model.embed_tokens(x)
        return tokens, grid, grid

    def run_blocks(self, tokens: Tensor, start: int, end: int, context: Any) -> Tensor:
        for block in self.model.blocks[start:end]:
            if self.training and self.gradient_checkpointing:
                tokens = torch_checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens

    def normalize_patch_tokens(self, tokens: Tensor) -> Tensor:
        # SegVol reads the trunk through its final LayerNorm before reshaping to the image embedding.
        return self.model.norm(tokens)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        embedding_parameters = [*self.model.patch_embed.parameters(), self.model.pos_embed]
        return make_parameter_layers(
            embedding_parameters,
            self.model.blocks,
            tuple(self.model.norm.parameters()),
        )


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    if weights is None:
        raise ValueError('SegVol ViT-Adapter requires a released SegVol checkpoint')
    if checkpoint_format is not None:
        raise ValueError('SegVol ViT-Adapter does not accept checkpoint_format')
    return {
        'feature_layers': list(_FEATURE_LAYERS),
        'gradient_checkpointing': gradient_checkpointing,
        **vit_adapter_config(EMBED_DIM),
    }


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> ViTAdapterEncoder3D:
    if tuple(int(layer) for layer in config['feature_layers']) != _FEATURE_LAYERS:
        raise ValueError(f'SegVol feature_layers must be {_FEATURE_LAYERS}')
    backbone = SegVolInteractiveBackbone(
        input_channels,
        config['vit_patch_size'],
        gradient_checkpointing=bool(config['gradient_checkpointing']),
    )
    return build_vit_adapter_encoder(
        backbone,
        plan,
        input_channels=input_channels,
        config=config,
    )


def _extract_encoder_state(checkpoint: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Pull the trunk out of either released format by its key prefix."""
    for prefix in ('model.image_encoder.', 'model.encoder.'):
        extracted = {
            key.removeprefix(prefix): value
            for key, value in checkpoint.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    raise RuntimeError(
        'checkpoint contains neither model.image_encoder.* nor model.encoder.* keys'
    )


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the SegVol trunk; prompt/mask/text components are dropped."""
    if weights is None:
        raise ValueError('SegVol ViT-Adapter requires a released SegVol checkpoint')
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    if not isinstance(encoder.backbone, SegVolInteractiveBackbone):
        raise TypeError(
            f'expected SegVolInteractiveBackbone, got {type(encoder.backbone).__name__}'
        )

    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    source = _extract_encoder_state(checkpoint)

    state_dict: dict[str, Tensor] = {}
    for key, value in source.items():
        if key == 'patch_embedding.position_embeddings':
            state_dict['pos_embed'] = value.view(1, *NATIVE_GRID, EMBED_DIM)
        elif key == 'patch_embedding.patch_embeddings.1.weight':
            state_dict[_INPUT_WEIGHT_KEY] = convert_patch_embedding_weight(value, 1)
        elif key == 'patch_embedding.patch_embeddings.1.bias':
            state_dict['patch_embed.bias'] = value
        else:
            # Transformer blocks and the final norm share their names with the vendored module.
            state_dict[key] = value

    model = encoder.backbone.model
    input_channels = model.patch_embed.in_channels
    if input_channels != 1:
        state_dict[_INPUT_WEIGHT_KEY] = repeat_single_channel_weight(
            state_dict[_INPUT_WEIGHT_KEY],
            input_channels,
        )
    model.load_state_dict(state_dict, strict=True)
