"""3DINO-initialized 3D ViT-Adapter for downstream segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import einops
import torch
from torch import Tensor, nn
from torch.utils import checkpoint as torch_checkpoint

from pumit.downstream.cls.backbones.three_dino import (
    PATCH,
    _HEAD_PREFIXES,
    _load_state_dict,
    build_encoder as build_three_dino,
)
from pumit.downstream.seg.adapters.checkpoint import (
    adapt_patch_embed_weight,
    repeat_single_channel_weight,
)
from pumit.downstream.seg.adapters.fixed_patch_vit import validate_fixed_patch_size
from pumit.downstream.seg.adapters.vit import make_parameter_layers, select_evenly_spaced_layers
from pumit.downstream.seg.adapters.vit_adapter import (
    InteractiveViTBackbone,
    VIT_ADAPTER_CONFIG_KEYS,
    VIT_ADAPTER_OPTIONAL_CONFIG_KEYS,
    ViTAdapterEncoder3D,
    build_vit_adapter_encoder,
    vit_adapter_config,
)
from pumit.downstream.seg.plan import EncoderPlan

_DEPTH = 24
_EMBED_DIM = 1024
_FEATURE_LAYERS = select_evenly_spaced_layers(_DEPTH)
_PATCH_WEIGHT_KEY = 'patch_embed.proj.weight'
CONFIG_KEYS = frozenset({
    'feature_layers',
    'gradient_checkpointing',
    'vit_patch_size',
    *VIT_ADAPTER_CONFIG_KEYS,
})
# Legacy plans omit architecture and retain zero drop path.
OPTIONAL_CONFIG_KEYS = VIT_ADAPTER_OPTIONAL_CONFIG_KEYS | {'architecture'}


def _transformer_blocks(model: nn.Module) -> tuple[nn.Module, ...]:
    if model.chunked_blocks:
        return tuple(
            block
            for chunk in model.blocks
            for block in chunk
            if not isinstance(block, nn.Identity)
        )
    return tuple(model.blocks)


class ThreeDinoInteractiveBackbone(InteractiveViTBackbone):
    """Expose 3DINO tokens in nnU-Net DHW order while preserving native HWD positions."""

    def __init__(
        self,
        input_channels: int,
        patch_size: Sequence[int],
        *,
        gradient_checkpointing: bool,
        drop_path_rate: float = 0.0,
    ):
        model = build_three_dino(input_channels, drop_path_rate=drop_path_rate)
        if model.n_blocks != _DEPTH:
            raise RuntimeError(f'expected {_DEPTH} 3DINO blocks, got {model.n_blocks}')
        super().__init__(
            embed_dim=model.embed_dim,
            feature_layers=_FEATURE_LAYERS,
            num_prefix_tokens=1,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.model = model
        self.input_channels = int(input_channels)
        self.patch_size = tuple(int(value) for value in patch_size)

        native_patch_size = (self.patch_size[1], self.patch_size[2], self.patch_size[0])
        source = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv3d(
            source.in_channels,
            source.out_channels,
            kernel_size=native_patch_size,
            stride=native_patch_size,
            bias=source.bias is not None,
        )
        model.patch_embed.patch_size = native_patch_size
        model.mask_token.requires_grad_(False)

    def prepare_tokens(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int], None]:
        native_input = einops.rearrange(x, 'b c d h w -> b c h w d')
        native_shape = tuple(
            int(size // step)
            for size, step in zip(
                native_input.shape[2:],
                self.model.patch_embed.patch_size,
                strict=True,
            )
        )
        if any(
            size % step
            for size, step in zip(
                native_input.shape[2:],
                self.model.patch_embed.patch_size,
                strict=True,
            )
        ):
            raise ValueError(
                f'input shape {tuple(x.shape[2:])} must be divisible by 3DINO patch size '
                f'{self.patch_size}'
            )
        patch_tokens = self.model.patch_embed(native_input)
        tokens = torch.cat(
            (self.model.cls_token.expand(x.shape[0], -1, -1), patch_tokens),
            dim=1,
        )
        # Position interpolation remains the released 3DINO implementation. Its input
        # sizes are synthesized from the actual grid because only the patch depth may differ.
        position_input_shape = tuple(size * PATCH for size in native_shape)
        tokens = tokens + self.model.interpolate_pos_encoding(
            tokens,
            *position_input_shape,
        )

        prefix_tokens = tokens[:, :1]
        patch_tokens = einops.rearrange(
            tokens[:, 1:],
            'b (h w d) c -> b (d h w) c',
            h=native_shape[0],
            w=native_shape[1],
            d=native_shape[2],
        )
        patch_shape = (native_shape[2], native_shape[0], native_shape[1])
        return torch.cat((prefix_tokens, patch_tokens), dim=1), patch_shape, None

    def run_blocks(
        self,
        tokens: Tensor,
        start: int,
        end: int,
        context: Any,
    ) -> Tensor:
        if context is not None:
            raise ValueError('3DINO blocks do not accept per-forward context')
        blocks = _transformer_blocks(self.model)
        for block in blocks[start:end]:
            if self.training and self.gradient_checkpointing:
                tokens = torch_checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens

    def normalize_patch_tokens(self, tokens: Tensor) -> Tensor:
        return self.model.norm(tokens)[:, 1:]

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return make_parameter_layers(
            (
                *self.model.patch_embed.parameters(),
                self.model.cls_token,
                self.model.pos_embed,
            ),
            _transformer_blocks(self.model),
            self.model.norm.parameters(),
        )


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    if weights is None:
        raise ValueError('3DINO ViT-Adapter requires the released checkpoint')
    if checkpoint_format is not None:
        raise ValueError('3DINO ViT-Adapter does not accept checkpoint_format')
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
        raise ValueError(f'3DINO feature_layers must be {_FEATURE_LAYERS}')
    patch_size = validate_fixed_patch_size(
        config['vit_patch_size'],
        in_plane_patch_size=PATCH,
    )
    architecture = config.get('architecture', {'drop_path_rate': 0.0})
    if not isinstance(architecture, Mapping):
        raise TypeError('3DINO architecture must be a mapping')
    backbone = ThreeDinoInteractiveBackbone(
        input_channels,
        patch_size,
        gradient_checkpointing=bool(config['gradient_checkpointing']),
        drop_path_rate=float(architecture['drop_path_rate']),
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
    if not isinstance(encoder, ViTAdapterEncoder3D):
        raise TypeError(f'expected ViTAdapterEncoder3D, got {type(encoder).__name__}')
    if not isinstance(encoder.backbone, ThreeDinoInteractiveBackbone):
        raise TypeError(
            f'expected ThreeDinoInteractiveBackbone, got '
            f'{type(encoder.backbone).__name__}'
        )
    state_dict = dict(_load_state_dict(weights))
    source_weight = state_dict[_PATCH_WEIGHT_KEY]
    input_channels = encoder.backbone.model.patch_embed.proj.in_channels
    if input_channels != source_weight.shape[1]:
        source_weight = repeat_single_channel_weight(source_weight, input_channels)
    target_native_patch = encoder.backbone.model.patch_embed.proj.kernel_size
    if tuple(source_weight.shape[2:]) != tuple(target_native_patch):
        source_dhw = source_weight.permute(0, 1, 4, 2, 3)
        target_dhw = (target_native_patch[2], target_native_patch[0], target_native_patch[1])
        source_weight = adapt_patch_embed_weight(source_dhw, target_dhw).permute(0, 1, 3, 4, 2)
    state_dict[_PATCH_WEIGHT_KEY] = source_weight.contiguous()

    missing, unexpected = encoder.backbone.model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f'3DINO checkpoint is missing encoder keys: {missing}')
    bad_unexpected = [key for key in unexpected if not key.startswith(_HEAD_PREFIXES)]
    if bad_unexpected:
        raise RuntimeError(f'3DINO checkpoint has unexpected non-head keys: {bad_unexpected}')
