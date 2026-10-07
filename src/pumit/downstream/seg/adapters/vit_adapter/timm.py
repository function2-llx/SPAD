"""Shared ViT-Adapter runtime for timm ViTs with learned 2D positions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pumit.downstream.seg.adapters.timm_vit import lift_2d_position_embedding
from pumit.downstream.seg.adapters.vit import make_parameter_layers
from pumit.downstream.seg.adapters.vit_adapter.backbone import InteractiveViTBackbone


class TimmAbsPosInteractiveBackbone(InteractiveViTBackbone):
    """Interleave adapter interactions with a single-CLS timm ViT block sequence."""

    def __init__(self, model: nn.Module, feature_layers: Sequence[int]):
        if (
            model.num_prefix_tokens != 1
            or model.cls_token is None
            or model.no_embed_class
        ):
            raise RuntimeError(
                'the timm ViT-Adapter runtime requires one CLS token with embedded positions'
            )
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        else:
            patch_size = tuple(int(value) for value in patch_size)
        if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
            raise ValueError(f'invalid timm 3D patch size: {patch_size}')
        super().__init__(
            embed_dim=model.embed_dim,
            feature_layers=feature_layers,
            num_prefix_tokens=model.num_prefix_tokens,
            gradient_checkpointing=model.grad_checkpointing,
        )
        self.patch_size = patch_size

    @property
    @abstractmethod
    def _timm_model(self) -> nn.Module:
        """Return the model under its backbone-specific registered attribute."""

    def _prepare_block_kwargs(
        self,
        tokens: Tensor,
        patch_shape: tuple[int, int, int],
    ) -> dict[str, Any]:
        return {}

    def prepare_tokens(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int], dict[str, Any]]:
        input_shape = tuple(int(value) for value in x.shape[2:])
        if any(
            size % step
            for size, step in zip(input_shape, self.patch_size, strict=True)
        ):
            raise ValueError(
                f'input shape {input_shape} must be divisible by timm ViT patch size '
                f'{self.patch_size}'
            )
        patch_shape = tuple(
            size // step
            for size, step in zip(input_shape, self.patch_size, strict=True)
        )
        model = self._timm_model
        tokens = model.patch_embed(x)
        tokens = torch.cat(
            (model.cls_token.expand(x.shape[0], -1, -1), tokens),
            dim=1,
        )
        tokens = tokens + lift_2d_position_embedding(
            model.pos_embed,
            patch_shape,
            num_prefix_tokens=model.num_prefix_tokens,
        )
        tokens = model.pos_drop(tokens)
        if model.patch_drop is not None:
            tokens = model.patch_drop(tokens)
        tokens = model.norm_pre(tokens)
        return tokens, patch_shape, self._prepare_block_kwargs(tokens, patch_shape)

    def run_blocks(
        self,
        tokens: Tensor,
        start: int,
        end: int,
        context: Any,
    ) -> Tensor:
        if not isinstance(context, dict):
            raise TypeError(f'timm block kwargs must be a dict, got {type(context).__name__}')
        for block in self._timm_model.blocks[start:end]:
            if self.training and self.gradient_checkpointing:
                tokens = checkpoint(
                    block,
                    tokens,
                    use_reentrant=False,
                    **context,
                )
            else:
                tokens = block(tokens, **context)
        return tokens

    def normalize_patch_tokens(self, tokens: Tensor) -> Tensor:
        return self._timm_model.norm(tokens)[:, self.num_prefix_tokens:]

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        model = self._timm_model
        return make_parameter_layers(
            (
                *model.patch_embed.parameters(),
                model.cls_token,
                model.pos_embed,
            ),
            model.blocks,
            model.norm.parameters(),
        )
