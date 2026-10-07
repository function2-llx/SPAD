"""Backbone contract for interleaving flat ViT blocks with 3D adapter interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn


class InteractiveViTBackbone(nn.Module, ABC):
    """Expose one flat ViT as tokens, block ranges, and normalized patch features."""

    def __init__(
        self,
        *,
        embed_dim: int,
        feature_layers: Sequence[int],
        num_prefix_tokens: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        feature_layers = tuple(int(layer) for layer in feature_layers)
        if (
            embed_dim <= 0
            or num_prefix_tokens < 0
            or not feature_layers
            or feature_layers[0] <= 0
            or tuple(sorted(set(feature_layers))) != feature_layers
        ):
            raise ValueError(
                f'invalid interactive ViT contract: embed_dim={embed_dim}, '
                f'feature_layers={feature_layers}, num_prefix_tokens={num_prefix_tokens}'
            )
        self.embed_dim = int(embed_dim)
        self.feature_layers = feature_layers
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    @abstractmethod
    def prepare_tokens(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int], Any]:
        """Return tokens in prefix-then-DHW order, the patch grid, and block context."""

    @abstractmethod
    def run_blocks(self, tokens: Tensor, start: int, end: int, context: Any) -> Tensor:
        """Run the half-open block range ``[start, end)``."""

    @abstractmethod
    def normalize_patch_tokens(self, tokens: Tensor) -> Tensor:
        """Normalize tokens and remove the prefix tokens."""

    @abstractmethod
    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return trainable backbone parameters from shallow to deep."""
