"""3D deformable interaction layers for ViT-Adapter."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.utils import checkpoint as torch_checkpoint

from pumit.downstream.seg.adapters.deformable_attention import (
    MultiScaleDeformableAttention3D,
    unflatten_multiscale_features_3d,
)


class MultiScaleConvFFN3D(nn.Module):
    """Feed-forward network with one depthwise 3D convolution per feature level."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.depthwise_conv = nn.Conv3d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        x = self.fc1(x)
        features = unflatten_multiscale_features_3d(x, spatial_shapes)
        levels = tuple(
            self.depthwise_conv(feature).flatten(2).transpose(1, 2)
            for feature in features
        )
        return self.fc2(self.activation(torch.cat(levels, dim=1)))


class Injector3D(nn.Module):
    """Inject multi-scale spatial priors into flat ViT patch tokens."""

    def __init__(
        self,
        vit_dim: int,
        adapter_dim: int,
        *,
        attention_dim: int,
        num_heads: int,
        num_points: int,
        num_spatial_levels: int,
        init_values: float,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.query_norm = nn.LayerNorm(vit_dim, eps=1e-6)
        self.value_norm = nn.LayerNorm(adapter_dim, eps=1e-6)
        self.attention = MultiScaleDeformableAttention3D(
            vit_dim,
            adapter_dim,
            vit_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_levels=num_spatial_levels,
            num_points=num_points,
        )
        self.gamma = nn.Parameter(torch.full((vit_dim,), float(init_values)))

    def _forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        update = self.attention(
            self.query_norm(query),
            reference_points,
            self.value_norm(value),
            spatial_shapes,
        )
        return query + self.gamma * update

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        if self.training and self.gradient_checkpointing:
            return torch_checkpoint.checkpoint(
                self._forward,
                query,
                reference_points,
                value,
                spatial_shapes,
                use_reentrant=False,
            )
        return self._forward(query, reference_points, value, spatial_shapes)


class Extractor3D(nn.Module):
    """Extract globally updated ViT semantics back into spatial-prior tokens."""

    def __init__(
        self,
        vit_dim: int,
        adapter_dim: int,
        *,
        attention_dim: int,
        num_heads: int,
        num_points: int,
        conv_ffn_hidden_dim: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.query_norm = nn.LayerNorm(adapter_dim, eps=1e-6)
        self.value_norm = nn.LayerNorm(vit_dim, eps=1e-6)
        self.attention = MultiScaleDeformableAttention3D(
            adapter_dim,
            vit_dim,
            adapter_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
        )
        self.ffn_norm = nn.LayerNorm(adapter_dim, eps=1e-6)
        self.ffn = MultiScaleConvFFN3D(adapter_dim, conv_ffn_hidden_dim)

    def _forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        query_spatial_shapes: Sequence[Sequence[int]],
        value_spatial_shape: Sequence[int],
    ) -> Tensor:
        query = query + self.attention(
            self.query_norm(query),
            reference_points,
            self.value_norm(value),
            (value_spatial_shape,),
        )
        return query + self.ffn(self.ffn_norm(query), query_spatial_shapes)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        query_spatial_shapes: Sequence[Sequence[int]],
        value_spatial_shape: Sequence[int],
    ) -> Tensor:
        if self.training and self.gradient_checkpointing:
            return torch_checkpoint.checkpoint(
                self._forward,
                query,
                reference_points,
                value,
                query_spatial_shapes,
                value_spatial_shape,
                use_reentrant=False,
            )
        return self._forward(
            query,
            reference_points,
            value,
            query_spatial_shapes,
            value_spatial_shape,
        )


class InteractionBlock3D(nn.Module):
    """One bidirectional exchange around a contiguous group of ViT blocks."""

    def __init__(
        self,
        vit_dim: int,
        adapter_dim: int,
        *,
        attention_dim: int,
        num_heads: int,
        num_points: int,
        num_spatial_levels: int,
        conv_ffn_hidden_dim: int,
        injector_init_values: float,
        num_extra_extractors: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.injector = Injector3D(
            vit_dim,
            adapter_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_points=num_points,
            num_spatial_levels=num_spatial_levels,
            init_values=injector_init_values,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.extractor = Extractor3D(
            vit_dim,
            adapter_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_points=num_points,
            conv_ffn_hidden_dim=conv_ffn_hidden_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.extra_extractors = nn.ModuleList(
            Extractor3D(
                vit_dim,
                adapter_dim,
                attention_dim=attention_dim,
                num_heads=num_heads,
                num_points=num_points,
                conv_ffn_hidden_dim=conv_ffn_hidden_dim,
                gradient_checkpointing=gradient_checkpointing,
            )
            for _ in range(num_extra_extractors)
        )
