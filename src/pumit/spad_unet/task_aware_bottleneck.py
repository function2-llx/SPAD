"""Task-Aware Bottleneck for Universal U-Nets."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


TAB_TOKENS_PER_DATASET = 16
TAB_DIM = 256
TAB_DEPTH = 2
TAB_HEADS = 8
TAB_MLP_DIM = 2048
TAB_ATTENTION_DOWNSAMPLE_RATE = 2
TAB_FOURIER_SCALE = 1.0
TAB_FOURIER_SEED = 20260801
TAB_MIN_INPLANE_DOWNSAMPLE = 16


class RandomFourierPositionEncoding3D(nn.Module):
    r"""Encode normalized 3D voxel-center coordinates with fixed random frequencies.

    For a coordinate \(c=(z,y,x)\in[0,1]^3\) and a fixed Gaussian matrix
    \(G\in\mathbb R^{3\times C/2}\), the encoding is

    \[\phi(c)=[\sin(2\pi(2c-1)G),\ \cos(2\pi(2c-1)G)]\in\mathbb R^C.\]
    """

    def __init__(
        self,
        embedding_dim: int = TAB_DIM,
        scale: float = TAB_FOURIER_SCALE,
        seed: int = TAB_FOURIER_SEED,
    ) -> None:
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2:
            raise ValueError('Fourier embedding_dim must be a positive even integer')
        if scale <= 0:
            raise ValueError('Fourier scale must be positive')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        gaussian_matrix = scale * torch.randn(
            3,
            embedding_dim // 2,
            generator=generator,
        )
        self.register_buffer('gaussian_matrix', gaussian_matrix)

    def forward(
        self,
        spatial_shape: Sequence[int],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one flattened patch-relative encoding with shape ``[1, D H W, C]``."""
        shape = tuple(int(size) for size in spatial_shape)
        if len(shape) != 3 or any(size < 1 for size in shape):
            raise ValueError(f'spatial_shape must contain three positive values, got {shape}')
        axes = tuple(
            (torch.arange(size, device=self.gaussian_matrix.device, dtype=torch.float32) + 0.5)
            / size
            for size in shape
        )
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing='ij'),
            dim=-1,
        )
        phases = 2 * math.pi * ((2 * coordinates - 1) @ self.gaussian_matrix.float())
        encoding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return encoding.reshape(1, math.prod(shape), -1).to(dtype=dtype)


class _Attention(nn.Module):
    """SAM-style multi-head attention backed by PyTorch SDPA."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        if downsample_rate < 1 or embedding_dim % downsample_rate:
            raise ValueError('attention downsample_rate must divide embedding_dim')
        internal_dim = embedding_dim // downsample_rate
        if internal_dim % num_heads:
            raise ValueError('attention internal dimension must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = internal_dim // num_heads
        self.q_proj = nn.Linear(embedding_dim, internal_dim)
        self.k_proj = nn.Linear(embedding_dim, internal_dim)
        self.v_proj = nn.Linear(embedding_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embedding_dim)

    def _separate_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = tensor.shape
        return tensor.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            channels // self.num_heads,
        )

    def _recombine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, num_heads, head_dim = tensor.shape
        return tensor.reshape(
            batch_size,
            num_tokens,
            num_heads * head_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query = self._separate_heads(self.q_proj(query))
        key = self._separate_heads(self.k_proj(key))
        value = self._separate_heads(self.v_proj(value))
        output = self._recombine_heads(
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
            ).transpose(1, 2)
        )
        return self.out_proj(output)


class _MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.activation(self.lin1(tokens)))


class TwoWayTaskBlock(nn.Module):
    """Update task tokens from image tokens, then image tokens from task tokens."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int,
        *,
        skip_first_task_pe: bool,
    ) -> None:
        super().__init__()
        self.task_self_attention = _Attention(embedding_dim, num_heads)
        self.task_to_image_attention = _Attention(
            embedding_dim,
            num_heads,
            attention_downsample_rate,
        )
        self.task_mlp = _MLPBlock(embedding_dim, mlp_dim)
        self.image_to_task_attention = _Attention(
            embedding_dim,
            num_heads,
            attention_downsample_rate,
        )
        self.task_self_norm = nn.LayerNorm(embedding_dim)
        self.task_to_image_norm = nn.LayerNorm(embedding_dim)
        self.task_mlp_norm = nn.LayerNorm(embedding_dim)
        self.image_to_task_norm = nn.LayerNorm(embedding_dim)
        self.skip_first_task_pe = skip_first_task_pe

    def forward(
        self,
        task_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        task_pe: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_task_pe:
            task_tokens = self.task_self_attention(
                task_tokens,
                task_tokens,
                task_tokens,
            )
        else:
            task_query = task_tokens + task_pe
            task_tokens = task_tokens + self.task_self_attention(
                task_query,
                task_query,
                task_tokens,
            )
        task_tokens = self.task_self_norm(task_tokens)

        task_tokens = self.task_to_image_norm(
            task_tokens
            + self.task_to_image_attention(
                task_tokens + task_pe,
                image_tokens + image_pe,
                image_tokens,
            )
        )
        task_tokens = self.task_mlp_norm(
            task_tokens + self.task_mlp(task_tokens)
        )
        image_tokens = self.image_to_task_norm(
            image_tokens
            + self.image_to_task_attention(
                image_tokens + image_pe,
                task_tokens + task_pe,
                task_tokens,
            )
        )
        return task_tokens, image_tokens


class TaskAwareBottleneck(nn.Module):
    """Condition one decoder bottleneck on an ordered dataset-token bank."""

    def __init__(
        self,
        bottleneck_channels: int,
        num_datasets: int,
        tokens_per_dataset: int = TAB_TOKENS_PER_DATASET,
        embedding_dim: int = TAB_DIM,
        depth: int = TAB_DEPTH,
        num_heads: int = TAB_HEADS,
        mlp_dim: int = TAB_MLP_DIM,
        attention_downsample_rate: int = TAB_ATTENTION_DOWNSAMPLE_RATE,
        fourier_scale: float = TAB_FOURIER_SCALE,
        fourier_seed: int = TAB_FOURIER_SEED,
    ) -> None:
        super().__init__()
        if bottleneck_channels < 1 or num_datasets < 1 or tokens_per_dataset < 1:
            raise ValueError('bottleneck channels, datasets, and task-token count must be positive')
        if depth < 1:
            raise ValueError('Two-Way depth must be positive')
        self.num_datasets = num_datasets
        self.input_projection = nn.Conv3d(
            bottleneck_channels,
            embedding_dim,
            kernel_size=1,
        )
        self.output_projection = nn.Conv3d(
            embedding_dim,
            bottleneck_channels,
            kernel_size=1,
        ).to(memory_format=torch.channels_last_3d)
        self.task_tokens = nn.Parameter(
            torch.empty(num_datasets, tokens_per_dataset, embedding_dim)
        )
        nn.init.normal_(self.task_tokens, std=0.02)
        self.position_encoding = RandomFourierPositionEncoding3D(
            embedding_dim,
            fourier_scale,
            fourier_seed,
        )
        self.blocks = nn.ModuleList(
            TwoWayTaskBlock(
                embedding_dim,
                num_heads,
                mlp_dim,
                attention_downsample_rate,
                skip_first_task_pe=block_index == 0,
            )
            for block_index in range(depth)
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        dataset_indices: torch.Tensor,
    ) -> torch.Tensor:
        if bottleneck.ndim != 5:
            raise ValueError(f'bottleneck must be 5D, got shape {tuple(bottleneck.shape)}')
        if dataset_indices.ndim != 1 or dataset_indices.shape[0] != bottleneck.shape[0]:
            raise ValueError(
                f'dataset_indices must have shape ({bottleneck.shape[0]},), '
                f'got {tuple(dataset_indices.shape)}'
            )
        if dataset_indices.dtype != torch.long:
            raise TypeError('dataset_indices must use torch.long')

        projected = self.input_projection(bottleneck)
        image_tokens = projected.flatten(2).transpose(1, 2)
        image_pe = self.position_encoding(
            bottleneck.shape[2:],
            dtype=image_tokens.dtype,
        )
        task_pe = self.task_tokens.index_select(0, dataset_indices)
        task_tokens = task_pe
        for block in self.blocks:
            task_tokens, image_tokens = block(
                task_tokens,
                image_tokens,
                task_pe,
                image_pe,
            )
        conditioned = image_tokens.transpose(1, 2).reshape_as(projected)
        return bottleneck + self.output_projection(conditioned)


class MultiScaleTaskAwareBottleneck(nn.Module):
    """Condition multiple encoder grids through one shared task-token exchange."""

    def __init__(
        self,
        feature_channels: Sequence[int],
        num_datasets: int,
        tokens_per_dataset: int = TAB_TOKENS_PER_DATASET,
        embedding_dim: int = TAB_DIM,
        depth: int = TAB_DEPTH,
        num_heads: int = TAB_HEADS,
        mlp_dim: int = TAB_MLP_DIM,
        attention_downsample_rate: int = TAB_ATTENTION_DOWNSAMPLE_RATE,
        fourier_scale: float = TAB_FOURIER_SCALE,
        fourier_seed: int = TAB_FOURIER_SEED,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in feature_channels)
        if not channels or any(value < 1 for value in channels):
            raise ValueError('feature_channels must contain positive values')
        if num_datasets < 1 or tokens_per_dataset < 1:
            raise ValueError('datasets and task-token count must be positive')
        if depth < 1:
            raise ValueError('Two-Way depth must be positive')
        self.num_datasets = num_datasets
        self.input_projections = nn.ModuleList(
            nn.Conv3d(channel, embedding_dim, kernel_size=1)
            for channel in channels
        )
        self.output_projections = nn.ModuleList(
            nn.Conv3d(embedding_dim, channel, kernel_size=1).to(
                memory_format=torch.channels_last_3d,
            )
            for channel in channels
        )
        self.level_embeddings = nn.Parameter(
            torch.empty(len(channels), embedding_dim)
        )
        nn.init.normal_(self.level_embeddings, std=0.02)
        self.task_tokens = nn.Parameter(
            torch.empty(num_datasets, tokens_per_dataset, embedding_dim)
        )
        nn.init.normal_(self.task_tokens, std=0.02)
        self.position_encoding = RandomFourierPositionEncoding3D(
            embedding_dim,
            fourier_scale,
            fourier_seed,
        )
        self.blocks = nn.ModuleList(
            TwoWayTaskBlock(
                embedding_dim,
                num_heads,
                mlp_dim,
                attention_downsample_rate,
                skip_first_task_pe=block_index == 0,
            )
            for block_index in range(depth)
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        dataset_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        if len(features) != len(self.input_projections):
            raise ValueError(
                f'expected {len(self.input_projections)} feature levels, '
                f'got {len(features)}'
            )
        if not features:
            raise ValueError('multi-scale TAB requires at least one feature level')
        batch_size = features[0].shape[0]
        if any(feature.ndim != 5 for feature in features):
            raise ValueError('every multi-scale TAB feature must be 5D')
        if any(feature.shape[0] != batch_size for feature in features):
            raise ValueError('multi-scale TAB features must share a batch size')
        if dataset_indices.ndim != 1 or dataset_indices.shape[0] != batch_size:
            raise ValueError(
                f'dataset_indices must have shape ({batch_size},), '
                f'got {tuple(dataset_indices.shape)}'
            )
        if dataset_indices.dtype != torch.long:
            raise TypeError('dataset_indices must use torch.long')

        projected_features = [
            projection(feature)
            for projection, feature in zip(
                self.input_projections,
                features,
                strict=True,
            )
        ]
        token_counts = [math.prod(feature.shape[2:]) for feature in features]
        image_tokens = torch.cat(
            [feature.flatten(2).transpose(1, 2) for feature in projected_features],
            dim=1,
        )
        image_pe = torch.cat(
            [
                self.position_encoding(
                    feature.shape[2:],
                    dtype=image_tokens.dtype,
                )
                + self.level_embeddings[level_index][None, None]
                for level_index, feature in enumerate(features)
            ],
            dim=1,
        )
        task_pe = self.task_tokens.index_select(0, dataset_indices)
        task_tokens = task_pe
        for block in self.blocks:
            task_tokens, image_tokens = block(
                task_tokens,
                image_tokens,
                task_pe,
                image_pe,
            )

        conditioned_tokens = image_tokens.split(token_counts, dim=1)
        return [
            feature
            + output_projection(
                tokens.transpose(1, 2).reshape_as(projected)
            )
            for feature, projected, tokens, output_projection in zip(
                features,
                projected_features,
                conditioned_tokens,
                self.output_projections,
                strict=True,
            )
        ]
