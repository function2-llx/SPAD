"""Shared multi-scale tensor transforms and 3D deformable attention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SpatialShape3D = tuple[int, int, int]


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def flatten_multiscale_features_3d(
    features: Sequence[Tensor],
    level_embeddings: Tensor | None = None,
) -> tuple[Tensor, tuple[SpatialShape3D, ...]]:
    """Flatten BCHWD feature levels into one concatenated BNC sequence."""
    features = tuple(features)
    spatial_shapes = tuple(
        tuple(int(value) for value in feature.shape[2:])
        for feature in features
    )
    flattened = tuple(
        feature.flatten(2).transpose(1, 2)
        for feature in features
    )
    if level_embeddings is not None:
        if level_embeddings.shape[0] != len(flattened):
            raise ValueError(
                f'level embeddings contain {level_embeddings.shape[0]} levels, '
                f'but features contain {len(flattened)}'
            )
        flattened = tuple(
            tokens + level_embedding[None, None]
            for tokens, level_embedding in zip(
                flattened,
                level_embeddings,
                strict=True,
            )
        )
    return torch.cat(flattened, dim=1), spatial_shapes


def unflatten_multiscale_features_3d(
    tokens: Tensor,
    spatial_shapes: Sequence[Sequence[int]],
) -> tuple[Tensor, ...]:
    """Split one concatenated BNC sequence into BCHWD feature levels."""
    features = []
    level_start = 0
    for spatial_shape in spatial_shapes:
        depth, height, width = (int(value) for value in spatial_shape)
        level_length = depth * height * width
        feature = tokens[:, level_start:level_start + level_length]
        features.append(
            feature.transpose(1, 2).reshape(
                tokens.shape[0],
                tokens.shape[2],
                depth,
                height,
                width,
            )
        )
        level_start += level_length
    if level_start != tokens.shape[1]:
        raise ValueError(
            f'token length {tokens.shape[1]} does not match spatial shapes '
            f'{tuple(tuple(int(value) for value in shape) for shape in spatial_shapes)}'
        )
    return tuple(features)


def reference_points_3d(
    spatial_shapes: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> Tensor:
    """Return voxel-center coordinates normalized to the full spatial extent."""
    references = []
    for spatial_shape in spatial_shapes:
        depth, height, width = (int(value) for value in spatial_shape)
        z, y, x = torch.meshgrid(
            (torch.arange(depth, device=device, dtype=torch.float32) + 0.5) / depth,
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing='ij',
        )
        references.append(torch.stack((x, y, z), dim=-1).reshape(-1, 3))
    return torch.cat(references, dim=0).unsqueeze(0).unsqueeze(2)


class MultiScaleDeformableAttention3D(nn.Module):
    """Sparse trilinear attention over one or more 3D feature levels."""

    def __init__(
        self,
        query_dim: int,
        value_dim: int,
        output_dim: int,
        *,
        attention_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError(
                f'attention_dim {attention_dim} must be divisible by num_heads {num_heads}'
            )
        if num_levels <= 0 or num_points <= 0:
            raise ValueError('num_levels and num_points must be positive')

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = attention_dim // num_heads
        self.sampling_offsets = nn.Linear(
            query_dim,
            num_heads * num_levels * num_points * 3,
        )
        self.attention_weights = nn.Linear(
            query_dim,
            num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(value_dim, attention_dim)
        self.output_proj = nn.Linear(attention_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)

        head_index = torch.arange(self.num_heads, dtype=torch.float32)
        z = 1 - 2 * (head_index + 0.5) / self.num_heads
        radius = torch.sqrt((1 - z.square()).clamp_min(0))
        angle = head_index * (math.pi * (3 - math.sqrt(5)))
        directions = torch.stack(
            (radius * torch.cos(angle), radius * torch.sin(angle), z),
            dim=-1,
        )
        directions /= directions.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        directions = directions[:, None, None, :].expand(
            self.num_heads,
            self.num_levels,
            self.num_points,
            3,
        )
        point_scales = torch.arange(1, self.num_points + 1, dtype=torch.float32)
        directions = directions * point_scales[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(directions.reshape(-1))

        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
    ) -> Tensor:
        if query.ndim != 3 or value.ndim != 3:
            raise ValueError('query and value must have shapes [B, N, C]')
        if query.shape[0] != value.shape[0]:
            raise ValueError('query and value batch sizes must match')
        if reference_points.dtype is not torch.float32:
            raise ValueError(
                f'reference points must use float32 geometry, got {reference_points.dtype}'
            )
        spatial_shapes = tuple(
            tuple(int(size) for size in shape)
            for shape in spatial_shapes
        )
        if len(spatial_shapes) != self.num_levels:
            raise ValueError(
                f'expected {self.num_levels} value levels, got {len(spatial_shapes)}'
            )
        if any(len(shape) != 3 or any(size <= 0 for size in shape) for shape in spatial_shapes):
            raise ValueError(f'invalid 3D spatial shapes: {spatial_shapes}')
        if sum(_shape_product(shape) for shape in spatial_shapes) != value.shape[1]:
            raise ValueError(
                f'value length {value.shape[1]} does not match spatial shapes {spatial_shapes}'
            )
        expected_reference_shape = (
            query.shape[0],
            query.shape[1],
            self.num_levels,
            3,
        )
        if reference_points.shape[0] == 1 and query.shape[0] != 1:
            reference_points = reference_points.expand(query.shape[0], -1, -1, -1)
        if tuple(reference_points.shape) != expected_reference_shape:
            raise ValueError(
                f'reference points must have shape {expected_reference_shape}, '
                f'got {tuple(reference_points.shape)}'
            )

        batch_size, query_length = query.shape[:2]
        projected_value = self.value_proj(value).reshape(
            batch_size,
            value.shape[1],
            self.num_heads,
            self.head_dim,
        )
        sampling_offsets = self.sampling_offsets(query).reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
            3,
        )
        attention_weights = self.attention_weights(query).reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(dim=-1).reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        geometry_dtype = (
            torch.float32
            if query.dtype in (torch.float16, torch.bfloat16)
            else query.dtype
        )
        sampling_offsets = sampling_offsets.to(dtype=geometry_dtype)
        offset_normalizers = torch.tensor(
            [(width, height, depth) for depth, height, width in spatial_shapes],
            device=query.device,
            dtype=geometry_dtype,
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizers[None, None, None, :, None, :]
        )

        output = query.new_zeros(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        level_start = 0
        for level, spatial_shape in enumerate(spatial_shapes):
            level_length = _shape_product(spatial_shape)
            depth, height, width = spatial_shape
            level_value = projected_value[:, level_start:level_start + level_length]
            level_value = level_value.reshape(
                batch_size,
                depth,
                height,
                width,
                self.num_heads,
                self.head_dim,
            )
            level_value = level_value.permute(0, 4, 5, 1, 2, 3).reshape(
                batch_size * self.num_heads,
                self.head_dim,
                depth,
                height,
                width,
            )

            level_grid = sampling_locations[:, :, :, level]
            level_grid = level_grid.permute(0, 2, 1, 3, 4).reshape(
                batch_size * self.num_heads,
                query_length,
                self.num_points,
                1,
                3,
            )
            sample_value = (
                level_value.float()
                if level_value.dtype in (torch.float16, torch.bfloat16)
                else level_value
            )
            with torch.autocast(device_type=level_value.device.type, enabled=False):
                sampled = F.grid_sample(
                    sample_value,
                    level_grid * 2 - 1,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )
            sampled = sampled.to(dtype=projected_value.dtype)
            sampled = sampled.reshape(
                batch_size,
                self.num_heads,
                self.head_dim,
                query_length,
                self.num_points,
            ).permute(0, 3, 1, 4, 2)
            level_weights = attention_weights[:, :, :, level].unsqueeze(-1)
            output = output + (sampled * level_weights).sum(dim=3)
            level_start += level_length

        return self.output_proj(output.reshape(batch_size, query_length, self.attention_dim))
