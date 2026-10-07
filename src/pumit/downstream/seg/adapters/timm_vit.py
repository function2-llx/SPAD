"""Runtime adaptation for timm ViTs on dynamic 3D patch grids."""

from __future__ import annotations

import math
from collections.abc import Sequence

import einops
import torch
from timm.layers import resample_patch_embed
from torch import Tensor, nn
from torch.nn import functional as F

from pumit.spadop.conv import uniform_inflator

from .checkpoint import adapt_input_weight

REVIEWED_SEG_TIMM_VERSION = '1.0.22'


def assert_reviewed_seg_timm() -> None:
    """Fail fast if nnU-Net's timm dependency differs from the reviewed version."""
    import timm

    if timm.__version__ != REVIEWED_SEG_TIMM_VERSION:
        raise RuntimeError(
            f'the dense adapters were reviewed against timm {REVIEWED_SEG_TIMM_VERSION}, '
            f'found {timm.__version__}'
        )


def inflate_patch_projection(
    source: nn.Conv2d,
    *,
    input_channels: int,
    patch_size: int | Sequence[int],
) -> nn.Conv3d:
    """Resize a 2D patch projection and uniformly inflate it onto a fixed 3D patch."""
    if source.kernel_size[0] != source.kernel_size[1] or source.stride != source.kernel_size:
        raise ValueError(f'expected a square non-overlapping patch projection, got {source}')
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * 3
    else:
        patch_size = tuple(int(value) for value in patch_size)
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError(f'patch_size must contain three positive values, got {patch_size}')
    depth, height, width = patch_size
    weight = source.weight.detach()
    if source.kernel_size != (height, width):
        weight = resample_patch_embed(weight, [height, width])
    weight = adapt_input_weight(weight, input_channels)
    weight = uniform_inflator(weight, depth).permute(1, 2, 0, 3, 4).contiguous()

    projection = nn.Conv3d(
        input_channels,
        source.out_channels,
        kernel_size=patch_size,
        stride=patch_size,
        bias=source.bias is not None,
    )
    projection.weight = nn.Parameter(weight)
    if source.bias is not None:
        projection.bias = nn.Parameter(source.bias.detach().clone())
    return projection


def lift_2d_position_embedding(
    position: Tensor,
    target_grid: Sequence[int],
    *,
    num_prefix_tokens: int,
) -> Tensor:
    """Interpolate a learned 2D table in-plane and replicate it along target depth."""
    target_grid = tuple(int(value) for value in target_grid)
    if len(target_grid) != 3 or any(value <= 0 for value in target_grid):
        raise ValueError(f'target_grid must contain three positive values, got {target_grid}')
    prefix = position[:, :num_prefix_tokens]
    patches = position[:, num_prefix_tokens:]
    source_edge = math.isqrt(patches.shape[1])
    if source_edge * source_edge != patches.shape[1]:
        raise ValueError(f'position table has a non-square 2D patch grid: {patches.shape[1]}')
    patches = einops.rearrange(
        patches,
        '1 (h w) c -> 1 c 1 h w',
        h=source_edge,
        w=source_edge,
    )
    patches = F.interpolate(
        patches,
        size=target_grid,
        mode='trilinear',
        align_corners=False,
    )
    patches = einops.rearrange(patches, '1 c d h w -> 1 (d h w) c')
    return torch.cat([prefix, patches], dim=1)


class DynamicPatchEmbed3D(nn.Module):
    """Fixed-kernel non-overlapping Conv3d patch embedding returning NLC tokens."""

    def __init__(self, projection: nn.Conv3d):
        super().__init__()
        if projection.stride != projection.kernel_size:
            raise ValueError(f'expected a stride-equals-kernel projection, got {projection}')
        self.proj = projection
        self.patch_size = tuple(int(value) for value in projection.kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)
