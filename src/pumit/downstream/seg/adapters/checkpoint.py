"""Pure tensor transforms for adapting pretrained encoder checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor


def repeat_single_channel_weight(weight: Tensor, input_channels: int) -> Tensor:
    """Repeat and scale a single-channel kernel while preserving equal-channel responses."""
    if weight.ndim != 5 or weight.shape[1] != 1:
        raise ValueError(
            f'expected [out, 1, depth, height, width] kernel, got {tuple(weight.shape)}'
        )
    if input_channels <= 0:
        raise ValueError(f'input_channels must be positive, got {input_channels}')
    return weight.repeat(1, input_channels, 1, 1, 1) / input_channels


def adapt_input_weight(weight: Tensor, input_channels: int) -> Tensor:
    """Adapt a patch projection while preserving its response to equal input channels."""
    if input_channels <= 0:
        raise ValueError(f'input_channels must be positive, got {input_channels}')
    if weight.shape[1] == input_channels:
        return weight
    collapsed = weight.sum(dim=1, keepdim=True)
    return collapsed.repeat(1, input_channels, *([1] * (weight.ndim - 2))) / input_channels


def _group_sum(weight: Tensor, dim: int, target: int, axis_name: str) -> Tensor:
    source = int(weight.shape[dim])
    if source == target:
        return weight
    if source < target or source % target:
        raise ValueError(
            f'patch kernel {axis_name} {source} must be an integer multiple of target {target}'
        )
    shape = list(weight.shape)
    shape[dim] = target
    shape.insert(dim + 1, source // target)
    return weight.reshape(shape).sum(dim=dim + 1)


def adapt_patch_embed_weight(
    weight: Tensor,
    target_patch_size: Sequence[int],
) -> Tensor:
    """Adapt a 2D or fixed 3D patch kernel to a smaller fixed patch size.

    Depth: a 2D kernel is repeated across the target depth and divided by it (mean over slices); a 3D
    kernel is reduced by contiguous group summation. In-plane: contiguous group summation, which is
    exactly equivalent to applying the source kernel to a nearest-neighbor upsampled input.
    """
    target_patch_size = tuple(int(value) for value in target_patch_size)
    if len(target_patch_size) != 3 or any(value <= 0 for value in target_patch_size):
        raise ValueError(f'target patch size must contain three positive values, got {target_patch_size}')
    target_depth, target_height, target_width = target_patch_size
    if weight.ndim == 4:
        weight = _group_sum(weight, 2, target_height, 'height')
        weight = _group_sum(weight, 3, target_width, 'width')
        return weight.unsqueeze(2).expand(-1, -1, target_depth, -1, -1).clone() / target_depth
    if weight.ndim != 5:
        raise ValueError(f'patch embedding weight must be 4D or 5D, got shape {tuple(weight.shape)}')
    weight = _group_sum(weight, 2, target_depth, 'depth')
    weight = _group_sum(weight, 3, target_height, 'height')
    return _group_sum(weight, 4, target_width, 'width')
