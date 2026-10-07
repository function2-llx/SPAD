"""Row-aligned bit packing for binary segmentation masks."""

from collections.abc import Sequence

import numpy as np

_BITORDER = 'big'


def _normalize_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    shape_3d = tuple(int(size) for size in shape)
    if len(shape_3d) != 3 or any(size < 1 for size in shape_3d):
        raise ValueError(f'mask shape must contain three positive dimensions, got {shape_3d!r}')
    return shape_3d


def _validate_packed_mask(packed: np.ndarray, shape: Sequence[int]) -> tuple[int, int, int]:
    shape_3d = _normalize_shape(shape)
    expected_shape = (*shape_3d[:-1], (shape_3d[-1] + 7) // 8)
    if packed.dtype != np.uint8 or packed.shape != expected_shape:
        raise ValueError(
            f'expected a row-aligned uint8 packed mask shaped {expected_shape}, got '
            f'dtype={packed.dtype} shape={packed.shape}; regenerate segmentation artifacts'
        )
    return shape_3d


def pack_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a ``(D, H, W)`` binary mask along W."""
    if mask.dtype != np.bool_:
        raise TypeError(f'binary mask must have bool dtype, got {mask.dtype}')
    _normalize_shape(mask.shape)
    return np.packbits(mask, axis=-1, bitorder=_BITORDER)


def unpack_binary_mask(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Unpack a complete row-aligned mask as uint8."""
    shape_3d = _validate_packed_mask(packed, shape)
    return np.unpackbits(packed, axis=-1, count=shape_3d[-1], bitorder=_BITORDER)


def unpack_binary_mask_crop(
    packed: np.ndarray,
    shape: Sequence[int],
    crop_slices: tuple[slice, slice, slice],
) -> np.ndarray:
    """Unpack only a spatial crop from a row-aligned mask."""
    shape_3d = _validate_packed_mask(packed, shape)
    if len(crop_slices) != 3:
        raise ValueError(f'mask crop must contain three slices, got {crop_slices!r}')

    bounds: list[tuple[int, int]] = []
    for axis, (axis_slice, size) in enumerate(zip(crop_slices, shape_3d, strict=True)):
        if not isinstance(axis_slice, slice) or axis_slice.step not in (None, 1):
            raise ValueError(f'mask crop axis {axis} must be a unit-step slice, got {axis_slice!r}')
        if axis_slice.start is None or axis_slice.stop is None:
            raise ValueError(f'mask crop axis {axis} must have explicit bounds, got {axis_slice!r}')
        start, stop = int(axis_slice.start), int(axis_slice.stop)
        if not 0 <= start < stop <= size:
            raise ValueError(f'mask crop axis {axis} has invalid bounds {axis_slice!r} for size {size}')
        bounds.append((start, stop))

    (z0, z1), (y0, y1), (x0, x1) = bounds
    byte_start = x0 // 8
    packed_crop = packed[z0:z1, y0:y1, byte_start:(x1 + 7) // 8]
    unpacked = np.unpackbits(packed_crop, axis=-1, bitorder=_BITORDER)
    bit_start = x0 - byte_start * 8
    return unpacked[..., bit_start:bit_start + x1 - x0]
