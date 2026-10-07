"""Packed storage for generation-time UCPT segmentation targets."""

from __future__ import annotations

from functools import cache
import os
from pathlib import Path
import time

import numpy as np
import torch
import zstandard as zstd

from pumit.segmentation_mask import pack_binary_mask


MASK_FRAME_KEY = '_positive_mask_frame'
MASK_REF_KEY = 'positive_mask_ref'
MASK_STORAGE = 'packbits-zstd1-v1'


@cache
def _compressor() -> zstd.ZstdCompressor:
    return zstd.ZstdCompressor(level=1)


def encode_positive_masks(masks: list[np.ndarray]) -> bytes:
    """Pack and compress selected positive masks in class order."""
    if not masks:
        return b''
    shape = masks[0].shape
    packed = []
    for mask in masks:
        if mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError(
                f'positive masks must share one boolean shape, got dtype={mask.dtype} shape={mask.shape}'
            )
        packed.append(pack_binary_mask(mask))
    return encode_packed_positive_masks(packed, shape=shape)


def encode_packed_positive_masks(
    packed_masks: list[np.ndarray],
    *,
    shape: list[int] | tuple[int, int, int],
) -> bytes:
    """Compress already-packed selected positive masks in class order."""
    if not packed_masks:
        return b''
    shape_3d = tuple(int(value) for value in shape)
    if len(shape_3d) != 3 or any(value < 1 for value in shape_3d):
        raise ValueError(f'invalid positive-mask shape: {shape!r}')
    expected_shape = (*shape_3d[:2], (shape_3d[2] + 7) // 8)
    for mask in packed_masks:
        if mask.dtype != np.uint8 or mask.shape != expected_shape:
            raise ValueError(
                f'packed positive masks must be uint8 arrays shaped {expected_shape}, '
                f'got dtype={mask.dtype} shape={mask.shape}'
            )
    return _compressor().compress(np.stack(packed_masks).tobytes())


def decode_positive_masks(
    frame: bytes,
    *,
    count: int,
    shape: list[int] | tuple[int, int, int],
) -> torch.Tensor:
    """Decode one occurrence frame to ``(K, D, H, W)`` boolean masks."""
    shape_3d = tuple(int(value) for value in shape)
    if len(shape_3d) != 3 or any(value < 1 for value in shape_3d):
        raise ValueError(f'invalid positive-mask shape: {shape!r}')
    if count < 0:
        raise ValueError(f'positive-mask count must be non-negative, got {count}')
    if count == 0:
        if frame:
            raise ValueError('zero positive masks must use an empty frame')
        return torch.empty((0, *shape_3d), dtype=torch.bool)
    if not frame:
        raise ValueError('positive masks require a non-empty frame')

    packed_width = (shape_3d[2] + 7) // 8
    expected_bytes = count * shape_3d[0] * shape_3d[1] * packed_width
    packed_bytes = zstd.decompress(frame)
    if len(packed_bytes) != expected_bytes:
        raise ValueError(
            f'positive-mask frame decoded to {len(packed_bytes)} bytes, expected {expected_bytes}'
        )
    packed = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(
        count,
        shape_3d[0],
        shape_3d[1],
        packed_width,
    )
    masks = np.unpackbits(
        packed,
        axis=-1,
        count=shape_3d[2],
        bitorder='big',
    )
    return torch.from_numpy(masks).bool()


def mask_shard_path(stream_dir: Path, shard_id: int) -> Path:
    """Return the positive-mask sidecar path for one stream shard."""
    return stream_dir / 'masks' / f'shard_{shard_id:05d}.bin'


def write_mask_shard(batches: list[dict], path: Path) -> dict[str, int]:
    """Write occurrence-level compressed frames and replace private frames with offsets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'refusing to overwrite existing mask shard: {path}')

    tmp = path.parent / f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp'
    offset = 0
    labeled_samples = 0
    positive_masks = 0
    with tmp.open('xb') as file:
        for batch in batches:
            for sample in batch['samples']:
                if not sample['labeled']:
                    if MASK_FRAME_KEY in sample or MASK_REF_KEY in sample:
                        raise ValueError('unlabeled sample cannot contain positive-mask storage')
                    continue
                labeled_samples += 1
                positive_count = sum(item['is_positive'] for item in sample['classes'])
                positive_masks += positive_count
                try:
                    frame = sample.pop(MASK_FRAME_KEY)
                except KeyError:
                    raise ValueError('fresh labeled sample lacks its positive-mask frame') from None
                if not isinstance(frame, bytes):
                    raise TypeError(f'positive-mask frame must be bytes, got {type(frame).__name__}')
                if bool(frame) != bool(positive_count):
                    raise ValueError(
                        f'positive-mask frame presence differs from positive query count {positive_count}'
                    )
                sample[MASK_REF_KEY] = [offset, len(frame)]
                file.write(frame)
                offset += len(frame)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return {
        'mask_labeled_samples': labeled_samples,
        'mask_positive_masks': positive_masks,
        'mask_storage_bytes': offset,
    }


def read_mask_frame(fd: int, reference: list[int] | tuple[int, int]) -> bytes:
    """Read one compressed frame from an open mask-shard descriptor."""
    if not isinstance(reference, list | tuple) or len(reference) != 2:
        raise ValueError(f'invalid positive-mask reference: {reference!r}')
    offset, length = map(int, reference)
    if offset < 0 or length < 0:
        raise ValueError(f'invalid positive-mask reference: {reference!r}')
    frame = os.pread(fd, length, offset)
    if len(frame) != length:
        raise ValueError(
            f'positive-mask sidecar returned {len(frame)} bytes, expected {length} at offset {offset}'
        )
    return frame
