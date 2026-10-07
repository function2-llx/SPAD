# src/pumit/ucpt/mask.py
"""Mask loading (training + viz) + affine application (viz-only) for UCPT.

Copied — does not import — from archive pumit.seg.replay_dataset. Standalone
functions (no class self). DATA_ROOT is a parameter, not a cwd-relative constant.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from monai import transforms as mt
from monai.utils import GridSampleMode, GridSamplePadMode

from pumit.segmentation_mask import unpack_binary_mask, unpack_binary_mask_crop
from pumit.ucpt.affine import stream_crop_size


def _find_mask_file(
    data_root: Path,
    dataset: str,
    key: str,
    source: str,
    cls_name: str,
) -> Path | None:
    label_dir = Path(data_root) / dataset / 'labels' / key / source
    if not label_dir.exists():
        return None
    # Preprocessing stores either <name> or <NN>__<name> and replaces '/' with '_'.
    # The '__' anchor prevents 'liver' from matching '00__liver_tumor.npy.zst'.
    stem = cls_name.replace('/', '_')
    for mask_file in label_dir.iterdir():
        base = mask_file.name[:-len('.npy.zst')] if mask_file.name.endswith('.npy.zst') else None
        if base is not None and (base == stem or base.endswith(f'__{stem}')):
            return mask_file
    return None


def _read_packed_mask(mask_file: Path) -> np.ndarray:
    dctx = zstd.ZstdDecompressor()
    return np.load(io.BytesIO(dctx.decompress(mask_file.read_bytes())))


def _load_mask(
    data_root: Path, dataset: str, key: str, source: str, cls_name: str,
    shape_3d: tuple[int, int, int],
    *,
    as_bool: bool = True,
) -> np.ndarray | None:
    """Load a zstd+packbits-compressed binary mask.

    Args:
        data_root: Root directory containing preprocessed datasets.
        dataset: Dataset name.
        key: Sample key.
        source: Label source.
        cls_name: Raw class name from the stream.
        shape_3d: Image voxel grid ``(D, H, W)``.
        as_bool: Convert unpacked bytes to bool instead of retaining uint8.
    Returns:
        The unpacked binary mask as bool (default) or uint8, or ``None`` when the label directory or class file is
        absent.
    """
    mask_file = _find_mask_file(data_root, dataset, key, source, cls_name)
    if mask_file is None:
        return None
    mask = unpack_binary_mask(_read_packed_mask(mask_file), shape_3d)
    return mask.astype(bool) if as_bool else mask


def _load_mask_crop(
    data_root: Path,
    dataset: str,
    key: str,
    source: str,
    cls_name: str,
    shape_3d: tuple[int, int, int],
    crop_slices: tuple[slice, slice, slice],
) -> np.ndarray | None:
    """Load only the requested crop of a compressed binary mask."""
    mask_file = _find_mask_file(data_root, dataset, key, source, cls_name)
    if mask_file is None:
        return None
    return unpack_binary_mask_crop(_read_packed_mask(mask_file), shape_3d, crop_slices)


def _apply_affine_to_mask(
    mask: np.ndarray, params: list,
) -> torch.Tensor:
    """Apply the image's spatial affine to a binary mask.

    Args:
        mask: Unpacked boolean mask shaped ``(D, H, W)``.
        params: Sampled transform parameters whose first entry is spatial.

    Returns:
        Nearest-resampled float mask shaped ``(1, *crop_size)`` with values in ``{0, 1}``.
    """
    sp = params[0]
    crop_size = stream_crop_size(sp)
    affine_flat = sp['affine']
    load_slice_start = sp['load_slice_start']
    load_slice_stop = sp['load_slice_stop']

    mask_t = torch.from_numpy(mask).unsqueeze(0).float()

    z0, y0, x0 = load_slice_start
    z1, y1, x1 = load_slice_stop
    crop = mask_t[:, z0:z1, y0:y1, x0:x1]

    affine_4x4 = np.array(affine_flat, dtype=np.float64).reshape(4, 4)
    result = mt.Affine(
        affine=affine_4x4,
        spatial_size=crop_size,
        image_only=True,
        mode=GridSampleMode.NEAREST,
        padding_mode=GridSamplePadMode.ZEROS,
        dtype=torch.float32,
    )(crop)

    if hasattr(result, 'as_tensor'):
        result = result.as_tensor()
    return (result > 0.5).float()
