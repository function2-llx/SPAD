from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Local MedMNIST archives used by training and feature extraction.
DEFAULT_DATA_ROOT = 'precompute/downstream/cls/medmnist_npz'


def resize_volume(x: Tensor, resize: int | None, *, is_3d: bool) -> Tensor:
    """Spatially resize a (B,3,D,H,W) encoder tensor to `resize` (no-op if None).

    2D (D==1): bilinear over (H,W), depth axis untouched -> (B,3,1,resize,resize).
    3D: trilinear over (D,H,W) -> (B,3,resize,resize,resize).
    Intended to run per-batch on-device (keeps CPU arrays at native size).
    """
    if resize is None:
        return x
    if is_3d:
        return F.interpolate(x, size=(resize, resize, resize),
                             mode='trilinear', align_corners=False)
    plane = F.interpolate(x[:, :, 0], size=(resize, resize),
                          mode='bilinear', align_corners=False)
    return plane.unsqueeze(2)


def to_3ch_volume(images: np.ndarray, *, is_3d: bool) -> Tensor:
    """uint8 MedMNIST images -> (B,3,D,H,W) float in [0,1], grayscale repeated to 3ch.

    2D input: (N,H,W) or (N,H,W,C) -> (N,C,1,H,W). 3D input: (N,D,H,W) -> (N,1,D,H,W).
    Size-invariant base for the per-backbone transform_batch normalization functions.
    """
    x = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0
    if is_3d:
        x = x.unsqueeze(1)  # (N, D, H, W) -> (N, 1, D, H, W)
    else:
        if x.ndim == 3:  # (N, H, W) -> (N, H, W, 1)
            x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2).unsqueeze(2)  # (N,H,W,C) -> (N,C,1,H,W)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1, 1)
    assert x.shape[1] == 3, f'expected 3 channels, got {x.shape[1]}'
    return x.contiguous()


def normalize_to_encoder(images: np.ndarray, *, is_3d: bool) -> Tensor:
    """uint8 MedMNIST images -> (B,3,D,H,W) float in [-1,1], matching PUMIT pretraining."""
    return to_3ch_volume(images, is_3d=is_3d) * 2 - 1


def _extract_npy_streaming(npz_path, member: str, out_path) -> None:
    """Stream a compressed .npy member out of a .npz to an uncompressed .npy, without
    loading the whole array into RAM (copies the decompressing stream in chunks)."""
    import zipfile
    with zipfile.ZipFile(npz_path) as z:
        with z.open(member) as src, open(out_path, 'wb') as out:
            while True:
                chunk = src.read(8 << 20)  # 8 MiB
                if not chunk:
                    break
                out.write(chunk)


def build_arrays(flag: str, size: int, split: str, *,
                 root: str | None = None, download: bool = False):
    """Load a MedMNIST split as (images, labels[int64,(N,)]).

    LAZY (download=False, the training path): the npz is `{root}/{flag}_{size}.npz`
    with members `{split}_images.npy` (DEFLATE-compressed). Each split is stream-extracted
    ONCE to an uncompressed sibling `.npy` (in `root/_memmap/`, cached) and returned as a
    `np.memmap` (mode='r'). Only accessed batch pages are resident -- RSS stays at the
    batch working set, not the whole split. Essential on the 64 GiB pod: the big-2D sets
    (oct/path/tissue ~5-16 GB uint8) would otherwise blow the cgroup when run concurrently.

    Eager (download=True, the one-time staging path): uses the medmnist DataClass (downloads
    + loads into RAM). When root is None, medmnist's default (~/.medmnist) is used.
    """
    root_path = Path(root or DEFAULT_DATA_ROOT)
    npz = root_path / f'{flag}_{size}.npz'
    if download or not npz.exists():
        import medmnist
        from medmnist import INFO
        DataClass = getattr(medmnist, INFO[flag]['python_class'])
        kwargs = {'split': split, 'download': download, 'size': size, 'root': str(root_path)}
        ds = DataClass(**kwargs)
        return ds.imgs, ds.labels.astype(np.int64).reshape(-1)

    import zipfile
    memmap_dir = root_path / '_memmap'
    memmap_dir.mkdir(parents=True, exist_ok=True)
    npy = memmap_dir / f'{flag}_{size}_{split}_images.npy'
    if not npy.exists():
        tmp = memmap_dir / f'.{flag}_{size}_{split}_images.part'
        _extract_npy_streaming(npz, f'{split}_images.npy', tmp)
        tmp.rename(npy)
    images = np.load(npy, mmap_mode='r')  # memmap: slicing copies only the accessed batch
    with zipfile.ZipFile(npz) as z:
        with z.open(f'{split}_labels.npy') as fh:
            labels = np.load(fh)
    return images, labels.astype(np.int64).reshape(-1)
