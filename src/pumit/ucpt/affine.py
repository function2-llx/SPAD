"""Affine geometry and replay kernels for UCPT crops."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


_NULL_SPACING_THRESHOLD = 1e5  # spacings >= this are sentinels, not physical
_INPLANE_ROUNDOFF_ULPS = 64


def _column_norm_spacing(M_3x3: np.ndarray, s0: np.ndarray) -> list[float]:
    """Compute per-axis spacing after an affine transform.

    Args:
        M_3x3: Realized affine matrix.
        s0: Source spacing; the 2D depth axis may be ``NaN``.

    Returns:
        Spacing-weighted column norms of ``M_3x3``.
    """
    # Convert finite 2D spacing sentinels to NaN so the downstream validity mask drops them.
    s0 = np.where(np.abs(s0) >= _NULL_SPACING_THRESHOLD, np.nan, s0)
    weighted = M_3x3 * s0[:, None]  # row j scaled by s0[j]
    # 0 * NaN = NaN in IEEE 754, so clamp zero entries to 0.0 to prevent
    # NaN from contaminating columns where the M entry genuinely contributes nothing.
    weighted[np.abs(M_3x3) < 1e-10] = 0.0
    return [float(np.sqrt((weighted[:, i] ** 2).sum())) for i in range(3)]


def _realized_output_to_source(affine_3x3: np.ndarray, crop_size: list[int]) -> np.ndarray:
    """Return the linear map actually realized by affine resampling."""
    inplane = _canonical_inplane_affine(affine_3x3)
    if inplane is not None:
        affine_3x3 = inplane
    if not _uses_interpolate(affine_3x3):
        return affine_3x3
    diag = np.diag(affine_3x3)
    extent = _affine_load_extent(affine_3x3, crop_size)
    return np.diag(np.copysign(extent / np.asarray(crop_size), diag))


def _uses_interpolate(affine_3x3: np.ndarray) -> bool:
    off_diag = affine_3x3 - np.diag(np.diag(affine_3x3))
    return bool(np.count_nonzero(off_diag) == 0)


def _canonical_inplane_affine(affine_3x3: np.ndarray) -> np.ndarray | None:
    """Return an exactly depth-separable affine when cross-axis terms are only fp64 roundoff."""
    scale = max(float(np.abs(affine_3x3).max()), 1.0)
    tolerance = _INPLANE_ROUNDOFF_ULPS * np.finfo(np.float64).eps * scale
    cross_axis = np.concatenate((affine_3x3[0, 1:], affine_3x3[1:, 0]))
    if np.abs(cross_axis).max() > tolerance:
        return None
    canonical = affine_3x3.copy()
    canonical[0, 1:] = 0.0
    canonical[1:, 0] = 0.0
    return canonical


def _canonical_inplane_grid_affine(affine_3x3: np.ndarray) -> np.ndarray | None:
    """Return the canonical matrix exactly when replay takes the in-plane grid branch."""
    if _uses_interpolate(affine_3x3):
        return None
    return _canonical_inplane_affine(affine_3x3)


def _affine_load_extent(
    affine_3x3: np.ndarray,
    reference_size: list[int] | np.ndarray,
) -> np.ndarray:
    """Return the pre-clamp source extent required by an affine crop."""
    reference = np.asarray(reference_size)
    inplane = _canonical_inplane_affine(affine_3x3)
    if inplane is not None:
        affine_3x3 = inplane
    if _uses_interpolate(affine_3x3):
        extent = np.abs(np.diag(affine_3x3)) * reference
    else:
        extent = np.abs(affine_3x3) @ reference
    return np.ceil(extent).astype(np.int64)


def _quantize_scale_to_load_extent(
    scale: list[float],
    reference_size: list[int],
) -> np.ndarray:
    """Return per-axis scales exactly realized by an integer load extent."""
    reference = np.asarray(reference_size, dtype=np.float64)
    extent = _affine_load_extent(np.diag(scale), reference)
    return extent / reference


def _pad_to_load_extent(
    crop: torch.Tensor,
    load_extent: np.ndarray,
    *,
    mode: str,
) -> torch.Tensor:
    """Pad a volume-clamped crop back to its requested source extent."""
    current = np.asarray(crop.shape[1:])
    pad_arg: list[int] = []
    for axis in (2, 1, 0):  # F.pad expects (W_lo,W_hi, H_lo,H_hi, D_lo,D_hi)
        deficit = max(int(load_extent[axis] - current[axis]), 0)
        lo = deficit // 2
        pad_arg += [lo, deficit - lo]
    crop = crop.float()
    if any(pad_arg):
        crop = F.pad(crop.unsqueeze(0), pad_arg, mode=mode).squeeze(0)
    return crop


def _grid_resample_3d(
    crop: torch.Tensor,
    affine_3x3: np.ndarray,
    output_size: list[int],
    reference_size: list[int],
    *,
    mode: str,
) -> torch.Tensor:
    """Apply the generic 3D affine grid used for genuinely oblique transforms."""
    load_dhw = list(crop.shape[1:])
    M_whd = affine_3x3[::-1, ::-1].copy()
    load_whd = [load_dhw[2], load_dhw[1], load_dhw[0]]
    reference_whd = [reference_size[2], reference_size[1], reference_size[0]]
    theta_3x3 = M_whd * np.array(reference_whd)[None, :] / np.array(load_whd)[:, None]
    theta_3x4 = torch.zeros(1, 3, 4, dtype=torch.float32)
    theta_3x4[0, :, :3] = torch.from_numpy(theta_3x3.astype(np.float32))
    grid_shape = [1, crop.shape[0], *output_size]
    grid = F.affine_grid(theta_3x4, grid_shape, align_corners=False).to(crop.device)
    return F.grid_sample(
        crop.unsqueeze(0),
        grid,
        mode=mode,
        padding_mode='border',
        align_corners=False,
    ).squeeze(0)


def _grid_resample_inplane(
    crop: torch.Tensor,
    affine_3x3: np.ndarray,
    output_size: list[int],
    reference_size: list[int],
    *,
    mode: str,
) -> torch.Tensor:
    """Apply depth resampling independently from a shared batched 2D in-plane grid."""
    C, load_d, load_h, load_w = crop.shape
    output_d, output_h, output_w = output_size

    # Treat channels and slices as channels of one 2D image so every slice uses exactly one shared grid.
    M_wh = affine_3x3[1:, 1:][::-1, ::-1].copy()
    theta_wh = M_wh * np.array([reference_size[2], reference_size[1]])[None, :]
    theta_wh /= np.array([load_w, load_h])[:, None]
    theta_2x3 = torch.zeros(1, 2, 3, dtype=torch.float32)
    theta_2x3[0, :, :2] = torch.from_numpy(theta_wh.astype(np.float32))
    xy_input = crop.reshape(1, C * load_d, load_h, load_w)
    xy_grid = F.affine_grid(
        theta_2x3,
        [1, C * load_d, output_h, output_w],
        align_corners=False,
    ).to(crop.device)
    xy = F.grid_sample(
        xy_input,
        xy_grid,
        mode=mode,
        padding_mode='border',
        align_corners=False,
    ).reshape(C, load_d, output_h, output_w)

    depth_input = xy.permute(0, 2, 3, 1).reshape(1, C * output_h * output_w, load_d)
    if affine_3x3[0, 0] < 0:
        depth_input = depth_input.flip(-1)
    interpolate_mode = 'nearest-exact' if mode == 'nearest' else 'linear'
    depth = F.interpolate(
        depth_input,
        size=output_d,
        mode=interpolate_mode,
        **({'align_corners': False} if interpolate_mode == 'linear' else {}),
    )
    return depth.reshape(C, output_h, output_w, output_d).permute(0, 3, 1, 2).contiguous()


def _affine_resample(
    crop: torch.Tensor,
    affine_4x4: np.ndarray,
    output_size: list[int],
    *,
    interp_mode: str,
    grid_mode: str,
    pad_mode: str = 'replicate',
    reference_size: list[int] | None = None,
) -> torch.Tensor:
    """Resample a cropped volume under an affine to ``output_size``.

    Images and masks share this implementation so they take the same affine and diagonal/rotation branch.

    The affine maps the reference FOV into source coordinates. Its integer load box contains that continuous
    FOV without changing the affine, while ``output_size`` only controls sampling density. The image call has
    reference == output; lower-resolution callers pass the image crop size as ``reference_size`` so both
    grids cover the same FOV.

    Args:
        crop: Writable tensor shaped ``(C, D, H, W)``.
        affine_4x4: Affine matrix in DHW order.
        output_size: ``[D, H, W]`` output grid size (resolution only).
        interp_mode: ``F.interpolate`` mode for the diagonal path.
        grid_mode: ``F.grid_sample`` mode for the rotation path.
        pad_mode: ``F.pad`` mode for the diagonal path.
        reference_size: ``[D, H, W]`` spatial size the affine was designed for. Defaults to ``output_size``.

    Returns:
        Resampled tensor with the input dtype and ``output_size`` spatial shape.
    """
    if reference_size is None:
        reference_size = output_size
    if not np.isfinite(affine_4x4).all():
        raise ValueError('affine matrix must contain only finite values')
    input_dtype = crop.dtype
    M_dhw = affine_4x4[:3, :3]
    diag = np.diag(M_dhw)
    use_interpolate = _uses_interpolate(M_dhw)
    load_extent = _affine_load_extent(M_dhw, reference_size)
    crop = _pad_to_load_extent(crop, load_extent, mode=pad_mode)
    if use_interpolate:
        flip_axes = [i + 1 for i, value in enumerate(diag) if value < 0]
        if flip_axes:
            crop = torch.flip(crop, dims=flip_axes)
        interp_kwargs: dict = {'mode': interp_mode}
        if interp_mode not in ('nearest', 'area', 'nearest-exact'):
            interp_kwargs['align_corners'] = False
        result = F.interpolate(
            crop.unsqueeze(0), size=output_size, **interp_kwargs,
        ).squeeze(0)
    else:
        # Construct coordinates on CPU so generation and replay use the same grid values, then move only the grid.
        # CPU grid_sample also needs fp32 input to retain sub-voxel coordinate precision.
        inplane = _canonical_inplane_grid_affine(M_dhw)
        if inplane is None:
            result = _grid_resample_3d(
                crop, M_dhw, output_size, reference_size, mode=grid_mode,
            )
        else:
            result = _grid_resample_inplane(
                crop, inplane, output_size, reference_size, mode=grid_mode,
            )
    return result.to(input_dtype)


def stream_crop_size(sp: dict) -> list[int]:
    """Return the image crop size from frozen stream spatial parameters."""
    if 'crop_size' in sp:
        return sp['crop_size']
    return sp['patch_size']


def replay_affine_patch(
    data: dict,
    *,
    da_enc: int | None,
    affine: list[float],
    load_slice_start: list[int],
    load_slice_stop: list[int],
    n_patches: int,
    spacing_label: list[float | None],
    patch_size: list[int] | None = None,
    crop_size: list[int] | None = None,
) -> dict:
    """Load and resample one image from frozen spatial parameters."""
    if (patch_size is None) == (crop_size is None):
        raise ValueError(
            f'exactly one of patch_size/crop_size must be set, got {patch_size=} {crop_size=}'
        )
    crop_size = crop_size if crop_size is not None else patch_size
    data = dict(data)
    img_path = data['img']
    img = np.load(img_path, 'r')
    start = np.asarray(load_slice_start, dtype=np.int64)
    stop = np.asarray(load_slice_stop, dtype=np.int64)
    volume_shape = np.asarray(img.shape[1:], dtype=np.int64)
    if (
        start.shape != (3,)
        or stop.shape != (3,)
        or volume_shape.shape != (3,)
        or np.any(start < 0)
        or np.any(stop <= start)
        or np.any(stop > volume_shape)
    ):
        raise ValueError(
            f'load slices [{start.tolist()}, {stop.tolist()}) are outside '
            f'volume shape {volume_shape.tolist()}'
        )
    z0, y0, x0 = start
    z1, y1, x1 = stop
    crop = img[:, z0:z1, y0:y1, x0:x1]

    affine_4x4 = np.array(affine, dtype=np.float64).reshape(4, 4)
    load_extent = _affine_load_extent(affine_4x4[:3, :3], crop_size)
    crop_shape = np.asarray(crop.shape[1:])
    if np.any(crop_shape > load_extent):
        raise ValueError(
            f'affine crop shape {tuple(int(value) for value in crop_shape)} exceeds '
            f'its load extent {tuple(int(value) for value in load_extent)}; '
            'migrate the frozen metadata'
        )
    # Force a writable contiguous copy because the source is a read-only memmap.
    crop = torch.from_numpy(np.array(crop, order='C'))
    result = _affine_resample(
        crop, affine_4x4, crop_size,
        interp_mode='trilinear', grid_mode='bilinear', pad_mode='replicate',
    )

    data['img'] = result
    data['da_enc'] = da_enc
    data['n_patches'] = n_patches

    del img
    fd = os.open(img_path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return data
