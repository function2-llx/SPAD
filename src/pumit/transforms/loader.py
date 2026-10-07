from collections.abc import Sequence

import numpy as np
import torch
from torch.types import Device

from monai import transforms as mt
from monai.data import get_random_patch
from monai.utils import GridSampleMode, GridSamplePadMode, ImageMetaKey

from .info import TransInfo

def get_rotation_matrix(axis: Sequence[float], θ: float) -> np.ndarray:
    cos = np.cos(θ)
    sin = np.sin(θ)
    x, y, z = axis
    return np.array((
        (cos + x * x * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin),
        (y * x * (1 - cos) + z * sin, cos + y * y * (1 - cos), y * z * (1 - cos) - x * sin),
        (z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z * z * (1 - cos)),
    ))

def smooth_for_resampling(img: torch.Tensor, downsample_scale: Sequence[int]):
    assert len(downsample_scale) == img.ndim - 1
    factors = torch.as_tensor(downsample_scale)
    # use the default sigma in skimage.transform.resize
    anti_aliasing_sigma = ((factors - 1) / 2).clamp(0).tolist()
    anti_aliasing_filter = mt.GaussianSmooth(anti_aliasing_sigma)
    smoothed_img = anti_aliasing_filter(img)
    return smoothed_img

class PUMITLoader(mt.Randomizable, mt.Transform):
    """
    This loader calculates affine transform before loading, then read content necessary and as little as possible from the disk
    """

    def __init__(self, rotate_p: float, rotate_axis_p: float, device: Device = 'cpu'):
        self.rotate_p = rotate_p
        self.rotate_axis_p = rotate_axis_p
        self.device = device

    def get_rotation(self, spacing: np.ndarray):
        if self.R.uniform() >= self.rotate_p:
            return None, None
        if np.any(np.isnan(spacing)):
            return None, None
        spacing_xy = spacing[1:].min()
        if spacing[0] < 3 * spacing_xy and self.R.uniform() < self.rotate_axis_p:
            # 3D rotation with restricted axis closing to z-axis
            axis_φ = self.R.uniform(7 * np.pi / 15, np.pi / 2)
        else:
            # dummy 2D rotation along z-axis
            axis_φ = np.pi / 2
        axis_z = np.sin(axis_φ)
        r_xy = np.cos(axis_φ)
        axis_θ = self.R.uniform(0, 2 * np.pi)
        axis = np.array((axis_z, r_xy * np.sin(axis_θ), r_xy * np.cos(axis_θ)))
        θ = self.R.uniform(0, 2 * np.pi)
        return axis, θ

    def __call__(self, data: dict):
        data = dict(data)
        trans_info: TransInfo = data['_trans']
        spacing = data['spacing']
        spacing_xy = spacing[1:].min()

        # Step 1: Random 3D rotation (or identity if skipped / NaN spacing)
        axis, θ = self.get_rotation(spacing)
        if axis is None:
            rotate_affine = np.eye(3)
        else:
            # Inverse transform: negate angle so resampling pulls from rotated source
            rotate_affine = get_rotation_matrix(axis, -θ)

        # Step 2: Scale from gen_trans_info (accounts for target patch size + augmentation)
        scale = trans_info['scale']
        scale_affine = np.diag(scale)

        # Step 3: Compose rotation and scale. Order matters for anisotropic data:
        # - Near-isotropic (z ≈ xy): randomly swap order for augmentation
        # - Anisotropic, unknown spacing, or 2D: scale first (preserves depth structure)
        if not np.any(np.isnan(spacing)) and spacing[0] < 1.5 * spacing_xy and self.R.uniform() < 0.5:
            affine = rotate_affine @ scale_affine
        else:
            affine = scale_affine @ rotate_affine

        # Step 4: Compute load region. We need to load a region large enough that
        # after affine transform, it covers the target patch_size.
        patch_size = trans_info['patch_size']
        load_size = np.ceil(np.abs(affine) @ patch_size).astype(np.int32)
        load_slice = get_random_patch(
            data['shape'], np.minimum(data['shape'], load_size), self.R,
        )

        # Step 5: Load from mmap'd .npy (only reads the crop region)
        img_path = data['img']
        img = np.load(img_path, 'r')
        img = torch.from_numpy(np.array(img[:, *load_slice]))

        # Step 6: Apply affine + random flips via MONAI lazy transforms
        # (computes the full affine chain, applies in a single resampling)
        affine_t = np.eye(4)
        affine_t[:3, :3] = affine
        patch_trans = mt.Compose(
            [
                mt.Affine(affine=affine_t, spatial_size=patch_size, image_only=True),
                *[mt.RandFlip(0.5, i) for i in range(3)],
            ],
            lazy=True,
            overrides={
                'mode': GridSampleMode.BILINEAR,
                'padding_mode': GridSamplePadMode.ZEROS,
                'dtype': torch.float32,
            }
        )
        patch_trans.set_random_state(state=self.R)
        patch = patch_trans(img)
        patch.meta[ImageMetaKey.FILENAME_OR_OBJ] = img_path
        data['img'] = patch
        return data

class PUMITLoaderV2:
    """Deterministic spatial transform: load crop from mmap'd .npy, apply 4x4 affine."""

    def __call__(self, data: dict, affine: np.ndarray, load_slice: tuple[slice, ...]) -> dict:
        data = dict(data)
        patch_size = data['_trans']['patch_size']
        img_path = data['img']
        img = np.load(img_path, 'r')
        img = torch.from_numpy(np.array(img[:, *load_slice]))

        affine_t = np.eye(4)
        affine_t[:3, :3] = affine[:3, :3]
        patch_trans = mt.Affine(
            affine=affine_t,
            spatial_size=patch_size,
            image_only=True,
            mode=GridSampleMode.BILINEAR,
            padding_mode=GridSamplePadMode.ZEROS,
            dtype=torch.float32,
        )
        patch = patch_trans(img)
        patch.meta[ImageMetaKey.FILENAME_OR_OBJ] = img_path
        data['img'] = patch
        return data


def _random_crop_slices(
    shape: np.ndarray, load_size: np.ndarray, rng: np.random.Generator,
) -> tuple[slice, ...]:
    slices = []
    for i in range(3):
        max_start = max(int(shape[i]) - int(load_size[i]), 0)
        start = int(rng.integers(0, max_start + 1))
        slices.append(slice(start, start + int(load_size[i])))
    return tuple(slices)


class RandPUMITLoaderV2:
    """Random spatial transform wrapper: samples params then delegates to PUMITLoaderV2."""

    def __init__(self, rotate_p: float, rotate_axis_p: float):
        self.rotate_p = rotate_p
        self.rotate_axis_p = rotate_axis_p
        self._inner = PUMITLoaderV2()
        self._affine: np.ndarray | None = None
        self._params: dict | None = None

    def _sample_rotation(
        self, spacing: np.ndarray, rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, float | None]:
        if rng.random() >= self.rotate_p:
            return None, None
        if np.any(np.isnan(spacing)):
            return None, None
        spacing_xy = spacing[1:].min()
        if spacing[0] < 3 * spacing_xy and rng.random() < self.rotate_axis_p:
            axis_phi = rng.uniform(7 * np.pi / 15, np.pi / 2)
        else:
            axis_phi = np.pi / 2
        axis_z = np.sin(axis_phi)
        r_xy = np.cos(axis_phi)
        axis_theta = rng.uniform(0, 2 * np.pi)
        axis = np.array((axis_z, r_xy * np.sin(axis_theta), r_xy * np.cos(axis_theta)))
        theta = rng.uniform(0, 2 * np.pi)
        return axis, theta

    def __call__(self, data: dict, rng: np.random.Generator) -> dict:
        data = dict(data)
        trans_info = data['_trans']
        spacing = data['spacing']
        spacing_xy = spacing[1:].min()

        axis, theta = self._sample_rotation(spacing, rng)
        if axis is None:
            rotate_3x3 = np.eye(3)
        else:
            rotate_3x3 = get_rotation_matrix(axis, -theta)

        scale_3x3 = np.diag(trans_info['scale'])

        if not np.any(np.isnan(spacing)) and spacing[0] < 1.5 * spacing_xy and rng.random() < 0.5:
            affine_3x3 = rotate_3x3 @ scale_3x3
        else:
            affine_3x3 = scale_3x3 @ rotate_3x3

        for axis_idx in range(3):
            if rng.random() < 0.5:
                affine_3x3[:, axis_idx] *= -1

        affine = np.eye(4)
        affine[:3, :3] = affine_3x3

        patch_size = np.array(trans_info['patch_size'])
        load_size = np.ceil(np.abs(affine_3x3) @ patch_size).astype(np.int32)
        shape = data['shape']
        load_slice = _random_crop_slices(shape, load_size, rng)

        self._affine = affine
        self._params = {
            'load_slice_start': [s.start for s in load_slice],
            'load_slice_stop': [s.stop for s in load_slice],
        }

        return self._inner(data, affine, load_slice)

    def get_params(self) -> dict:
        assert self._params is not None
        return dict(self._params)

    def get_affine(self) -> np.ndarray:
        assert self._affine is not None
        return self._affine.copy()

    def replay(self, data: dict, params: dict, affine: np.ndarray) -> dict:
        load_slice = tuple(
            slice(s, e) for s, e in zip(params['load_slice_start'], params['load_slice_stop'])
        )
        return self._inner(data, affine, load_slice)


if __name__ == '__main__':
    R = get_rotation_matrix(np.array((1, 0, 0)), np.pi / 3)
    print(R)
