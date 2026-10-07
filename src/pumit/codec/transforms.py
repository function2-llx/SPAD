"""Codec transforms following the sample_params/apply protocol.

Each transform exposes:
  - sample_params(state, rng) -> dict | None
  - apply(data, **kwargs) -> dict
"""

from __future__ import annotations

import math

import numpy as np
import torch
from monai import transforms as mt
from monai.utils import GridSampleMode, GridSamplePadMode

from pumit.data.config import DepthTierConfig
from pumit.transforms.augmentation import (
    AdjustContrastTransform,
    GammaCorrectionTransform,
    NormalizeTransform,
    ScaleIntensityTransform,
)
from pumit.transforms.loader import get_rotation_matrix
from pumit.transforms.pipeline import TransformPipeline

from .datamodule import TransformConf


# ---------------------------------------------------------------------------
# SpatialTransform
# ---------------------------------------------------------------------------


class SpatialTransform:
    """Spatial augmentation: scale, rotation, DA decomposition, crop, and affine resampling.

    Absorbs the logic from gen_trans_info + RandPUMITLoaderV2 into a single
    sample_params/apply pair.
    """

    def __init__(
        self,
        *,
        conf: TransformConf,
        depth_tiers: dict[int | None, DepthTierConfig],
        max_da: int = 4,
        smooth_spad: bool = True,
    ):
        self.conf = conf
        self.depth_tiers = depth_tiers
        self.max_da = max_da
        self.smooth_spad = smooth_spad

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict | None:
        shape = np.asarray(state['shape'])
        spacing = np.asarray(state['spacing'], dtype=np.float64)
        spacing_z = spacing[0]
        spacing_xy = spacing[1:].min()
        size_xy = self.conf.size_xy

        # Scale XY
        origin_size_xy = shape[1:].min().item()
        scale_xy_hi = min(origin_size_xy / size_xy, self.conf.scale_xy[1])
        if rng.random() < self.conf.scale_xy_p and scale_xy_hi > self.conf.scale_xy[0]:
            scale_xy = float(rng.uniform(self.conf.scale_xy[0], scale_xy_hi))
        else:
            scale_xy = 1.0

        # --- 2D early return ---
        if shape[0] == 1:
            patch_size = [1, size_xy, size_xy]
            affine = np.eye(4)
            # Random crop in XY plane
            load_slice_start = [0, 0, 0]
            load_slice_stop = [1, 0, 0]
            for axis in range(1, 3):
                needed = int(math.ceil(size_xy * scale_xy))
                max_start = max(int(shape[axis]) - needed, 0)
                start = int(rng.integers(0, max_start + 1))
                load_slice_start[axis] = start
                load_slice_stop[axis] = start + needed
            return {
                'da_enc': None,
                'da_dec': None,
                'patch_size': patch_size,
                'affine': affine.ravel().tolist(),
                'load_slice_start': load_slice_start,
                'load_slice_stop': load_slice_stop,
            }

        # --- 3D path ---
        # Scale Z (only if spacing_z < 3*spacing_xy)
        if spacing_z < 3 * spacing_xy and rng.random() < self.conf.scale_z_p:
            scale_z = float(rng.uniform(*self.conf.scale_z))
        else:
            scale_z = 1.0

        # DA decomposition (stochastic)
        ratio = float(np.clip(spacing_z * scale_z / (spacing_xy * scale_xy), 1, 1 << self.max_da))
        log_r = math.log2(ratio)
        floor_da = min(int(log_r), self.max_da)
        t = log_r - floor_da

        if self.smooth_spad and t > 0 and floor_da < self.max_da:
            da_enc = floor_da + 1 if rng.random() < t else floor_da
            da_dec = floor_da + 1 if rng.random() < t else floor_da
        else:
            da_enc = floor_da
            da_dec = floor_da

        # Depth tier routing
        tier_key = min(da_enc, self.max_da)
        tier_cfg = self.depth_tiers[tier_key]
        raw_depth = int(shape[0])

        # Drop check: too-thin 3D volumes
        if raw_depth < tier_cfg.tiers[0] // 2:
            return None

        # Find smallest tier >= raw_depth; overflow -> largest tier (random crop)
        size_z = tier_cfg.tiers[-1]
        for tier_depth in tier_cfg.tiers:
            if tier_depth >= raw_depth:
                size_z = tier_depth
                break

        patch_size = [size_z, size_xy, size_xy]
        scale = [scale_z, scale_xy, scale_xy]

        # Rotation
        rotate_3x3 = np.eye(3)
        if rng.random() < self.conf.rotate.prob and not np.any(np.isnan(spacing)):
            if spacing_z < 3 * spacing_xy and rng.random() < self.conf.rotate.axis_prob:
                axis_phi = float(rng.uniform(7 * np.pi / 15, np.pi / 2))
            else:
                axis_phi = np.pi / 2
            axis_z = np.sin(axis_phi)
            r_xy = np.cos(axis_phi)
            axis_theta = float(rng.uniform(0, 2 * np.pi))
            axis = (axis_z, r_xy * np.sin(axis_theta), r_xy * np.cos(axis_theta))
            theta = float(rng.uniform(0, 2 * np.pi))
            rotate_3x3 = get_rotation_matrix(axis, -theta)

        scale_3x3 = np.diag(scale)

        # Affine composition order
        if not np.any(np.isnan(spacing)) and spacing_z < 1.5 * spacing_xy and rng.random() < 0.5:
            affine_3x3 = rotate_3x3 @ scale_3x3
        else:
            affine_3x3 = scale_3x3 @ rotate_3x3

        # 3 random flips
        for i in range(3):
            if rng.random() < 0.5:
                affine_3x3[:, i] *= -1

        affine = np.eye(4)
        affine[:3, :3] = affine_3x3

        # Load region
        load_size = np.ceil(np.abs(affine_3x3) @ np.array(patch_size)).astype(np.int32)
        # Clamp load_size to shape
        load_size = np.minimum(load_size, shape)
        load_slice_start = []
        load_slice_stop = []
        for i in range(3):
            max_start = max(int(shape[i]) - int(load_size[i]), 0)
            start = int(rng.integers(0, max_start + 1))
            load_slice_start.append(start)
            load_slice_stop.append(start + int(load_size[i]))

        return {
            'da_enc': da_enc,
            'da_dec': da_dec,
            'patch_size': patch_size,
            'affine': affine.ravel().tolist(),
            'load_slice_start': load_slice_start,
            'load_slice_stop': load_slice_stop,
        }

    def __call__(
        self,
        data: dict,
        *,
        da_enc: int | None,
        da_dec: int | None,
        patch_size: list[int],
        affine: list[float],
        load_slice_start: list[int],
        load_slice_stop: list[int],
    ) -> dict:
        data = dict(data)
        img_path = data['img']
        img = np.load(img_path, 'r')
        z0, y0, x0 = load_slice_start
        z1, y1, x1 = load_slice_stop
        crop = img[:, z0:z1, y0:y1, x0:x1]
        crop = torch.from_numpy(np.array(crop))

        affine_4x4 = np.array(affine, dtype=np.float64).reshape(4, 4)
        result = mt.Affine(
            affine=affine_4x4,
            spatial_size=patch_size,
            image_only=True,
            mode=GridSampleMode.BILINEAR,
            padding_mode=GridSamplePadMode.ZEROS,
            dtype=torch.float32,
        )(crop)

        # Unwrap MetaTensor to plain tensor
        if hasattr(result, 'as_tensor'):
            result = result.as_tensor()

        data['img'] = result
        data['da_enc'] = da_enc
        data['da_dec'] = da_dec
        return data


# ---------------------------------------------------------------------------
# build_codec_pipeline
# ---------------------------------------------------------------------------


def build_codec_pipeline(
    conf: TransformConf | None = None,
    *,
    depth_tiers: dict[int | None, DepthTierConfig] | None = None,
    max_da: int = 4,
    smooth_spad: bool = True,
) -> TransformPipeline:
    """Build the codec transform pipeline with all 5 transforms in order."""
    if conf is None:
        conf = TransformConf()
    if depth_tiers is None:
        raise ValueError("depth_tiers is required")

    return TransformPipeline([
        SpatialTransform(
            conf=conf,
            depth_tiers=depth_tiers,
            max_da=max_da,
            smooth_spad=smooth_spad,
        ),
        ScaleIntensityTransform(
            prob=conf.scale_intensity_p,
            factor_range=(-conf.scale_intensity, conf.scale_intensity),
            channel_wise=True,
        ),
        AdjustContrastTransform(
            prob=conf.adjust_contrast.prob,
            contrast_range=conf.adjust_contrast.range,
            preserve_range=conf.adjust_contrast.preserve_intensity_range,
        ),
        GammaCorrectionTransform(
            prob=conf.gamma_correction.prob,
            gamma_range=conf.gamma_correction.range,
            prob_invert=conf.gamma_correction.prob_invert,
        ),
        NormalizeTransform(),
    ])
