"""Transform info generation: DA computation, spatial crop sizing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from monai import transforms as mt

from pumit.transforms import TransInfo
from pumit.types import tuple2_t

from .config import DepthTierConfig


@dataclass(kw_only=True)
class TransformConf:
    # spatial
    depth_schedule: tuple[int, ...]
    size_xy: int

    @dataclass
    class Rotate:
        prob: float
        axis_prob: float
    rotate: Rotate

    scale_z: tuple2_t[float]
    scale_z_p: float
    scale_xy: tuple2_t[float]
    scale_xy_p: float
    # intensity
    scale_intensity: float
    scale_intensity_p: float

    @dataclass
    class AdjustContrast:
        prob: float
        range: tuple2_t[float]
        preserve_intensity_range: bool
    adjust_contrast: AdjustContrast

    @dataclass
    class GammaCorrection:
        prob: float
        range: tuple2_t[float]
        prob_invert: float
    gamma_correction: GammaCorrection

    # Gaussian noise
    noise_p: float
    noise_std: tuple2_t[float]

    # Gaussian blur
    blur_p: float
    blur_sigma: tuple2_t[float]


def gen_trans_info(
    data: dict,
    trans_conf: TransformConf,
    R: np.random.RandomState,
    *,
    max_da: int,
    smooth_spad: bool = True,
    depth_tiers: dict[int, DepthTierConfig] | None = None,
) -> TransInfo | None:
    shape = np.array(data['shape'])
    origin_size_xy = shape[1:].min().item()
    spacing = data['spacing']
    spacing_z = spacing[0]
    spacing_xy = spacing[1:].min()
    size_xy = trans_conf.size_xy

    # Scale XY augmentation (applied for both 2D and 3D)
    if R.uniform() < trans_conf.scale_xy_p:
        scale_xy = R.uniform(
            trans_conf.scale_xy[0],
            min(origin_size_xy / size_xy, trans_conf.scale_xy[1]),
        )
    else:
        scale_xy = 1.

    # 2D early return
    if shape[0] == 1:
        return {
            'da_enc': None,
            'da_dec': None,
            't': 0.0,
            'scale': (1., scale_xy, scale_xy),
            'patch_size': (1, size_xy, size_xy),
        }

    # 3D path
    if spacing_z < 3 * spacing_xy and R.uniform() < trans_conf.scale_z_p:
        scale_z = R.uniform(*trans_conf.scale_z)
    else:
        scale_z = 1.
    ratio = np.clip(spacing_z * scale_z / (spacing_xy * scale_xy), 1, 1 << max_da)

    # Stochastic DA decomposition
    log_r = float(np.log2(ratio))
    floor_da = min(int(log_r), max_da)
    t = log_r - floor_da  # frac(log2(ratio)), in [0, 1)

    if smooth_spad and t > 0 and floor_da < max_da:
        da_enc = floor_da + 1 if R.uniform() < t else floor_da
        da_dec = floor_da + 1 if R.uniform() < t else floor_da
    else:
        da_enc = floor_da
        da_dec = floor_da

    if depth_tiers is not None:
        # Tier routing: map da_enc to tier key
        tier_key = None if da_enc is None else min(da_enc, max_da)
        tier_cfg = depth_tiers[tier_key]
        raw_depth = int(shape[0])

        # Drop check: too-thin 3D volumes
        if tier_key != -1 and raw_depth < tier_cfg.tiers[0] // 2:
            return None

        # Find smallest tier >= raw_depth; overflow -> largest tier (random crop)
        size_z = tier_cfg.tiers[-1]  # default to largest
        for tier_depth in tier_cfg.tiers:
            if tier_depth >= raw_depth:
                size_z = tier_depth
                break
    else:
        # Legacy depth_schedule logic
        if da_enc is None:
            size_z = 1
        else:
            size_z = trans_conf.depth_schedule[min(da_enc, len(trans_conf.depth_schedule) - 1)]

    return {
        'da_enc': da_enc,
        'da_dec': da_dec,
        't': t,
        'scale': (scale_z, scale_xy, scale_xy),
        'patch_size': (size_z, size_xy, size_xy),
    }


class GenTransInfo(mt.Randomizable, mt.Transform):
    """Compute DA, scale, and patch_size for a sample. Monai Randomizable transform."""

    def __init__(self, trans_conf: TransformConf, *, max_da: int):
        self.trans_conf = trans_conf
        self.max_da = max_da

    def __call__(self, data):
        data = dict(data)
        data['_trans'] = gen_trans_info(data, self.trans_conf, self.R, max_da=self.max_da)
        return data
