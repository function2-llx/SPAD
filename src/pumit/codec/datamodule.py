"""Codec-specific data pipeline: TransformConf defaults and transform builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch.types import Device
from monai import transforms as mt

from pumit.transforms import InputTransformD, PUMITLoader
from pumit.transforms import augmentation as lt
from pumit.data.trans_info import TransformConf as _TransformConf
from pumit.types import tuple2_t


@dataclass(kw_only=True)
class TransformConf(_TransformConf):
    """TransformConf with codec defaults (all fields have values)."""

    depth_schedule: tuple[int, ...] = (64, 48, 32, 24, 16)
    size_xy: int = 256
    rotate: _TransformConf.Rotate = field(default_factory=lambda: _TransformConf.Rotate(prob=0.3, axis_prob=0.2))
    scale_z: tuple2_t[float] = (3 / 4, 4 / 3)
    scale_z_p: float = 0.25
    scale_xy: tuple2_t[float] = (0.75, 2)
    scale_xy_p: float = 0.5
    scale_intensity: float = 0.2
    scale_intensity_p: float = 0.1
    adjust_contrast: _TransformConf.AdjustContrast = field(
        default_factory=lambda: _TransformConf.AdjustContrast(prob=0.1, range=(0.75, 1.25), preserve_intensity_range=True)
    )
    gamma_correction: _TransformConf.GammaCorrection = field(
        default_factory=lambda: _TransformConf.GammaCorrection(prob=0.15, range=(0.7, 1.5), prob_invert=0.15)
    )
    noise_p: float = 0.0
    noise_std: tuple2_t[float] = (0.0, 0.0)
    blur_p: float = 0.0
    blur_sigma: tuple2_t[float] = (0.0, 0.0)


def build_train_transform(conf: TransformConf, device: Device = 'cpu') -> Callable:
    """Build the codec training transform pipeline (spatial + intensity augmentation)."""
    return mt.Compose(
        [
            PUMITLoader(conf.rotate.prob, conf.rotate.axis_prob, device),
            mt.RandScaleIntensityD('img', conf.scale_intensity, prob=conf.scale_intensity_p, channel_wise=True),
            lt.RandDictWrapper(
                'img',
                lt.RandAdjustContrast(
                    conf.adjust_contrast.prob,
                    conf.adjust_contrast.range,
                    conf.adjust_contrast.preserve_intensity_range,
                ),
            ),
            lt.ClampIntensityD('img'),
            lt.RandDictWrapper(
                'img',
                lt.RandGammaCorrection(
                    conf.gamma_correction.prob,
                    conf.gamma_correction.range,
                    conf.gamma_correction.prob_invert,
                    False,
                ),
            ),
            InputTransformD(),
        ],
    )
