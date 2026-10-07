"""Replayable spatial and intensity transforms for UCPT."""

from __future__ import annotations

import math

import numpy as np

from pumit.transforms.augmentation import (
    AdjustContrastTransform,
    GammaCorrectionTransform,
    NormalizeTransform,
    ScaleIntensityTransform,
)
from pumit.transforms.loader import get_rotation_matrix
from pumit.transforms.pipeline import TransformPipeline
from pumit.ucpt.affine import (
    _affine_load_extent,
    _column_norm_spacing,
    _quantize_scale_to_load_extent,
    _realized_output_to_source,
    replay_affine_patch,
)
from pumit.ucpt.fg_coords import place_crop_start
from pumit.ucpt.input import InputNormalizer


class AffinePatchLoader:
    """Sample and replay spatial augmentation for UCPT.

    Handles 2D/3D crop sizing, scale, rotation, SPAD depth adaptation, foreground-centered crops, and affine resampling.
    """

    def __init__(
        self,
        *,
        size_xy_choices: list[int],
        size_xy_choices_2d: list[int],
        max_depth_per_da: dict[int, int],
        max_da: int = 4,
        rotate_prob: float = 0.3,
        rotate_axis_prob: float = 0.2,
        scale_z: tuple[float, float] = (3 / 4, 4 / 3),
        scale_z_p: float = 0.25,
        scale_xy: tuple[float, float] = (3 / 4, 4 / 3),
        scale_xy_p: float = 0.5,
        labeled_size_xy_max: int | None = None,
        labeled_size_xy_max_2d: int | None = None,
        fg_cache=None,
        force_fraction: float = 0.5,
        input_normalizer: InputNormalizer | None = None,
    ):
        """
        Args:
            size_xy_choices: Candidate in-plane sizes for 3D samples.
            size_xy_choices_2d: Candidate in-plane sizes for 2D samples.
            max_depth_per_da: Maximum crop depth by SPAD adaptation level.
            max_da: Maximum SPAD adaptation level.
            rotate_prob: Rotation probability for 3D and labeled 2D draws.
            rotate_axis_prob: Probability of sampling a non-axial rotation axis.
            scale_z: Depth-scale range.
            scale_z_p: Probability of sampling depth scale.
            scale_xy: In-plane scale range.
            scale_xy_p: Probability of sampling in-plane scale.
            labeled_size_xy_max: Optional 3D in-plane cap for labeled samples.
            labeled_size_xy_max_2d: Optional 2D in-plane cap for labeled samples.
            fg_cache: Optional foreground-coordinate cache.
            force_fraction: Probability of centering a labeled crop on foreground.
            input_normalizer: Optional source-metadata routing for the new UCPT input recipe.
        """
        # 2D images (X-ray, pathology, fundus) are often much larger in-plane than
        # 3D volumes, so they get their own (typically larger) size choices.
        self.size_xy_choices = size_xy_choices
        self.size_xy_choices_2d = size_xy_choices_2d
        self.max_depth_per_da = max_depth_per_da
        self.max_da = max_da
        self.rotate_prob = rotate_prob
        self.rotate_axis_prob = rotate_axis_prob
        self.scale_z = scale_z
        self.scale_z_p = scale_z_p
        self.scale_xy = scale_xy
        self.scale_xy_p = scale_xy_p
        # Labeled draws may use a smaller crop ceiling to limit segmentation compute and memory.
        self.labeled_size_xy_max = labeled_size_xy_max
        self.labeled_size_xy_max_2d = labeled_size_xy_max_2d
        self.fg_cache = fg_cache
        self.force_fraction = force_fraction
        self.foreground_jitter_fraction = 1 / 4
        self.input_normalizer = input_normalizer

    def _fg_start(self, state, crop_size, load_size, output_to_source, rng):
        """Sample the start of a foreground-centered crop.

        Args:
            state: Sample metadata.
            crop_size: Final model-visible crop size.
            load_size: Requested source crop size.
            output_to_source: Realized affine map from final crop to source coordinates.
            rng: Sampling generator.

        Returns:
            Crop start, or ``None`` to use class-agnostic random placement.
        """
        if self.fg_cache is None or not state.get('labeled_draw', False):
            return None
        center = state.get('_center_class')
        if center is None:
            return None
        if rng.random() >= self.force_fraction:
            return None
        source, cls = center
        v = self.fg_cache.sample_voxel(state['dataset'], state['key'], source, cls, rng)
        shape = np.asarray(state['shape'])
        return place_crop_start(
            v,
            np.asarray(load_size),
            shape,
            np.asarray(crop_size),
            output_to_source,
            rng,
            jitter_fraction=self.foreground_jitter_fraction,
        )

    def _select_size_xy(self, state: dict, shape: np.ndarray) -> int:
        is_2d = shape[0] == 1
        choices = self.size_xy_choices_2d if is_2d else self.size_xy_choices
        if state.get('labeled_draw', False):
            cap = self.labeled_size_xy_max_2d if is_2d else self.labeled_size_xy_max
            if cap is not None:
                choices = [size for size in choices if size <= cap]
                assert choices, f'labeled size cap {cap} excludes all size_xy choices'
        eligible = [size for size in choices if size <= int(shape[1:].min())]
        return max(eligible) if eligible else min(choices)

    def _sample_scale_xy(self, shape: np.ndarray, size_xy: int, rng) -> float:
        origin_size_xy = shape[1:].min().item()
        scale_xy_hi = min(origin_size_xy / size_xy, self.scale_xy[1])
        if rng.random() < self.scale_xy_p and scale_xy_hi > self.scale_xy[0]:
            return float(rng.uniform(self.scale_xy[0], scale_xy_hi))
        return 1.0

    def _sample_load_slices(
        self,
        shape: np.ndarray,
        load_size: np.ndarray,
        rng,
        *,
        random_axes: tuple[int, ...],
    ) -> tuple[list[int], list[int]]:
        starts = [0, 0, 0]
        stops = [0, 0, 0]
        for axis in range(3):
            if axis not in random_axes:
                stops[axis] = int(load_size[axis])
                continue
            max_start = max(int(shape[axis]) - int(load_size[axis]), 0)
            start = int(rng.integers(0, max_start + 1))
            starts[axis] = start
            stops[axis] = start + int(load_size[axis])
        return starts, stops

    def _sample_load_region(
        self,
        state: dict,
        crop_size: list[int],
        load_size: np.ndarray,
        output_to_source: np.ndarray,
        rng,
        *,
        random_axes: tuple[int, ...],
    ) -> tuple[list[int], list[int], bool]:
        fg_start = self._fg_start(state, crop_size, load_size, output_to_source, rng)
        if fg_start is not None:
            starts = [int(value) for value in fg_start]
            stops = [int(fg_start[i] + load_size[i]) for i in range(3)]
            return starts, stops, True
        starts, stops = self._sample_load_slices(shape=np.asarray(state['shape']), load_size=load_size, rng=rng, random_axes=random_axes)
        return starts, stops, False

    def _pack_params(
        self,
        *,
        da_enc: int | None,
        crop_size: list[int],
        affine_3x3: np.ndarray,
        load_slice_start: list[int],
        load_slice_stop: list[int],
        n_patches: int,
        spacing: np.ndarray,
        output_to_source: np.ndarray | None = None,
        foreground_forced: bool,
    ) -> dict:
        if output_to_source is None:
            output_to_source = affine_3x3
        affine = np.eye(4)
        affine[:3, :3] = affine_3x3
        return {
            'da_enc': da_enc,
            'crop_size': crop_size,
            'affine': affine.ravel().tolist(),
            'load_slice_start': load_slice_start,
            'load_slice_stop': load_slice_stop,
            'n_patches': n_patches,
            'spacing_label': _column_norm_spacing(output_to_source, spacing),
            'foreground_forced': foreground_forced,
        }

    def _sample_2d_params(
        self, state: dict, shape: np.ndarray, spacing: np.ndarray, size_xy: int, scale_xy: float, rng
    ) -> dict:
        crop_size = [1, size_xy, size_xy]
        n_patches = (size_xy // 16) * (size_xy // 16)
        needed_xy = int(math.ceil(size_xy * scale_xy))
        load_size = np.array([1, min(needed_xy, int(shape[1])), min(needed_xy, int(shape[2]))])
        initial_scale = load_size / np.asarray(crop_size, dtype=np.float64)
        affine_3x3 = np.diag(initial_scale)
        output_to_source = affine_3x3
        if state.get('labeled_draw', False):
            # Keep existing unlabeled geometry; new labeled views share one image/mask affine.
            if rng.random() < self.rotate_prob:
                theta = float(rng.uniform(0, 2 * np.pi))
                cos, sin = np.cos(theta), np.sin(theta)
                affine_3x3[1:, 1:] = affine_3x3[1:, 1:] @ np.array([
                    [cos, sin],
                    [-sin, cos],
                ])
            for axis in (1, 2):
                if rng.random() < 0.5:
                    affine_3x3[:, axis] *= -1
            output_to_source = _realized_output_to_source(affine_3x3, crop_size)
            load_size = np.minimum(_affine_load_extent(affine_3x3, crop_size), shape)
        starts, stops, foreground_forced = self._sample_load_region(
            state, crop_size, load_size, output_to_source, rng, random_axes=(1, 2)
        )
        return self._pack_params(
            da_enc=None,
            crop_size=crop_size,
            affine_3x3=affine_3x3,
            load_slice_start=starts,
            load_slice_stop=stops,
            n_patches=n_patches,
            spacing=spacing,
            output_to_source=output_to_source,
            foreground_forced=foreground_forced,
        )

    def _sample_scale_z(self, spacing_z: float, spacing_xy: float, rng) -> float:
        if spacing_z < 3 * spacing_xy and rng.random() < self.scale_z_p:
            return float(rng.uniform(*self.scale_z))
        return 1.0

    def _sample_rotation(
        self, spacing: np.ndarray, spacing_z: float, spacing_xy: float, rng
    ) -> np.ndarray:
        rotate_3x3 = np.eye(3)
        if rng.random() < self.rotate_prob and not np.any(np.isnan(spacing)):
            if spacing_z < 3 * spacing_xy and rng.random() < self.rotate_axis_prob:
                axis_phi = float(rng.uniform(7 * np.pi / 15, np.pi / 2))
                axis_z = np.sin(axis_phi)
                r_xy = np.cos(axis_phi)
                axis_theta = float(rng.uniform(0, 2 * np.pi))
                axis = (axis_z, r_xy * np.sin(axis_theta), r_xy * np.cos(axis_theta))
                theta = float(rng.uniform(0, 2 * np.pi))
                return get_rotation_matrix(axis, -theta)

            theta = float(rng.uniform(0, 2 * np.pi))
            cos = np.cos(theta)
            sin = np.sin(theta)
            rotate_3x3 = np.array((
                (1.0, 0.0, 0.0),
                (0.0, cos, sin),
                (0.0, -sin, cos),
            ))
        return rotate_3x3

    def _sample_3d_params(
        self,
        state: dict,
        shape: np.ndarray,
        spacing: np.ndarray,
        spacing_z: float,
        spacing_xy: float,
        size_xy: int,
        scale_xy: float,
        rng,
    ) -> dict:
        scale_z = self._sample_scale_z(spacing_z, spacing_xy, rng)
        ratio = float(np.clip(spacing_z * scale_z / (spacing_xy * scale_xy), 1, 1 << self.max_da))
        da_enc = min(int(math.log2(ratio)), self.max_da)
        tier_key = min(da_enc, self.max_da)
        patch_d = 16 >> min(da_enc, self.max_da)
        size_z = int(math.ceil(min(int(shape[0]), self.max_depth_per_da[tier_key]) / patch_d) * patch_d)
        crop_size = [size_z, size_xy, size_xy]
        n_patches = (size_z // patch_d) * (size_xy // 16) * (size_xy // 16)

        rotation = self._sample_rotation(spacing, spacing_z, spacing_xy, rng)
        scale_3x3 = np.diag(_quantize_scale_to_load_extent(
            [scale_z, scale_xy, scale_xy], crop_size
        ))
        if not np.any(np.isnan(spacing)) and spacing_z < 1.5 * spacing_xy and rng.random() < 0.5:
            affine_3x3 = rotation @ scale_3x3
        else:
            affine_3x3 = scale_3x3 @ rotation

        for axis in range(3):
            if rng.random() < 0.5:
                affine_3x3[:, axis] *= -1

        output_to_source = _realized_output_to_source(affine_3x3, crop_size)
        load_size = np.minimum(_affine_load_extent(affine_3x3, crop_size), shape)
        starts, stops, foreground_forced = self._sample_load_region(
            state, crop_size, load_size, output_to_source, rng, random_axes=(0, 1, 2)
        )
        return self._pack_params(
            da_enc=da_enc,
            crop_size=crop_size,
            affine_3x3=affine_3x3,
            load_slice_start=starts,
            load_slice_stop=stops,
            n_patches=n_patches,
            spacing=spacing,
            output_to_source=output_to_source,
            foreground_forced=foreground_forced,
        )

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict | None:
        shape = np.asarray(state['shape'])
        spacing = np.asarray(state['spacing'], dtype=np.float64)
        spacing_z = spacing[0]
        spacing_xy = spacing[1:].min()
        size_xy = self._select_size_xy(state, shape)
        scale_xy = self._sample_scale_xy(shape, size_xy, rng)
        if shape[0] == 1:
            return self._sample_2d_params(state, shape, spacing, size_xy, scale_xy, rng)
        return self._sample_3d_params(
            state, shape, spacing, spacing_z, spacing_xy, size_xy, scale_xy, rng
        )

    def __call__(
        self,
        data: dict,
        *,
        da_enc: int,
        affine: list[float],
        load_slice_start: list[int],
        load_slice_stop: list[int],
        n_patches: int,
        spacing_label: list[float],
        foreground_forced: bool = False,
        patch_size: list[int] | None = None,
        crop_size: list[int] | None = None,
    ) -> dict:
        if self.input_normalizer is not None:
            data = {**data, 'input_scheme': self.input_normalizer.scheme(data['img'])}
        return replay_affine_patch(
            data,
            da_enc=da_enc,
            affine=affine,
            load_slice_start=load_slice_start,
            load_slice_stop=load_slice_stop,
            n_patches=n_patches,
            spacing_label=spacing_label,
            patch_size=patch_size,
            crop_size=crop_size,
        )


class _MedicalScale(ScaleIntensityTransform):
    def __init__(self, normalizer: InputNormalizer):
        super().__init__(prob=0.1, factor_range=(-0.2, 0.2), channel_wise=True)
        self.normalizer = normalizer

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict:
        if self.normalizer.scheme(state['img']) == 'rgb':
            return {'enabled': False}
        return super().sample_params(state, rng)

    def __call__(self, data: dict, *, enabled: bool, factors: list[float] | None = None) -> dict:
        if not enabled or data['input_scheme'] == 'rgb':
            return data
        return {**data, 'img': self._inner(data['img'].float(), factors)}


class _DisabledContrast:
    """Retain the contrast parameter slot without applying independent contrast."""

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict:
        return {'enabled': False}

    def __call__(self, data: dict, *, enabled: bool, factor: float | None = None) -> dict:
        return data


class _MedicalGamma(GammaCorrectionTransform):
    def __init__(self, normalizer: InputNormalizer):
        super().__init__(prob=0.15, gamma_range=(0.8, 1.2), prob_invert=0.0)
        self.normalizer = normalizer

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict:
        if self.normalizer.scheme(state['img']) == 'rgb':
            return {'enabled': False}
        return super().sample_params(state, rng)

    def __call__(
        self, data: dict, *, enabled: bool, gammas: list[float] | None = None, invert: bool = False,
    ) -> dict:
        if not enabled or data['input_scheme'] == 'rgb':
            return data
        gamma = gammas[0] if len(gammas) == 1 else gammas
        image = self._inner(data['img'].float(), gamma, invert, True)
        return {**data, 'img': image}


def build_ucpt_pipeline(
    *,
    size_xy_choices: list[int],
    size_xy_choices_2d: list[int],
    max_depth_per_da: dict[int, int],
    max_da: int = 4,
    scale_xy: tuple[float, float] = (3 / 4, 4 / 3),
    labeled_size_xy_max: int | None = None,
    labeled_size_xy_max_2d: int | None = None,
    fg_cache=None,
    force_fraction: float = 0.5,
    scale_intensity_p: float = 0.1,
    scale_intensity: float = 0.2,
    adjust_contrast_prob: float = 0.1,
    adjust_contrast_range: tuple[float, float] = (0.75, 1.25),
    gamma_prob: float = 0.15,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    gamma_invert_prob: float = 0.15,
    input_normalizer: InputNormalizer | None = None,
) -> TransformPipeline:
    """Build the UCPT spatial and intensity augmentation pipeline.

    Args:
        size_xy_choices: Candidate in-plane sizes for 3D samples.
        size_xy_choices_2d: Candidate in-plane sizes for 2D samples.
        max_depth_per_da: Maximum crop depth by SPAD adaptation level.
        max_da: Maximum SPAD adaptation level.
        scale_xy: In-plane scale range.
        labeled_size_xy_max: Optional 3D in-plane cap for labeled samples.
        labeled_size_xy_max_2d: Optional 2D in-plane cap for labeled samples.
        fg_cache: Optional foreground-coordinate cache.
        force_fraction: Probability of centering labeled crops on foreground.
        scale_intensity_p: Probability of intensity scaling.
        scale_intensity: Maximum intensity-scale delta.
        adjust_contrast_prob: Probability of contrast adjustment.
        adjust_contrast_range: Contrast-factor range.
        gamma_prob: Probability of gamma correction.
        gamma_range: Gamma range.
        gamma_invert_prob: Probability of inverted gamma correction.
        input_normalizer: Enable the fixed medical/display input recipe; omitted for legacy codec replay.

    Returns:
        Replayable transform pipeline.
    """
    spatial = AffinePatchLoader(
        size_xy_choices=size_xy_choices,
        size_xy_choices_2d=size_xy_choices_2d,
        max_depth_per_da=max_depth_per_da,
        max_da=max_da,
        scale_xy=scale_xy,
        labeled_size_xy_max=labeled_size_xy_max,
        labeled_size_xy_max_2d=labeled_size_xy_max_2d,
        fg_cache=fg_cache,
        force_fraction=force_fraction,
        input_normalizer=input_normalizer,
    )
    if input_normalizer is not None:
        return TransformPipeline([
            spatial,
            _MedicalScale(input_normalizer),
            _DisabledContrast(),
            _MedicalGamma(input_normalizer),
            input_normalizer,
        ])
    return TransformPipeline([
        spatial,
        ScaleIntensityTransform(
            prob=scale_intensity_p,
            factor_range=(-scale_intensity, scale_intensity),
            channel_wise=True,
        ),
        AdjustContrastTransform(
            prob=adjust_contrast_prob,
            contrast_range=adjust_contrast_range,
        ),
        GammaCorrectionTransform(
            prob=gamma_prob,
            gamma_range=gamma_range,
            prob_invert=gamma_invert_prob,
        ),
        NormalizeTransform(),
    ])
