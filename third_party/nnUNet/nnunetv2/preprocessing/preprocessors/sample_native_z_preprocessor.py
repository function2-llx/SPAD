from __future__ import annotations

from copy import deepcopy

from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager


SAMPLE_NATIVE_Z_MINIMUM_SPACING_MM = 1.0
SAMPLE_NATIVE_Z_INPLANE_SPACING_MM = 1.0
SPACING_AFTER_RESAMPLING_KEY = 'spacing_after_resampling'
SHAPE_AFTER_RESAMPLING_KEY = 'shape_after_resampling'


def sample_native_z_target_spacing(
    transposed_original_spacing,
) -> tuple[float, float, float]:
    """Return the per-case Sample-Native-Z target spacing in (D, H, W)."""
    spacing = tuple(float(value) for value in transposed_original_spacing)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f'expected positive 3D spacing, got {spacing}')
    return (
        max(spacing[0], SAMPLE_NATIVE_Z_MINIMUM_SPACING_MM),
        SAMPLE_NATIVE_Z_INPLANE_SPACING_MM,
        SAMPLE_NATIVE_Z_INPLANE_SPACING_MM,
    )


def sample_native_z_planned_xy_target_spacing(
    transposed_original_spacing,
    planned_spacing,
) -> tuple[float, float, float]:
    """Combine case-native z with a plan's dataset-specific XY spacing."""
    original = tuple(float(value) for value in transposed_original_spacing)
    planned = tuple(float(value) for value in planned_spacing)
    if len(original) != 3 or any(value <= 0 for value in original):
        raise ValueError(f'expected positive original 3D spacing, got {original}')
    if len(planned) != 3 or any(value <= 0 for value in planned):
        raise ValueError(f'expected positive planned 3D spacing, got {planned}')
    return (
        max(original[0], SAMPLE_NATIVE_Z_MINIMUM_SPACING_MM),
        planned[1],
        planned[2],
    )


def sample_native_z_fixed_xy_target_spacing(
    transposed_original_spacing,
    configured_spacing,
) -> tuple[float, float, float]:
    """Keep native z down to the fixed in-plane spacing configured by the plan."""
    original = tuple(float(value) for value in transposed_original_spacing)
    configured = tuple(float(value) for value in configured_spacing)
    if len(original) != 3 or any(value <= 0 for value in original):
        raise ValueError(f'expected positive original 3D spacing, got {original}')
    if len(configured) != 3 or any(value <= 0 for value in configured):
        raise ValueError(
            f'expected positive configured 3D spacing, got {configured}'
        )
    if configured[1] != configured[2]:
        raise ValueError(
            f'fixed-XY preprocessing requires equal in-plane spacing, '
            f'got {configured[1:]}'
        )
    return (
        max(original[0], configured[1]),
        configured[1],
        configured[2],
    )


class SampleNativeZPreprocessor(DefaultPreprocessor):
    """Preprocess each case on its own Sample-Native-Z grid."""

    def run_case_npy(
        self,
        data,
        seg,
        properties,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        transposed_original_spacing = tuple(
            properties['spacing'][axis]
            for axis in plans_manager.transpose_forward
        )
        target_spacing = sample_native_z_target_spacing(
            transposed_original_spacing
        )
        case_configuration = deepcopy(configuration_manager.configuration)
        case_configuration['spacing'] = list(target_spacing)
        case_configuration_manager = ConfigurationManager(case_configuration)

        data, seg, properties = super().run_case_npy(
            data,
            seg,
            properties,
            plans_manager,
            case_configuration_manager,
            dataset_json,
        )
        properties[SPACING_AFTER_RESAMPLING_KEY] = list(target_spacing)
        return data, seg, properties


class _SampleNativeZConfiguredXYPreprocessor(DefaultPreprocessor):
    """Shared execution path for configured-XY Sample-Native-Z grids."""

    def _target_spacing(
        self,
        transposed_original_spacing,
        configured_spacing,
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def run_case_npy(
        self,
        data,
        seg,
        properties,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        transposed_original_spacing = tuple(
            properties['spacing'][axis]
            for axis in plans_manager.transpose_forward
        )
        target_spacing = self._target_spacing(
            transposed_original_spacing,
            configuration_manager.spacing,
        )
        case_configuration = deepcopy(configuration_manager.configuration)
        case_configuration['spacing'] = list(target_spacing)
        case_configuration_manager = ConfigurationManager(case_configuration)

        data, seg, properties = super().run_case_npy(
            data,
            seg,
            properties,
            plans_manager,
            case_configuration_manager,
            dataset_json,
        )
        properties[SPACING_AFTER_RESAMPLING_KEY] = list(target_spacing)
        properties[SHAPE_AFTER_RESAMPLING_KEY] = list(data.shape[1:])
        return data, seg, properties


class SampleNativeZPlannedXYPreprocessor(
    _SampleNativeZConfiguredXYPreprocessor
):
    """Keep case-native z while using the plan's dataset-specific XY spacing."""

    def _target_spacing(
        self,
        transposed_original_spacing,
        configured_spacing,
    ) -> tuple[float, float, float]:
        return sample_native_z_planned_xy_target_spacing(
            transposed_original_spacing,
            configured_spacing,
        )


class SampleNativeZFixedXYPreprocessor(
    _SampleNativeZConfiguredXYPreprocessor
):
    """Keep case-native z down to the plan's fixed isotropic XY spacing."""

    def _target_spacing(
        self,
        transposed_original_spacing,
        configured_spacing,
    ) -> tuple[float, float, float]:
        return sample_native_z_fixed_xy_target_spacing(
            transposed_original_spacing,
            configured_spacing,
        )
