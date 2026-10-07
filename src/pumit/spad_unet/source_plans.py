"""Source-plan adaptations for controlled SPAD U-Net experiments."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from nnunetv2.experiment_planning.experiment_planners.network_topology import (
    get_pool_and_conv_props,
)

from pumit.spad_unet.geometry import compute_da_schedule


SEVEN_STAGE_INPLANE_MIN_PATCH = 256
SEVEN_STAGE_INPLANE_DIVISIBILITY = 64
SEVEN_STAGE_INPLANE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans7StageInplane'
)
SEVEN_STAGE_FOV_MINIMUM_MM = 192.0
SEVEN_STAGE_FOV_PATCH_DIVISIBILITY = (16, 64, 64)
SEVEN_STAGE_FOV_PLANS_IDENTIFIER = 'nnUNetResEncUNetLPlans7StageFOV192'
SIX_STAGE_PLANNED_Z_INPLANE1_FOV_MM = 192.0
SIX_STAGE_PLANNED_Z_INPLANE1_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StagePlannedZ1x1FOV192'
)
SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZ1x1FOV192'
)
SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_FOV224_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZ1x1FOV224'
)
SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE0P9_FOV192_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZ0p9x0p9FOV192'
)
SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZPlannedXYMaxB8'
)
SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_P224_MAXB7_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZPlannedXYP224MaxB7'
)
SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_P224_MAXB7_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZ1x1P224MaxB7'
)
SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_INPLANE_PATCH = 256
SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_BOTTLENECK_MIN = 4
SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_BOTTLENECK_MAX = 8
SIX_STAGE_LEGACY_NATIVE_Z1_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageNativeZ1x1FOV192'
)
SIX_STAGE_N_STAGES = 6


def adapt_plan_to_six_stage_planned_z_fixed_xy(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = SIX_STAGE_PLANNED_Z_INPLANE1_PLANS_IDENTIFIER,
    configuration_name: str = '3d_fullres',
    target_fov_mm: float = SIX_STAGE_PLANNED_Z_INPLANE1_FOV_MM,
    inplane_spacing_mm: float = 1.0,
    reference_z_spacing_mm: float | None = None,
    preprocessed_data_identifier: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use dataset-planned z spacing and a fixed-XY six-stage grid.

    Args:
        source_plans: The dataset's nnU-Net-generated source plan.
        dataset_id: Three-digit source dataset identifier.
        source_plans_identifier: Expected source plan identifier.
        derived_plans_identifier: Identifier for the derived plan.
        configuration_name: Configuration to adapt.
        target_fov_mm: Minimum physical FOV on each axis.
        inplane_spacing_mm: Fixed spacing for both in-plane axes.
        reference_z_spacing_mm: Optional z spacing for the plan's reference
            geometry. The dataset-planned z spacing is preserved when omitted.
        preprocessed_data_identifier: Existing compatible array directory to
            reuse. If omitted, the derived plan receives its own data identifier
            and requires preprocessing.
    """
    plans = deepcopy(source_plans)
    if plans.get('plans_name') != source_plans_identifier:
        raise ValueError(
            f'dataset {dataset_id} source plan name '
            f'{plans.get("plans_name")!r} does not match '
            f'{source_plans_identifier!r}'
        )
    if target_fov_mm <= 0:
        raise ValueError('target_fov_mm must be positive')
    if not math.isfinite(inplane_spacing_mm) or inplane_spacing_mm <= 0:
        raise ValueError('inplane_spacing_mm must be finite and positive')
    if reference_z_spacing_mm is not None and (
        not math.isfinite(reference_z_spacing_mm)
        or reference_z_spacing_mm <= 0
    ):
        raise ValueError(
            'reference_z_spacing_mm must be finite and positive or null'
        )
    if preprocessed_data_identifier is not None and (
        not isinstance(preprocessed_data_identifier, str)
        or not preprocessed_data_identifier
    ):
        raise ValueError(
            'preprocessed_data_identifier must be a non-empty string or null'
        )
    if configuration_name not in plans.get('configurations', {}):
        raise KeyError(
            f'dataset {dataset_id} has no {configuration_name!r} configuration'
        )

    configuration = plans['configurations'][configuration_name]
    old_spacing = tuple(float(value) for value in configuration['spacing'])
    old_patch = tuple(int(value) for value in configuration['patch_size'])
    if len(old_spacing) != 3 or len(old_patch) != 3:
        raise ValueError(
            f'dataset {dataset_id} requires 3D spacing and patch, got '
            f'{old_spacing} and {old_patch}'
        )
    if any(value <= 0 for value in (*old_spacing, *old_patch)):
        raise ValueError(
            f'dataset {dataset_id} spacing and patch must be positive, got '
            f'{old_spacing} and {old_patch}'
        )

    new_spacing = (
        old_spacing[0]
        if reference_z_spacing_mm is None
        else reference_z_spacing_mm,
        inplane_spacing_mm,
        inplane_spacing_mm,
    )
    continuous_da = max(
        0.0,
        math.log2(new_spacing[0] / inplane_spacing_mm),
    )
    floor_da = math.floor(continuous_da)
    depth_divisibility = 2 ** max(
        0,
        SIX_STAGE_N_STAGES - 1 - floor_da,
    )
    inplane_patch = _ceil_to_multiple(
        target_fov_mm / inplane_spacing_mm,
        2 ** (SIX_STAGE_N_STAGES - 1),
    )
    new_patch = (
        _ceil_to_multiple(target_fov_mm / new_spacing[0], depth_divisibility),
        inplane_patch,
        inplane_patch,
    )
    (
        _,
        strides,
        kernel_sizes,
        topology_patch,
        _,
    ) = get_pool_and_conv_props(
        new_spacing,
        new_patch,
        4,
        999999,
    )
    topology_patch = tuple(int(value) for value in topology_patch)
    if topology_patch != new_patch:
        raise RuntimeError(
            f'dataset {dataset_id} topology changed requested patch '
            f'{new_patch} to {topology_patch}'
        )
    if len(strides) != SIX_STAGE_N_STAGES:
        raise RuntimeError(
            f'dataset {dataset_id} did not produce six stages: {strides}'
        )

    architecture_kwargs = configuration['architecture']['arch_kwargs']
    features = list(architecture_kwargs['features_per_stage'])
    blocks = list(architecture_kwargs['n_blocks_per_stage'])
    features.extend(
        [features[-1]] * (SIX_STAGE_N_STAGES - len(features))
    )
    blocks.extend([blocks[-1]] * (SIX_STAGE_N_STAGES - len(blocks)))
    architecture_kwargs.update(
        {
            'n_stages': SIX_STAGE_N_STAGES,
            'features_per_stage': features[:SIX_STAGE_N_STAGES],
            'kernel_sizes': [list(values) for values in kernel_sizes],
            'strides': [list(values) for values in strides],
            'n_blocks_per_stage': blocks[:SIX_STAGE_N_STAGES],
            'n_conv_per_stage_decoder': [1]
            * (SIX_STAGE_N_STAGES - 1),
        }
    )
    median_shape = tuple(
        float(value) for value in configuration['median_image_size_in_voxels']
    )
    new_median_shape = tuple(
        median_shape[axis] * old_spacing[axis] / new_spacing[axis] for axis in range(3)
    )
    configuration.update(
        {
            'data_identifier': (
                preprocessed_data_identifier
                or f'{derived_plans_identifier}_{configuration_name}'
            ),
            'batch_size': 4,
            'batch_dice': False,
            'spacing': list(new_spacing),
            'patch_size': list(new_patch),
            'median_image_size_in_voxels': list(new_median_shape),
        }
    )
    plans['plans_name'] = derived_plans_identifier

    old_fov = tuple(old_spacing[axis] * old_patch[axis] for axis in range(3))
    new_fov = tuple(new_spacing[axis] * new_patch[axis] for axis in range(3))
    record = {
        'dataset_id': dataset_id,
        'adapted': old_spacing != new_spacing or old_patch != new_patch,
        'source_plans_identifier': source_plans_identifier,
        'derived_plans_identifier': derived_plans_identifier,
        'configuration': configuration_name,
        'rule': (
            f'dataset_planned_z_fixed_xy_{inplane_spacing_mm:g}mm_'
            f'six_stage_fov{target_fov_mm:g}'
        ),
        'z_spacing_policy': 'nnunet_dataset_target',
        'sample_native_z': False,
        'target_fov_mm': target_fov_mm,
        'planned_z_spacing_unchanged': math.isclose(
            new_spacing[0],
            old_spacing[0],
            rel_tol=0,
            abs_tol=1e-6,
        ),
        'preprocessed_data_reused': preprocessed_data_identifier is not None,
        'preprocessed_data_identifier': configuration['data_identifier'],
        'old_spacing': list(old_spacing),
        'new_spacing': list(new_spacing),
        'old_patch_size': list(old_patch),
        'new_patch_size': list(new_patch),
        'old_fov_mm': list(old_fov),
        'new_fov_mm': list(new_fov),
        'continuous_da': continuous_da,
        'floor_da': floor_da,
        'depth_patch_divisibility': depth_divisibility,
    }
    plans['pumit_spad_unet_source_adaptation'] = record
    return plans, record


def adapt_plan_to_six_stage_planned_z_inplane1(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = SIX_STAGE_PLANNED_Z_INPLANE1_PLANS_IDENTIFIER,
    configuration_name: str = '3d_fullres',
    target_fov_mm: float = SIX_STAGE_PLANNED_Z_INPLANE1_FOV_MM,
    preprocessed_data_identifier: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the legacy fixed-1-mm Planned-Z six-stage plan."""
    return adapt_plan_to_six_stage_planned_z_fixed_xy(
        source_plans,
        dataset_id,
        source_plans_identifier=source_plans_identifier,
        derived_plans_identifier=derived_plans_identifier,
        configuration_name=configuration_name,
        target_fov_mm=target_fov_mm,
        inplane_spacing_mm=1.0,
        preprocessed_data_identifier=preprocessed_data_identifier,
    )


def adapt_plan_to_six_stage_sample_native_z_fixed_xy(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = (
        SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER
    ),
    configuration_name: str = '3d_fullres',
    target_fov_mm: float = SIX_STAGE_PLANNED_Z_INPLANE1_FOV_MM,
    inplane_spacing_mm: float = 1.0,
    preprocessed_data_identifier: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a fixed-XY six-stage plan with case-specific runtime z."""
    if not math.isfinite(target_fov_mm) or target_fov_mm <= 0:
        raise ValueError('target_fov_mm must be finite and positive')
    if not math.isfinite(inplane_spacing_mm) or inplane_spacing_mm <= 0:
        raise ValueError('inplane_spacing_mm must be finite and positive')
    requested_minimum_fov_mm = target_fov_mm
    inplane_patch = _ceil_to_multiple(
        requested_minimum_fov_mm / inplane_spacing_mm,
        2 ** (SIX_STAGE_N_STAGES - 1),
    )
    target_fov_mm = inplane_patch * inplane_spacing_mm
    plans, record = adapt_plan_to_six_stage_planned_z_fixed_xy(
        source_plans,
        dataset_id,
        source_plans_identifier=source_plans_identifier,
        derived_plans_identifier=derived_plans_identifier,
        configuration_name=configuration_name,
        target_fov_mm=target_fov_mm,
        inplane_spacing_mm=inplane_spacing_mm,
        reference_z_spacing_mm=inplane_spacing_mm,
        preprocessed_data_identifier=preprocessed_data_identifier,
    )
    configuration = plans['configurations'][configuration_name]
    configuration.update(
        {
            'preprocessor_name': (
                'SampleNativeZPreprocessor'
                if math.isclose(inplane_spacing_mm, 1.0)
                else 'SampleNativeZFixedXYPreprocessor'
            ),
        }
    )
    record.update(
        {
            'rule': (
                f'sample_native_z_fixed_xy_{inplane_spacing_mm:g}mm_'
                f'six_stage_min_fov{requested_minimum_fov_mm:g}'
            ),
            'z_spacing_policy': 'sample_native_with_minimum',
            'sample_native_z': True,
            'minimum_z_spacing_mm': inplane_spacing_mm,
            'requested_minimum_fov_mm': requested_minimum_fov_mm,
            'target_fov_mm': target_fov_mm,
            'inplane_spacing_policy': 'fixed',
            'inplane_spacing_mm': [
                inplane_spacing_mm,
                inplane_spacing_mm,
            ],
            'per_case_patch_size': True,
            'configuration_geometry_role': 'isotropic_fixed_xy_reference',
            'runtime_spacing_property': 'spacing_after_resampling',
            'preprocessed_data_reused': (
                preprocessed_data_identifier is not None
            ),
        }
    )
    record.pop('planned_z_spacing_unchanged')
    plans['pumit_spad_unet_source_adaptation'] = record
    return plans, record


def adapt_plan_to_six_stage_sample_native_z_inplane1(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = (
        SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER
    ),
    configuration_name: str = '3d_fullres',
    target_fov_mm: float = SIX_STAGE_PLANNED_Z_INPLANE1_FOV_MM,
    preprocessed_data_identifier: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the legacy fixed-1-mm Sample-Native-Z six-stage plan."""
    return adapt_plan_to_six_stage_sample_native_z_fixed_xy(
        source_plans,
        dataset_id,
        source_plans_identifier=source_plans_identifier,
        derived_plans_identifier=derived_plans_identifier,
        configuration_name=configuration_name,
        target_fov_mm=target_fov_mm,
        inplane_spacing_mm=1.0,
        preprocessed_data_identifier=preprocessed_data_identifier,
    )


def adapt_plan_to_six_stage_sample_native_z_planned_xy(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str,
    derived_plans_identifier: str,
    inplane_patch_size: int,
    floor_bottleneck_min: int,
    floor_bottleneck_max: int,
    rule: str,
    configuration_name: str = '3d_fullres',
    preprocessed_data_identifier: str | None = None,
    fixed_inplane_spacing_mm: float | None = None,
    preprocessor_name: str = 'SampleNativeZPlannedXYPreprocessor',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use dataset-planned XY spacing and image-driven Sample-Native-Z patches.

    `fixed_inplane_spacing_mm` overrides the dataset-planned XY spacing with a
    shared fixed value while keeping the bottleneck-window z contract.
    """
    if inplane_patch_size <= 0:
        raise ValueError('inplane_patch_size must be positive')
    if fixed_inplane_spacing_mm is not None and (
        not math.isfinite(fixed_inplane_spacing_mm)
        or fixed_inplane_spacing_mm <= 0
    ):
        raise ValueError(
            'fixed_inplane_spacing_mm must be finite and positive'
        )
    if not 0 < floor_bottleneck_min <= floor_bottleneck_max:
        raise ValueError(
            'floor bottleneck bounds must satisfy '
            '0 < floor_bottleneck_min <= floor_bottleneck_max'
        )
    if preprocessed_data_identifier is not None and (
        not isinstance(preprocessed_data_identifier, str)
        or not preprocessed_data_identifier
    ):
        raise ValueError(
            'preprocessed_data_identifier must be a non-empty string or null'
        )
    plans = deepcopy(source_plans)
    if plans.get('plans_name') != source_plans_identifier:
        raise ValueError(
            f'dataset {dataset_id} source plan name '
            f'{plans.get("plans_name")!r} does not match '
            f'{source_plans_identifier!r}'
        )
    if configuration_name not in plans.get('configurations', {}):
        raise KeyError(
            f'dataset {dataset_id} has no {configuration_name!r} configuration'
        )
    configuration = plans['configurations'][configuration_name]
    old_spacing = tuple(float(value) for value in configuration['spacing'])
    old_patch = tuple(int(value) for value in configuration['patch_size'])
    if len(old_spacing) != 3 or len(old_patch) != 3:
        raise ValueError(
            f'dataset {dataset_id} requires 3D spacing and patch, got '
            f'{old_spacing} and {old_patch}'
        )
    if any(value <= 0 for value in (*old_spacing, *old_patch)):
        raise ValueError(
            f'dataset {dataset_id} spacing and patch must be positive, got '
            f'{old_spacing} and {old_patch}'
        )
    if fixed_inplane_spacing_mm is None and not math.isclose(
        old_spacing[1],
        old_spacing[2],
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f'dataset {dataset_id} planned XY spacing must be equal, got '
            f'{old_spacing[1:]}'
        )

    new_spacing = (
        old_spacing
        if fixed_inplane_spacing_mm is None
        else (
            old_spacing[0],
            fixed_inplane_spacing_mm,
            fixed_inplane_spacing_mm,
        )
    )
    continuous_da = max(
        0.0,
        math.log2(new_spacing[0] / min(new_spacing[1:])),
    )
    floor_da = math.floor(continuous_da)
    depth_divisibility = 2 ** max(
        0,
        SIX_STAGE_N_STAGES - 1 - floor_da,
    )
    median_shape = tuple(
        float(value) for value in configuration['median_image_size_in_voxels']
    )
    reference_bottleneck = min(
        max(
            math.ceil(median_shape[0] / depth_divisibility),
            floor_bottleneck_min,
        ),
        floor_bottleneck_max,
    )
    new_patch = (
        reference_bottleneck * depth_divisibility,
        inplane_patch_size,
        inplane_patch_size,
    )

    da_schedule = compute_da_schedule(
        floor_da,
        SIX_STAGE_N_STAGES,
    )
    strides = [
        (1, 1, 1)
        if stage == 0
        else ((1, 2, 2) if da >= 1 else (2, 2, 2))
        for stage, da in enumerate(da_schedule)
    ]
    kernel_sizes = [
        (1, 3, 3) if da >= 2 else (3, 3, 3)
        for da in da_schedule
    ]
    architecture_kwargs = configuration['architecture']['arch_kwargs']
    features = list(architecture_kwargs['features_per_stage'])
    blocks = list(architecture_kwargs['n_blocks_per_stage'])
    features.extend(
        [features[-1]] * (SIX_STAGE_N_STAGES - len(features))
    )
    blocks.extend(
        [blocks[-1]] * (SIX_STAGE_N_STAGES - len(blocks))
    )
    architecture_kwargs.update({
        'n_stages': SIX_STAGE_N_STAGES,
        'features_per_stage': features[:SIX_STAGE_N_STAGES],
        'kernel_sizes': [list(values) for values in kernel_sizes],
        'strides': [list(values) for values in strides],
        'n_blocks_per_stage': blocks[:SIX_STAGE_N_STAGES],
        'n_conv_per_stage_decoder': [1]
        * (SIX_STAGE_N_STAGES - 1),
    })
    configuration.update({
        'data_identifier': (
            preprocessed_data_identifier
            or f'{derived_plans_identifier}_{configuration_name}'
        ),
        'preprocessor_name': preprocessor_name,
        'batch_size': 2,
        'batch_dice': False,
        'spacing': list(new_spacing),
        'patch_size': list(new_patch),
    })
    plans['plans_name'] = derived_plans_identifier

    old_fov = tuple(
        old_spacing[axis] * old_patch[axis] for axis in range(3)
    )
    reference_fov = tuple(
        new_spacing[axis] * new_patch[axis] for axis in range(3)
    )
    record = {
        'dataset_id': dataset_id,
        'adapted': True,
        'source_plans_identifier': source_plans_identifier,
        'derived_plans_identifier': derived_plans_identifier,
        'configuration': configuration_name,
        'rule': rule,
        'z_spacing_policy': 'sample_native_with_minimum',
        'sample_native_z': True,
        'minimum_z_spacing_mm': 1.0,
        'inplane_spacing_policy': (
            'nnunet_dataset_target'
            if fixed_inplane_spacing_mm is None
            else 'fixed'
        ),
        'inplane_spacing_mm': list(new_spacing[1:]),
        'per_case_patch_size': True,
        'runtime_patch_policy': 'image_depth_floor_bottleneck_cap',
        'runtime_spacing_property': 'spacing_after_resampling',
        'runtime_shape_property': 'shape_after_resampling',
        'inplane_patch_size': [
            inplane_patch_size,
            inplane_patch_size,
        ],
        'floor_bottleneck_min': floor_bottleneck_min,
        'floor_bottleneck_max': floor_bottleneck_max,
        'configuration_geometry_role': (
            'dataset_planned_reference'
            if fixed_inplane_spacing_mm is None
            else 'fixed_xy_reference'
        ),
        'preprocessed_data_reused': preprocessed_data_identifier is not None,
        'preprocessed_data_identifier': configuration['data_identifier'],
        'old_spacing': list(old_spacing),
        'new_spacing': list(new_spacing),
        'old_patch_size': list(old_patch),
        'reference_patch_size': list(new_patch),
        'old_fov_mm': list(old_fov),
        'reference_fov_mm': list(reference_fov),
        'continuous_da': continuous_da,
        'floor_da': floor_da,
        'depth_patch_divisibility': depth_divisibility,
    }
    plans['pumit_spad_unet_source_adaptation'] = record
    return plans, record


def adapt_plan_to_six_stage_sample_native_z_planned_xy_max8(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = (
        SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER
    ),
    configuration_name: str = '3d_fullres',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the original 256-patch, maximum-bottleneck-8 contract."""
    return adapt_plan_to_six_stage_sample_native_z_planned_xy(
        source_plans,
        dataset_id,
        source_plans_identifier=source_plans_identifier,
        derived_plans_identifier=derived_plans_identifier,
        inplane_patch_size=SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_INPLANE_PATCH,
        floor_bottleneck_min=(
            SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_BOTTLENECK_MIN
        ),
        floor_bottleneck_max=(
            SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_BOTTLENECK_MAX
        ),
        rule='sample_native_z_dataset_planned_xy_six_stage_max_bottleneck_8',
        configuration_name=configuration_name,
    )


def adapt_plan_to_six_stage_sample_native_z_planned_xy_p224_maxb7(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = (
        SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_P224_MAXB7_PLANS_IDENTIFIER
    ),
    configuration_name: str = '3d_fullres',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use a 224 in-plane patch and maximum floor bottleneck depth 7."""
    return adapt_plan_to_six_stage_sample_native_z_planned_xy(
        source_plans,
        dataset_id,
        source_plans_identifier=source_plans_identifier,
        derived_plans_identifier=derived_plans_identifier,
        inplane_patch_size=224,
        floor_bottleneck_min=4,
        floor_bottleneck_max=7,
        rule='sample_native_z_dataset_planned_xy_six_stage_p224_max_bottleneck_7',
        configuration_name=configuration_name,
        preprocessed_data_identifier=(
            f'{SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER}_'
            f'{configuration_name}'
        ),
    )


def adapt_plan_to_six_stage_sample_native_z_inplane1_p224_maxb7(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = (
        SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_P224_MAXB7_PLANS_IDENTIFIER
    ),
    configuration_name: str = '3d_fullres',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fix 1 mm in-plane spacing while keeping the MaxB7 image-driven z patches.

    Reuses the shared 1 mm Sample-Native-Z arrays, so only patch topology and
    runtime metadata differ from the fixed-1-mm square-FOV plans.
    """
    return adapt_plan_to_six_stage_sample_native_z_planned_xy(
        source_plans,
        dataset_id,
        source_plans_identifier=source_plans_identifier,
        derived_plans_identifier=derived_plans_identifier,
        inplane_patch_size=224,
        floor_bottleneck_min=4,
        floor_bottleneck_max=7,
        rule='sample_native_z_fixed_xy_1mm_six_stage_p224_max_bottleneck_7',
        configuration_name=configuration_name,
        preprocessed_data_identifier=(
            f'{SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER}_'
            f'{configuration_name}'
        ),
        fixed_inplane_spacing_mm=1.0,
        preprocessor_name='SampleNativeZPreprocessor',
    )


def _ceil_to_multiple(value: float, multiple: int) -> int:
    return int(math.ceil((value - 1e-8) / multiple) * multiple)


def adapt_plan_to_seven_stage_inplane(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = SEVEN_STAGE_INPLANE_PLANS_IDENTIFIER,
    configuration_name: str = '3d_fullres',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve FOV while making both in-plane patch axes at least 256.

    Unaffected datasets retain their existing preprocessed data identifier.
    Adapted datasets receive a new target spacing and data identifier.
    """
    plans = deepcopy(source_plans)
    if plans.get('plans_name') != source_plans_identifier:
        raise ValueError(
            f'dataset {dataset_id} source plan name '
            f'{plans.get("plans_name")!r} does not match '
            f'{source_plans_identifier!r}'
        )
    if configuration_name not in plans.get('configurations', {}):
        raise KeyError(
            f'dataset {dataset_id} has no {configuration_name!r} configuration'
        )

    configuration = plans['configurations'][configuration_name]
    old_spacing = tuple(float(value) for value in configuration['spacing'])
    old_patch = tuple(int(value) for value in configuration['patch_size'])
    if len(old_spacing) != 3 or len(old_patch) != 3:
        raise ValueError(
            f'dataset {dataset_id} requires 3D spacing and patch, got '
            f'{old_spacing} and {old_patch}'
        )
    if not math.isclose(old_spacing[1], old_spacing[2], rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            f'dataset {dataset_id} requires equal in-plane spacing, got '
            f'{old_spacing[1:]}'
        )

    plans['plans_name'] = derived_plans_identifier
    adapted = min(old_patch[1:]) < SEVEN_STAGE_INPLANE_MIN_PATCH
    if adapted:
        spacing_scale = (
            min(old_patch[1:]) / SEVEN_STAGE_INPLANE_MIN_PATCH
        )
        new_spacing = (
            old_spacing[0],
            old_spacing[1] * spacing_scale,
            old_spacing[2] * spacing_scale,
        )
        new_patch = (
            old_patch[0],
            *(
                _ceil_to_multiple(
                    old_patch[axis] * old_spacing[axis] / new_spacing[axis],
                    SEVEN_STAGE_INPLANE_DIVISIBILITY,
                )
                for axis in (1, 2)
            ),
        )
        median_shape = tuple(
            float(value)
            for value in configuration['median_image_size_in_voxels']
        )
        new_median_shape = tuple(
            median_shape[axis] * old_spacing[axis] / new_spacing[axis]
            for axis in range(3)
        )
        (
            _,
            strides,
            kernel_sizes,
            topology_patch,
            _,
        ) = get_pool_and_conv_props(
            new_spacing,
            new_patch,
            4,
            999999,
        )
        topology_patch = tuple(int(value) for value in topology_patch)
        if topology_patch != new_patch:
            raise RuntimeError(
                f'dataset {dataset_id} topology changed requested patch '
                f'{new_patch} to {topology_patch}'
            )
        if len(strides) != 7 or tuple(strides[-1]) == (1, 1, 1):
            raise RuntimeError(
                f'dataset {dataset_id} did not produce a true seventh stage: '
                f'{strides}'
            )

        architecture = configuration['architecture']
        architecture_kwargs = architecture['arch_kwargs']
        num_stages = len(strides)
        features = list(architecture_kwargs['features_per_stage'])
        blocks = list(architecture_kwargs['n_blocks_per_stage'])
        features.extend([features[-1]] * (num_stages - len(features)))
        blocks.extend([blocks[-1]] * (num_stages - len(blocks)))
        architecture_kwargs.update({
            'n_stages': num_stages,
            'features_per_stage': features[:num_stages],
            'kernel_sizes': [list(values) for values in kernel_sizes],
            'strides': [list(values) for values in strides],
            'n_blocks_per_stage': blocks[:num_stages],
            'n_conv_per_stage_decoder': [1] * (num_stages - 1),
        })
        configuration.update({
            'data_identifier': (
                f'{derived_plans_identifier}_{configuration_name}'
            ),
            'spacing': list(new_spacing),
            'patch_size': list(new_patch),
            'median_image_size_in_voxels': list(new_median_shape),
        })
    else:
        new_spacing = old_spacing
        new_patch = old_patch

    record = {
        'dataset_id': dataset_id,
        'adapted': adapted,
        'source_plans_identifier': source_plans_identifier,
        'derived_plans_identifier': derived_plans_identifier,
        'configuration': configuration_name,
        'rule': 'preserve_inplane_fov_with_minimum_patch_256',
        'z_spacing_unchanged': True,
        'old_spacing': list(old_spacing),
        'new_spacing': list(new_spacing),
        'old_patch_size': list(old_patch),
        'new_patch_size': list(new_patch),
    }
    plans['pumit_spad_unet_source_adaptation'] = record
    return plans, record


def adapt_plan_to_seven_stage_fov(
    source_plans: dict[str, Any],
    dataset_id: str,
    *,
    source_plans_identifier: str = 'nnUNetResEncUNetLPlans',
    derived_plans_identifier: str = SEVEN_STAGE_FOV_PLANS_IDENTIFIER,
    configuration_name: str = '3d_fullres',
    minimum_fov_mm: float = SEVEN_STAGE_FOV_MINIMUM_MM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Increase the native patch to a true seven-stage, minimum-FOV grid.

    Target spacing and preprocessed arrays remain native. Patch axes are rounded
    up to the divisibility required by the seven-stage topology.
    """
    plans = deepcopy(source_plans)
    if plans.get('plans_name') != source_plans_identifier:
        raise ValueError(
            f'dataset {dataset_id} source plan name '
            f'{plans.get("plans_name")!r} does not match '
            f'{source_plans_identifier!r}'
        )
    if minimum_fov_mm <= 0:
        raise ValueError('minimum_fov_mm must be positive')
    if configuration_name not in plans.get('configurations', {}):
        raise KeyError(
            f'dataset {dataset_id} has no {configuration_name!r} configuration'
        )

    configuration = plans['configurations'][configuration_name]
    old_spacing = tuple(float(value) for value in configuration['spacing'])
    old_patch = tuple(int(value) for value in configuration['patch_size'])
    if len(old_spacing) != 3 or len(old_patch) != 3:
        raise ValueError(
            f'dataset {dataset_id} requires 3D spacing and patch, got '
            f'{old_spacing} and {old_patch}'
        )
    if any(value <= 0 for value in (*old_spacing, *old_patch)):
        raise ValueError(
            f'dataset {dataset_id} spacing and patch must be positive, got '
            f'{old_spacing} and {old_patch}'
        )

    new_patch = tuple(
        _ceil_to_multiple(
            max(old_patch[axis], minimum_fov_mm / old_spacing[axis]),
            SEVEN_STAGE_FOV_PATCH_DIVISIBILITY[axis],
        )
        for axis in range(3)
    )
    (
        _,
        strides,
        kernel_sizes,
        topology_patch,
        _,
    ) = get_pool_and_conv_props(
        old_spacing,
        new_patch,
        4,
        999999,
    )
    topology_patch = tuple(int(value) for value in topology_patch)
    if topology_patch != new_patch:
        raise RuntimeError(
            f'dataset {dataset_id} topology changed requested patch '
            f'{new_patch} to {topology_patch}'
        )
    if len(strides) != 7 or tuple(strides[-1]) == (1, 1, 1):
        raise RuntimeError(
            f'dataset {dataset_id} did not produce a true seventh stage: '
            f'{strides}'
        )

    architecture_kwargs = configuration['architecture']['arch_kwargs']
    num_stages = len(strides)
    features = list(architecture_kwargs['features_per_stage'])
    blocks = list(architecture_kwargs['n_blocks_per_stage'])
    features.extend([features[-1]] * (num_stages - len(features)))
    blocks.extend([blocks[-1]] * (num_stages - len(blocks)))
    architecture_kwargs.update({
        'n_stages': num_stages,
        'features_per_stage': features[:num_stages],
        'kernel_sizes': [list(values) for values in kernel_sizes],
        'strides': [list(values) for values in strides],
        'n_blocks_per_stage': blocks[:num_stages],
        'n_conv_per_stage_decoder': [1] * (num_stages - 1),
    })
    configuration['patch_size'] = list(new_patch)
    plans['plans_name'] = derived_plans_identifier

    old_fov = tuple(
        old_spacing[axis] * old_patch[axis]
        for axis in range(3)
    )
    new_fov = tuple(
        old_spacing[axis] * new_patch[axis]
        for axis in range(3)
    )
    record = {
        'dataset_id': dataset_id,
        'adapted': new_patch != old_patch,
        'source_plans_identifier': source_plans_identifier,
        'derived_plans_identifier': derived_plans_identifier,
        'configuration': configuration_name,
        'rule': 'native_spacing_minimum_fov_192_true_seven_stage',
        'minimum_fov_mm': minimum_fov_mm,
        'spacing_unchanged': True,
        'preprocessed_data_reused': True,
        'old_spacing': list(old_spacing),
        'new_spacing': list(old_spacing),
        'old_patch_size': list(old_patch),
        'new_patch_size': list(new_patch),
        'old_fov_mm': list(old_fov),
        'new_fov_mm': list(new_fov),
    }
    plans['pumit_spad_unet_source_adaptation'] = record
    return plans, record
