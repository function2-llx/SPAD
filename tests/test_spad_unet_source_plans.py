"""Tests for SPAD source-plan adaptations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from batchgenerators.utilities.file_and_folder_operations import save_pickle
from nnunetv2.preprocessing.preprocessors import (
    sample_native_z_preprocessor as sample_native_z_preprocessor_module,
)
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.preprocessing.preprocessors.sample_native_z_preprocessor import (
    SampleNativeZFixedXYPreprocessor,
    SampleNativeZPreprocessor,
    SampleNativeZPlannedXYPreprocessor,
    sample_native_z_planned_xy_target_spacing,
    sample_native_z_fixed_xy_target_spacing,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from pumit.spad_unet.data import (
    _load_sample_native_case_geometries,
    derive_sample_native_max_bottleneck_patch_size,
    derive_sample_native_patch_size,
)
from pumit.spad_unet.source_plans import (
    SIX_STAGE_PLANNED_Z_INPLANE1_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_FOV224_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_P224_MAXB7_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE0P9_FOV192_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER,
    SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_P224_MAXB7_PLANS_IDENTIFIER,
    adapt_plan_to_six_stage_planned_z_inplane1,
    adapt_plan_to_six_stage_sample_native_z_fixed_xy,
    adapt_plan_to_six_stage_sample_native_z_inplane1,
    adapt_plan_to_six_stage_sample_native_z_inplane1_p224_maxb7,
    adapt_plan_to_six_stage_sample_native_z_planned_xy_max8,
    adapt_plan_to_six_stage_sample_native_z_planned_xy_p224_maxb7,
    adapt_plan_to_seven_stage_fov,
    adapt_plan_to_seven_stage_inplane,
)


@pytest.mark.parametrize(
    ('dataset_id', 'spacing', 'patch_size', 'expected_patch'),
    [
        ('503', (1.0, 0.767578125, 0.767578125), (192, 192, 192), (192, 192, 192)),
        (
            '506',
            (1.2449799776, 0.78515625, 0.78515625),
            (112, 256, 256),
            (160, 192, 192),
        ),
        ('507', (2.5, 0.802734, 0.802734), (56, 320, 256), (80, 192, 192)),
        ('509', (1.6000100374, 0.792969, 0.792969), (80, 256, 256), (128, 192, 192)),
        ('510', (3.0, 0.78125, 0.78125), (80, 256, 256), (64, 192, 192)),
    ],
)
def test_six_stage_planned_z_inplane1_uses_dataset_target_z_and_controlled_fov(
    dataset_id,
    spacing,
    patch_size,
    expected_patch,
):
    adapted, record = adapt_plan_to_six_stage_planned_z_inplane1(
        source_plans(spacing, patch_size),
        dataset_id,
    )
    configuration = adapted['configurations']['3d_fullres']
    architecture = configuration['architecture']['arch_kwargs']

    assert adapted['plans_name'] == SIX_STAGE_PLANNED_Z_INPLANE1_PLANS_IDENTIFIER
    assert tuple(configuration['spacing']) == (spacing[0], 1.0, 1.0)
    assert tuple(configuration['patch_size']) == expected_patch
    assert configuration['batch_size'] == 4
    assert configuration['batch_dice'] is False
    assert architecture['n_stages'] == 6
    assert len(architecture['features_per_stage']) == 6
    assert len(architecture['n_blocks_per_stage']) == 6
    assert len(architecture['n_conv_per_stage_decoder']) == 5
    assert len(architecture['strides']) == 6
    assert all(value >= 192.0 - 1e-6 for value in record['new_fov_mm'])
    assert record['planned_z_spacing_unchanged']
    assert record['z_spacing_policy'] == 'nnunet_dataset_target'
    assert record['sample_native_z'] is False
    assert not record['preprocessed_data_reused']


def test_six_stage_planned_z_inplane1_can_reuse_compatible_arrays():
    data_identifier = 'nnUNetResEncUNetLPlans6StageNativeZ1x1FOV192_3d_fullres'
    adapted, record = adapt_plan_to_six_stage_planned_z_inplane1(
        source_plans((2.5, 0.8, 0.8), (56, 320, 256)),
        '507',
        preprocessed_data_identifier=data_identifier,
    )

    configuration = adapted['configurations']['3d_fullres']
    assert configuration['data_identifier'] == data_identifier
    assert record['preprocessed_data_reused']
    assert record['preprocessed_data_identifier'] == data_identifier


def test_six_stage_sample_native_z_uses_isolated_case_geometry_preprocessing():
    adapted, record = adapt_plan_to_six_stage_sample_native_z_inplane1(
        source_plans((2.5, 0.8, 0.8), (56, 320, 256)),
        '507',
    )

    configuration = adapted['configurations']['3d_fullres']
    assert adapted['plans_name'] == (
        SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER
    )
    assert configuration['data_identifier'] == (
        f'{SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER}_3d_fullres'
    )
    assert configuration['preprocessor_name'] == 'SampleNativeZPreprocessor'
    assert tuple(configuration['spacing']) == (1.0, 1.0, 1.0)
    assert tuple(configuration['patch_size']) == (192, 192, 192)
    assert record['z_spacing_policy'] == 'sample_native_with_minimum'
    assert record['sample_native_z'] is True
    assert record['minimum_z_spacing_mm'] == 1.0
    assert record['per_case_patch_size'] is True
    assert record['runtime_spacing_property'] == 'spacing_after_resampling'
    assert not record['preprocessed_data_reused']
    assert (
        PlansManager(adapted)
        .get_configuration('3d_fullres')
        .preprocessor_class
        is SampleNativeZPreprocessor
    )


def test_six_stage_sample_native_z_fov224_reuses_fov192_arrays():
    data_identifier = (
        f'{SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER}_3d_fullres'
    )
    adapted, record = adapt_plan_to_six_stage_sample_native_z_inplane1(
        source_plans((2.5, 0.8, 0.8), (56, 320, 256)),
        '507',
        derived_plans_identifier=(
            SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_FOV224_PLANS_IDENTIFIER
        ),
        target_fov_mm=224.0,
        preprocessed_data_identifier=data_identifier,
    )

    configuration = adapted['configurations']['3d_fullres']
    assert adapted['plans_name'] == (
        SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_FOV224_PLANS_IDENTIFIER
    )
    assert configuration['data_identifier'] == data_identifier
    assert tuple(configuration['spacing']) == (1.0, 1.0, 1.0)
    assert tuple(configuration['patch_size']) == (224, 224, 224)
    assert record['target_fov_mm'] == 224.0
    assert record['preprocessed_data_reused']
    assert derive_sample_native_patch_size(
        (2.5, 1.0, 1.0),
        target_fov_mm=224.0,
        n_stages=6,
    ) == (96, 224, 224)


def test_six_stage_sample_native_z_fixed_0p9_uses_common_realized_fov():
    inplane_spacing_mm = 0.9
    adapted, record = adapt_plan_to_six_stage_sample_native_z_fixed_xy(
        source_plans((2.5, 0.8, 0.8), (56, 320, 256)),
        '507',
        derived_plans_identifier=(
            SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE0P9_FOV192_PLANS_IDENTIFIER
        ),
        inplane_spacing_mm=inplane_spacing_mm,
    )

    configuration = adapted['configurations']['3d_fullres']
    assert tuple(configuration['spacing']) == (0.9, 0.9, 0.9)
    assert tuple(configuration['patch_size']) == (224, 224, 224)
    assert configuration['preprocessor_name'] == (
        'SampleNativeZFixedXYPreprocessor'
    )
    assert record['inplane_spacing_policy'] == 'fixed'
    assert record['inplane_spacing_mm'] == [
        inplane_spacing_mm,
        inplane_spacing_mm,
    ]
    assert record['minimum_z_spacing_mm'] == inplane_spacing_mm
    assert record['requested_minimum_fov_mm'] == 192.0
    assert record['target_fov_mm'] == pytest.approx(201.6)
    assert record['new_fov_mm'] == pytest.approx([201.6, 201.6, 201.6])
    assert record['configuration_geometry_role'] == (
        'isotropic_fixed_xy_reference'
    )
    assert (
        PlansManager(adapted)
        .get_configuration('3d_fullres')
        .preprocessor_class
        is SampleNativeZFixedXYPreprocessor
    )
    assert derive_sample_native_patch_size(
        (2.5, inplane_spacing_mm, inplane_spacing_mm),
        target_fov_mm=record['target_fov_mm'],
        n_stages=6,
    ) == (96, 224, 224)


def test_sample_native_z_fixed_xy_preprocessor_uses_xy_as_minimum_z(
    monkeypatch,
):
    seen = {}

    def fake_run_case_npy(
        self,
        data,
        seg,
        properties,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        seen['spacing'] = tuple(configuration_manager.spacing)
        return data, seg, properties

    monkeypatch.setattr(DefaultPreprocessor, 'run_case_npy', fake_run_case_npy)
    monkeypatch.setattr(
        sample_native_z_preprocessor_module,
        'ConfigurationManager',
        lambda configuration: SimpleNamespace(
            spacing=tuple(configuration['spacing'])
        ),
    )
    preprocessor = SampleNativeZFixedXYPreprocessor()
    data = np.zeros((1, 3, 5, 7), dtype=np.float32)
    seg = np.zeros((1, 3, 5, 7), dtype=np.int8)
    properties = {'spacing': [0.7, 0.6, 0.6]}
    configuration_manager = SimpleNamespace(
        spacing=(1.0, 0.9, 0.9),
        configuration={'spacing': [1.0, 0.9, 0.9]},
    )

    _, _, output_properties = preprocessor.run_case_npy(
        data,
        seg,
        properties,
        SimpleNamespace(transpose_forward=(0, 1, 2)),
        configuration_manager,
        {},
    )

    assert sample_native_z_fixed_xy_target_spacing(
        (0.7, 0.6, 0.6),
        (1.0, 0.9, 0.9),
    ) == pytest.approx((0.9, 0.9, 0.9))
    assert seen['spacing'] == pytest.approx((0.9, 0.9, 0.9))
    assert output_properties['spacing_after_resampling'] == pytest.approx(
        (0.9, 0.9, 0.9)
    )
    assert output_properties['shape_after_resampling'] == [3, 5, 7]


def test_sample_native_case_geometry_rejects_stale_minimum_z_arrays(tmp_path):
    save_pickle(
        {
            'spacing': [0.7, 0.6, 0.6],
            'spacing_after_resampling': [1.0, 0.9, 0.9],
        },
        tmp_path / 'case.pkl',
    )

    with pytest.raises(ValueError, match=r'instead of 0\.9'):
        _load_sample_native_case_geometries(
            tmp_path,
            ['case'],
            spacing_property='spacing_after_resampling',
            expected_inplane_spacing=(0.9, 0.9),
            minimum_z_spacing_mm=0.9,
            transpose_forward=(0, 1, 2),
            target_fov_mm=192.0,
            n_stages=6,
        )


@pytest.mark.parametrize(
    ('z_spacing', 'expected_depth'),
    [
        (1.0, 192),
        (1.25, 160),
        (2.5, 80),
        (5.0, 40),
        (8.0, 24),
    ],
)
def test_sample_native_patch_depth_is_derived_without_a_manual_cap(
    z_spacing,
    expected_depth,
):
    assert derive_sample_native_patch_size(
        (z_spacing, 1.0, 1.0),
        target_fov_mm=192.0,
        n_stages=6,
    ) == (expected_depth, 192, 192)


def test_sample_native_z_planned_xy_uses_case_z_and_planned_xy():
    assert sample_native_z_planned_xy_target_spacing(
        (0.7, 0.6, 0.6),
        (2.5, 0.802734, 0.802734),
    ) == pytest.approx((1.0, 0.802734, 0.802734))
    assert sample_native_z_planned_xy_target_spacing(
        (5.0, 0.7, 0.7),
        (2.5, 0.802734, 0.802734),
    ) == pytest.approx((5.0, 0.802734, 0.802734))


def test_six_stage_sample_native_z_planned_xy_max8_plan_contract():
    adapted, record = adapt_plan_to_six_stage_sample_native_z_planned_xy_max8(
        source_plans((2.5, 0.802734, 0.802734), (56, 320, 256)),
        '507',
    )
    configuration = adapted['configurations']['3d_fullres']
    architecture = configuration['architecture']['arch_kwargs']

    assert adapted['plans_name'] == (
        SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER
    )
    assert configuration['preprocessor_name'] == (
        'SampleNativeZPlannedXYPreprocessor'
    )
    assert tuple(configuration['spacing']) == pytest.approx(
        (2.5, 0.802734, 0.802734)
    )
    assert tuple(configuration['patch_size']) == (128, 256, 256)
    assert architecture['n_stages'] == 6
    assert len(architecture['strides']) == 6
    assert record['runtime_patch_policy'] == (
        'image_depth_floor_bottleneck_cap'
    )
    assert record['runtime_shape_property'] == 'shape_after_resampling'
    assert record['floor_bottleneck_min'] == 4
    assert record['floor_bottleneck_max'] == 8
    assert record['inplane_patch_size'] == [256, 256]
    assert (
        PlansManager(adapted)
        .get_configuration('3d_fullres')
        .preprocessor_class
        is SampleNativeZPlannedXYPreprocessor
    )


def test_six_stage_sample_native_z_planned_xy_p224_maxb7_reuses_preprocessing():
    adapted, record = (
        adapt_plan_to_six_stage_sample_native_z_planned_xy_p224_maxb7(
            source_plans((2.5, 0.802734, 0.802734), (56, 320, 256)),
            '507',
        )
    )
    configuration = adapted['configurations']['3d_fullres']

    assert adapted['plans_name'] == (
        SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_P224_MAXB7_PLANS_IDENTIFIER
    )
    assert tuple(configuration['patch_size']) == (112, 224, 224)
    assert configuration['data_identifier'] == (
        f'{SIX_STAGE_SAMPLE_NATIVE_Z_PLANNED_XY_MAX8_PLANS_IDENTIFIER}_'
        '3d_fullres'
    )
    assert record['inplane_patch_size'] == [224, 224]
    assert record['floor_bottleneck_min'] == 4
    assert record['floor_bottleneck_max'] == 7
    assert record['preprocessed_data_reused'] is True
    assert record['preprocessed_data_identifier'] == (
        configuration['data_identifier']
    )


def test_six_stage_sample_native_z_inplane1_p224_maxb7_fixes_xy_spacing():
    adapted, record = (
        adapt_plan_to_six_stage_sample_native_z_inplane1_p224_maxb7(
            source_plans((2.5, 0.802734, 0.802734), (56, 320, 256)),
            '507',
        )
    )
    configuration = adapted['configurations']['3d_fullres']

    assert adapted['plans_name'] == (
        SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_P224_MAXB7_PLANS_IDENTIFIER
    )
    assert tuple(configuration['spacing']) == (2.5, 1.0, 1.0)
    assert tuple(configuration['patch_size']) == (112, 224, 224)
    assert configuration['preprocessor_name'] == 'SampleNativeZPreprocessor'
    assert configuration['data_identifier'] == (
        f'{SIX_STAGE_SAMPLE_NATIVE_Z_INPLANE1_PLANS_IDENTIFIER}_3d_fullres'
    )
    assert record['inplane_spacing_policy'] == 'fixed'
    assert record['inplane_spacing_mm'] == [1.0, 1.0]
    assert record['runtime_patch_policy'] == 'image_depth_floor_bottleneck_cap'
    assert record['inplane_patch_size'] == [224, 224]
    assert record['floor_bottleneck_min'] == 4
    assert record['floor_bottleneck_max'] == 7
    assert record['configuration_geometry_role'] == 'fixed_xy_reference'
    assert record['preprocessed_data_reused'] is True
    assert (
        PlansManager(adapted)
        .get_configuration('3d_fullres')
        .preprocessor_class
        is SampleNativeZPreprocessor
    )


def test_sample_native_z_planned_xy_preprocessor_records_realized_geometry(
    monkeypatch,
):
    seen = {}

    def fake_run_case_npy(
        self,
        data,
        seg,
        properties,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        seen['spacing'] = tuple(configuration_manager.spacing)
        return data, seg, properties

    monkeypatch.setattr(DefaultPreprocessor, 'run_case_npy', fake_run_case_npy)
    monkeypatch.setattr(
        sample_native_z_preprocessor_module,
        'ConfigurationManager',
        lambda configuration: SimpleNamespace(
            spacing=tuple(configuration['spacing'])
        ),
    )
    preprocessor = SampleNativeZPlannedXYPreprocessor()
    data = np.zeros((1, 3, 5, 7), dtype=np.float32)
    seg = np.zeros((1, 3, 5, 7), dtype=np.int8)
    properties = {'spacing': [0.7, 0.6, 0.6]}
    configuration_manager = SimpleNamespace(
        spacing=(2.5, 0.802734, 0.802734),
        configuration={'spacing': [2.5, 0.802734, 0.802734]},
    )

    _, _, output_properties = preprocessor.run_case_npy(
        data,
        seg,
        properties,
        SimpleNamespace(transpose_forward=(0, 1, 2)),
        configuration_manager,
        {},
    )

    assert seen['spacing'] == pytest.approx((1.0, 0.802734, 0.802734))
    assert output_properties['spacing_after_resampling'] == pytest.approx(
        (1.0, 0.802734, 0.802734)
    )
    assert output_properties['shape_after_resampling'] == [3, 5, 7]


@pytest.mark.parametrize(
    ('spacing', 'image_shape', 'expected_patch'),
    [
        ((5.0, 0.8, 0.8), (90, 512, 512), (64, 256, 256)),
        ((5.0, 0.8, 0.8), (31, 512, 512), (32, 256, 256)),
        ((1.5, 0.8, 0.8), (168, 512, 512), (192, 256, 256)),
    ],
)
def test_sample_native_max_bottleneck_patch_uses_resampled_image_depth(
    spacing,
    image_shape,
    expected_patch,
):
    assert derive_sample_native_max_bottleneck_patch_size(
        spacing,
        image_shape,
        inplane_patch_size=256,
        n_stages=6,
        floor_bottleneck_min=4,
        floor_bottleneck_max=8,
    ) == expected_patch


def source_plans(
    spacing: tuple[float, float, float],
    patch_size: tuple[int, int, int],
) -> dict:
    return {
        'plans_name': 'nnUNetResEncUNetLPlans',
        'configurations': {
            '3d_fullres': {
                'data_identifier': 'nnUNetResEncUNetLPlans_3d_fullres',
                'batch_size': 2,
                'spacing': list(spacing),
                'patch_size': list(patch_size),
                'median_image_size_in_voxels': [300.0, 512.0, 512.0],
                'architecture': {
                    'arch_kwargs': {
                        'n_stages': 6,
                        'features_per_stage': [32, 64, 128, 256, 320, 320],
                        'kernel_sizes': [[3, 3, 3]] * 6,
                        'strides': [[1, 1, 1], *([[2, 2, 2]] * 5)],
                        'n_blocks_per_stage': [1, 3, 4, 6, 6, 6],
                        'n_conv_per_stage_decoder': [1] * 5,
                    },
                },
            },
        },
    }


@pytest.mark.parametrize(
    ('dataset_id', 'spacing', 'patch_size', 'expected_spacing', 'expected_patch'),
    [
        (
            '503',
            (1.0, 0.767578125, 0.767578125),
            (192, 192, 192),
            (1.0, 0.57568359375, 0.57568359375),
            (192, 256, 256),
        ),
        (
            '546',
            (1.0, 0.8172264993190765, 0.8172264993190765),
            (128, 256, 224),
            (1.0, 0.715073186904192, 0.715073186904192),
            (128, 320, 256),
        ),
        (
            '564',
            (0.78126, 0.78125, 0.78125),
            (192, 192, 192),
            (0.78126, 0.5859375, 0.5859375),
            (192, 256, 256),
        ),
    ],
)
def test_adaptation_preserves_fov_and_builds_true_seventh_stage(
    dataset_id,
    spacing,
    patch_size,
    expected_spacing,
    expected_patch,
):
    adapted, record = adapt_plan_to_seven_stage_inplane(
        source_plans(spacing, patch_size),
        dataset_id,
    )
    configuration = adapted['configurations']['3d_fullres']

    assert record['adapted']
    assert tuple(configuration['spacing']) == pytest.approx(expected_spacing)
    assert tuple(configuration['patch_size']) == expected_patch
    assert configuration['architecture']['arch_kwargs']['n_stages'] == 7
    assert tuple(
        configuration['architecture']['arch_kwargs']['strides'][-1]
    ) != (1, 1, 1)
    for axis in (1, 2):
        old_fov = spacing[axis] * patch_size[axis]
        new_fov = configuration['spacing'][axis] * configuration['patch_size'][axis]
        assert new_fov >= old_fov - 1e-6


def test_adaptation_reuses_existing_data_when_patch_is_already_admissible():
    plans = source_plans((2.5, 0.8, 0.8), (80, 320, 256))

    adapted, record = adapt_plan_to_seven_stage_inplane(plans, '507')
    configuration = adapted['configurations']['3d_fullres']

    assert not record['adapted']
    assert configuration['spacing'] == [2.5, 0.8, 0.8]
    assert configuration['patch_size'] == [80, 320, 256]
    assert configuration['data_identifier'] == (
        'nnUNetResEncUNetLPlans_3d_fullres'
    )


@pytest.mark.parametrize(
    ('dataset_id', 'spacing', 'patch_size', 'expected_patch'),
    [
        ('503', (1.0, 0.767578125, 0.767578125), (192, 192, 192), (192, 256, 256)),
        ('506', (1.2449799776, 0.78515625, 0.78515625), (112, 256, 256), (160, 256, 256)),
        ('718', (2.0, 0.6845703125, 0.6845703125), (96, 224, 224), (96, 320, 320)),
        ('725', (3.0, 0.9765625, 0.9765625), (96, 224, 224), (96, 256, 256)),
        ('220', (1.0, 0.78125, 0.78125), (160, 224, 192), (192, 256, 256)),
    ],
)
def test_fov_adaptation_preserves_native_spacing_and_reuses_preprocessing(
    dataset_id,
    spacing,
    patch_size,
    expected_patch,
):
    adapted, record = adapt_plan_to_seven_stage_fov(
        source_plans(spacing, patch_size),
        dataset_id,
    )
    configuration = adapted['configurations']['3d_fullres']

    assert tuple(configuration['spacing']) == spacing
    assert tuple(configuration['patch_size']) == expected_patch
    assert configuration['data_identifier'] == 'nnUNetResEncUNetLPlans_3d_fullres'
    assert configuration['architecture']['arch_kwargs']['n_stages'] == 7
    assert record['preprocessed_data_reused']
    assert all(value >= 192 - 1e-6 for value in record['new_fov_mm'])
