"""Tests for the dataset-planned-grid SPAD Universal trainer path."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from pumit.nnunet.compile_cache import CompileCacheMixin
from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.nnunet.model_ema import ModelEmaMixin
import pumit.spad_unet.experiments.spad_universal as spad_universal_module
import pumit.spad_unet.architecture as spad_architecture_module
import pumit.spad_unet.evaluation as spad_evaluation_module
from pumit.spad_unet.architecture import (
    FGCReturnPrefilter,
    LEGACY_UNIVERSAL_SPAD_BLOCKS,
    LEGACY_UNIVERSAL_SPAD_DECODER_CONVS,
    LEGACY_UNIVERSAL_SPAD_FEATURES,
    LEGACY_UNIVERSAL_SPAD_N_STAGES,
    UNIVERSAL_SPAD_BLOCKS,
    UNIVERSAL_SPAD_DECODER_CONVS,
    UNIVERSAL_SPAD_FEATURES,
    UNIVERSAL_SPAD_MIN_BOTTLENECK,
    UNIVERSAL_SPAD_N_STAGES,
    UNIVERSAL_SPAD_NETWORK_CLASS_NAME,
    UNIVERSAL_SPAD_TAB_FEATURE_LEVEL_INDICES,
    UniversalSPADResEncUNet,
    align_spad_feature,
    build_universal_spad_architecture,
)
from pumit.spad_unet.experiments.spad_universal import (
    SPADUniversalTrainer,
    _SPADActiveRegionNetwork,
    _num_epochs_for_updates,
    collate_spad_universal_samples,
    resolve_spad_universal_dataset_objective,
    spad_universal_dataset_loss_weights,
    validate_spad_universal_dataset_sampling,
    validate_spad_universal_inference_mode,
    validate_spad_universal_replay_metadata,
)
from pumit.spad_unet.data import UniversalCaseGeometry
from pumit.spad_unet.geometry import (
    SUPPORTED_FGC_STAGES,
    compute_fgc_geometry,
    decompose_continuous_da,
)
from pumit.spad_unet.loss import (
    REGION_BALANCED_LOSS_NORMALIZATION,
    SAMPLE_MEAN_LOSS_NORMALIZATION,
    SIGMOID_REGIONS,
    SOFTMAX_LABELS,
    UniversalPartialLabelLoss,
)
from pumit.spad_unet.replay import (
    COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    SQRT_DATASET_REPLAY_RULE,
    UniversalSample,
)
from pumit.spad_unet.task_aware_bottleneck import (
    TAB_ATTENTION_DOWNSAMPLE_RATE,
    TAB_DEPTH,
    TAB_DIM,
    TAB_FOURIER_SCALE,
    TAB_FOURIER_SEED,
    TAB_HEADS,
    TAB_MLP_DIM,
    TAB_TOKENS_PER_DATASET,
    MultiScaleTaskAwareBottleneck,
    TaskAwareBottleneck,
)
from pumit.spad_unet.universal import (
    CanonicalRegionRegistry,
    build_dataset_output_contract,
    foreground_region_values,
)


def test_spad_universal_retains_periodic_checkpoints():
    assert (
        SPADUniversalTrainer.save_checkpoint
        is RetainPeriodicCheckpointsMixin.save_checkpoint
    )


def test_spad_universal_uses_runtime_compile_cache():
    assert issubclass(SPADUniversalTrainer, CompileCacheMixin)


def test_spad_universal_supports_rank_zero_model_ema():
    assert issubclass(SPADUniversalTrainer, ModelEmaMixin)


@pytest.mark.parametrize(
    ('num_updates', 'expected_num_epochs'),
    [(250_000, 1_000), (500_000, 2_000)],
)
def test_spad_universal_derives_epochs_from_update_budget(
    num_updates,
    expected_num_epochs,
):
    assert _num_epochs_for_updates(num_updates, 250) == expected_num_epochs


def test_spad_universal_rejects_partial_epoch_update_budget():
    with pytest.raises(ValueError, match='not divisible'):
        _num_epochs_for_updates(500_001, 250)


def _source_architecture() -> dict:
    return {
        'network_class_name': (
            'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
        ),
        'arch_kwargs': {
            'n_stages': 3,
            'features_per_stage': [4, 8, 16],
            'conv_op': 'torch.nn.modules.conv.Conv3d',
            'kernel_sizes': [[3, 3, 3]] * 3,
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            'n_blocks_per_stage': [1, 1, 1],
            'n_conv_per_stage_decoder': [1, 1],
            'conv_bias': True,
            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True},
        },
        '_kw_requires_import': [],
    }


def _tab_kwargs(num_datasets: int = 2) -> dict:
    return {
        'num_task_datasets': num_datasets,
        'tab_tokens_per_dataset': TAB_TOKENS_PER_DATASET,
        'tab_dim': TAB_DIM,
        'tab_depth': TAB_DEPTH,
        'tab_heads': TAB_HEADS,
        'tab_mlp_dim': TAB_MLP_DIM,
        'tab_attention_downsample_rate': TAB_ATTENTION_DOWNSAMPLE_RATE,
        'tab_fourier_scale': TAB_FOURIER_SCALE,
        'tab_fourier_seed': TAB_FOURIER_SEED,
    }


def _universal_source_architecture(n_stages: int = UNIVERSAL_SPAD_N_STAGES) -> dict:
    source = _source_architecture()
    kwargs = source['arch_kwargs']
    if n_stages == UNIVERSAL_SPAD_N_STAGES:
        features = UNIVERSAL_SPAD_FEATURES
        blocks = UNIVERSAL_SPAD_BLOCKS
        decoder_convs = UNIVERSAL_SPAD_DECODER_CONVS
    elif n_stages == LEGACY_UNIVERSAL_SPAD_N_STAGES:
        features = LEGACY_UNIVERSAL_SPAD_FEATURES
        blocks = LEGACY_UNIVERSAL_SPAD_BLOCKS
        decoder_convs = LEGACY_UNIVERSAL_SPAD_DECODER_CONVS
    else:
        raise ValueError(n_stages)
    kwargs.update({
        'n_stages': n_stages,
        'features_per_stage': list(features),
        'kernel_sizes': [[3, 3, 3]] * n_stages,
        'strides': [[1, 1, 1], *([[2, 2, 2]] * (n_stages - 1))],
        'n_blocks_per_stage': list(blocks),
        'n_conv_per_stage_decoder': list(decoder_convs),
    })
    return source


def test_dataset_output_contract_recovers_native_modes_and_packs_backgrounds():
    dataset_counts = {
        '503': 2,
        '506': 1,
        '507': 2,
        '508': 2,
        '509': 1,
        '510': 1,
        '718': 15,
        '725': 16,
        '555': 4,
        '562': 1,
        '220': 3,
        '518': 4,
    }
    region_datasets = {'503', '507', '508', '220'}
    dataset_jsons = {}
    datasets = {}
    for dataset_id, count in dataset_counts.items():
        labels = {'background': 0}
        for index in range(1, count + 1):
            value = [index]
            if dataset_id in region_datasets and index == 1:
                value = list(range(1, count + 1))
            labels[f'region-{index}'] = value
        dataset_json = {
            'labels': labels,
            'regions_class_order': list(range(1, count + 1)),
        }
        dataset_jsons[dataset_id] = dataset_json
        datasets[dataset_id] = SimpleNamespace(
            plans={},
            dataset_json=dataset_json,
            region_values=foreground_region_values(dataset_json),
        )
    registry = CanonicalRegionRegistry(dataset_jsons, {})

    contract = build_dataset_output_contract(registry, datasets)

    assert len(registry) == 52
    assert contract['num_packed_output_channels'] == 60
    assert {
        dataset_id
        for dataset_id, dataset_contract in contract['datasets'].items()
        if dataset_contract['prediction_mode'] == SIGMOID_REGIONS
    } == region_datasets
    softmax_contracts = [
        dataset_contract
        for dataset_contract in contract['datasets'].values()
        if dataset_contract['prediction_mode'] == SOFTMAX_LABELS
    ]
    assert [contract['packed_output_rows'][0] for contract in softmax_contracts] == (
        list(range(52, 60))
    )
    assert all(
        contract['packed_output_rows'][1:]
        == contract['canonical_foreground_rows']
        for contract in softmax_contracts
    )


def test_universal_spad_plan_declares_canonical_output_network():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
    )

    assert architecture['network_class_name'] == UNIVERSAL_SPAD_NETWORK_CLASS_NAME
    kwargs = architecture['arch_kwargs']
    assert kwargs['num_canonical_regions'] == 41
    assert kwargs['n_stages'] == UNIVERSAL_SPAD_N_STAGES
    assert kwargs['features_per_stage'] == UNIVERSAL_SPAD_FEATURES
    assert kwargs['n_blocks_per_stage'] == UNIVERSAL_SPAD_BLOCKS
    assert kwargs['n_conv_per_stage_decoder'] == UNIVERSAL_SPAD_DECODER_CONVS
    assert kwargs['min_bottleneck'] == UNIVERSAL_SPAD_MIN_BOTTLENECK
    assert kwargs['num_task_datasets'] == 12
    assert kwargs['tab_tokens_per_dataset'] == 16
    assert kwargs['tab_feature_level_indices'] == (
        UNIVERSAL_SPAD_TAB_FEATURE_LEVEL_INDICES
    )
    assert kwargs['learnable_kernel_reduction'] is False
    assert kwargs['full_kernel_dynamic_stride'] is False
    assert 'num_packed_output_channels' not in kwargs


def test_universal_spad_plan_can_declare_packed_output_bank():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=52,
        num_task_datasets=12,
        num_packed_output_channels=60,
    )

    assert architecture['arch_kwargs']['num_canonical_regions'] == 52
    assert architecture['arch_kwargs']['num_packed_output_channels'] == 60


def test_universal_spad_plan_can_enable_lkr_explicitly():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
        learnable_kernel_reduction=True,
    )

    assert architecture['arch_kwargs']['learnable_kernel_reduction'] is True


def test_universal_spad_plan_can_enable_fkds_explicitly():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
        full_kernel_dynamic_stride=True,
    )

    assert architecture['arch_kwargs']['full_kernel_dynamic_stride'] is True


@pytest.mark.parametrize('stage', SUPPORTED_FGC_STAGES)
def test_universal_spad_plan_can_enable_fgc_placement(stage):
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
        feature_grid_canonicalization_stage=stage,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='area',
    )

    assert architecture['arch_kwargs']['feature_grid_canonicalization_stage'] == stage
    assert architecture['arch_kwargs']['feature_grid_canonicalization_return'] == 'late'
    assert (
        architecture['arch_kwargs'][
            'feature_grid_canonicalization_return_downsample_mode'
        ]
        == 'area'
    )
    assert (
        architecture['arch_kwargs'][
            'feature_grid_canonicalization_return_prefilter'
        ]
        is False
    )


@pytest.mark.parametrize('downsample_mode', ['area', 'trilinear'])
def test_universal_spad_plan_can_enable_fgc_return_prefilter(downsample_mode):
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
        feature_grid_canonicalization_stage=2,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode=downsample_mode,
        feature_grid_canonicalization_return_prefilter=True,
    )

    assert (
        architecture['arch_kwargs'][
            'feature_grid_canonicalization_return_prefilter'
        ]
        is True
    )


@pytest.mark.parametrize(
    ('return_mode', 'downsample_mode', 'error'),
    [
        ('early', 'area', 'requires late return'),
        ('early', 'trilinear', 'requires late return'),
    ],
)
def test_universal_spad_plan_rejects_incompatible_return_prefilter(
    return_mode,
    downsample_mode,
    error,
):
    with pytest.raises(ValueError, match=error):
        build_universal_spad_architecture(
            _universal_source_architecture(),
            inference_da=1,
            num_canonical_regions=41,
            num_task_datasets=12,
            feature_grid_canonicalization_stage=2,
            feature_grid_canonicalization_return=return_mode,
            feature_grid_canonicalization_return_downsample_mode=(
                downsample_mode
            ),
            feature_grid_canonicalization_return_prefilter=True,
        )


def test_universal_spad_plan_defaults_to_area_return_downsampling():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
        feature_grid_canonicalization_stage=2,
    )

    assert (
        architecture['arch_kwargs']['feature_grid_canonicalization_return']
        == 'post_upconv'
    )
    assert (
        architecture['arch_kwargs'][
            'feature_grid_canonicalization_return_downsample_mode'
        ]
        == 'area'
    )


def test_universal_spad_plan_can_request_trilinear_return_downsampling():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
        feature_grid_canonicalization_stage=2,
        feature_grid_canonicalization_return_downsample_mode='trilinear',
    )

    assert (
        architecture['arch_kwargs'][
            'feature_grid_canonicalization_return_downsample_mode'
        ]
        == 'trilinear'
    )


def test_universal_spad_plan_rejects_legacy_return_outside_s2():
    with pytest.raises(ValueError, match='legacy FGC@S2'):
        build_universal_spad_architecture(
            _universal_source_architecture(),
            inference_da=1,
            num_canonical_regions=41,
            num_task_datasets=12,
            feature_grid_canonicalization_stage=1,
            feature_grid_canonicalization_return='post_upconv',
        )


def test_universal_spad_plan_preserves_legacy_seven_stage_topology():
    architecture = build_universal_spad_architecture(
        _universal_source_architecture(LEGACY_UNIVERSAL_SPAD_N_STAGES),
        inference_da=1,
        num_canonical_regions=41,
        num_task_datasets=12,
    )

    kwargs = architecture['arch_kwargs']
    assert kwargs['n_stages'] == LEGACY_UNIVERSAL_SPAD_N_STAGES
    assert kwargs['features_per_stage'] == LEGACY_UNIVERSAL_SPAD_FEATURES
    assert kwargs['n_blocks_per_stage'] == LEGACY_UNIVERSAL_SPAD_BLOCKS
    assert kwargs['n_conv_per_stage_decoder'] == (
        LEGACY_UNIVERSAL_SPAD_DECODER_CONVS
    )
    assert kwargs['tab_feature_level_indices'] == (4, 5, 6)


def test_universal_spad_selects_multiscale_tab_without_changing_legacy_default():
    common_kwargs = {
        'input_channels': 1,
        'num_classes': 1,
        'num_canonical_regions': 2,
        'n_stages': 3,
        'features_per_stage': (4, 8, 16),
        'n_blocks_per_stage': (1, 1, 1),
        'n_conv_per_stage_decoder': (1, 1),
        'inference_da': 0,
        'deep_supervision': False,
        **_tab_kwargs(),
    }
    legacy = UniversalSPADResEncUNet(**common_kwargs)
    multiscale = UniversalSPADResEncUNet(
        **common_kwargs,
        tab_feature_level_indices=(0, 1, 2),
    )

    assert isinstance(
        legacy.decoder.task_aware_bottleneck,
        TaskAwareBottleneck,
    )
    assert isinstance(
        multiscale.decoder.task_aware_bottleneck,
        MultiScaleTaskAwareBottleneck,
    )
    assert [
        projection.in_channels
        for projection in multiscale.decoder.task_aware_bottleneck.input_projections
    ] == [4, 8, 16]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='xFormers attention requires CUDA',
)
def test_universal_spad_network_keeps_native_sample_shapes_separate():
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=5,
        n_stages=3,
        features_per_stage=(4, 8, 16),
        n_blocks_per_stage=(1, 1, 1),
        n_conv_per_stage_decoder=(1, 1),
        inference_da=0,
        deep_supervision=False,
        **_tab_kwargs(),
    ).cuda()

    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = network((
            (
                torch.randn(1, 1, 16, 16, 16, device='cuda'),
                0,
                0,
                torch.tensor([0], device='cuda'),
            ),
            (
                torch.randn(1, 1, 8, 16, 16, device='cuda'),
                1,
                1,
                torch.tensor([1], device='cuda'),
            ),
        ))

    assert [tuple(output.shape) for output in outputs] == [
        (1, 5, 16, 16, 16),
        (1, 5, 8, 16, 16),
    ]


@pytest.mark.parametrize(
    ('da_encoder', 'da_decoder', 'second_output_shape'),
    [
        (0, 1, (16, 8, 8)),
        (1, 0, (8, 8, 8)),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='xFormers attention requires CUDA',
)
def test_universal_spad_network_aligns_cross_da_feature_hierarchy(
    da_encoder,
    da_decoder,
    second_output_shape,
):
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=5,
        n_stages=3,
        features_per_stage=(4, 8, 16),
        n_blocks_per_stage=(1, 1, 1),
        n_conv_per_stage_decoder=(1, 1),
        inference_da=0,
        deep_supervision=True,
        **_tab_kwargs(),
    ).cuda()

    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = network(((
            torch.randn(1, 1, 16, 16, 16, device='cuda'),
            da_encoder,
            da_decoder,
            torch.tensor([1], device='cuda'),
        ),))[0]
    sum(output.mean() for output in outputs).backward()

    assert [tuple(output.shape[2:]) for output in outputs] == [
        (16, 16, 16),
        second_output_shape,
    ]
    assert network.encoder.stem.conv.weight.grad is not None
    assert network.decoder.post_cat_convs[0].conv.weight.grad is not None
    assert network.decoder.seg_layers[0].weight.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='xFormers attention requires CUDA',
)
def test_universal_spad_fgc_s2_forward_backward():
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=5,
        n_stages=6,
        features_per_stage=(4, 8, 16, 32, 32, 32),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=2,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='area',
        tab_feature_level_indices=(4, 5),
        deep_supervision=True,
        **_tab_kwargs(),
    ).cuda()
    geometry = compute_fgc_geometry(
        np.log2(5.0),
        (32, 128, 128),
        stage=2,
    )

    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = network(((
            torch.randn(1, 1, 32, 128, 128, device='cuda'),
            geometry.route_da,
            geometry.route_da,
            torch.tensor([1], device='cuda'),
            geometry.canonical_shape,
        ),))[0]
        loss = sum(output.mean() for output in outputs)
    loss.backward()

    assert [tuple(output.shape[2:]) for output in outputs] == [
        (32, 128, 128),
        (32, 64, 64),
        (40, 32, 32),
        (20, 16, 16),
        (10, 8, 8),
    ]
    assert network.encoder.stem.conv.weight.grad is not None
    assert network.decoder.post_cat_convs[0].conv.weight.grad is not None
    assert network.decoder.seg_layers[0].weight.grad is not None


def test_universal_spad_fgc_tied_ceil_forward_uses_the_five_mm_endpoint_geometry():
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=5,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=2,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='trilinear',
        feature_grid_canonicalization_return_prefilter=True,
        tab_feature_level_indices=(4, 5),
        deep_supervision=False,
        **_tab_kwargs(),
    )
    floor_geometry = compute_fgc_geometry(np.log2(5.0), (40, 192, 192), stage=2)
    ceil_geometry = compute_fgc_geometry(
        np.log2(5.0),
        (40, 192, 192),
        stage=2,
        route_da=3,
    )
    assert floor_geometry.canonical_shape == (48, 96, 96)
    assert ceil_geometry.canonical_shape == (52, 96, 96)

    skips = network.encode_sample(
        torch.zeros(1, 1, 40, 192, 192),
        ceil_geometry.route_da,
        ceil_geometry.canonical_shape,
    )
    logits = network.forward_sample(
        torch.zeros(1, 1, 40, 192, 192),
        ceil_geometry.route_da,
        ceil_geometry.route_da,
        torch.tensor([1]),
        ceil_geometry.canonical_shape,
    )

    assert tuple(skips[1].shape[2:]) == (52, 96, 96)
    assert tuple(skips[-1].shape[2:]) == (13, 6, 6)
    assert tuple(logits.shape[2:]) == (40, 192, 192)


def test_universal_spad_fgc_s5_tied_ceil_deep_supervision_backward():
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=5,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=5,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='trilinear',
        feature_grid_canonicalization_return_prefilter=True,
        tab_feature_level_indices=(4, 5),
        deep_supervision=True,
        **_tab_kwargs(),
    )
    network.apply(network.initialize)
    geometry = compute_fgc_geometry(
        np.log2(5.0),
        (32, 128, 128),
        stage=5,
        route_da=3,
    )

    outputs = network.forward_sample(
        torch.randn(1, 1, 32, 128, 128),
        geometry.route_da,
        geometry.route_da,
        torch.tensor([1]),
        geometry.canonical_shape,
    )
    assert isinstance(outputs, list)
    assert tuple(outputs[0].shape[2:]) == (32, 128, 128)
    sum(output.square().mean() for output in outputs).backward()

    assert network.encoder.stem.conv.weight.grad is not None
    assert (
        network.decoder.spatial_decoder.return_prefilter.depthwise.weight.grad
        is not None
    )
    assert network.decoder.task_aware_bottleneck.task_tokens.grad is not None


def test_spad_feature_alignment_uses_common_fov_cell_grid():
    source = torch.tensor([0.0, 1.0]).reshape(1, 1, 2, 1, 1)

    aligned = align_spad_feature(source, (4, 1, 1))

    torch.testing.assert_close(
        aligned[:, :, :, 0, 0],
        torch.tensor([[[0.0, 0.25, 0.75, 1.0]]])
    )
    assert align_spad_feature(source, source.shape[2:]) is source


def test_spad_feature_alignment_uses_area_for_downsampling():
    source = torch.arange(4.0).reshape(1, 1, 4, 1, 1)

    aligned = align_spad_feature(source, (2, 1, 1), mode='area')

    torch.testing.assert_close(
        aligned[:, :, :, 0, 0],
        torch.tensor([[[0.5, 2.5]]]),
    )


def test_spad_feature_area_alignment_rejects_upsampling():
    source = torch.ones(1, 1, 2, 1, 1)

    with pytest.raises(ValueError, match='area alignment requires downsampling'):
        align_spad_feature(source, (4, 1, 1), mode='area')


def test_fgc_return_prefilter_starts_as_identity_and_keeps_identity_grad():
    prefilter = FGCReturnPrefilter(2)
    prefilter.depthwise.weight.data.fill_(1)
    prefilter.apply(UniversalSPADResEncUNet.initialize)
    source = torch.randn(1, 2, 4, 2, 2, requires_grad=True)

    identity = prefilter(source, source.shape[2:])

    torch.testing.assert_close(identity, source, rtol=0, atol=0)
    identity.sum().backward()
    torch.testing.assert_close(
        prefilter.depthwise.weight.grad,
        torch.zeros_like(prefilter.depthwise.weight),
        rtol=0,
        atol=0,
    )


def test_fgc_return_prefilter_learns_before_trilinear_downsampling():
    prefilter = FGCReturnPrefilter(2)
    source = torch.randn(1, 2, 4, 2, 2, requires_grad=True)

    filtered = prefilter(source, (2, 2, 2))
    aligned = align_spad_feature(filtered, (2, 2, 2), mode='trilinear')

    torch.testing.assert_close(
        aligned,
        align_spad_feature(source, (2, 2, 2), mode='trilinear'),
        rtol=0,
        atol=0,
    )
    aligned.square().mean().backward()
    assert prefilter.depthwise.weight.grad is not None
    assert torch.count_nonzero(prefilter.depthwise.weight.grad) > 0


def test_spad_feature_alignment_preserves_source_dtype(monkeypatch):
    source = torch.ones(1, 1, 2, 1, 1, dtype=torch.float16)
    monkeypatch.setattr(
        spad_architecture_module.F,
        'interpolate',
        lambda *args, **kwargs: torch.ones(1, 1, 4, 1, 1, dtype=torch.float32),
    )

    aligned = align_spad_feature(source, (4, 1, 1))

    assert aligned.dtype == source.dtype


def test_universal_spad_network_rejects_nonadjacent_cross_da():
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=2,
        n_stages=3,
        features_per_stage=(4, 8, 16),
        n_blocks_per_stage=(1, 1, 1),
        n_conv_per_stage_decoder=(1, 1),
        inference_da=0,
        deep_supervision=False,
        **_tab_kwargs(),
    )

    with pytest.raises(ValueError, match='equal or adjacent'):
        network(((
            torch.randn(1, 1, 16, 16, 16),
            0,
            2,
            torch.tensor([0]),
        ),))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='xFormers attention requires CUDA',
)
def test_seven_stage_network_executes_size_clamped_deepest_stages():
    network = UniversalSPADResEncUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=5,
        n_stages=7,
        features_per_stage=(4, 8, 12, 16, 16, 16, 16),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
        inference_da=0,
        min_bottleneck=4,
        deep_supervision=True,
        **_tab_kwargs(),
    ).cuda()

    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = network(((
            torch.randn(1, 1, 32, 32, 32, device='cuda'),
            0,
            0,
            torch.tensor([0], device='cuda'),
        ),))[0]
        target = torch.randint(
            0,
            2,
            (1, 5, 32, 32, 32),
            device='cuda',
        ).float()
        loss = UniversalPartialLabelLoss(
            batch_dice=False,
            keep_last_for_ddp=True,
        ).sample_loss(outputs, target)
    loss.backward()

    assert [tuple(output.shape[2:]) for output in outputs] == [
        (32, 32, 32),
        (16, 16, 16),
        (8, 8, 8),
        (4, 4, 4),
        (4, 4, 4),
        (4, 4, 4),
    ]
    assert tuple(dict(network.named_children())) == ('encoder', 'decoder')
    assert not any(
        name.endswith('conv_fallback') for name, _ in network.named_modules()
    )
    assert all(layer.weight.grad is not None for layer in network.decoder.seg_layers)


def test_spad_collator_does_not_stack_native_grids():
    samples = (
        UniversalSample(
            '503',
            torch.zeros(1, 1, 8, 8, 8),
            torch.zeros(1, 2, 8, 8, 8),
            torch.tensor([0, 1]),
        ),
        UniversalSample(
            '506',
            torch.zeros(1, 1, 4, 8, 8),
            torch.zeros(1, 1, 4, 8, 8),
            torch.tensor([2]),
        ),
    )

    batch = collate_spad_universal_samples(samples)

    assert batch == {'samples': samples}


def test_spad_universal_uses_native_compile_enablement():
    assert '_do_i_compile' not in SPADUniversalTrainer.__dict__


def test_spad_universal_compiles_encoder_and_decoder_separately():
    class CompileRecorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compile_kwargs = None

        def compile(self, **kwargs):
            self.compile_kwargs = kwargs

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = CompileRecorder()
            self.decoder = CompileRecorder()

    trainer = object.__new__(SPADUniversalTrainer)
    network = Network()

    compiled = trainer._compile_network(network)

    assert compiled is network
    assert network.encoder.compile_kwargs == {
        'dynamic': False,
        'mode': 'default',
    }
    assert network.decoder.compile_kwargs == {
        'dynamic': False,
        'mode': 'default',
    }


def test_spad_universal_ddp_loss_keeps_lowest_resolution_head_connected():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.is_ddp = True
    trainer.registry = tuple(range(52))

    assert trainer._build_loss().keep_last_for_ddp is True


def test_spad_universal_partial_group_uses_dynamic_region_count(monkeypatch):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.is_ddp = True
    trainer.registry = tuple(range(52))
    trainer.loss_normalization = (
        spad_universal_module.REGION_BALANCED_LOSS_NORMALIZATION
    )
    trainer.global_batch_size = 4
    trainer.num_active_regions_per_global_batch = None
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)

    loss = trainer._build_loss()

    assert loss.global_active_region_count is None
    assert loss.ddp_world_size == 1


def test_spad_universal_full_group_preserves_fixed_region_count(monkeypatch):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.is_ddp = True
    trainer.registry = tuple(range(52))
    trainer.loss_normalization = (
        spad_universal_module.REGION_BALANCED_LOSS_NORMALIZATION
    )
    trainer.global_batch_size = 12
    trainer.num_active_regions_per_global_batch = 52
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)

    loss = trainer._build_loss()

    assert loss.global_active_region_count == 52
    assert loss.ddp_world_size == 4


def test_spad_universal_static_component_paths_cover_floor_and_ceil():
    class Dataset:
        def __init__(self, patch_size):
            self.patch_size = patch_size

    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {
        'a': Dataset((8, 16, 16)),
        'b': Dataset((8, 16, 16)),
        'c': Dataset((12, 16, 16)),
    }
    trainer.continuous_da = {'a': 0.25, 'b': 0.75, 'c': 1.0}
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = False

    assert trainer._static_component_paths() == (
        ((8, 16, 16), 0),
        ((8, 16, 16), 1),
        ((12, 16, 16), 1),
    )


def test_spad_universal_cross_da_component_paths_do_not_form_endpoint_product():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {'a': type('Dataset', (), {'patch_size': (8, 16, 16)})()}
    trainer.continuous_da = {'a': 0.25}
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = True

    assert trainer._static_component_paths() == (
        ((8, 16, 16), 0),
        ((8, 16, 16), 1),
    )


def test_spad_universal_static_paths_use_sample_native_case_geometry():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {
        'a': SimpleNamespace(
            patch_size=(80, 192, 192),
            case_geometries={
                'thin': UniversalCaseGeometry(
                    spacing=(5.0, 1.0, 1.0),
                    patch_size=(40, 192, 192),
                    continuous_da=np.log2(5.0),
                ),
                'isotropic': UniversalCaseGeometry(
                    spacing=(1.0, 1.0, 1.0),
                    patch_size=(192, 192, 192),
                    continuous_da=0.0,
                ),
            },
        ),
    }
    trainer.continuous_da = {'a': 1.0}
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = True

    assert trainer._static_component_paths() == (
        ((40, 192, 192), 2),
        ((40, 192, 192), 3),
        ((192, 192, 192), 0),
    )


def test_spad_universal_warmup_ranking_is_dataset_balanced_and_deduplicates_integer_da():
    class Dataset:
        def __init__(self, geometries):
            self.case_geometries = geometries
            self.training_identifiers = tuple(geometries)

        def geometry_for_case(self, case_id):
            return self.case_geometries[case_id]

    fractional_geometry = UniversalCaseGeometry(
        spacing=(2 ** 0.25, 1.0, 1.0),
        patch_size=(8, 16, 16),
        continuous_da=0.25,
    )
    integer_geometry = UniversalCaseGeometry(
        spacing=(2.0, 1.0, 1.0),
        patch_size=(12, 16, 16),
        continuous_da=1.0,
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {
        'many': Dataset({
            'many_0': fractional_geometry,
            'many_1': fractional_geometry,
        }),
        'one': Dataset({'one_0': integer_geometry}),
    }
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = False
    trainer.feature_grid_canonicalization_stage = None

    paths = trainer._ranked_compile_warmup_paths()

    assert [
        (path.patch_size, path.da_encoder, path.da_decoder)
        for path in paths
    ] == [
        ((12, 16, 16), 1, 1),
        ((8, 16, 16), 0, 0),
        ((8, 16, 16), 1, 1),
    ]
    assert [path.dataset_id for path in paths] == ['one', 'many', 'many']
    assert [path.exposure for path in paths] == pytest.approx([
        0.5,
        0.375,
        0.125,
    ])


def test_spad_universal_warmup_ranking_keeps_every_reachable_path():
    geometries = {
        f'case_{index}': UniversalCaseGeometry(
            spacing=(1.0, 1.0, 1.0),
            patch_size=(4 + index, 16, 16),
            continuous_da=0.0,
        )
        for index in range(20)
    }
    dataset = SimpleNamespace(
        training_identifiers=tuple(geometries),
        geometry_for_case=lambda case_id: geometries[case_id],
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {'503': dataset}
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = False
    trainer.feature_grid_canonicalization_stage = None

    paths = trainer._ranked_compile_warmup_paths()

    assert len(paths) == 20
    assert {path.patch_size for path in paths} == {
        geometry.patch_size for geometry in geometries.values()
    }
    assert sum(path.exposure for path in paths) == pytest.approx(1)


def test_spad_universal_warmup_paths_bind_each_fgc_route_canonical_shape():
    geometry = UniversalCaseGeometry(
        spacing=(5.0, 1.0, 1.0),
        patch_size=(40, 192, 192),
        continuous_da=np.log2(5.0),
    )
    dataset = SimpleNamespace(
        training_identifiers=('case',),
        geometry_for_case=lambda case_id: geometry,
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {'503': dataset}
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = False
    trainer.feature_grid_canonicalization_stage = 2
    trainer.spad_n_stages = 6

    paths = trainer._ranked_compile_warmup_paths()

    assert [
        (path.da_encoder, path.da_decoder, path.canonical_shape)
        for path in paths
    ] == [
        (2, 2, (48, 96, 96)),
        (3, 3, (52, 96, 96)),
    ]


def test_spad_universal_validation_warmup_uses_validation_cases_and_floor_route():
    geometries = {
        'train': UniversalCaseGeometry(
            spacing=(1.0, 1.0, 1.0),
            patch_size=(16, 16, 16),
            continuous_da=0.0,
        ),
        'validation': UniversalCaseGeometry(
            spacing=(2 ** 1.75, 1.0, 1.0),
            patch_size=(8, 16, 16),
            continuous_da=1.75,
        ),
    }
    dataset = SimpleNamespace(
        training_identifiers=('train',),
        validation_identifiers=('validation',),
        geometry_for_case=lambda case_id: geometries[case_id],
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {'503': dataset}
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = False
    trainer.feature_grid_canonicalization_stage = None

    (path,) = trainer._ranked_compile_warmup_paths(validation=True)

    assert path.patch_size == (8, 16, 16)
    assert (path.da_encoder, path.da_decoder) == (1, 1)
    assert path.exposure == 1


@pytest.mark.parametrize(
    ('cache_was_missing', 'expected_archive_calls'),
    [(False, 0), (True, 1)],
)
def test_spad_initialize_always_warms_and_only_snapshots_missing_cache(
    monkeypatch,
    cache_was_missing,
    expected_archive_calls,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer._compile_cache_needs_archive = cache_was_missing
    trainer._do_i_compile = lambda: True
    trainer._static_component_paths = lambda: (((4, 4, 4), 0),)
    trainer._ranked_compile_warmup_paths = (
        lambda validation=False: ('validation',) if validation else ('training',)
    )
    warmed = []
    trainer._warmup_compile_paths = lambda paths: warmed.append(('train', paths))
    trainer._warmup_validation_compile_paths = (
        lambda paths: warmed.append(('validation', paths))
    )
    trainer.print_to_log_file = lambda message: None
    archive_calls = []
    trainer.archive_compile_cache_now = lambda: archive_calls.append(True)
    monkeypatch.setattr(
        spad_universal_module.CompileCacheMixin,
        'initialize',
        lambda self: None,
    )

    trainer.initialize()

    assert warmed == [
        ('train', ('training',)),
        ('validation', ('validation',)),
    ]
    assert len(archive_calls) == expected_archive_calls


def test_spad_refreshes_compile_cache_only_after_epoch_zero(monkeypatch):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer._do_i_compile = lambda: True
    archive_calls = []
    trainer.archive_compile_cache_now = (
        lambda **kwargs: archive_calls.append(kwargs)
    )
    monkeypatch.setattr(
        spad_universal_module.CompileCacheMixin,
        'on_epoch_end',
        lambda self: None,
    )

    trainer.current_epoch = 0
    trainer.on_epoch_end()
    trainer.current_epoch = 1
    trainer.on_epoch_end()

    assert archive_calls == [{'force': True}]


def test_spad_universal_warmup_rotates_every_path_per_rank(monkeypatch):
    class Registry:
        @staticmethod
        def indices(dataset_id, device):
            assert dataset_id == '503'
            return torch.tensor([0], device=device)

    class RecordingNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []
            self.no_sync_calls = 0
            self.no_sync_active = False
            self.backward_no_sync_states = []
            self.scale.register_hook(self._record_backward_no_sync_state)

        def _record_backward_no_sync_state(self, gradient):
            self.backward_no_sync_states.append(self.no_sync_active)
            return gradient

        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            self.no_sync_active = True
            try:
                yield
            finally:
                self.no_sync_active = False

        def forward(self, sample_inputs):
            ((data, da_encoder, da_decoder, dataset_indices),) = sample_inputs
            self.calls.append((
                tuple(data.shape[2:]),
                da_encoder,
                da_decoder,
                dataset_indices.tolist(),
            ))
            return ((data * self.scale,),)

    network = RecordingNetwork()
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network
    trainer.optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    trainer.loss = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=(
            spad_universal_module.REGION_BALANCED_LOSS_NORMALIZATION
        ),
    )
    trainer.registry = Registry()
    trainer.dataset_index_by_id = {'503': 0}
    trainer.num_input_channels = 1
    trainer.sample_native_z = True
    trainer.feature_grid_canonicalization_stage = None
    trainer.is_ddp = True
    trainer.global_rank = 3
    paths = tuple(
        spad_universal_module._SPADCompileWarmupPath(
            patch_size=(4 + index, 4, 4),
            da_encoder=index,
            da_decoder=index,
            canonical_shape=None,
            dataset_id='503',
            continuous_da=float(index),
            exposure=1 / 16,
        )
        for index in range(16)
    )
    barriers = []
    monkeypatch.setenv('LOCAL_RANK', '3')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '8')
    monkeypatch.setattr(spad_universal_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(spad_universal_module.dist, 'get_rank', lambda: 3)
    monkeypatch.setattr(spad_universal_module.dist, 'get_world_size', lambda: 8)
    monkeypatch.setattr(
        spad_universal_module.dist,
        'barrier',
        lambda: barriers.append(True),
    )
    monkeypatch.setattr(
        spad_universal_module.dist,
        'all_reduce',
        lambda *args, **kwargs: pytest.fail(
            'compile warmup must not synchronize dynamic loss normalization'
        ),
    )
    before = network.scale.detach().clone()

    trainer._warmup_compile_paths(paths)

    expected_indices = [*range(3, 16), *range(3)]
    assert network.calls == [
        ((4 + index, 4, 4), index, index, [0])
        for index in expected_indices
    ]
    assert network.no_sync_calls == 16
    assert network.backward_no_sync_states == [True] * 16
    assert barriers == [True] * 16
    assert network.scale.grad is None
    assert torch.equal(network.scale, before)


def test_spad_universal_validation_warmup_is_sharded_no_grad_eval(monkeypatch):
    class Registry:
        @staticmethod
        def indices(dataset_id, device):
            return torch.tensor([0], device=device)

    class RecordingNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, sample_inputs):
            ((data, da_encoder, da_decoder, dataset_indices),) = sample_inputs
            self.calls.append((
                tuple(data.shape[2:]),
                da_encoder,
                da_decoder,
                dataset_indices.tolist(),
                torch.is_grad_enabled(),
                self.training,
            ))
            return ((data * self.scale,),)

    network = RecordingNetwork()
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network
    trainer.registry = Registry()
    trainer.dataset_index_by_id = {'503': 0}
    trainer.num_input_channels = 1
    trainer.sample_native_z = True
    trainer.feature_grid_canonicalization_stage = None
    trainer.global_rank = 2
    paths = tuple(
        spad_universal_module._SPADCompileWarmupPath(
            patch_size=(4 + index, 4, 4),
            da_encoder=index,
            da_decoder=index,
            canonical_shape=None,
            dataset_id='503',
            continuous_da=float(index),
            exposure=1 / 8,
        )
        for index in range(8)
    )
    barriers = []
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '4')
    monkeypatch.setattr(spad_universal_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(spad_universal_module.dist, 'get_rank', lambda: 2)
    monkeypatch.setattr(spad_universal_module.dist, 'get_world_size', lambda: 4)
    monkeypatch.setattr(
        spad_universal_module.dist,
        'barrier',
        lambda: barriers.append(True),
    )

    trainer._warmup_validation_compile_paths(paths)

    expected_indices = [*range(2, 8), *range(2)]
    assert network.calls == [
        (
            (4 + index, 4, 4),
            index,
            index,
            [0],
            False,
            False,
        )
        for index in expected_indices
    ]
    assert barriers == [True] * 8
    assert network.training
    assert network.scale.grad is None


@pytest.mark.parametrize(
    ('stage', 'canonical_shape'),
    [
        (1, (48, 192, 192)),
        (2, (48, 96, 96)),
        (3, (48, 48, 48)),
        (4, (24, 24, 24)),
        (5, (12, 12, 12)),
    ],
)
def test_spad_universal_fgc_static_paths_include_canonical_shape(
    stage,
    canonical_shape,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {
        'a': SimpleNamespace(
            patch_size=(40, 192, 192),
            case_geometries={
                'thick': UniversalCaseGeometry(
                    spacing=(5.0, 1.0, 1.0),
                    patch_size=(40, 192, 192),
                    continuous_da=np.log2(5.0),
                ),
            },
        ),
    }
    trainer.continuous_da = {'a': np.log2(5.0)}
    trainer.cross_da = False
    trainer.da_discretization = 'floor'
    trainer.feature_grid_canonicalization_stage = stage
    trainer.feature_grid_canonicalization_return = 'late'
    trainer.spad_n_stages = 6

    assert trainer._static_component_paths() == ((
        (40, 192, 192),
        2,
        canonical_shape,
    ),)


def test_spad_universal_counts_every_static_path():
    dataset_contract = {
        '503': ((192, 192, 192), (1.0, 0.767578125, 0.767578125)),
        '506': ((112, 256, 256), (1.244979977607727, 0.78515625, 0.78515625)),
        '507': ((56, 320, 256), (2.5, 0.8027340173721313, 0.8027340173721313)),
        '508': ((80, 320, 256), (1.5, 0.7988280057907104, 0.7988280057907104)),
        '509': ((80, 256, 256), (1.6000100374221802, 0.7929689884185791, 0.7929689884185791)),
        '510': ((80, 256, 256), (3.0, 0.78125, 0.78125)),
        '517': ((80, 256, 256), (3.0, 0.7578124403953552, 0.7578124403953552)),
        '546': ((128, 256, 224), (1.0, 0.8172264993190765, 0.8172264993190765)),
        '555': ((80, 256, 256), (2.5, 0.9765620231628418, 0.9765620231628418)),
        '562': ((112, 256, 256), (1.0, 0.859375, 0.859375)),
        '564': ((192, 192, 192), (0.78126, 0.78125, 0.78125)),
        '518': ((80, 256, 256), (2.5, 0.9765625, 0.9765625)),
    }
    datasets = {
        dataset_id: SimpleNamespace(patch_size=patch_size, spacing=spacing)
        for dataset_id, (patch_size, spacing) in dataset_contract.items()
    }
    metadata = {
        'global_batch_size': 12,
        'samples_per_dataset': 1,
        'sampling_rule': (
            spad_universal_module.COMPLEMENTARY_FOREGROUND_REPLAY_RULE
        ),
        'loss_normalization': 'mean_dataset_local_losses_global_samples',
    }

    validate_spad_universal_replay_metadata(metadata, 4)
    validate_spad_universal_replay_metadata(metadata, 2)
    with pytest.raises(ValueError, match='not divisible'):
        validate_spad_universal_replay_metadata(metadata, 5)
    validate_spad_universal_replay_metadata(
        {**metadata, 'global_batch_size': 4},
        4,
    )
    validate_spad_universal_replay_metadata(
        {**metadata, 'global_batch_size': 8},
        4,
    )
    with pytest.raises(ValueError, match='not divisible'):
        validate_spad_universal_replay_metadata(
            {**metadata, 'global_batch_size': 4},
            3,
        )

    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = datasets
    trainer.continuous_da = {
        dataset_id: spad_universal_module.compute_continuous_da(dataset.spacing)
        for dataset_id, dataset in datasets.items()
    }
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = True
    # Any rank can draw any dataset, so one path set covers the whole suite.
    assert len(trainer._static_component_paths()) == 12


def test_spad_universal_compile_cache_paths_are_archive_specific(
    tmp_path,
    monkeypatch,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path / 'results')
    cache_root = tmp_path / 'cache'
    monkeypatch.setenv('SPAD_UNIVERSAL_COMPILE_CACHE_ROOT', str(cache_root))
    monkeypatch.delenv('SPAD_UNIVERSAL_COMPILE_CACHE_ARCHIVE', raising=False)

    archive, cache_dir = trainer._compile_cache_paths()

    assert archive == tmp_path / 'results' / 'torchinductor-runtime-cache.tar.zst'
    assert cache_dir.parent == cache_root
    assert len(cache_dir.name) == 16


def test_spad_universal_train_step_uses_one_outer_forward_and_one_backward(monkeypatch):
    class RecordingNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv3d(1, 4, 1)
            self.calls = []

        def forward(self, sample_inputs):
            self.calls.append(tuple(
                (tuple(data.shape), da_encoder, da_decoder)
                for data, da_encoder, da_decoder, _ in sample_inputs
            ))
            return tuple(
                [self.head(data), self.head(data)[:, :, ::2, ::2, ::2]]
                for data, _, _, _ in sample_inputs
            )

    network = RecordingNetwork()
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network
    trainer.optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    trainer.grad_scaler = None
    trainer.loss = UniversalPartialLabelLoss(batch_dice=False)
    trainer.continuous_da = {'503': 0.0, '506': 0.0}
    trainer.sample_native_z = True
    trainer.da_discretization = 'stochastic'
    trainer.cross_da = False
    trainer.dataset_index_by_id = {'503': 0, '506': 1}
    monkeypatch.setattr(
        'pumit.spad_unet.experiments.spad_universal.select_da_pair',
        lambda continuous_da, **kwargs: (
            (0, 0) if continuous_da < 1 else (2, 2)
        ),
    )
    samples = (
        UniversalSample(
            '503',
            torch.randn(1, 1, 8, 8, 8),
            torch.randint(0, 2, (1, 2, 8, 8, 8)).float(),
            torch.tensor([0, 1]),
            0.25,
        ),
        UniversalSample(
            '506',
            torch.randn(1, 1, 4, 8, 8),
            torch.randint(0, 2, (1, 1, 4, 8, 8)).float(),
            torch.tensor([3]),
            1.75,
        ),
    )
    before = network.head.weight.detach().clone()

    result = trainer.train_step({'samples': samples})

    assert result['loss'].ndim == 0
    assert network.calls == [(
        ((1, 1, 8, 8, 8), 0, 0),
        ((1, 1, 4, 8, 8), 2, 2),
    )]
    assert not torch.equal(before, network.head.weight)


def test_spad_universal_stochastic_validation_uses_floor_route():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.da_discretization = 'stochastic'

    assert trainer._validation_da_pair(1.75) == (1, 1)


@pytest.mark.parametrize(
    ('stage', 'canonical_shape'),
    [
        (1, (40, 128, 128)),
        (2, (40, 64, 64)),
        (3, (40, 32, 32)),
    ],
)
def test_spad_universal_fgc_sample_input_uses_floor_and_canonical_shape(
    stage,
    canonical_shape,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.sample_native_z = True
    trainer.cross_da = False
    trainer.da_discretization = 'floor'
    trainer.feature_grid_canonicalization_stage = stage
    trainer.feature_grid_canonicalization_return = 'late'
    trainer.spad_n_stages = 6
    trainer.dataset_index_by_id = {'503': 0}
    sample = UniversalSample(
        '503',
        torch.randn(1, 1, 32, 128, 128),
        torch.zeros(1, 1, 32, 128, 128),
        torch.tensor([0]),
        np.log2(5.0),
    )

    sample_input = trainer._network_input_for_sample(
        sample,
        trainer._select_da_pair(sample.continuous_da),
    )

    assert sample_input[1:3] == (2, 2)
    assert sample_input[4] == canonical_shape


def test_spad_universal_native_output_sample_input_passes_packed_rows():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.feature_grid_canonicalization_stage = None
    trainer.dataset_index_by_id = {'503': 0}
    trainer.packed_output_rows_by_dataset = {
        '503': torch.tensor([52, 0, 1]),
    }
    sample = UniversalSample(
        '503',
        torch.randn(1, 1, 8, 8, 8),
        torch.zeros(1, 2, 8, 8, 8),
        torch.tensor([0, 1]),
        0.0,
    )

    sample_input = trainer._network_input_for_sample(sample, (0, 0))

    assert len(sample_input) == 6
    assert sample_input[4] is None
    assert torch.equal(sample_input[5], torch.tensor([52, 0, 1]))


def test_spad_universal_disables_unused_parameter_detection():
    trainer = object.__new__(SPADUniversalTrainer)

    assert trainer._get_ddp_kwargs() == {'find_unused_parameters': False}


def test_spad_universal_train_step_defines_zero_gradient_for_inactive_lkr(
    monkeypatch,
):
    class NetworkWithInactiveLKR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv3d(1, 1, 1)
            self.kernel_reduction_delta = torch.nn.Parameter(
                torch.zeros(2, 3, 3, 3)
            )

        def forward(self, sample_inputs):
            return tuple([self.head(data)] for data, _, _, _ in sample_inputs)

    network = NetworkWithInactiveLKR()
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network
    trainer.optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    trainer.grad_scaler = None
    trainer.loss = UniversalPartialLabelLoss(batch_dice=False)
    trainer.continuous_da = {'503': 0.0}
    trainer.da_discretization = 'floor'
    trainer.cross_da = False
    trainer.dataset_index_by_id = {'503': 0}
    monkeypatch.setattr(
        'pumit.spad_unet.experiments.spad_universal.select_da_pair',
        lambda continuous_da, **kwargs: (0, 0),
    )
    sample = UniversalSample(
        '503',
        torch.randn(1, 1, 4, 4, 4),
        torch.randint(0, 2, (1, 1, 4, 4, 4)).float(),
        torch.tensor([0]),
    )

    trainer.train_step({'samples': (sample,)})

    assert network.kernel_reduction_delta.grad is not None
    assert torch.count_nonzero(network.kernel_reduction_delta.grad) == 0


def test_spad_validation_scatter_maps_native_grid_metrics_to_canonical_rows():
    registry = CanonicalRegionRegistry(
        {
            '503': {'labels': {'background': 0, 'a': 1}},
            '506': {'labels': {'background': 0, 'b': 1}},
        },
        shared_regions={},
    )

    class Network(torch.nn.Module):
        def forward(self, sample_inputs):
            outputs = []
            for sample_index, (data, _, _, _) in enumerate(sample_inputs):
                logits = torch.full((1, 2, *data.shape[2:]), -10.0)
                logits[:, sample_index] = 10
                outputs.append(logits)
            return tuple(outputs)

    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = Network()
    trainer.loss = UniversalPartialLabelLoss(batch_dice=False)
    trainer.registry = registry
    trainer.continuous_da = {'503': 0.0, '506': 1.0}
    trainer.dataset_index_by_id = {'503': 0, '506': 1}
    samples = (
        UniversalSample(
            '503',
            torch.zeros(1, 1, 2, 2, 2),
            torch.ones(1, 1, 2, 2, 2),
            registry.indices('503'),
        ),
        UniversalSample(
            '506',
            torch.zeros(1, 1, 1, 2, 2),
            torch.zeros(1, 1, 1, 2, 2),
            registry.indices('506'),
        ),
    )

    result = trainer.validation_step({'samples': samples})

    assert result['tp_hard'].tolist() == [8, 0]
    assert result['fp_hard'].tolist() == [0, 4]
    assert result['fn_hard'].tolist() == [0, 0]


def test_softmax_patch_metrics_use_argmax_and_exclude_background():
    target = torch.zeros(1, 2, 1, 1, 3)
    target[:, 0, :, :, 1] = 1
    target[:, 1, :, :, 2] = 1
    logits = torch.tensor([
        [
            [[[[8.0, 0.0, 0.0]]]],
            [[[[0.0, 8.0, 1.0]]]],
            [[[[0.0, 1.0, 8.0]]]],
        ]
    ]).reshape(1, 3, 1, 1, 3)

    tp, fp, fn = spad_universal_module.aggregate_canonical_region_confusion(
        ((logits, target, torch.tensor([2, 5]), SOFTMAX_LABELS),),
        6,
    )

    assert tp.tolist() == [0, 0, 1, 0, 0, 1]
    assert torch.count_nonzero(fp) == 0
    assert torch.count_nonzero(fn) == 0


def test_softmax_source_grid_restoration_fills_cropped_area_as_background():
    logits = torch.tensor([[[[0.0]]], [[[1.0]]], [[[-1.0]]]])
    properties = {
        'spacing': [1.0, 1.0, 1.0],
        'shape_after_cropping_and_before_resampling': [1, 1, 1],
        'shape_before_cropping': [3, 3, 3],
        'bbox_used_for_cropping': [[1, 2], [1, 2], [1, 2]],
    }
    plans_manager = SimpleNamespace(
        transpose_forward=(0, 1, 2),
        transpose_backward=(0, 1, 2),
    )
    configuration_manager = SimpleNamespace(
        spacing=[1.0, 1.0, 1.0],
        resampling_fn_probabilities=(
            lambda values, *_args: values
        ),
    )

    probabilities = spad_evaluation_module.restore_region_probabilities(
        logits,
        properties,
        plans_manager,
        configuration_manager,
        SOFTMAX_LABELS,
    )

    assert probabilities.shape == (3, 3, 3, 3)
    assert probabilities[0, 0, 0, 0] == 1
    assert probabilities[1:, 0, 0, 0].sum() == 0
    assert np.isclose(probabilities[:, 1, 1, 1].sum(), 1)


def test_spad_inference_adapter_binds_dataset_da_and_rows():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received_da = None

        def forward_sample(self, data, da_encoder, da_decoder, dataset_indices):
            self.received_da = (da_encoder, da_decoder, dataset_indices.tolist())
            logits = torch.arange(
                4,
                dtype=data.dtype,
                device=data.device,
            )[None, :, None, None, None].expand(data.shape[0], 4, *data.shape[2:])
            return logits

    network = Network()
    adapter = _SPADActiveRegionNetwork(
        network,
        torch.tensor([1, 3]),
        da_encoder=1,
        da_decoder=2,
        dataset_index=7,
    )

    output = adapter(torch.zeros(1, 1, 3, 3, 3))

    assert network.received_da == (1, 2, [7])
    assert output.shape == (1, 2, 3, 3, 3)
    assert torch.equal(output[:, 0], torch.ones_like(output[:, 0]))
    assert torch.equal(output[:, 1], torch.full_like(output[:, 1], 3))


def test_spad_inference_adapter_passes_packed_rows_before_projection():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received_rows = None

        def forward_sample(
            self,
            data,
            da_encoder,
            da_decoder,
            dataset_indices,
            canonical_shape,
            packed_output_rows,
        ):
            self.received_rows = packed_output_rows
            return torch.zeros(
                data.shape[0],
                len(packed_output_rows),
                *data.shape[2:],
            )

    network = Network()
    packed_rows = torch.tensor([5, 1, 3])
    adapter = _SPADActiveRegionNetwork(
        network,
        torch.tensor([1, 3]),
        da_encoder=1,
        da_decoder=1,
        dataset_index=2,
        packed_output_rows=packed_rows,
    )

    output = adapter(torch.zeros(1, 1, 3, 3, 3))

    assert torch.equal(network.received_rows, packed_rows)
    assert output.shape == (1, 3, 3, 3, 3)


def test_spad_inference_adapter_binds_fgc_canonical_shape():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.canonical_shape = None

        def forward_sample(
            self,
            data,
            da_encoder,
            da_decoder,
            dataset_indices,
            canonical_shape,
        ):
            self.canonical_shape = canonical_shape
            return torch.zeros(data.shape[0], 2, *data.shape[2:])

    network = Network()
    adapter = _SPADActiveRegionNetwork(
        network,
        torch.tensor([0]),
        da_encoder=2,
        da_decoder=2,
        dataset_index=0,
        canonical_shape=(48, 96, 96),
    )

    adapter(torch.zeros(1, 1, 40, 192, 192))

    assert network.canonical_shape == (48, 96, 96)


def test_spad_fgc_case_inference_binds_floor_and_canonical_shape(
    monkeypatch,
    tmp_path,
):
    case_geometry = UniversalCaseGeometry(
        spacing=(5.0, 1.0, 1.0),
        patch_size=(40, 192, 192),
        continuous_da=np.log2(5.0),
    )
    dataset = SimpleNamespace(
        geometry_for_case=lambda case_id: case_geometry,
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.current_epoch = 1000
    trainer.datasets = {'503': dataset}
    trainer.sample_native_z = True
    trainer.feature_grid_canonicalization_stage = 2
    trainer.spad_n_stages = 6
    trainer.device = torch.device('cpu')
    base_configuration = SimpleNamespace(configuration={
        'spacing': [1.0, 1.0, 1.0],
        'patch_size': [192, 192, 192],
        'architecture': {},
    })
    continuous_da, case_configuration = trainer._case_inference_geometry(
        '503',
        'case',
        base_configuration,
    )
    network_calls = []
    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        lambda dataset_id, *, da_encoder, da_decoder, canonical_shape: (
            network_calls.append((
                dataset_id,
                da_encoder,
                da_decoder,
                canonical_shape,
            ))
            or object()
        ),
    )
    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        lambda **kwargs: (np.zeros((1, 2, 2, 2), dtype=np.float32),),
    )

    trainer._ensure_inference_component_probability_cache(
        '503',
        'case',
        'ff',
        continuous_da=continuous_da,
        data=torch.zeros(1, 1, 2, 2, 2),
        properties={},
        plans_manager=SimpleNamespace(),
        configuration_manager=case_configuration,
    )

    assert tuple(case_configuration.spacing) == (5.0, 1.0, 1.0)
    assert tuple(case_configuration.patch_size) == (40, 192, 192)
    assert network_calls == [('503', 2, 2, (48, 96, 96))]


def test_spad_full_volume_validation_records_inference_mode(monkeypatch):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.inference_mode = 'ceil'
    calls = []
    monkeypatch.setattr(
        spad_universal_module,
        'perform_universal_full_volume_validation',
        lambda *args, **kwargs: calls.append((args, kwargs)) or {'ok': True},
    )

    assert trainer.perform_actual_validation(True) == {'ok': True}
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (trainer, True)
    assert kwargs['output_folder_name'] == 'validation_ceil'
    assert kwargs['summary_metadata'] == {
        'inference_mode': 'ceil',
        'tile_step_size': 0.5,
        'sliding_window_batch_size': 1,
        'case_evaluation_workers': 4,
        'model_weights': 'raw',
    }
    assert kwargs['summary_log_prefix'] == 'final_val'
    assert kwargs['max_pending_case_predictions'] == 4
    assert kwargs['case_predictor'].func == trainer._predict_inference_case
    assert kwargs['case_predictor'].keywords['save_probabilities'] is True
    assert callable(
        kwargs['case_predictor'].keywords['evaluation_executor'].submit
    )


def test_spad_nondefault_tile_step_has_distinct_output_and_cache_names(
    monkeypatch,
    tmp_path,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.inference_mode = 'endpoints'
    trainer.inference_tile_step_size = 0.4
    trainer.inference_sliding_window_batch_size = 8
    calls = []
    monkeypatch.setattr(
        spad_universal_module,
        'perform_universal_full_volume_validation',
        lambda *args, **kwargs: calls.append((args, kwargs)) or {'ok': True},
    )

    assert trainer.perform_actual_validation() == {'ok': True}
    _, kwargs = calls[0]
    assert kwargs['output_folder_name'] == (
        'validation_endpoints_step0.4_batch8'
    )
    assert kwargs['summary_metadata'] == {
        'inference_mode': 'endpoints',
        'tile_step_size': 0.4,
        'sliding_window_batch_size': 8,
        'case_evaluation_workers': 4,
        'model_weights': 'raw',
    }
    assert kwargs['summary_log_prefix'] == 'final_val'
    assert trainer._inference_probability_cache_path(
        'ff',
        '503',
        'case',
    ) == (
        tmp_path
        / 'inference_probability_cache'
        / 'step0.4_batch8'
        / 'ff'
        / 'dataset-503'
        / 'case.npz'
    )


def test_spad_model_ema_validation_uses_distinct_output_and_cache_names(
    monkeypatch,
    tmp_path,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.inference_mode = 'floor'
    trainer.inference_tile_step_size = 0.5
    trainer.inference_sliding_window_batch_size = 1
    trainer.model_ema_decay = 0.9998
    trainer.print_to_log_file = lambda *args, **kwargs: None
    calls = []

    @contextmanager
    def model_ema_weights():
        yield

    monkeypatch.setattr(
        trainer,
        'model_ema_weights',
        model_ema_weights,
    )
    monkeypatch.setattr(
        spad_universal_module,
        'perform_universal_full_volume_validation',
        lambda *args, **kwargs: calls.append(kwargs) or kwargs['output_folder_name'],
    )

    assert trainer.perform_actual_validation() == {
        'raw': 'validation_floor',
        'model_ema': 'validation_floor_model_ema',
    }
    assert [call['summary_metadata']['model_weights'] for call in calls] == [
        'raw',
        'model_ema',
    ]
    assert [call['summary_log_prefix'] for call in calls] == [
        'final_val',
        'final_val/model_ema',
    ]
    assert trainer._inference_probability_cache_path('ff', '503', 'case') == (
        tmp_path / 'inference_probability_cache' / 'ff' / 'dataset-503' / 'case.npz'
    )
    trainer._inference_model_weights = 'model_ema'
    assert trainer._inference_probability_cache_path('ff', '503', 'case') == (
        tmp_path
        / 'inference_probability_cache'
        / 'model_ema'
        / 'ff'
        / 'dataset-503'
        / 'case.npz'
    )


def test_spad_validation_weights_raw_skips_ema_pass(monkeypatch, tmp_path):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.inference_mode = 'floor'
    trainer.model_ema_decay = 0.9998
    trainer.inference_validation_weights = 'raw'
    calls = []
    monkeypatch.setattr(
        spad_universal_module,
        'perform_universal_full_volume_validation',
        lambda *args, **kwargs: calls.append(kwargs) or kwargs['output_folder_name'],
    )

    assert trainer.perform_actual_validation() == 'validation_floor'
    assert [call['summary_metadata']['model_weights'] for call in calls] == ['raw']


def test_spad_validation_weights_model_ema_skips_raw_pass(monkeypatch, tmp_path):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.inference_mode = 'floor'
    trainer.model_ema_decay = 0.9998
    trainer.inference_validation_weights = 'model_ema'
    trainer.print_to_log_file = lambda *args, **kwargs: None
    calls = []

    @contextmanager
    def model_ema_weights():
        yield

    monkeypatch.setattr(trainer, 'model_ema_weights', model_ema_weights)
    monkeypatch.setattr(
        spad_universal_module,
        'perform_universal_full_volume_validation',
        lambda *args, **kwargs: calls.append(kwargs) or kwargs['output_folder_name'],
    )

    assert trainer.perform_actual_validation() == 'validation_floor_model_ema'
    assert [call['summary_metadata']['model_weights'] for call in calls] == [
        'model_ema',
    ]
    assert trainer._inference_model_weights == 'raw'


def test_spad_validation_weights_model_ema_requires_decay():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.inference_mode = 'floor'
    trainer.model_ema_decay = None
    trainer.inference_validation_weights = 'model_ema'

    with pytest.raises(ValueError, match='requires plans with a model EMA decay'):
        trainer.perform_actual_validation()


def test_spad_tta_mirroring_names_and_axes(monkeypatch, tmp_path):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.inference_mode = 'endpoints'
    trainer.inference_sliding_window_batch_size = 8
    trainer.inference_tta_mirroring = True
    trainer.inference_allowed_mirroring_axes = (0, 1, 2)
    trainer.model_ema_decay = None
    calls = []
    monkeypatch.setattr(
        spad_universal_module,
        'perform_universal_full_volume_validation',
        lambda *args, **kwargs: calls.append(kwargs) or {'ok': True},
    )

    assert trainer.perform_actual_validation() == {'ok': True}
    assert calls[0]['output_folder_name'] == 'validation_endpoints_batch8_mirror'
    assert calls[0]['mirroring_axes'] == (0, 1, 2)
    assert trainer._inference_probability_cache_path('ff', '503', 'case') == (
        tmp_path
        / 'inference_probability_cache'
        / 'batch8_mirror'
        / 'ff'
        / 'dataset-503'
        / 'case.npz'
    )


def test_spad_tta_mirroring_requires_training_axes():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.inference_mode = 'floor'
    trainer.inference_tta_mirroring = True
    trainer.model_ema_decay = None

    with pytest.raises(ValueError, match='requires the training mirror axes'):
        trainer.perform_actual_validation()


def test_spad_inference_probability_cache_saves_fp16_and_loads_fp32(tmp_path):
    cache_path = tmp_path / 'probabilities.npz'
    probabilities = np.array([0.123456, 0.654321], dtype=np.float32)

    SPADUniversalTrainer._save_inference_probabilities(
        cache_path,
        probabilities,
        1000,
    )

    with np.load(cache_path, allow_pickle=False) as artifact:
        saved_probabilities = artifact['probabilities']
        assert saved_probabilities.dtype == np.float16
        np.testing.assert_array_equal(
            saved_probabilities,
            probabilities.astype(np.float16),
        )
    assert SPADUniversalTrainer._cached_probabilities_match(cache_path, 1000)
    assert not SPADUniversalTrainer._cached_probabilities_match(cache_path, 1001)
    loaded_probabilities = SPADUniversalTrainer._load_inference_probabilities(
        cache_path
    )
    assert loaded_probabilities.dtype == np.float32
    np.testing.assert_array_equal(
        loaded_probabilities,
        probabilities.astype(np.float16).astype(np.float32),
    )


def test_spad_fresh_component_payload_writes_cache_and_matches_load_path(
    monkeypatch,
    tmp_path,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.current_epoch = 1000
    monkeypatch.setattr(
        trainer,
        '_dataset_prediction_mode',
        lambda dataset_id: 'sigmoid_regions',
        raising=False,
    )
    captured = []
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: captured.append(kwargs['probabilities']) or {'ok': True},
    )
    cache_path = tmp_path / 'ff' / 'case.npz'
    payload = np.random.default_rng(0).random((2, 3, 3, 3)).astype(np.float32)
    common = dict(
        weighted_components=(('ff', 1.0),),
        properties={},
        plans_manager=SimpleNamespace(),
        dataset=SimpleNamespace(dataset_id='503'),
        case_id='case',
        output_folder=tmp_path / 'out',
        save_probabilities=False,
    )

    fresh = trainer._evaluate_inference_component_caches(
        component_cache_paths={'ff': cache_path},
        component_probabilities={'ff': payload.astype(np.float16)},
        **common,
    )
    assert fresh == {'ok': True}
    assert SPADUniversalTrainer._cached_probabilities_match(cache_path, 1000)

    cached = trainer._evaluate_inference_component_caches(
        component_cache_paths={'ff': cache_path},
        **common,
    )
    assert cached == {'ok': True}
    np.testing.assert_array_equal(captured[0], captured[1])
    assert captured[0].dtype == captured[1].dtype == np.float32


def test_spad_sample_native_full_volume_uses_case_geometry(
    monkeypatch,
    tmp_path,
):
    geometry = UniversalCaseGeometry(
        spacing=(2.5, 1.0, 1.0),
        patch_size=(80, 192, 192),
        continuous_da=np.log2(2.5),
    )
    geometry_calls = []

    def geometry_for_case(case_id):
        geometry_calls.append(case_id)
        return geometry

    dataset = SimpleNamespace(
        dataset_id='503',
        region_names=('organ',),
        geometry_for_case=geometry_for_case,
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.datasets = {'503': dataset}
    trainer.continuous_da = {'503': 0.0}
    trainer.sample_native_z = True
    trainer.inference_mode = 'cross'
    trainer.inference_tile_step_size = 0.4
    trainer.inference_sliding_window_batch_size = 8
    trainer.configuration_name = '3d_fullres'
    trainer.device = torch.device('cpu')
    trainer.current_epoch = 1000

    base_configuration = {
        'architecture': {},
        'spacing': [1.0, 1.0, 1.0],
        'patch_size': [192, 192, 192],
    }
    monkeypatch.setattr(
        spad_universal_module,
        'load_universal_dataset_case',
        lambda **kwargs: (
            torch.zeros(1, 1, 2, 2, 2),
            {},
            SimpleNamespace(),
            SimpleNamespace(configuration=base_configuration),
        ),
    )
    pair_calls = []
    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        lambda dataset_id, *, da_encoder, da_decoder: pair_calls.append(
            (da_encoder, da_decoder)
        ) or object(),
    )
    prediction_geometry = {}

    def predict(**kwargs):
        configuration = kwargs['configuration_manager']
        prediction_geometry.update({
            'spacing': tuple(configuration.spacing),
            'patch_size': tuple(configuration.patch_size),
            'tile_step_size': kwargs['tile_step_size'],
            'sliding_window_batch_size': kwargs[
                'sliding_window_batch_size'
            ],
        })
        return (np.zeros((1, 2, 2, 2), dtype=np.float32),)

    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        predict,
    )
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: {'ok': True},
    )

    assert trainer._predict_inference_case(
        '503',
        'case',
        tmp_path / 'cross',
        save_probabilities=False,
    ) == {'ok': True}
    assert geometry_calls == ['case']
    assert pair_calls == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert prediction_geometry == {
        'spacing': (2.5, 1.0, 1.0),
        'patch_size': (80, 192, 192),
        'tile_step_size': 0.4,
        'sliding_window_batch_size': 8,
    }
    assert base_configuration == {
        'architecture': {},
        'spacing': [1.0, 1.0, 1.0],
        'patch_size': [192, 192, 192],
    }


def test_spad_planned_z_full_volume_preserves_dataset_configuration():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.sample_native_z = False
    trainer.continuous_da = {'503': 1.25}
    configuration_manager = object()

    continuous_da, case_configuration = trainer._case_inference_geometry(
        '503',
        'case',
        configuration_manager,
    )

    assert continuous_da == 1.25
    assert case_configuration is configuration_manager


def test_spad_cross_inference_reuses_components_and_predicts_only_missing_paths(
    monkeypatch,
    tmp_path,
):
    dataset = SimpleNamespace(
        dataset_id='503',
        region_names=('organ',),
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.datasets = {'503': dataset}
    trainer.continuous_da = {'503': 1.25}
    trainer.inference_mode = 'cross'
    trainer.configuration_name = '3d_fullres'
    trainer.device = torch.device('cpu')
    trainer.current_epoch = 1000
    for component, value in (('ff', 1.0), ('fc', 2.0), ('cc', 3.0)):
        path = tmp_path / 'inference_probability_cache' / component / 'dataset-503'
        path.mkdir(parents=True)
        np.savez(
            path / 'case.npz',
            probabilities=np.full((1, 2, 2, 2), value),
            epoch=np.asarray(1000, dtype=np.int64),
        )

    monkeypatch.setattr(
        spad_universal_module,
        'load_universal_dataset_case',
        lambda **kwargs: (
            torch.zeros(1, 1, 2, 2, 2),
            {},
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    pair_calls = []
    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        lambda dataset_id, *, da_encoder, da_decoder: pair_calls.append(
            (da_encoder, da_decoder)
        ) or object(),
    )
    prediction_calls = []

    def predict(**kwargs):
        prediction_calls.append(kwargs['inference_networks'])
        return (np.full((1, 2, 2, 2), 4.001, dtype=np.float32),)

    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        predict,
    )
    captured = {}
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: captured.update(kwargs) or {'ok': True},
    )

    assert trainer._predict_inference_case(
        '503',
        'case',
        tmp_path / 'cross',
        save_probabilities=False,
    ) == {'ok': True}
    assert pair_calls == [(2, 1)]
    assert len(prediction_calls) == 1
    assert len(prediction_calls[0]) == 1
    np.testing.assert_allclose(
        captured['probabilities'],
        np.full(
            (1, 2, 2, 2),
            0.5625 * 1.0
            + 0.1875 * 2.0
            + 0.1875 * np.float16(4.001).astype(np.float32)
            + 0.0625 * 3.0,
        ),
    )
    cached_cf = (
        tmp_path
        / 'inference_probability_cache'
        / 'cf'
        / 'dataset-503'
        / 'case.npz'
    )
    with np.load(cached_cf, allow_pickle=False) as artifact:
        assert artifact['probabilities'].dtype == np.float16
        np.testing.assert_array_equal(
            artifact['probabilities'],
            np.full((1, 2, 2, 2), np.float16(4.001)),
        )
    assert list(cached_cf.parent.glob('*.tmp')) == []


def test_spad_integer_da_cross_inference_computes_one_component(monkeypatch, tmp_path):
    dataset = SimpleNamespace(dataset_id='503', region_names=('organ',))
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.datasets = {'503': dataset}
    trainer.continuous_da = {'503': 1.0}
    trainer.inference_mode = 'cross'
    trainer.configuration_name = '3d_fullres'
    trainer.device = torch.device('cpu')
    trainer.current_epoch = 1000
    monkeypatch.setattr(
        spad_universal_module,
        'load_universal_dataset_case',
        lambda **kwargs: (
            torch.zeros(1, 1, 2, 2, 2),
            {},
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    pair_calls = []
    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        lambda dataset_id, *, da_encoder, da_decoder: pair_calls.append(
            (da_encoder, da_decoder)
        ) or object(),
    )
    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        lambda **kwargs: (np.full((1, 2, 2, 2), 5.0),),
    )
    captured = {}
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: captured.update(kwargs) or {'ok': True},
    )

    assert trainer._predict_inference_case(
        '503',
        'case',
        tmp_path / 'cross',
        save_probabilities=False,
    ) == {'ok': True}
    assert pair_calls == [(1, 1)]
    np.testing.assert_array_equal(
        captured['probabilities'],
        np.full((1, 2, 2, 2), 5.0),
    )
    cache_root = tmp_path / 'inference_probability_cache'
    assert (cache_root / 'ff' / 'dataset-503' / 'case.npz').is_file()
    assert not (cache_root / 'fc').exists()
    assert not (cache_root / 'cf').exists()
    assert not (cache_root / 'cc').exists()


def test_spad_integer_da_ceil_inference_reuses_ff_cache(monkeypatch, tmp_path):
    dataset = SimpleNamespace(dataset_id='503', region_names=('organ',))
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.datasets = {'503': dataset}
    trainer.continuous_da = {'503': 1.0}
    trainer.inference_mode = 'ceil'
    trainer.configuration_name = '3d_fullres'
    trainer.device = torch.device('cpu')
    trainer.current_epoch = 1000
    ff_cache = (
        tmp_path
        / 'inference_probability_cache'
        / 'ff'
        / 'dataset-503'
        / 'case.npz'
    )
    ff_cache.parent.mkdir(parents=True)
    np.savez(
        ff_cache,
        probabilities=np.full((1, 2, 2, 2), 7.0),
        epoch=np.asarray(1000, dtype=np.int64),
    )
    monkeypatch.setattr(
        spad_universal_module,
        'load_universal_dataset_case',
        lambda **kwargs: (
            torch.zeros(1, 1, 2, 2, 2),
            {},
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        lambda *args, **kwargs: pytest.fail('integer DA must reuse ff cache'),
    )
    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        lambda **kwargs: pytest.fail('integer DA must not predict cc'),
    )
    captured = {}
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: captured.update(kwargs) or {'ok': True},
    )

    assert trainer._predict_inference_case(
        '503',
        'case',
        tmp_path / 'ceil',
        save_probabilities=False,
    ) == {'ok': True}
    np.testing.assert_array_equal(
        captured['probabilities'],
        np.full((1, 2, 2, 2), 7.0),
    )
    assert not (
        tmp_path
        / 'inference_probability_cache'
        / 'cc'
        / 'dataset-503'
        / 'case.npz'
    ).exists()


@pytest.mark.parametrize('inference_mode', ['floor', 'endpoints'])
def test_spad_floor_plans_accept_floor_and_endpoint_inference(inference_mode):
    validate_spad_universal_inference_mode(inference_mode, 'floor')


@pytest.mark.parametrize(
    'inference_mode',
    ['ceil', 'cross'],
)
def test_spad_floor_plans_still_reject_untied_route_inference(inference_mode):
    with pytest.raises(ValueError, match='floor or endpoint'):
        validate_spad_universal_inference_mode(inference_mode, 'floor')


def test_spad_inference_mode_validation_rejects_unknown_modes():
    with pytest.raises(ValueError, match='unsupported'):
        validate_spad_universal_inference_mode('ff_cc', 'stochastic')


def test_spad_stochastic_plans_accept_endpoint_inference():
    validate_spad_universal_inference_mode('endpoints', 'stochastic')


@pytest.mark.parametrize(
    ('raw_value', 'expected'),
    [
        (None, 0.5),
        ('0.5', 0.5),
        ('0.40', 0.4),
    ],
)
def test_spad_resolves_inference_tile_step_size(raw_value, expected):
    assert spad_universal_module._resolve_inference_tile_step_size(
        raw_value
    ) == expected


@pytest.mark.parametrize(
    'raw_value',
    ['abc', 'nan', 'inf', '0', '-0.1', '1.1'],
)
def test_spad_rejects_invalid_inference_tile_step_size(raw_value):
    with pytest.raises(ValueError):
        spad_universal_module._resolve_inference_tile_step_size(raw_value)


@pytest.mark.parametrize(
    ('raw_value', 'expected'),
    [(None, 1), ('1', 1), ('8', 8)],
)
def test_spad_resolves_sliding_window_batch_size(raw_value, expected):
    assert spad_universal_module._resolve_sliding_window_batch_size(
        raw_value
    ) == expected


@pytest.mark.parametrize('raw_value', ['abc', '1.5', '0', '-1'])
def test_spad_rejects_invalid_sliding_window_batch_size(raw_value):
    with pytest.raises(ValueError):
        spad_universal_module._resolve_sliding_window_batch_size(raw_value)


@pytest.mark.parametrize(
    ('raw_value', 'expected'),
    [(None, 4), ('1', 1), ('8', 8)],
)
def test_spad_resolves_case_evaluation_workers(raw_value, expected):
    assert spad_universal_module._resolve_case_evaluation_workers(
        raw_value
    ) == expected


@pytest.mark.parametrize(
    ('tile_step_size', 'batch_size', 'expected'),
    [
        (0.5, 1, 'endpoints'),
        (0.4, 1, 'endpoints_step0.4'),
        (0.375, 8, 'endpoints_step0.375_batch8'),
    ],
)
def test_spad_inference_variant_names_tile_step(
    tile_step_size,
    batch_size,
    expected,
):
    assert spad_universal_module._inference_variant_name(
        'endpoints',
        tile_step_size,
        batch_size,
    ) == expected


@pytest.mark.parametrize('tile_step_size', [0, -0.1, 1.1, float('nan'), True])
def test_spad_inference_variant_rejects_invalid_tile_step(tile_step_size):
    with pytest.raises(ValueError, match='tile_step_size'):
        spad_universal_module._inference_variant_name(
            'endpoints',
            tile_step_size,
        )


@pytest.mark.parametrize(
    ('inference_mode', 'da_discretization', 'expected_paths'),
    [
        ('floor', 'floor', (((40, 192, 192), 2, (48, 96, 96)),)),
        (
            'endpoints',
            'floor',
            (
                ((40, 192, 192), 2, (48, 96, 96)),
                ((40, 192, 192), 3, (52, 96, 96)),
            ),
        ),
        (
            'floor',
            'stochastic',
            (
                ((40, 192, 192), 2, (48, 96, 96)),
                ((40, 192, 192), 3, (52, 96, 96)),
            ),
        ),
    ],
)
def test_spad_universal_fgc_static_paths_follow_the_inference_mode(
    inference_mode,
    da_discretization,
    expected_paths,
):
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.datasets = {
        'a': SimpleNamespace(
            patch_size=(40, 192, 192),
            case_geometries={
                'thick': UniversalCaseGeometry(
                    spacing=(5.0, 1.0, 1.0),
                    patch_size=(40, 192, 192),
                    continuous_da=np.log2(5.0),
                ),
                'dyadic': UniversalCaseGeometry(
                    spacing=(4.0, 1.0, 1.0),
                    patch_size=(48, 192, 192),
                    continuous_da=2.0,
                ),
            },
        ),
    }
    trainer.continuous_da = {'a': np.log2(5.0)}
    trainer.cross_da = False
    trainer.da_discretization = da_discretization
    trainer.inference_mode = inference_mode
    trainer.feature_grid_canonicalization_stage = 2
    trainer.feature_grid_canonicalization_return = 'late'
    trainer.spad_n_stages = 6

    # Integer DA has one endpoint, so it contributes one path under either mode.
    assert trainer._static_component_paths() == (
        *expected_paths,
        ((48, 192, 192), 2, (48, 96, 96)),
    )


def test_spad_universal_stochastic_fgc_binds_the_sampled_route_geometry():
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.dataset_index_by_id = {'503': 0}
    trainer.sample_native_z = True
    trainer.feature_grid_canonicalization_stage = 2
    trainer.spad_n_stages = 6
    sample = SimpleNamespace(
        data=torch.zeros(1, 1, 40, 192, 192),
        dataset_id='503',
        continuous_da=np.log2(5.0),
    )

    floor_input = trainer._network_input_for_sample(sample, (2, 2))
    ceil_input = trainer._network_input_for_sample(sample, (3, 3))

    assert floor_input[-1] == (48, 96, 96)
    assert ceil_input[-1] == (52, 96, 96)
    with pytest.raises(ValueError, match='tied encoder and decoder DA'):
        trainer._network_input_for_sample(sample, (2, 3))


def _fgc_endpoint_trainer(
    tmp_path,
    continuous_da,
    patch_size,
    inference_mode='endpoints',
):
    case_geometry = UniversalCaseGeometry(
        spacing=(5.0, 1.0, 1.0),
        patch_size=patch_size,
        continuous_da=continuous_da,
    )
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.output_folder = str(tmp_path)
    trainer.current_epoch = 1000
    trainer.datasets = {
        '503': SimpleNamespace(
            dataset_id='503',
            region_names=('organ',),
            geometry_for_case=lambda case_id: case_geometry,
        ),
    }
    trainer.continuous_da = {'503': continuous_da}
    trainer.sample_native_z = True
    trainer.inference_mode = inference_mode
    trainer.configuration_name = '3d_fullres'
    trainer.device = torch.device('cpu')
    trainer.feature_grid_canonicalization_stage = 2
    trainer.spad_n_stages = 6
    return trainer


def test_spad_fgc_endpoint_inference_fuses_only_the_tied_endpoint_routes(
    monkeypatch,
    tmp_path,
):
    continuous_da = np.log2(5.0)
    trainer = _fgc_endpoint_trainer(
        tmp_path,
        continuous_da,
        (40, 192, 192),
        'endpoints',
    )
    monkeypatch.setattr(
        spad_universal_module,
        'load_universal_dataset_case',
        lambda **kwargs: (
            torch.zeros(1, 1, 2, 2, 2),
            {},
            SimpleNamespace(),
            SimpleNamespace(configuration={
                'architecture': {},
                'spacing': [1.0, 1.0, 1.0],
                'patch_size': [192, 192, 192],
            }),
        ),
    )
    network_calls = []
    route_by_network = {}

    def build_inference_network(dataset_id, *, da_encoder, da_decoder, canonical_shape):
        network_calls.append((da_encoder, da_decoder, canonical_shape))
        network = object()
        route_by_network[id(network)] = da_encoder
        return network

    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        build_inference_network,
    )

    def predict(**kwargs):
        (network,) = kwargs['inference_networks']
        value = 1.0 if route_by_network[id(network)] == 2 else 3.0
        return (np.full((1, 2, 2, 2), value, dtype=np.float32),)

    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        predict,
    )
    captured = {}
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: captured.update(kwargs) or {'ok': True},
    )

    assert trainer._predict_inference_case(
        '503',
        'case',
        tmp_path / 'endpoints',
        save_probabilities=False,
    ) == {'ok': True}

    assert network_calls == [
        (2, 2, (48, 96, 96)),
        (3, 3, (52, 96, 96)),
    ]
    _, _, ceil_weight = decompose_continuous_da(continuous_da)
    floor_weight = 1.0 - ceil_weight
    np.testing.assert_allclose(
        captured['probabilities'],
        np.full(
            (1, 2, 2, 2),
            floor_weight * 1.0 + ceil_weight * 3.0,
        ),
    )
    cache_root = tmp_path / 'inference_probability_cache'
    assert (cache_root / 'ff' / 'dataset-503' / 'case.npz').is_file()
    assert (cache_root / 'cc' / 'dataset-503' / 'case.npz').is_file()
    assert not (cache_root / 'fc').exists()
    assert not (cache_root / 'cf').exists()


def test_spad_fgc_endpoint_inference_collapses_integer_da_to_floor(
    monkeypatch,
    tmp_path,
):
    trainer = _fgc_endpoint_trainer(tmp_path, 2.0, (48, 192, 192))
    monkeypatch.setattr(
        spad_universal_module,
        'load_universal_dataset_case',
        lambda **kwargs: (
            torch.zeros(1, 1, 2, 2, 2),
            {},
            SimpleNamespace(),
            SimpleNamespace(configuration={
                'architecture': {},
                'spacing': [1.0, 1.0, 1.0],
                'patch_size': [192, 192, 192],
            }),
        ),
    )
    network_calls = []
    monkeypatch.setattr(
        trainer,
        'build_dataset_inference_network',
        lambda dataset_id, *, da_encoder, da_decoder, canonical_shape: (
            network_calls.append((da_encoder, da_decoder, canonical_shape))
            or object()
        ),
    )
    monkeypatch.setattr(
        spad_universal_module,
        'predict_source_grid_probabilities',
        lambda **kwargs: (np.full((1, 2, 2, 2), 5.0, dtype=np.float32),),
    )
    captured = {}
    monkeypatch.setattr(
        spad_universal_module,
        'evaluate_universal_dataset_case',
        lambda **kwargs: captured.update(kwargs) or {'ok': True},
    )

    assert trainer._predict_inference_case(
        '503',
        'case',
        tmp_path / 'endpoints',
        save_probabilities=False,
    ) == {'ok': True}

    assert network_calls == [(2, 2, (48, 96, 96))]
    np.testing.assert_array_equal(
        captured['probabilities'],
        np.full((1, 2, 2, 2), 5.0),
    )
    assert not (tmp_path / 'inference_probability_cache' / 'cc').exists()


@pytest.mark.parametrize('component', ['fc', 'cf'])
def test_spad_fgc_inference_rejects_untied_components(tmp_path, component):
    trainer = _fgc_endpoint_trainer(tmp_path, np.log2(5.0), (40, 192, 192))

    with pytest.raises(ValueError, match='tied encoder and decoder DA'):
        trainer._ensure_inference_component_probability_cache(
            '503',
            'case',
            component,
            continuous_da=np.log2(5.0),
            data=torch.zeros(1, 1, 2, 2, 2),
            properties={},
            plans_manager=SimpleNamespace(),
            configuration_manager=SimpleNamespace(patch_size=[40, 192, 192]),
        )


# Frozen v6 sqrt-replay sampling probabilities shared by the FG1/2 and FG1/3
# SPADCTUniversalV2 streams (seed 20260822).
SQRT_STREAM_PROBABILITIES = {
    '503': 0.0802563041883688,
    '506': 0.055647734601999815,
    '507': 0.11778405342481231,
    '508': 0.12242501612439959,
    '509': 0.044518187681599854,
    '510': 0.07869778098948671,
    '718': 0.12191807806230427,
    '725': 0.07710776292541786,
    '555': 0.044518187681599854,
    '562': 0.06295822479158937,
    '220': 0.15561478806571255,
    '518': 0.03855388146270893,
}
SQRT_STREAM_EXPECTED_UNIFORM_DATASET_WEIGHTS = {
    '503': 1.038,
    '506': 1.498,
    '507': 0.708,
    '508': 0.681,
    '509': 1.872,
    '510': 1.059,
    '718': 0.684,
    '725': 1.081,
    '555': 1.872,
    '562': 1.324,
    '220': 0.536,
    '518': 2.161,
}
SQRT_STREAM_DIRECTORIES = tuple(
    Path(__file__).resolve().parents[1]
    / 'nnUNet_data/preprocessed/Dataset591_SPADCTUniversalV2'
    / name
    for name in (
        'SPADCTUniversalV2SqrtNFG1of2ReplaySeed20260822MaxGB24',
        'SPADCTUniversalV2SqrtNFG1of3ReplaySeed20260822MaxGB24',
    )
)


def test_spad_universal_dataset_objective_defaults_preserve_legacy_semantics():
    assert resolve_spad_universal_dataset_objective(
        None,
        REGION_BALANCED_LOSS_NORMALIZATION,
    ) == 'uniform_region'
    assert resolve_spad_universal_dataset_objective(
        None,
        SAMPLE_MEAN_LOSS_NORMALIZATION,
    ) == 'sampling_matched'


def test_spad_universal_dataset_objective_accepts_matching_normalization():
    assert resolve_spad_universal_dataset_objective(
        'uniform_region',
        REGION_BALANCED_LOSS_NORMALIZATION,
    ) == 'uniform_region'
    assert resolve_spad_universal_dataset_objective(
        'uniform_region',
        SAMPLE_MEAN_LOSS_NORMALIZATION,
    ) == 'uniform_region'
    assert resolve_spad_universal_dataset_objective(
        'sampling_matched',
        SAMPLE_MEAN_LOSS_NORMALIZATION,
    ) == 'sampling_matched'
    assert resolve_spad_universal_dataset_objective(
        'uniform_dataset',
        SAMPLE_MEAN_LOSS_NORMALIZATION,
    ) == 'uniform_dataset'


def test_spad_universal_dataset_objective_rejects_unknown_modes():
    with pytest.raises(ValueError, match='unsupported dataset objective'):
        resolve_spad_universal_dataset_objective(
            'focal_dataset',
            SAMPLE_MEAN_LOSS_NORMALIZATION,
        )


def test_spad_universal_dataset_objective_rejects_mismatched_normalization():
    with pytest.raises(ValueError, match='sample-level'):
        resolve_spad_universal_dataset_objective(
            'uniform_dataset',
            REGION_BALANCED_LOSS_NORMALIZATION,
        )
    with pytest.raises(ValueError, match='sample-level'):
        resolve_spad_universal_dataset_objective(
            'sampling_matched',
            REGION_BALANCED_LOSS_NORMALIZATION,
        )


def test_spad_universal_dataset_sampling_validates_against_replay():
    validate_spad_universal_dataset_sampling(
        None,
        ('503',),
        COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        None,
    )
    validate_spad_universal_dataset_sampling(
        'sqrt_training_cases',
        ('503', '506'),
        SQRT_DATASET_REPLAY_RULE,
        {'503': 0.25, '506': 0.75},
    )
    with pytest.raises(ValueError, match='unsupported dataset sampling'):
        validate_spad_universal_dataset_sampling(
            'uniform_datasets_per_group',
            ('503',),
            COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
            None,
        )
    with pytest.raises(ValueError, match='sampling_rule'):
        validate_spad_universal_dataset_sampling(
            'sqrt_training_cases',
            ('503',),
            COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
            None,
        )


def test_spad_universal_unweighted_objective_combinations_use_unit_weights():
    assert spad_universal_dataset_loss_weights(
        'uniform_region',
        ('503', '506'),
        COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        None,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
        dataset_region_counts=None,
    ) == {}
    assert spad_universal_dataset_loss_weights(
        'sampling_matched',
        ('503', '506'),
        COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        None,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        dataset_region_counts=None,
    ) == {}


def test_spad_universal_uniform_dataset_rejects_unusable_replay_probabilities():
    probabilities = {'503': 0.25, '506': 0.75}
    unusable = (
        ('sampling_rule', COMPLEMENTARY_FOREGROUND_REPLAY_RULE, probabilities),
        ('every dataset exactly once', SQRT_DATASET_REPLAY_RULE, None),
        ('every dataset exactly once', SQRT_DATASET_REPLAY_RULE, {'503': 1.0}),
        ('positive', SQRT_DATASET_REPLAY_RULE, {'503': 0.0, '506': 1.0}),
        ('sum to one', SQRT_DATASET_REPLAY_RULE, {'503': 0.25, '506': 0.5}),
    )
    for match, sampling_rule, replay_probabilities in unusable:
        with pytest.raises(ValueError, match=match):
            spad_universal_dataset_loss_weights(
                'uniform_dataset',
                ('503', '506'),
                sampling_rule,
                replay_probabilities,
                loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
                dataset_region_counts=None,
            )


def test_spad_universal_uniform_dataset_weights_match_sqrt_stream_table():
    weights = spad_universal_dataset_loss_weights(
        'uniform_dataset',
        tuple(SQRT_STREAM_PROBABILITIES),
        SQRT_DATASET_REPLAY_RULE,
        SQRT_STREAM_PROBABILITIES,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        dataset_region_counts=None,
    )

    assert weights == pytest.approx(
        SQRT_STREAM_EXPECTED_UNIFORM_DATASET_WEIGHTS,
        abs=6e-4,
    )
    assert sum(
        SQRT_STREAM_PROBABILITIES[dataset_id] * weight
        for dataset_id, weight in weights.items()
    ) == pytest.approx(1.0)


@pytest.mark.skipif(
    not all(
        (directory / 'meta.yaml').is_file()
        for directory in SQRT_STREAM_DIRECTORIES
    ),
    reason='frozen sqrt replay artifacts are not available',
)
def test_spad_universal_fg_streams_derive_identical_uniform_dataset_weights():
    metas = [
        yaml.safe_load((directory / 'meta.yaml').read_text())
        for directory in SQRT_STREAM_DIRECTORIES
    ]
    weight_maps = [
        spad_universal_dataset_loss_weights(
            'uniform_dataset',
            tuple(meta['dataset_ids']),
            meta['sampling_rule'],
            meta['dataset_sampling_probabilities'],
            loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
            dataset_region_counts=None,
        )
        for meta in metas
    ]

    assert metas[0]['dataset_sampling_probabilities'] == SQRT_STREAM_PROBABILITIES
    assert weight_maps[0] == weight_maps[1]


def test_spad_universal_uniform_dataset_weights_recover_uniform_dataset_mean():
    """An exact schedule matching p reproduces the uniform dataset mean."""
    weights = spad_universal_dataset_loss_weights(
        'uniform_dataset',
        ('503', '506'),
        SQRT_DATASET_REPLAY_RULE,
        {'503': 0.25, '506': 0.75},
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        dataset_region_counts=None,
    )
    assert weights == pytest.approx({'503': 2.0, '506': 2 / 3})

    dataset_losses = {'503': [3.0], '506': [1.5, 4.5, 6.0]}
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        global_sample_count=4,
    )
    weighted_losses = [
        torch.tensor(sample_loss) * weights[dataset_id]
        for dataset_id, sample_losses in dataset_losses.items()
        for sample_loss in sample_losses
    ]

    result = loss_module.mean_sample_loss(weighted_losses)

    uniform_dataset_mean = sum(
        sum(sample_losses) / len(sample_losses)
        for sample_losses in dataset_losses.values()
    ) / len(dataset_losses)
    assert torch.allclose(result, torch.tensor(uniform_dataset_mean))


# Foreground region counts of the frozen SPADCTUniversalV2 suite (52 total).
SUITE_REGION_COUNTS = {
    '503': 2,
    '506': 1,
    '507': 2,
    '508': 2,
    '509': 1,
    '510': 1,
    '718': 15,
    '725': 16,
    '555': 4,
    '562': 1,
    '220': 3,
    '518': 4,
}
EQUAL_STREAM_EXPECTED_UNIFORM_REGION_WEIGHTS = {
    '503': 0.4615,
    '506': 0.2308,
    '507': 0.4615,
    '508': 0.4615,
    '509': 0.2308,
    '510': 0.2308,
    '718': 3.4615,
    '725': 3.6923,
    '555': 0.9231,
    '562': 0.2308,
    '220': 0.6923,
    '518': 0.9231,
}
SQRT_STREAM_EXPECTED_UNIFORM_REGION_WEIGHTS = {
    '503': 0.4792,
    '506': 0.3456,
    '507': 0.3265,
    '508': 0.3142,
    '509': 0.4320,
    '510': 0.2444,
    '718': 2.3660,
    '725': 3.9904,
    '555': 1.7279,
    '562': 0.3055,
    '220': 0.3707,
    '518': 1.9952,
}


def test_spad_universal_sample_level_uniform_region_weights_match_region_mass():
    equal_weights = spad_universal_dataset_loss_weights(
        'uniform_region',
        tuple(SUITE_REGION_COUNTS),
        COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        None,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        dataset_region_counts=SUITE_REGION_COUNTS,
    )
    sqrt_weights = spad_universal_dataset_loss_weights(
        'uniform_region',
        tuple(SUITE_REGION_COUNTS),
        SQRT_DATASET_REPLAY_RULE,
        SQRT_STREAM_PROBABILITIES,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        dataset_region_counts=SUITE_REGION_COUNTS,
    )

    assert equal_weights == pytest.approx(
        EQUAL_STREAM_EXPECTED_UNIFORM_REGION_WEIGHTS,
        abs=6e-4,
    )
    assert sqrt_weights == pytest.approx(
        SQRT_STREAM_EXPECTED_UNIFORM_REGION_WEIGHTS,
        abs=6e-4,
    )
    total_regions = sum(SUITE_REGION_COUNTS.values())
    for weights, probabilities in (
        (equal_weights, {d: 1 / 12 for d in SUITE_REGION_COUNTS}),
        (sqrt_weights, SQRT_STREAM_PROBABILITIES),
    ):
        assert sum(
            probabilities[dataset_id] * weight
            for dataset_id, weight in weights.items()
        ) == pytest.approx(1.0)
        for dataset_id, weight in weights.items():
            assert probabilities[dataset_id] * weight == pytest.approx(
                SUITE_REGION_COUNTS[dataset_id] / total_regions
            )


def test_spad_universal_sample_level_uniform_region_matches_legacy_balance():
    """An expectation-matched batch reproduces the legacy region-balanced mean."""
    region_counts = {'503': 2, '506': 1}
    weights = spad_universal_dataset_loss_weights(
        'uniform_region',
        ('503', '506'),
        COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        None,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        dataset_region_counts=region_counts,
    )
    assert weights == pytest.approx({'503': 4 / 3, '506': 2 / 3})

    sample_losses = {'503': 3.0, '506': 1.5}
    weighted = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        global_sample_count=2,
    ).mean_sample_loss([
        torch.tensor(sample_losses[dataset_id]) * weights[dataset_id]
        for dataset_id in region_counts
    ])
    legacy = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
    ).mean_sample_loss(
        [
            torch.tensor(sample_losses[dataset_id]) * region_counts[dataset_id]
            for dataset_id in region_counts
        ],
        [region_counts[dataset_id] for dataset_id in region_counts],
    )

    assert torch.allclose(weighted, legacy)


def test_spad_universal_sample_level_uniform_region_rejects_unusable_inputs():
    region_counts = {'503': 2, '506': 1}
    with pytest.raises(ValueError, match='does not define'):
        spad_universal_dataset_loss_weights(
            'uniform_region',
            ('503', '506'),
            'adhoc_rule',
            None,
            loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
            dataset_region_counts=region_counts,
        )
    unusable_counts = (
        ('region counts', None),
        ('region counts', {'503': 2}),
        ('positive', {'503': 2, '506': 0}),
        ('positive', {'503': 2, '506': True}),
    )
    for match, counts in unusable_counts:
        with pytest.raises(ValueError, match=match):
            spad_universal_dataset_loss_weights(
                'uniform_region',
                ('503', '506'),
                COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
                None,
                loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
                dataset_region_counts=counts,
            )


def test_spad_universal_weighted_sample_mean_is_ddp_equivalent():
    weighted_losses = [
        torch.tensor(2.0 * 3.0),
        torch.tensor((2 / 3) * 1.5),
        torch.tensor((2 / 3) * 4.5),
        torch.tensor((2 / 3) * 6.0),
    ]
    single_process = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        global_sample_count=4,
    ).mean_sample_loss(weighted_losses)
    ddp_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
        global_sample_count=4,
        ddp_world_size=2,
    )

    rank_results = [
        ddp_module.mean_sample_loss(weighted_losses[:2]),
        ddp_module.mean_sample_loss(weighted_losses[2:]),
    ]

    assert torch.allclose(torch.stack(rank_results).mean(), single_process)


def test_spad_universal_dataset_objective_weight_is_identity_without_weights():
    trainer = object.__new__(SPADUniversalTrainer)
    sample_loss = torch.tensor(3.0)

    assert trainer._apply_dataset_objective_weight(sample_loss, '503') is sample_loss
    trainer.dataset_loss_weights = {}
    assert trainer._apply_dataset_objective_weight(sample_loss, '503') is sample_loss
    trainer.dataset_loss_weights = {'503': 2.0}
    assert torch.allclose(
        trainer._apply_dataset_objective_weight(sample_loss, '503'),
        torch.tensor(6.0),
    )


def _dataset_objective_weighted_trainer() -> (
    tuple[SPADUniversalTrainer, tuple[UniversalSample, ...]]
):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv3d(1, 4, 1)

        def forward(self, sample_inputs):
            return tuple([self.head(data)] for data, _, _, _ in sample_inputs)

    torch.manual_seed(0)
    network = Network()
    trainer = object.__new__(SPADUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network
    trainer.optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    trainer.grad_scaler = None
    trainer.loss = UniversalPartialLabelLoss(batch_dice=False)
    trainer.registry = CanonicalRegionRegistry(
        {
            '503': {'labels': {'background': 0, 'a': 1, 'b': 2}},
            '506': {'labels': {'background': 0, 'c': 1}},
        },
        shared_regions={},
    )
    trainer.continuous_da = {'503': 0.0, '506': 0.0}
    trainer.da_discretization = 'floor'
    trainer.cross_da = False
    trainer.dataset_index_by_id = {'503': 0, '506': 1}
    trainer.dataset_loss_weights = {'503': 2.0, '506': 0.5}
    samples = (
        UniversalSample(
            '503',
            torch.randn(1, 1, 4, 4, 4),
            torch.randint(0, 2, (1, 2, 4, 4, 4)).float(),
            trainer.registry.indices('503'),
        ),
        UniversalSample(
            '506',
            torch.randn(1, 1, 4, 4, 4),
            torch.randint(0, 2, (1, 1, 4, 4, 4)).float(),
            trainer.registry.indices('506'),
        ),
    )
    return trainer, samples


def _expected_dataset_objective_weighted_loss(
    trainer: SPADUniversalTrainer,
    samples: tuple[UniversalSample, ...],
) -> torch.Tensor:
    with torch.no_grad():
        outputs_per_sample = trainer.network(tuple(
            trainer._network_input_for_sample(sample, (0, 0))
            for sample in samples
        ))
        weighted_losses = [
            trainer.loss.sample_loss(
                [
                    output.index_select(1, sample.region_indices)
                    for output in outputs
                ],
                sample.target,
            ) * trainer.dataset_loss_weights[sample.dataset_id]
            for sample, outputs in zip(
                samples,
                outputs_per_sample,
                strict=True,
            )
        ]
        return trainer.loss.mean_sample_loss(
            weighted_losses,
            [int(sample.region_indices.numel()) for sample in samples],
        )


def test_spad_universal_train_step_applies_dataset_objective_weights():
    trainer, samples = _dataset_objective_weighted_trainer()
    expected = _expected_dataset_objective_weighted_loss(trainer, samples)

    result = trainer.train_step({'samples': samples})

    assert np.allclose(result['loss'], expected.numpy())


def test_spad_universal_validation_step_applies_same_dataset_objective_weights():
    trainer, samples = _dataset_objective_weighted_trainer()
    expected = _expected_dataset_objective_weighted_loss(trainer, samples)

    result = trainer.validation_step({'samples': samples})

    assert np.allclose(result['loss'], expected.numpy())
