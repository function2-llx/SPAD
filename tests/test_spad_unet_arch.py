"""Tests for the primary six-stage SPADResEncUNet architecture."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.spadop.conv import (
    SPADConv3d_K3S1,
    SPADConv3d_K3S2,
    SPADConvTranspose3d_K2S2,
)
from pumit.spad_unet.architecture import (
    LEGACY_UNIVERSAL_SPAD_BLOCKS,
    LEGACY_UNIVERSAL_SPAD_DECODER_CONVS,
    LEGACY_UNIVERSAL_SPAD_FEATURES,
    LEGACY_UNIVERSAL_SPAD_N_STAGES,
    UNIVERSAL_SPAD_BLOCKS,
    UNIVERSAL_SPAD_DECODER_CONVS,
    UNIVERSAL_SPAD_FEATURES,
    UNIVERSAL_SPAD_N_STAGES,
    SPADResEncUNet,
    SPADUNetDecoder,
    build_spad_architecture,
    build_universal_spad_architecture,
)
from pumit.spad_unet.geometry import (
    SUPPORTED_FGC_STAGES,
    compute_fgc_geometry,
    compute_stride_schedule,
)
from pumit.spad_unet import prepare
from pumit.spad_unet.experiments.spad import SPADUNetTrainer


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
    }


@pytest.fixture
def model():
    return SPADResEncUNet(input_channels=1, num_classes=4, inference_da=0).cuda()


class TestForwardShapes:
    """Verify output shapes for various DA and patch size combinations."""

    def test_isotropic_large(self, model):
        """KiTS-like: da=0, 160x224x192."""
        x = torch.randn(1, 1, 160, 224, 192, device='cuda')
        features = model.forward_features(x, da=0)
        assert len(features) == 5
        assert features[0].shape == (1, 32, 160, 224, 192)
        assert features[-1].shape[2:] == torch.Size([10, 14, 12])

    def test_anisotropic_thin(self, model):
        """ACDC-like: da=2, 20x256x224."""
        x = torch.randn(1, 1, 20, 256, 224, device='cuda')
        features = model.forward_features(x, da=2)
        assert len(features) == 5
        assert features[0].shape[2:] == torch.Size([20, 256, 224])
        # Depth preserved at stages 1-2 (da>=1), then downsampled
        assert features[1].shape[2] == 20  # stage 1: stride (1,2,2)

    def test_small_isotropic_clamped(self, model):
        """Small patch triggers clamping at later stages."""
        x = torch.randn(1, 1, 64, 64, 64, device='cuda')
        features = model.forward_features(x, da=0)
        assert len(features) == 5
        # Bottleneck should be 4x4x4 (clamped, not 2x2x2)
        assert features[-1].shape[2:] == torch.Size([4, 4, 4])

    def test_very_small_patch_clamped(self, model):
        """32^3 patch: aggressive clamping with a decoder channel change."""
        x = torch.randn(1, 1, 32, 32, 32, device='cuda')
        features = model.forward_features(x, da=0)
        assert len(features) == 5
        assert features[0].shape == (1, 32, 32, 32, 32)
        assert features[-1].shape[2:] == torch.Size([4, 4, 4])

    def test_batch_size_2(self, model):
        """Batched input works."""
        x = torch.randn(2, 1, 128, 128, 128, device='cuda')
        features = model.forward_features(x, da=0)
        assert features[0].shape[0] == 2

    def test_multichannel_input(self):
        """Multi-channel input (e.g., BraTS 4-channel)."""
        model = SPADResEncUNet(input_channels=4, num_classes=4, inference_da=0).cuda()
        x = torch.randn(1, 4, 64, 64, 64, device='cuda')
        features = model.forward_features(x, da=0)
        assert len(features) == 5
        assert features[0].shape[:2] == (1, 32)


def test_native_nnunet_builder_constructs_spad_from_plans_schema():
    architecture = build_spad_architecture(_source_architecture(), inference_da=0)
    configuration_manager = SimpleNamespace(
        network_arch_class_name=architecture['network_class_name'],
        network_arch_init_kwargs=architecture['arch_kwargs'],
        network_arch_init_kwargs_req_import=architecture['_kw_requires_import'],
    )

    network = nnUNetTrainer.build_network_architecture(
        plans_manager=None,
        configuration_manager=configuration_manager,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )
    outputs = network(torch.randn(1, 1, 16, 16, 16), da=0)

    assert isinstance(network, SPADResEncUNet)
    assert len(outputs) == 2
    assert all(output.shape[1] == 3 for output in outputs)
    assert any(key.startswith('decoder.seg_layers.') for key in network.state_dict())

    network.decoder.deep_supervision = False
    inference_output = network(torch.randn(1, 1, 16, 16, 16), da=0)
    assert isinstance(inference_output, torch.Tensor)
    assert inference_output.shape == (1, 3, 16, 16, 16)


def test_packed_segmentation_rows_match_full_projection_and_keep_dense_gradient():
    decoder = SPADUNetDecoder(
        n_stages=3,
        features_per_stage=(4, 8, 16),
        num_classes=6,
        deep_supervision=True,
    )
    features = [
        torch.randn(1, 4, 8, 8, 8),
        torch.randn(1, 8, 4, 4, 4),
    ]
    output_rows = torch.tensor([5, 1, 3])

    full_outputs = [
        head(feature)
        for head, feature in zip(
            decoder.seg_layers,
            features,
            strict=True,
        )
    ]
    packed_outputs = decoder.segment(features, output_rows)

    assert isinstance(packed_outputs, list)
    for full, packed in zip(full_outputs, packed_outputs, strict=True):
        assert torch.allclose(
            packed,
            full.index_select(1, output_rows),
            atol=1e-6,
            rtol=1e-6,
        )
    sum(output.sum() for output in packed_outputs).backward()
    for head in decoder.seg_layers:
        assert head.weight.grad is not None
        assert torch.count_nonzero(head.weight.grad[[0, 2, 4]]) == 0
        assert torch.count_nonzero(head.weight.grad[output_rows]) > 0


def test_lkr_architecture_flag_reaches_every_reduction_operator():
    architecture = build_spad_architecture(
        _source_architecture(),
        inference_da=0,
        learnable_kernel_reduction=True,
    )
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        **architecture['arch_kwargs'],
    )
    reduction_operators = [
        module
        for module in network.modules()
        if isinstance(
            module,
            (SPADConv3d_K3S1, SPADConv3d_K3S2, SPADConvTranspose3d_K2S2),
        )
    ]

    assert architecture['arch_kwargs']['learnable_kernel_reduction'] is True
    assert reduction_operators
    assert all(
        module.kernel_reduction_delta is not None
        for module in reduction_operators
    )
    assert sum(
        module.kernel_reduction_delta.numel()
        for module in reduction_operators
    ) == 2_040


def test_lkr_architecture_default_preserves_parameter_set():
    architecture = build_spad_architecture(_source_architecture(), inference_da=0)
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        **architecture['arch_kwargs'],
    )

    assert architecture['arch_kwargs']['learnable_kernel_reduction'] is False
    assert architecture['arch_kwargs']['full_kernel_dynamic_stride'] is False
    assert not any(
        name.endswith('kernel_reduction_delta')
        for name, _ in network.named_parameters()
    )


def test_fkds_keeps_route_shapes_and_forces_full_k3_operator_modes():
    common_kwargs = {
        'input_channels': 1,
        'num_classes': 2,
        'n_stages': 6,
        'features_per_stage': (2, 4, 8, 16, 16, 16),
        'n_blocks_per_stage': (1, 1, 1, 1, 1, 1),
        'n_conv_per_stage_decoder': (1, 1, 1, 1, 1),
        'inference_da': 0,
        'deep_supervision': False,
    }
    standard = SPADResEncUNet(**common_kwargs)
    fkds = SPADResEncUNet(
        **common_kwargs,
        full_kernel_dynamic_stride=True,
    )
    fkds.load_state_dict(standard.state_dict())
    standard_operator_das = []
    fkds_operator_das = []
    fkds_upsample_modes = []

    def record_da(target):
        return lambda module, args: target.append(args[1])

    for module in standard.modules():
        if isinstance(module, (SPADConv3d_K3S1, SPADConv3d_K3S2)):
            module.register_forward_pre_hook(record_da(standard_operator_das))
    for module in fkds.modules():
        if isinstance(module, (SPADConv3d_K3S1, SPADConv3d_K3S2)):
            module.register_forward_pre_hook(record_da(fkds_operator_das))
        elif isinstance(module, SPADConvTranspose3d_K2S2):
            module.register_forward_pre_hook(
                lambda module, args, kwargs: fkds_upsample_modes.append(
                    (args[1], kwargs['stride_override'])
                ),
                with_kwargs=True,
            )

    x = torch.randn(1, 1, 32, 64, 64)
    standard_features = standard.forward_features(x, da=3)
    fkds_features = fkds.forward_features(x, da=3)

    assert any(da >= 2 for da in standard_operator_das)
    assert fkds_operator_das
    assert all(da in (0, 1) for da in fkds_operator_das)
    assert (0, (2, 2, 2)) in fkds_upsample_modes
    assert (1, (1, 2, 2)) in fkds_upsample_modes
    assert [feature.shape for feature in fkds_features] == [
        feature.shape for feature in standard_features
    ]
    assert standard.state_dict().keys() == fkds.state_dict().keys()
    assert sum(p.numel() for p in standard.parameters()) == sum(
        p.numel() for p in fkds.parameters()
    )


def test_fkds_rejects_learnable_kernel_reduction():
    with pytest.raises(
        ValueError,
        match='FKDS is incompatible with learnable kernel reduction',
    ):
        build_spad_architecture(
            _source_architecture(),
            inference_da=0,
            learnable_kernel_reduction=True,
            full_kernel_dynamic_stride=True,
        )


def test_fkds_rejects_fgc_in_schema_and_network():
    with pytest.raises(ValueError, match='FKDS does not support FGC'):
        build_universal_spad_architecture(
            _source_architecture(),
            inference_da=0,
            num_canonical_regions=2,
            num_task_datasets=1,
            full_kernel_dynamic_stride=True,
            feature_grid_canonicalization_stage=2,
            feature_grid_canonicalization_return='late',
        )
    with pytest.raises(ValueError, match='FKDS does not support FGC'):
        SPADResEncUNet(
            input_channels=1,
            num_classes=2,
            inference_da=0,
            full_kernel_dynamic_stride=True,
            feature_grid_canonicalization_stage=2,
            feature_grid_canonicalization_return='late',
        )


def test_universal_lkr_parameter_count_is_exact():
    with torch.device('meta'):
        network = SPADResEncUNet(
            input_channels=1,
            num_classes=52,
            n_stages=UNIVERSAL_SPAD_N_STAGES,
            features_per_stage=UNIVERSAL_SPAD_FEATURES,
            n_blocks_per_stage=UNIVERSAL_SPAD_BLOCKS,
            n_conv_per_stage_decoder=UNIVERSAL_SPAD_DECODER_CONVS,
            inference_da=0,
            learnable_kernel_reduction=True,
        )

    assert sum(
        parameter.numel()
        for name, parameter in network.named_parameters()
        if name.endswith('kernel_reduction_delta')
    ) == 358_912


def test_legacy_seven_stage_universal_lkr_parameter_count_is_exact():
    with torch.device('meta'):
        network = SPADResEncUNet(
            input_channels=1,
            num_classes=52,
            n_stages=LEGACY_UNIVERSAL_SPAD_N_STAGES,
            features_per_stage=LEGACY_UNIVERSAL_SPAD_FEATURES,
            n_blocks_per_stage=LEGACY_UNIVERSAL_SPAD_BLOCKS,
            n_conv_per_stage_decoder=LEGACY_UNIVERSAL_SPAD_DECODER_CONVS,
            inference_da=0,
            learnable_kernel_reduction=True,
        )

    assert sum(
        parameter.numel()
        for name, parameter in network.named_parameters()
        if name.endswith('kernel_reduction_delta')
    ) == 473_792


def test_spad_schema_accepts_native_anisotropic_3d_plans():
    source = _source_architecture()
    source['arch_kwargs']['kernel_sizes'] = [
        [1, 3, 3],
        [1, 3, 3],
        [3, 3, 3],
    ]
    source['arch_kwargs']['strides'] = [
        [1, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
    ]

    architecture = build_spad_architecture(source, inference_da=2)

    assert architecture['arch_kwargs']['inference_da'] == 2


def test_spad_schema_rejects_2d_source_plans():
    source = _source_architecture()
    source['arch_kwargs']['conv_op'] = 'torch.nn.modules.conv.Conv2d'

    with pytest.raises(ValueError, match='source architecture mismatch'):
        build_spad_architecture(source, inference_da=0)


@pytest.mark.parametrize('da', [0, 1, 2, None])
def test_size_clamped_runtime_stride_matches_k3s1_with_shared_weights(da):
    dynamic = SPADConv3d_K3S2(4, 8, kernel_size=3, stride=2)
    reference = SPADConv3d_K3S1(4, 8, kernel_size=3, padding=1)
    with torch.no_grad():
        reference.weight.copy_(dynamic.weight)
        reference.bias.copy_(dynamic.bias)
    x = torch.randn(1, 4, 7, 11, 13)

    actual = dynamic(x, da=da, stride_override=(1, 1, 1))
    expected = reference(x, da=da)

    assert torch.equal(actual, expected)


def test_runtime_stride_must_match_da_mode():
    conv = SPADConv3d_K3S2(4, 8, kernel_size=3, stride=2)

    with pytest.raises(ValueError, match='requires runtime stride'):
        conv(
            torch.randn(1, 4, 7, 11, 13),
            da=0,
            stride_override=(1, 2, 2),
        )


def test_spad_trainer_inherits_native_network_and_checkpoint_lifecycle():
    inherited_methods = {
        'build_network_architecture',
        'initialize',
        'set_deep_supervision_enabled',
        'save_checkpoint',
        'load_checkpoint',
        'perform_actual_validation',
    }

    assert inherited_methods.isdisjoint(SPADUNetTrainer.__dict__)
    assert SPADUNetTrainer.save_checkpoint is RetainPeriodicCheckpointsMixin.save_checkpoint


def test_native_predictor_uses_planned_inference_da():
    predictor = object.__new__(nnUNetPredictor)
    predictor.network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=3,
        features_per_stage=(4, 8, 16),
        n_blocks_per_stage=(1, 1, 1),
        n_conv_per_stage_decoder=(1, 1),
        inference_da=0,
        deep_supervision=False,
    )
    predictor.allowed_mirroring_axes = None
    predictor.use_mirroring = False

    output = predictor._internal_maybe_mirror_and_predict(torch.randn(1, 1, 16, 16, 16))

    assert output.shape == (1, 3, 16, 16, 16)


def test_spad_trainer_enables_dynamic_graph_ddp_detection():
    trainer = object.__new__(SPADUNetTrainer)

    assert trainer._get_ddp_kwargs() == {'find_unused_parameters': True}


def test_spad_validation_uses_floor_da():
    class RecordingNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.da = None

        def forward(self, data, da):
            self.da = da
            return torch.zeros(data.shape[0], 2, *data.shape[2:])

    network = RecordingNetwork()
    trainer = object.__new__(SPADUNetTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = network
    trainer.loss = lambda output, target: output.sum()
    trainer.continuous_da = 1.75
    trainer.label_manager = SimpleNamespace(
        has_regions=False,
        has_ignore_label=False,
    )

    trainer.validation_step({
        'data': torch.zeros(1, 1, 2, 2, 2),
        'target': torch.zeros(1, 1, 2, 2, 2, dtype=torch.long),
    })

    assert network.da == 1


@pytest.mark.parametrize(
    ('has_regions', 'expected_loss_type'),
    [(False, DC_and_CE_loss), (True, DC_and_BCE_loss)],
)
def test_spad_loss_reuses_native_label_semantics(has_regions, expected_loss_type):
    trainer = object.__new__(SPADUNetTrainer)
    trainer.label_manager = SimpleNamespace(has_regions=has_regions, ignore_label=None)
    trainer.configuration_manager = SimpleNamespace(
        batch_dice=False,
        network_arch_init_kwargs={'n_stages': 3},
    )
    trainer.is_ddp = True
    trainer.enable_deep_supervision = True
    trainer._do_i_compile = lambda: False

    loss = trainer._build_loss()

    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, expected_loss_type)
    assert loss.weight_factors[-1] > 0


def test_prepare_spad_plans_writes_native_training_contract(tmp_path, monkeypatch):
    dataset_name = 'Dataset999_Test'
    dataset_folder = tmp_path / dataset_name
    dataset_folder.mkdir()
    source_plans = {
        'plans_name': 'SourcePlans',
        'configurations': {
            '3d_fullres': {
                'spacing': [3.0, 1.0, 1.0],
                'architecture': _source_architecture(),
            },
        },
    }
    (dataset_folder / 'SourcePlans.json').write_text(json.dumps(source_plans) + '\n')
    monkeypatch.setattr(prepare, 'nnUNet_preprocessed', str(tmp_path))

    plan_path = prepare.prepare_spad_plans(
        dataset_name,
        '3d_fullres',
        source_plans_identifier='SourcePlans',
    )

    derived = json.loads(plan_path.read_text())
    architecture = derived['configurations']['3d_fullres']['architecture']
    assert derived['plans_name'] == prepare.SPAD_PLANS_IDENTIFIER
    assert architecture['network_class_name'].endswith('.SPADResEncUNet')
    assert architecture['arch_kwargs']['inference_da'] == 1


class TestStrideSchedule:
    """Verify stride schedule produces correct values."""

    def test_isotropic_no_clamp(self):
        """Large isotropic patch: 6 active strides."""
        ss = compute_stride_schedule(0, 6, (256, 256, 256))
        assert ss[0] == (1, 1, 1)
        assert all(s == (2, 2, 2) for s in ss[1:])

    def test_anisotropic_da2(self):
        """DA=2: first two stages skip depth."""
        ss = compute_stride_schedule(2, 6, (64, 256, 256))
        assert ss[1] == (1, 2, 2)  # da=2
        assert ss[2] == (1, 2, 2)  # da=1
        assert ss[3] == (2, 2, 2)  # da=0

    def test_clamping_small_patch(self):
        """Small patch triggers clamping."""
        ss = compute_stride_schedule(0, 6, (64, 64, 64))
        # 64 -> 32 -> 16 -> 8 -> 4 (stop). Stages 1-4 active, stage 5 clamped.
        assert ss[5] == (1, 1, 1)

    def test_only_three_stride_values(self):
        """All strides are (1,1,1), (1,2,2), or (2,2,2)."""
        for da in range(4):
            for patch in [(32, 128, 128), (64, 64, 64), (160, 224, 192)]:
                ss = compute_stride_schedule(da, 6, patch)
                for s in ss:
                    assert s in ((1, 1, 1), (1, 2, 2), (2, 2, 2))


class TestGradientFlow:
    """Verify gradients flow through all DA modes."""

    def test_gradient_da0(self, model):
        x = torch.randn(1, 1, 64, 64, 64, device='cuda', requires_grad=True)
        logits = model(x, da=0)
        logits[0].sum().backward()
        assert x.grad is not None

    def test_gradient_da2(self, model):
        x = torch.randn(1, 1, 32, 128, 128, device='cuda', requires_grad=True)
        logits = model(x, da=2)
        logits[0].sum().backward()
        assert x.grad is not None


def test_size_clamped_decoder_has_no_unused_parameters():
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=4,
        features_per_stage=(2, 4, 8, 16),
        n_blocks_per_stage=(1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1),
        inference_da=0,
    )
    outputs = network(torch.randn(1, 1, 16, 16, 16), da=0)

    sum(output.sum() for output in outputs).backward()

    assert not hasattr(network.decoder, 'skip_projections')
    assert all(
        parameter.grad is not None
        for parameter in network.parameters()
        if parameter.requires_grad
    )


def test_fgc_s2_builds_mixed_hierarchy_and_returns_to_native_grid():
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=2,
    )
    geometry = compute_fgc_geometry(
        math.log2(5.0),
        (32, 128, 128),
        stage=2,
    )
    skips = network.encode_sample(
        torch.randn(1, 1, 32, 128, 128),
        geometry.route_da,
        geometry.canonical_shape,
    )
    features = network.decoder(
        skips,
        list(geometry.da_schedule),
        list(geometry.stride_schedule),
        geometry.native_bridge_shape,
    )[::-1]
    outputs = network.decoder.segment(features)

    assert [tuple(skip.shape[2:]) for skip in skips] == list(
        geometry.feature_shapes
    )
    assert [tuple(output.shape[2:]) for output in outputs] == [
        (32, 128, 128),
        (32, 64, 64),
        (40, 32, 32),
        (20, 16, 16),
        (10, 8, 8),
    ]

    with pytest.raises(RuntimeError, match='UniversalSPADResEncUNet'):
        network(torch.randn(1, 1, 32, 128, 128), da=geometry.route_da)


@pytest.mark.parametrize('stage', SUPPORTED_FGC_STAGES)
@pytest.mark.parametrize('return_mode', ['early', 'late'])
def test_fgc_return_closes_native_output_for_each_placement(
    stage,
    return_mode,
):
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=stage,
        feature_grid_canonicalization_return=return_mode,
        feature_grid_canonicalization_return_downsample_mode='area',
    )
    geometry = compute_fgc_geometry(
        math.log2(5.0),
        (32, 128, 128),
        stage=stage,
    )
    skips = network.encode_sample(
        torch.randn(1, 1, 32, 128, 128),
        geometry.route_da,
        geometry.canonical_shape,
    )
    features = network.decoder(
        skips,
        list(geometry.da_schedule),
        list(geometry.stride_schedule),
        geometry.native_bridge_shape,
    )[::-1]
    outputs = network.decoder.segment(features)
    sum(output.mean() for output in outputs).backward()

    expected_skip_shapes = list(geometry.feature_shapes)
    if return_mode == 'late':
        expected_skip_shapes[stage - 1] = geometry.canonical_shape
    assert [tuple(skip.shape[2:]) for skip in skips] == expected_skip_shapes
    assert tuple(outputs[0].shape[2:]) == (32, 128, 128)
    assert tuple(outputs[stage - 1].shape[2:]) == (
        geometry.native_bridge_shape
    )
    assert network.encoder.stem.conv.weight.grad is not None
    assert network.decoder.post_cat_convs[0].conv.weight.grad is not None
    assert network.decoder.seg_layers[0].weight.grad is not None


@pytest.mark.parametrize('stage', SUPPORTED_FGC_STAGES)
def test_fgc_tied_ceil_route_closes_native_output_for_each_placement(stage):
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=stage,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='trilinear',
        feature_grid_canonicalization_return_prefilter=True,
    )
    network.apply(network.initialize)
    geometry = compute_fgc_geometry(
        math.log2(5.0),
        (32, 128, 128),
        stage=stage,
        route_da=3,
    )
    skips = network.encode_sample(
        torch.randn(1, 1, 32, 128, 128),
        geometry.route_da,
        geometry.canonical_shape,
    )
    features = network.decoder(
        skips,
        list(geometry.da_schedule),
        list(geometry.stride_schedule),
        geometry.native_bridge_shape,
    )[::-1]
    outputs = network.decoder.segment(features)
    sum(output.mean() for output in outputs).backward()

    expected_skip_shapes = list(geometry.feature_shapes)
    expected_skip_shapes[stage - 1] = geometry.canonical_shape
    assert [tuple(skip.shape[2:]) for skip in skips] == expected_skip_shapes
    assert tuple(outputs[0].shape[2:]) == (32, 128, 128)
    assert tuple(outputs[stage - 1].shape[2:]) == geometry.native_bridge_shape
    assert network.decoder.return_prefilter.depthwise.weight.grad is not None


def test_fgc_rejects_the_ceil_route_on_the_floor_canonical_shape():
    """A route that reaches the boundary with fewer z strides cannot reuse the floor grid."""
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=0,
        feature_grid_canonicalization_stage=2,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='trilinear',
        feature_grid_canonicalization_return_prefilter=True,
    )
    floor_geometry = compute_fgc_geometry(
        math.log2(1.5),
        (128, 128, 128),
        stage=2,
    )
    ceil_geometry = compute_fgc_geometry(
        math.log2(1.5),
        (128, 128, 128),
        stage=2,
        route_da=1,
    )
    assert floor_geometry.native_bridge_shape[0] == 64
    assert floor_geometry.canonical_shape[0] == 96
    assert ceil_geometry.native_bridge_shape[0] == 128

    skips = network.encode_sample(
        torch.randn(1, 1, 128, 128, 128),
        ceil_geometry.route_da,
        floor_geometry.canonical_shape,
    )

    with pytest.raises(ValueError, match='requires z-only downsampling'):
        network.decoder(
            skips,
            list(ceil_geometry.da_schedule),
            list(ceil_geometry.stride_schedule),
            ceil_geometry.native_bridge_shape,
        )


def test_fgc_late_return_prefilter_participates_before_trilinear_return():
    network = SPADResEncUNet(
        input_channels=1,
        num_classes=3,
        n_stages=6,
        features_per_stage=(2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        inference_da=2,
        feature_grid_canonicalization_stage=2,
        feature_grid_canonicalization_return='late',
        feature_grid_canonicalization_return_downsample_mode='trilinear',
        feature_grid_canonicalization_return_prefilter=True,
    )
    network.apply(network.initialize)
    geometry = compute_fgc_geometry(
        math.log2(5.0),
        (32, 128, 128),
        stage=2,
    )
    skips = network.encode_sample(
        torch.randn(1, 1, 32, 128, 128),
        geometry.route_da,
        geometry.canonical_shape,
    )
    features = network.decoder(
        skips,
        list(geometry.da_schedule),
        list(geometry.stride_schedule),
        geometry.native_bridge_shape,
    )

    sum(feature.square().mean() for feature in features).backward()

    prefilter = network.decoder.return_prefilter
    assert prefilter is not None
    assert tuple(prefilter.depthwise.weight.shape) == (2, 1, 3, 1, 1)
    assert prefilter.depthwise.weight.grad is not None
    assert torch.count_nonzero(prefilter.depthwise.weight.grad) > 0


class TestParamCount:
    """Verify parameter counts are reasonable."""

    def test_6stage_param_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        # Six-stage primary model remains close to the ResEnc L capacity target.
        assert 100_000_000 < n_params < 200_000_000

    def test_encoder_downsampling_blocks_share_one_runtime_stride_conv(self, model):
        """Size-clamped and downsampling paths share one SPAD K3 weight."""
        for s in range(1, 6):
            first_block = model.encoder.stages[s][0]
            assert first_block.conv1.has_stride
            assert first_block.conv1.conv is not None
            assert not hasattr(first_block.conv1, 'conv_fallback')

    def test_decoder_all_runtime_strides_share_one_transposed_conv(self):
        with torch.device('meta'):
            network = SPADResEncUNet(
                input_channels=1,
                num_classes=4,
                inference_da=0,
            )

        assert not hasattr(network.decoder, 'skip_projections')
        assert len(network.decoder.upsample_convs) == network.n_stages - 1
        assert all(
            isinstance(conv, SPADConvTranspose3d_K2S2)
            for conv in network.decoder.upsample_convs
        )
