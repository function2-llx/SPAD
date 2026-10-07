import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import pumit.downstream.seg.initialize as initialize_module
import pumit.downstream.seg.prepare as prepare_module


def _source_architecture() -> dict:
    return {
        'network_class_name': 'source.Network',
        'arch_kwargs': {
            'n_stages': 3,
            'features_per_stage': [4, 8, 16],
            'conv_op': 'torch.nn.Conv3d',
            'kernel_sizes': [[3, 3, 3]] * 3,
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            'n_blocks_per_stage': [1, 3, 4],
            'n_conv_per_stage_decoder': [1, 1],
            'conv_bias': True,
            'norm_op': 'torch.nn.InstanceNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True},
        },
        '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
    }


@pytest.mark.parametrize(
    'backbone_name',
    [
        '3dino-vit-adapter',
        'biomedclip-vit-adapter',
        'dinov3-vit-adapter',
        'eva02-l-vit-adapter',
        'pumit-vit-adapter',
        'sam-med3d',
    ],
)
def test_configure_vit_patch_size_records_only_explicit_fixed_geometry(backbone_name):
    config = prepare_module._configure_vit_patch_size(
        backbone_name,
        {'architecture': {'hidden_size': 2}},
        (4, 16, 16),
        (20, 256, 224),
    )

    assert config['vit_patch_size'] == [4, 16, 16]


def test_configure_vit_patch_size_rejects_missing_or_non_power_of_two_depth():
    with pytest.raises(ValueError, match='requires an explicit'):
        prepare_module._configure_vit_patch_size(
            'dinov3-vit-adapter',
            {},
            None,
            (20, 256, 224),
        )
    with pytest.raises(ValueError, match='D in'):
        prepare_module._configure_vit_patch_size(
            'dinov3-vit-adapter',
            {},
            (6, 16, 16),
            (24, 256, 224),
        )


def test_configure_vit_patch_size_rejects_truncated_input_grid():
    with pytest.raises(ValueError, match='must be divisible'):
        prepare_module._configure_vit_patch_size(
            'dinov3-vit-adapter',
            {},
            (8, 16, 16),
            (20, 256, 224),
        )


def test_configure_features_per_stage_replaces_the_single_plan_channel_source():
    architecture = prepare_module._configure_features_per_stage(
        _source_architecture(),
        (8, 16, 32),
    )

    assert architecture['arch_kwargs']['features_per_stage'] == [8, 16, 32]
    assert _source_architecture()['arch_kwargs']['features_per_stage'] == [4, 8, 16]


def test_configure_features_per_stage_requires_the_source_stage_count():
    with pytest.raises(ValueError, match='must contain 3 positive values'):
        prepare_module._configure_features_per_stage(
            _source_architecture(),
            (8, 16),
        )


def test_truncate_architecture_stages_keeps_the_shallowest_stages():
    architecture = prepare_module._truncate_architecture_stages(_source_architecture(), 2)

    kwargs = architecture['arch_kwargs']
    assert kwargs['n_stages'] == 2
    assert kwargs['features_per_stage'] == [4, 8]
    assert kwargs['kernel_sizes'] == [[3, 3, 3]] * 2
    assert kwargs['strides'] == [[1, 1, 1], [2, 2, 2]]
    assert kwargs['n_blocks_per_stage'] == [1, 3]
    assert kwargs['n_conv_per_stage_decoder'] == [1]
    assert _source_architecture()['arch_kwargs']['features_per_stage'] == [4, 8, 16]


def test_truncate_architecture_stages_rejects_out_of_range_stage_counts():
    for num_stages in (1, 4):
        with pytest.raises(ValueError, match='num_stages must be in'):
            prepare_module._truncate_architecture_stages(_source_architecture(), num_stages)


def test_configure_vit_adapter_records_explicit_narrow_architecture(monkeypatch):
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {'test': SimpleNamespace(is_vit_adapter=True)},
    )

    config = prepare_module._configure_vit_adapter(
        'test',
        {
            'adapter_dim': 1024,
            'deform_attention_dim': 512,
            'deform_num_heads': 16,
            'conv_ffn_hidden_dim': 256,
        },
        256,
        256,
        8,
        64,
        (256, 256, 256, 256),
        'project-then-add',
    )

    assert config == {
        'adapter_dim': 256,
        'deform_attention_dim': 256,
        'deform_num_heads': 8,
        'conv_ffn_hidden_dim': 64,
        'spatial_prior_channels': [256, 256, 256, 256],
        'output_fusion': 'project-then-add',
    }


def test_configure_vit_adapter_paths_separates_native_and_shared_stems():
    config = {
        'spatial_prior_input': 'raw',
        'with_high_resolution_stem': True,
    }

    native_without_refiner = prepare_module._configure_vit_adapter_paths(
        'pumit-vit-adapter',
        config,
        readout='mask2former',
        spatial_prior_input='raw',
    )
    shared_with_refiner = prepare_module._configure_vit_adapter_paths(
        'pumit-vit-adapter',
        config,
        readout='unet',
        spatial_prior_input='p1',
    )

    assert native_without_refiner['spatial_prior_input'] == 'raw'
    assert native_without_refiner['with_high_resolution_stem'] is False
    assert shared_with_refiner['spatial_prior_input'] == 'p1'
    assert shared_with_refiner['with_high_resolution_stem'] is True


@pytest.mark.parametrize(
    ('readout', 'expected'),
    [('unet', 'p1'), ('unet-no-refiner', 'raw'), ('mask2former', 'raw')],
)
def test_configure_vit_adapter_paths_resolves_default_for_readout(monkeypatch, readout, expected):
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {'test': SimpleNamespace(is_vit_adapter=True, can_drop_stem=True)},
    )
    config = prepare_module._configure_vit_adapter_paths(
        'test', {}, readout=readout, spatial_prior_input=None,
    )
    assert config['spatial_prior_input'] == expected
    assert config['with_high_resolution_stem'] is (readout == 'unet')


def test_configure_vit_adapter_paths_keeps_non_adapter_default(monkeypatch):
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {'test': SimpleNamespace(is_vit_adapter=False, can_drop_stem=False)},
    )
    assert prepare_module._configure_vit_adapter_paths(
        'test', {}, readout='unet', spatial_prior_input=None,
    ) == {}


def test_configure_vit_adapter_paths_rejects_p1_without_refiner():
    with pytest.raises(ValueError, match='requires the U-Net high-resolution refiner'):
        prepare_module._configure_vit_adapter_paths(
            'pumit-vit-adapter',
            {},
            readout='unet-no-refiner',
            spatial_prior_input='p1',
        )


def test_configure_vit_adapter_paths_rejects_adapter_option_for_other_backbone(
    monkeypatch,
):
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {'test': SimpleNamespace(is_vit_adapter=False, can_drop_stem=False)},
    )
    with pytest.raises(ValueError, match='has no ViT-Adapter spatial prior'):
        prepare_module._configure_vit_adapter_paths(
            'test',
            {},
            readout='unet',
            spatial_prior_input='p1',
        )


def test_build_native_architecture_selects_suprem_readout():
    architecture = prepare_module.build_native_architecture(
        _source_architecture(), 'suprem-unet', {}, readout='suprem',
    )
    assert architecture['network_class_name'] == (
        'pumit.downstream.seg.suprem.SupremSegmentationNetwork'
    )
    with pytest.raises(ValueError, match='requires the suprem-unet backbone'):
        prepare_module.build_native_architecture(
            _source_architecture(), 'dinov3', {}, readout='suprem',
        )


def test_build_native_architecture_selects_mask2former_readout():
    architecture = prepare_module.build_native_architecture(
        _source_architecture(),
        'pumit-vit-adapter',
        {'architecture': {'hidden_size': 2}},
        readout='mask2former',
        readout_kwargs={
            'mask_attention_mode': 'classes',
            'query_feature_schedule': 'cycle',
        },
    )

    assert architecture['network_class_name'] == (
        'pumit.downstream.seg.mask2former.PlanAlignedMask2FormerSegmentationNetwork'
    )
    assert architecture['arch_kwargs']['backbone_name'] == 'pumit-vit-adapter'
    assert architecture['arch_kwargs']['mask_attention_mode'] == 'classes'
    assert architecture['arch_kwargs']['query_feature_schedule'] == 'cycle'


def test_build_native_architecture_selects_unet_without_refiner():
    architecture = prepare_module.build_native_architecture(
        _source_architecture(),
        'pumit-vit-adapter',
        {'architecture': {'hidden_size': 2}},
        readout='unet-no-refiner',
    )

    assert architecture['network_class_name'] == (
        'pumit.downstream.seg.network.'
        'PlanAlignedUNetWithoutRefinerSegmentationNetwork'
    )


def test_build_native_architecture_rejects_mask2former_for_other_backbones():
    with pytest.raises(ValueError, match='requires a ViT-Adapter backbone'):
        prepare_module.build_native_architecture(
            _source_architecture(),
            'sam-med3d',
            {'vit_patch_size': [16, 16, 16]},
            readout='mask2former',
        )


def test_build_native_architecture_unet_without_refiner_needs_an_optional_stem():
    architecture = prepare_module.build_native_architecture(
        _source_architecture(),
        'sam-med3d',
        {'vit_patch_size': [16, 16, 16], 'with_high_resolution_stem': False},
        readout='unet-no-refiner',
    )
    assert architecture['network_class_name'].endswith('PlanAlignedUNetWithoutRefinerSegmentationNetwork')

    with pytest.raises(ValueError, match='drop its high-resolution stem'):
        prepare_module.build_native_architecture(
            _source_architecture(),
            'unimiss',
            {},
            readout='unet-no-refiner',
        )


def test_configure_vit_adapter_paths_drops_the_stem_of_simple_fpn_backbones_without_refiner():
    config = prepare_module._configure_vit_adapter_paths(
        'sam-med3d',
        {'vit_patch_size': [16, 16, 16]},
        readout='unet-no-refiner',
        spatial_prior_input='raw',
    )
    assert config == {'vit_patch_size': [16, 16, 16], 'with_high_resolution_stem': False}

    config = prepare_module._configure_vit_adapter_paths(
        'sam-med3d',
        {'vit_patch_size': [16, 16, 16]},
        readout='unet',
        spatial_prior_input='raw',
    )
    assert config == {'vit_patch_size': [16, 16, 16]}


class _FakeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.body = nn.Linear(2, 2)
        self.decoder.seg_layers = nn.Linear(2, 1)

    def load_pretrained(self, weights):
        self.loaded_weights = weights


def test_prepare_experiment_writes_official_nnunet_plan(tmp_path, monkeypatch):
    dataset_folder = tmp_path / 'Dataset001_Test'
    dataset_folder.mkdir()
    source_plans = {
        'plans_name': 'source',
        'dataset_name': 'Dataset001_Test',
        'experiment_planner_used': 'AnyPlanner',
        'configurations': {
            '3d_fullres': {
                'data_identifier': 'nnUNetPlans_3d_fullres',
                'architecture': _source_architecture(),
            }
        },
    }
    dataset_json = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'target': 1}}
    loaded = {
        dataset_folder / 'nnUNetResEncUNetLPlans.json': source_plans,
        dataset_folder / 'dataset.json': dataset_json,
    }
    saved_json = {}

    monkeypatch.setattr(prepare_module, 'nnUNet_preprocessed', tmp_path)
    monkeypatch.setattr(
        prepare_module,
        'maybe_convert_to_dataset_name',
        lambda dataset: 'Dataset001_Test',
    )
    monkeypatch.setattr(prepare_module, 'load_json', lambda path: copy.deepcopy(loaded[path]))
    monkeypatch.setattr(
        prepare_module,
        'save_json',
        lambda value, path, sort_keys: saved_json.update({path: copy.deepcopy(value)}),
    )
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {
            'test': SimpleNamespace(
                requires_vit_patch_size=False,
                is_vit_adapter=False,
                can_drop_stem=False,
                prepare_config=lambda weights, checkpoint_format, gradient_checkpointing: {
                    'architecture': {'hidden_size': 2},
                    'checkpoint_format': checkpoint_format,
                },
                validate_config=lambda config: None,
            )
        },
    )

    plans_identifier = prepare_module.prepare_experiment(
        '1',
        '3d_fullres',
        source_plans_identifier='nnUNetResEncUNetLPlans',
        backbone_name='test',
        weights=tmp_path / 'model.pt',
        checkpoint_format='ucpt',
        gradient_checkpointing=False,
        optimization={
            'optimizer': 'adamw',
            'learning_rate': 3e-4,
            'layer_decay': 0.95,
            'warmup_epochs': 50,
            'weight_decay': 0.05,
            'weight_decay_policy': 'vit_standard',
            'amsgrad': False,
            'lr_scheduler': 'poly',
            'poly_exponent': 0.9,
        },
        oversample_foreground_percent=0.25,
    )

    assert plans_identifier == 'nnUNetResEncUNetLPlans_test'
    derived_plans = saved_json[dataset_folder / f'{plans_identifier}.json']
    assert derived_plans['plans_name'] == plans_identifier
    assert derived_plans['experiment_planner_used'] == 'AnyPlanner'
    assert derived_plans['configurations']['3d_fullres']['data_identifier'] == 'nnUNetPlans_3d_fullres'
    assert derived_plans['configurations']['3d_fullres']['optimization'] == {
        'optimizer': 'adamw',
        'learning_rate': 3e-4,
        'layer_decay': 0.95,
        'warmup_epochs': 50,
        'weight_decay': 0.05,
        'weight_decay_policy': 'vit_standard',
        'amsgrad': False,
        'lr_scheduler': 'poly',
        'poly_exponent': 0.9,
    }
    assert (
        derived_plans['configurations']['3d_fullres'][
            'oversample_foreground_percent'
        ]
        == 0.25
    )
    assert 'downstream_segmentation' not in derived_plans
    assert 'downstream_segmentation_preparation' not in derived_plans
    assert derived_plans['configurations']['3d_fullres']['architecture'] == {
        'network_class_name': prepare_module.NETWORK_CLASS_NAMES['unet'],
        'arch_kwargs': {
            'backbone_name': 'test',
            'backbone_config': {
                'architecture': {'hidden_size': 2},
                'checkpoint_format': 'ucpt',
            },
            'features_per_stage': [4, 8, 16],
            'conv_op': 'torch.nn.Conv3d',
            'kernel_sizes': [[3, 3, 3]] * 3,
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            'n_blocks_per_stage': [1, 3, 4],
            'n_conv_per_stage_decoder': [1, 1],
            'conv_bias': True,
            'norm_op': 'torch.nn.InstanceNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True},
        },
        '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
    }
    assert not (
        dataset_folder / 'nnUNetResEncUNetLPlans_test_3d_fullres_initialization.pth'
    ).exists()

    (dataset_folder / 'nnUNetResEncUNetLPlans_test.json').write_text('{}')
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='test',
            weights=tmp_path / 'model.pt',
            checkpoint_format='ucpt',
            gradient_checkpointing=False,
            optimization={},
        )

    (dataset_folder / 'nnUNetResEncUNetLPlans_test.json').unlink()
    (dataset_folder / 'nnUNetResEncUNetLPlans_test_3d_fullres_initialization.pth').touch()
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='test',
            weights=tmp_path / 'model.pt',
            checkpoint_format='ucpt',
            gradient_checkpointing=False,
            optimization={},
        )


def test_prepare_experiment_rejects_reserved_identifier_separator():
    with pytest.raises(ValueError, match='source_plans_identifier'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='bad__identifier',
            backbone_name='test',
            weights=None,
            checkpoint_format=None,
            gradient_checkpointing=False,
            optimization={},
        )


@pytest.mark.parametrize('value', [-0.1, 1.1, True, '0.25'])
def test_prepare_experiment_rejects_invalid_foreground_oversampling(value):
    with pytest.raises(ValueError, match='must be a number in'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='test',
            weights=None,
            checkpoint_format=None,
            gradient_checkpointing=False,
            optimization={},
            oversample_foreground_percent=value,
        )


@pytest.mark.parametrize(
    ('batch_dice', 'expected'),
    [(None, True), (True, True), (False, False)],
)
def test_prepare_experiment_overrides_batch_dice_only_when_requested(
    tmp_path,
    monkeypatch,
    batch_dice,
    expected,
):
    dataset_folder = tmp_path / 'Dataset001_Test'
    dataset_folder.mkdir()
    source_plans = {
        'plans_name': 'source',
        'dataset_name': 'Dataset001_Test',
        'experiment_planner_used': 'AnyPlanner',
        'configurations': {
            '3d_fullres': {
                'data_identifier': 'nnUNetPlans_3d_fullres',
                'batch_dice': True,
                'architecture': _source_architecture(),
            }
        },
    }
    loaded = {
        dataset_folder / 'nnUNetResEncUNetLPlans.json': source_plans,
        dataset_folder / 'dataset.json': {
            'channel_names': {'0': 'CT'},
            'labels': {'background': 0, 'target': 1},
        },
    }
    saved_json = {}

    monkeypatch.setattr(prepare_module, 'nnUNet_preprocessed', tmp_path)
    monkeypatch.setattr(
        prepare_module,
        'maybe_convert_to_dataset_name',
        lambda dataset: 'Dataset001_Test',
    )
    monkeypatch.setattr(prepare_module, 'load_json', lambda path: copy.deepcopy(loaded[path]))
    monkeypatch.setattr(
        prepare_module,
        'save_json',
        lambda value, path, sort_keys: saved_json.update({path: copy.deepcopy(value)}),
    )
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {
            'test': SimpleNamespace(
                requires_vit_patch_size=False,
                is_vit_adapter=False,
                can_drop_stem=False,
                prepare_config=lambda weights, checkpoint_format, gradient_checkpointing: {
                    'architecture': {'hidden_size': 2},
                },
                validate_config=lambda config: None,
            )
        },
    )

    plans_identifier = prepare_module.prepare_experiment(
        '1',
        '3d_fullres',
        source_plans_identifier='nnUNetResEncUNetLPlans',
        backbone_name='test',
        weights=None,
        checkpoint_format=None,
        gradient_checkpointing=False,
        optimization={},
        batch_dice=batch_dice,
    )

    derived = saved_json[dataset_folder / f'{plans_identifier}.json']
    assert derived['configurations']['3d_fullres']['batch_dice'] is expected


def test_initialize_from_existing_plan_uses_backbone_loader_without_rewriting_plan(
    tmp_path,
    monkeypatch,
):
    dataset_folder = tmp_path / 'Dataset001_Test'
    dataset_folder.mkdir()
    plans_identifier = 'manual_plan'
    plans_path = dataset_folder / f'{plans_identifier}.json'
    plans = {
        'plans_name': plans_identifier,
        'dataset_name': 'Dataset001_Test',
        'configurations': {
            '3d_fullres': {
                'architecture': {
                    'network_class_name': prepare_module.NETWORK_CLASS_NAMES['unet'],
                    'arch_kwargs': {
                        'backbone_name': 'test',
                        'backbone_config': {'architecture': {'hidden_size': 2}},
                    },
                },
            },
        },
    }
    dataset_json = {
        'channel_names': {'0': 'CT'},
        'labels': {'background': 0, 'target': 1},
    }
    plans_path.write_text(json.dumps(plans))
    (dataset_folder / 'dataset.json').write_text(json.dumps(dataset_json))
    original_plan = plans_path.read_bytes()
    fake_network = _FakeNetwork()
    captured = {}

    monkeypatch.setattr(initialize_module, 'nnUNet_preprocessed', tmp_path)
    monkeypatch.setattr(
        initialize_module,
        'maybe_convert_to_dataset_name',
        lambda dataset: 'Dataset001_Test',
    )
    monkeypatch.setattr(
        initialize_module,
        'BACKBONES',
        {'test': SimpleNamespace()},
    )
    monkeypatch.setattr(
        initialize_module,
        'PlansManager',
        lambda value: SimpleNamespace(
            plans=value,
            get_configuration=lambda name: SimpleNamespace(
                network_arch_class_name=prepare_module.NETWORK_CLASS_NAMES['unet'],
            ),
            get_label_manager=lambda value: SimpleNamespace(num_segmentation_heads=2),
        ),
    )
    monkeypatch.setattr(initialize_module, 'determine_num_input_channels', lambda *args: 1)

    def build_network(*args, **kwargs):
        captured['initialization_seed'] = torch.initial_seed()
        return fake_network

    monkeypatch.setattr(
        initialize_module.nnUNetTrainer,
        'build_network_architecture',
        staticmethod(build_network),
    )
    weights = tmp_path / 'source.pt'

    initialization_path = initialize_module.initialize_from_plan(
        '1',
        '3d_fullres',
        plans_identifier,
        weights=weights,
    )

    assert plans_path.read_bytes() == original_plan
    assert captured['initialization_seed'] == initialize_module.INITIALIZATION_SEED
    assert fake_network.loaded_weights == weights
    checkpoint = torch.load(initialization_path, map_location='cpu', weights_only=True)
    assert checkpoint['network_weights'].keys() == {
        'encoder.weight',
        'encoder.bias',
        'decoder.body.weight',
        'decoder.body.bias',
    }
    assert not list(dataset_folder.glob('.*initialization*.tmp'))

    with pytest.raises(FileExistsError, match='refusing to overwrite initialization'):
        initialize_module.initialize_from_plan(
            '1',
            '3d_fullres',
            plans_identifier,
            weights=weights,
        )


def test_save_initialization_does_not_clobber_concurrent_winner(tmp_path, monkeypatch):
    initialization_path = tmp_path / 'initialization.pth'
    winner = b'concurrent winner'

    def concurrent_link(staging_path, destination_path):
        destination_path.write_bytes(winner)
        raise FileExistsError(destination_path)

    monkeypatch.setattr(initialize_module.os, 'link', concurrent_link)

    with pytest.raises(FileExistsError):
        initialize_module._save_initialization_checkpoint(
            {'network_weights': {}},
            initialization_path,
        )

    assert initialization_path.read_bytes() == winner
    assert not list(tmp_path.glob('.*initialization*.tmp'))


def test_prepare_experiment_writes_the_trunk_only_probe_readout(tmp_path, monkeypatch):
    dataset_folder = tmp_path / 'Dataset001_Test'
    dataset_folder.mkdir()
    loaded = {
        dataset_folder / 'nnUNetResEncUNetLPlans.json': {
            'plans_name': 'source',
            'dataset_name': 'Dataset001_Test',
            'experiment_planner_used': 'AnyPlanner',
            'configurations': {
                '3d_fullres': {
                    'data_identifier': 'nnUNetPlans_3d_fullres',
                    'architecture': _source_architecture(),
                }
            },
        },
        dataset_folder / 'dataset.json': {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'target': 1}},
    }
    saved_json = {}
    monkeypatch.setattr(prepare_module, 'nnUNet_preprocessed', tmp_path)
    monkeypatch.setattr(prepare_module, 'maybe_convert_to_dataset_name', lambda dataset: 'Dataset001_Test')
    monkeypatch.setattr(prepare_module, 'load_json', lambda path: copy.deepcopy(loaded[path]))
    monkeypatch.setattr(
        prepare_module,
        'save_json',
        lambda value, path, sort_keys: saved_json.update({path: copy.deepcopy(value)}),
    )
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {
            'test': SimpleNamespace(
                config_keys=frozenset({'architecture', 'feature_layers'}),
                optional_config_keys=frozenset({'with_high_resolution_stem', 'pyramid_branch'}),
                requires_vit_patch_size=False,
                is_vit_adapter=False,
                can_drop_stem=True,
                prepare_config=lambda weights, checkpoint_format, gradient_checkpointing: {
                    'architecture': {'hidden_size': 2},
                    'feature_layers': [6, 12, 18, 24],
                },
                validate_config=lambda config: None,
            ),
            'pyramid': SimpleNamespace(
                config_keys=frozenset(),
                optional_config_keys=frozenset(),
                requires_vit_patch_size=False,
                is_vit_adapter=False,
                can_drop_stem=False,
                prepare_config=lambda weights, checkpoint_format, gradient_checkpointing: {},
                validate_config=lambda config: None,
            ),
        },
    )

    plans_identifier = prepare_module.prepare_experiment(
        '1',
        '3d_fullres',
        source_plans_identifier='nnUNetResEncUNetLPlans',
        backbone_name='test',
        weights=None,
        checkpoint_format=None,
        gradient_checkpointing=False,
        optimization={'optimizer': 'adamw', 'learning_rate': 1e-4},
        readout='unet-no-refiner',
        pyramid_branch='resample',
        num_epochs=200,
        deep_supervision=False,
    )

    configuration = saved_json[dataset_folder / f'{plans_identifier}.json']['configurations']['3d_fullres']
    assert configuration['num_epochs'] == 200
    assert configuration['deep_supervision'] is False
    assert configuration['architecture']['network_class_name'] == prepare_module.NETWORK_CLASS_NAMES['unet-no-refiner']
    assert configuration['architecture']['arch_kwargs']['backbone_config'] == {
        'architecture': {'hidden_size': 2},
        'feature_layers': [6, 12, 18, 24],
        'with_high_resolution_stem': False,
        'pyramid_branch': 'resample',
    }
    # The channel schedule stays the source plan's: the readout projects every trunk onto it.
    assert configuration['architecture']['arch_kwargs']['features_per_stage'] == (
        _source_architecture()['arch_kwargs']['features_per_stage']
    )

    with pytest.raises(ValueError, match='no pyramid branch'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='pyramid',
            weights=None,
            checkpoint_format=None,
            gradient_checkpointing=False,
            optimization={'optimizer': 'adamw', 'learning_rate': 1e-4},
            pyramid_branch='resample',
            output_plans_identifier='other',
        )
    with pytest.raises(ValueError, match='requires --pyramid-branch'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='test',
            weights=None,
            checkpoint_format=None,
            gradient_checkpointing=False,
            optimization={'optimizer': 'adamw', 'learning_rate': 1e-4},
            readout='unet-no-refiner',
            output_plans_identifier='other',
        )


def test_prepare_experiment_sets_the_vit_drop_path_rate(tmp_path, monkeypatch):
    dataset_folder = tmp_path / 'Dataset001_Test'
    dataset_folder.mkdir()
    loaded = {
        dataset_folder / 'nnUNetResEncUNetLPlans.json': {
            'plans_name': 'source',
            'dataset_name': 'Dataset001_Test',
            'experiment_planner_used': 'AnyPlanner',
            'configurations': {
                '3d_fullres': {
                    'data_identifier': 'nnUNetPlans_3d_fullres',
                    'architecture': _source_architecture(),
                }
            },
        },
        dataset_folder / 'dataset.json': {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'target': 1}},
    }
    saved_json = {}
    monkeypatch.setattr(prepare_module, 'nnUNet_preprocessed', tmp_path)
    monkeypatch.setattr(prepare_module, 'maybe_convert_to_dataset_name', lambda dataset: 'Dataset001_Test')
    monkeypatch.setattr(prepare_module, 'load_json', lambda path: copy.deepcopy(loaded[path]))
    monkeypatch.setattr(
        prepare_module,
        'save_json',
        lambda value, path, sort_keys: saved_json.update({path: copy.deepcopy(value)}),
    )
    monkeypatch.setattr(
        prepare_module,
        'BACKBONES',
        {
            'vit': SimpleNamespace(
                config_keys=frozenset({'architecture'}),
                optional_config_keys=frozenset(),
                requires_vit_patch_size=False,
                is_vit_adapter=False,
                can_drop_stem=False,
                prepare_config=lambda weights, checkpoint_format, gradient_checkpointing: {
                    'architecture': {'hidden_size': 2, 'drop_path_rate': 0.0},
                },
                validate_config=lambda config: None,
            ),
            'pyramid': SimpleNamespace(
                config_keys=frozenset(),
                optional_config_keys=frozenset(),
                requires_vit_patch_size=False,
                is_vit_adapter=False,
                can_drop_stem=False,
                prepare_config=lambda weights, checkpoint_format, gradient_checkpointing: {},
                validate_config=lambda config: None,
            ),
        },
    )

    plans_identifier = prepare_module.prepare_experiment(
        '1',
        '3d_fullres',
        source_plans_identifier='nnUNetResEncUNetLPlans',
        backbone_name='vit',
        weights=None,
        checkpoint_format=None,
        gradient_checkpointing=False,
        optimization={'optimizer': 'adamw', 'learning_rate': 1e-4},
        drop_path_rate=0.1,
    )

    configuration = saved_json[dataset_folder / f'{plans_identifier}.json']['configurations']['3d_fullres']
    assert configuration['architecture']['arch_kwargs']['backbone_config'] == {
        'architecture': {'hidden_size': 2, 'drop_path_rate': 0.1},
    }

    with pytest.raises(ValueError, match='no serialized ViT drop_path_rate'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='pyramid',
            weights=None,
            checkpoint_format=None,
            gradient_checkpointing=False,
            optimization={'optimizer': 'adamw', 'learning_rate': 1e-4},
            drop_path_rate=0.1,
            output_plans_identifier='other',
        )
    with pytest.raises(ValueError, match=r'drop_path_rate must be a number in \[0, 1\)'):
        prepare_module.prepare_experiment(
            '1',
            '3d_fullres',
            source_plans_identifier='nnUNetResEncUNetLPlans',
            backbone_name='vit',
            weights=None,
            checkpoint_format=None,
            gradient_checkpointing=False,
            optimization={'optimizer': 'adamw', 'learning_rate': 1e-4},
            drop_path_rate=1.0,
            output_plans_identifier='other',
        )
