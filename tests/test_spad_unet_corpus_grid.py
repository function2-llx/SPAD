"""Tests for the Corpus-grid Universal training contract."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.experiment_planning.experiment_planners.corpus_grid_planner import (
    nnUNetPlannerResEncLCorpusGridIso0p9P224,
    nnUNetPlannerResEncLCorpusGridIso1,
    nnUNetPlannerResEncLCorpusGridIso1P224,
)
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    nnUNetPlannerResEncL,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_crossval_split

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.spad_unet import data as universal_data_module
from pumit.spad_unet import replay as replay_module
from pumit.spad_unet.experiments.corpus_grid import (
    CORPUS_GRID_0P9_P224_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_1P0_P224_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_CONFIGURATION_NAME,
    CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
    collate_corpus_grid_samples,
    validate_corpus_grid_configurations,
    validate_corpus_grid_replay_metadata,
)
from pumit.spad_unet.data import (
    UNIVERSAL_NNUNET_DATASET_NAME,
    UniversalExperimentManifest,
    expected_universal_splits,
    load_universal_datasets,
    prepare_universal_nnunet_namespace,
)
from pumit.spad_unet.experiments.corpus_grid import CorpusGridUniversalTrainer
from pumit.spad_unet.loss import (
    REGION_BALANCED_LOSS_NORMALIZATION,
    UniversalPartialLabelBatch,
    UniversalPartialLabelLoss,
    deep_supervision_weights,
)
from pumit.spad_unet.replay import (
    PrescribedForegroundDataLoader,
    ReplayDataLoader,
    ReplayRecord,
    UniversalReplayBatchLoader,
    UniversalValidationBatchLoader,
    native_augmentation_parameters,
)
from pumit.spad_unet.universal import (
    CanonicalRegionRegistry,
    UniversalResidualEncoderUNet,
    build_universal_resenc_architecture,
)


def _keep_universal_samples(samples):
    return {'samples': samples}


class _FlatReader:
    """Replay reader over an explicit flat record list, recording every slice read."""

    def __init__(self, records: list[ReplayRecord], logical_batch_size: int):
        if len(records) % logical_batch_size:
            raise ValueError('records must fill whole groups')
        self.records = records
        self.logical_batch_size = logical_batch_size
        self.reads = []

    @property
    def total_records(self) -> int:
        return len(self.records)

    def records_at(self, start: int, count: int) -> list[ReplayRecord]:
        if not 0 <= start <= self.total_records - count:
            raise IndexError(f'records [{start}, {start + count}) out of range')
        self.reads.append(start)
        return self.records[start:start + count]


def _group_records(
    groups: tuple[tuple[str, ...], ...],
    foreground: frozenset[str] = frozenset(),
) -> list[ReplayRecord]:
    return [
        ReplayRecord(dataset_id, f'{dataset_id}-case', dataset_id in foreground)
        for group in groups
        for dataset_id in group
    ]


def test_multitalent_subset_with_nnunet_v2_update_budget():
    config_path = Path('configs/downstream/spad_unet/multitalent_ct_universal.json')
    experiment = UniversalExperimentManifest.load(config_path)
    raw = json.loads(config_path.read_text())

    assert len(experiment.datasets) == 12
    assert sum(spec.expected_num_training for spec in experiment.datasets.values()) == 1407
    assert sum(len(spec.regions) for spec in experiment.datasets.values()) == 41
    assert '551' not in experiment.datasets
    assert raw['excluded_datasets']['551']['expected_num_training'] == 50
    assert raw['excluded_cases']['562']['case_identifiers'] == [
        'Pancreas-CT_0025',
        'Pancreas-CT_0070',
    ]
    assert raw['excluded_cases']['546']['case_identifiers'] == ['PANCREAS_0025']
    assert experiment.shared_regions == {}
    assert experiment.fold == 0
    assert experiment.num_updates == 250_000
    assert 'plans_name' not in raw
    assert 'configuration' not in raw
    assert {
        'spacing',
        'patch_size',
        'logical_batch_size',
        'batch_dice',
        'dataset_sampling',
        'cross_validation',
        'training',
    }.isdisjoint(raw)
    assert experiment.datasets['546'].split_policy == 'multitalent_task046'
    assert all(
        spec.split_policy == 'nnunet_seeded'
        for dataset_id, spec in experiment.datasets.items()
        if dataset_id != '546'
    )


def test_spad_ct_universal_v2_manifest_freezes_replacement_suite():
    experiment = UniversalExperimentManifest.load(
        Path('configs/downstream/spad_unet/spad_ct_universal_v2.json')
    )

    assert experiment.dataset_ids == (
        '503',
        '506',
        '507',
        '508',
        '509',
        '510',
        '718',
        '725',
        '555',
        '562',
        '220',
        '518',
    )
    assert sum(
        spec.expected_num_training for spec in experiment.datasets.values()
    ) == 2_004
    assert sum(len(spec.regions) for spec in experiment.datasets.values()) == 52
    assert experiment.datasets['555'].regions == (
        'aorta',
        'esophagus',
        'heart',
        'trachea',
    )
    assert experiment.universal_dataset_id == 591
    assert experiment.universal_dataset_name == 'Dataset591_SPADCTUniversalV2'
    assert all(
        spec.split_policy == 'nnunet_seeded'
        for spec in experiment.datasets.values()
    )


def test_manifest_invariants_apply_to_direct_dataclass_construction():
    manifest = UniversalExperimentManifest.load(
        Path('configs/downstream/spad_unet/multitalent_ct_universal.json')
    )
    with pytest.raises(ValueError, match='fold 0 only'):
        replace(manifest, fold=1)


def test_universal_nnunet_namespace_owns_metadata_and_links_carrier_data(
    tmp_path: Path,
):
    real_root = tmp_path / 'real_preprocessed'
    real_root.mkdir()
    exposed_root = tmp_path / 'preprocessed'
    exposed_root.symlink_to(real_root, target_is_directory=True)
    source_data_folder = real_root / 'Dataset503_Source' / 'carrier_data'
    source_data_folder.mkdir(parents=True)
    namespace = prepare_universal_nnunet_namespace(
        exposed_root,
        source_data_folder,
        ('dataset-503/liver', 'dataset-506/lung_nodule'),
        ('503', '506'),
    )

    assert namespace == exposed_root / UNIVERSAL_NNUNET_DATASET_NAME
    assert (namespace / 'carrier_data').is_symlink()
    assert (namespace / 'carrier_data').resolve() == source_data_folder.resolve()
    dataset_json = json.loads((namespace / 'dataset.json').read_text())
    assert dataset_json['numTraining'] == 0
    assert dataset_json['labels'] == {
        'background': 0,
        'dataset-503/liver': [1],
        'dataset-506/lung_nodule': [2],
    }
    assert dataset_json['regions_class_order'] == [1, 2]
    assert dataset_json['pumit_spad_unet']['source_dataset_ids'] == ['503', '506']


def test_universal_nnunet_namespace_rejects_conflicting_carrier_link(
    tmp_path: Path,
):
    first = tmp_path / 'Dataset503_Source' / 'carrier_data'
    second = tmp_path / 'Dataset506_Source' / 'carrier_data'
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    prepare_universal_nnunet_namespace(
        tmp_path,
        first,
        ('dataset-503/liver',),
        ('503',),
    )

    with pytest.raises(ValueError, match='points to'):
        prepare_universal_nnunet_namespace(
            tmp_path,
            second,
            ('dataset-503/liver',),
            ('503',),
        )


def test_shared_loader_allows_spad_dataset_specific_grids(monkeypatch, tmp_path: Path):
    identifiers = [f'case-{index}' for index in range(5)]
    raw_manifest = json.loads(
        Path('configs/downstream/spad_unet/multitalent_ct_universal.json').read_text()
    )
    patch_sizes = {'503': (32, 32, 32), '506': (48, 32, 32)}
    for dataset_id in patch_sizes:
        raw_manifest['datasets'][dataset_id]['expected_num_training'] = len(identifiers)
        raw_manifest['datasets'][dataset_id]['regions'] = ['organ']
    manifest = UniversalExperimentManifest.from_dict(raw_manifest)
    for dataset_id, patch_size in patch_sizes.items():
        base = tmp_path / f'Dataset{dataset_id}_Test'
        data_folder = base / 'data'
        data_folder.mkdir(parents=True)
        (base / 'dataset.json').write_text(json.dumps({
            'numTraining': len(identifiers),
            'labels': {'background': 0, 'organ': 1},
        }))
        (base / 'plans.json').write_text(json.dumps({
            'data_identifier': 'data',
            'patch_size': patch_size,
        }))
        (base / 'splits_final.json').write_text(json.dumps(
            generate_crossval_split(identifiers)
        ))

    class FakePlansManager:
        def __init__(self, plans):
            self.plans = plans

        def get_configuration(self, name):
            assert name == '3d_fullres'
            return SimpleNamespace(
                network_arch_init_kwargs={'network': 'same'},
                data_identifier=self.plans['data_identifier'],
                spacing=(1.0, 1.0, 1.0),
                patch_size=self.plans['patch_size'],
                batch_size=8,
                batch_dice=False,
            )

    class FakeDataset:
        @staticmethod
        def get_identifiers(folder):
            return identifiers

    monkeypatch.setattr(universal_data_module, 'PlansManager', FakePlansManager)
    monkeypatch.setattr(universal_data_module, 'infer_dataset_class', lambda folder: FakeDataset)

    datasets = load_universal_datasets(
        manifest,
        tmp_path,
        tuple(patch_sizes),
        plans_name='plans',
        configuration_name='3d_fullres',
    )
    assert {dataset_id: dataset.patch_size for dataset_id, dataset in datasets.items()} == patch_sizes
    with pytest.raises(ValueError, match='does not share the frozen Corpus-grid configuration'):
        validate_corpus_grid_configurations({
            dataset_id: dataset.plans
            for dataset_id, dataset in datasets.items()
        })


def test_corpus_grid_validator_compares_complete_configuration():
    reference = {
        'architecture': {
            'network_class_name': 'example.FirstNetwork',
            'arch_kwargs': {'width': 32},
            '_kw_requires_import': ['conv_op'],
        },
        'data_identifier': 'shared',
        'spacing': [1.5, 1.0, 1.0],
        'patch_size': [128, 192, 192],
        'batch_size': 8,
        'batch_dice': False,
        'median_image_size_in_voxels': [200, 210, 220],
    }
    equivalent = {
        **reference,
        'median_image_size_in_voxels': [300, 310, 320],
    }
    validate_corpus_grid_configurations({'503': reference, '506': equivalent})

    different_network = {
        **equivalent,
        'architecture': {
            **equivalent['architecture'],
            'network_class_name': 'example.SecondNetwork',
        },
    }
    with pytest.raises(ValueError, match='does not share the frozen Corpus-grid configuration'):
        validate_corpus_grid_configurations({'503': reference, '506': different_network})


def test_deep_supervision_weights_follow_nnunet_v2():
    weights = deep_supervision_weights(3, torch.device('cpu'))
    expected = torch.tensor([1.0, 0.5, 0.0]) / 1.5
    torch.testing.assert_close(weights, expected)


def test_single_deep_supervision_output_keeps_unit_weight():
    weights = deep_supervision_weights(1, torch.device('cpu'))
    torch.testing.assert_close(weights, torch.ones(1))


def test_eager_ddp_keeps_native_nonzero_last_deep_supervision_weight():
    weights = deep_supervision_weights(
        3,
        torch.device('cpu'),
        keep_last_for_ddp=True,
    )
    expected = torch.tensor([1.0, 0.5, 1e-6])
    torch.testing.assert_close(weights, expected / expected.sum())


def test_default_corpus_grid_source_plan_is_isotropic_1mm():
    planner = object.__new__(nnUNetPlannerResEncLCorpusGridIso1)
    planner.plans_identifier = CORPUS_GRID_SOURCE_PLANS_IDENTIFIER

    assert CORPUS_GRID_SOURCE_PLANS_IDENTIFIER == 'nnUNetResEncUNetLPlans1x1x1FOV192'
    assert planner.batch_size == 4
    assert planner.target_spacing == (1.0, 1.0, 1.0)
    assert planner.patch_size == (192, 192, 192)
    assert tuple(np.multiply(planner.target_spacing, planner.patch_size)) == (
        192.0,
        192.0,
        192.0,
    )
    assert (
        planner.generate_data_identifier('3d_fullres')
        == f'{CORPUS_GRID_SOURCE_PLANS_IDENTIFIER}_3d_fullres'
    )


def test_fixed_0p9_corpus_grid_uses_a_224_cube():
    planner = object.__new__(nnUNetPlannerResEncLCorpusGridIso0p9P224)
    planner.plans_identifier = CORPUS_GRID_0P9_P224_SOURCE_PLANS_IDENTIFIER

    assert planner.batch_size == 2
    assert planner.gpu_memory_target_in_gb_default == 40.0
    assert planner.target_spacing == (0.9, 0.9, 0.9)
    assert planner.patch_size == (224, 224, 224)
    assert tuple(np.multiply(planner.target_spacing, planner.patch_size)) == (
        201.6,
        201.6,
        201.6,
    )
    assert (
        planner.generate_data_identifier('3d_fullres')
        == f'{CORPUS_GRID_0P9_P224_SOURCE_PLANS_IDENTIFIER}_3d_fullres'
    )


def test_fixed_1mm_corpus_grid_uses_a_224_cube():
    planner = object.__new__(nnUNetPlannerResEncLCorpusGridIso1P224)
    planner.plans_identifier = CORPUS_GRID_1P0_P224_SOURCE_PLANS_IDENTIFIER

    assert planner.batch_size == 2
    assert planner.gpu_memory_target_in_gb_default == 40.0
    assert planner.target_spacing == (1.0, 1.0, 1.0)
    assert planner.patch_size == (224, 224, 224)
    assert tuple(np.multiply(planner.target_spacing, planner.patch_size)) == (
        224.0,
        224.0,
        224.0,
    )
    assert (
        planner.generate_data_identifier('3d_fullres')
        == f'{CORPUS_GRID_1P0_P224_SOURCE_PLANS_IDENTIFIER}_3d_fullres'
    )


@pytest.mark.parametrize(
    'planner_cls',
    [
        nnUNetPlannerResEncLCorpusGridIso0p9P224,
        nnUNetPlannerResEncLCorpusGridIso1P224,
    ],
)
def test_fixed_p224_corpus_grid_passes_its_40gb_target_to_resenc_planner(
    monkeypatch,
    planner_cls,
):
    seen = {}

    def fake_init(
        self,
        dataset_name_or_id,
        gpu_memory_target_in_gb,
        preprocessor_name,
        plans_name,
        overwrite_target_spacing,
        suppress_transpose,
    ):
        seen['gpu_memory_target_in_gb'] = gpu_memory_target_in_gb

    monkeypatch.setattr(nnUNetPlannerResEncL, '__init__', fake_init)

    planner_cls('503')

    assert seen['gpu_memory_target_in_gb'] == 40.0
    with pytest.raises(ValueError, match='frozen at the ResEnc L 40 GB target'):
        planner_cls(
            '503',
            gpu_memory_target_in_gb=24,
        )


def test_corpus_grid_trainer_only_overrides_native_integration_hooks():
    assert issubclass(CorpusGridUniversalTrainer, nnUNetTrainer)
    assert 'build_network_architecture' not in CorpusGridUniversalTrainer.__dict__
    assert 'configure_optimizers' not in CorpusGridUniversalTrainer.__dict__
    assert '_compile_network' in CorpusGridUniversalTrainer.__dict__
    assert 'train_step' in CorpusGridUniversalTrainer.__dict__
    assert 'run_training' not in CorpusGridUniversalTrainer.__dict__
    assert 'save_checkpoint' not in CorpusGridUniversalTrainer.__dict__
    assert 'load_checkpoint' not in CorpusGridUniversalTrainer.__dict__
    assert (
        CorpusGridUniversalTrainer.save_checkpoint
        is RetainPeriodicCheckpointsMixin.save_checkpoint
    )


def test_corpus_grid_plan_accepts_a_batch_smaller_than_the_dataset_count():
    metadata = {
        'global_batch_size': 4,
        'samples_per_dataset': 1,
        'sampling_rule': replay_module.COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        'loss_normalization': REGION_BALANCED_LOSS_NORMALIZATION,
    }

    validate_corpus_grid_replay_metadata(metadata, 4)
    validate_corpus_grid_replay_metadata(metadata, 2)
    validate_corpus_grid_replay_metadata(
        {**metadata, 'global_batch_size': 8},
        4,
    )
    validate_corpus_grid_replay_metadata({**metadata, 'global_batch_size': 12}, 4)

    with pytest.raises(ValueError, match='not divisible'):
        validate_corpus_grid_replay_metadata(metadata, 3)
    with pytest.raises(ValueError, match='samples_per_dataset'):
        validate_corpus_grid_replay_metadata({**metadata, 'samples_per_dataset': 2}, 4)
    with pytest.raises(ValueError, match='complementary-foreground'):
        validate_corpus_grid_replay_metadata({**metadata, 'sampling_rule': 'other'}, 4)
    with pytest.raises(ValueError, match='loss normalization'):
        validate_corpus_grid_replay_metadata({**metadata, 'loss_normalization': 'x'}, 4)


def test_corpus_grid_architecture_carries_region_balanced_normalization():
    source_architecture = {
        'network_class_name': (
            'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
        ),
        'arch_kwargs': {
            'strides': [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
        },
    }

    architecture = build_universal_resenc_architecture(
        source_architecture,
        num_canonical_regions=41,
        num_samples_per_global_batch=12,
        batch_dice=False,
        num_task_datasets=12,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
        num_active_regions_per_global_batch=41,
    )

    kwargs = architecture['arch_kwargs']
    assert kwargs['loss_normalization'] == REGION_BALANCED_LOSS_NORMALIZATION
    assert kwargs['num_active_regions_per_global_batch'] == 41
    assert kwargs['tab_feature_level_indices'] == [4, 5]


def test_corpus_grid_compiles_only_the_shape_static_components():
    compiled = []
    network = SimpleNamespace(
        encoder=torch.nn.Identity(),
        task_aware_bottleneck=torch.nn.Identity(),
        decoder=torch.nn.Identity(),
    )
    for name in ('encoder', 'task_aware_bottleneck', 'decoder'):
        module = getattr(network, name)
        module.compile = lambda _name=name, **kwargs: compiled.append((_name, kwargs))
    trainer = object.__new__(CorpusGridUniversalTrainer)

    assert trainer._compile_network(network) is network
    assert compiled == [
        ('encoder', {'dynamic': False, 'mode': 'default'}),
        ('task_aware_bottleneck', {'dynamic': False, 'mode': 'default'}),
        ('decoder', {'dynamic': False, 'mode': 'default'}),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='xFormers attention requires CUDA',
)
def test_native_builder_constructs_canonical_region_bank_from_plans_schema():
    source_architecture = {
        'network_class_name': (
            'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
        ),
        'arch_kwargs': {
            'n_stages': 2,
            'features_per_stage': [4, 8],
            'conv_op': 'torch.nn.Conv3d',
            'kernel_sizes': [[3, 3, 3], [3, 3, 3]],
            'strides': [[1, 1, 1], [2, 2, 2]],
            'n_blocks_per_stage': [1, 1],
            'n_conv_per_stage_decoder': [1],
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
    architecture = build_universal_resenc_architecture(
        source_architecture,
        num_canonical_regions=5,
        num_samples_per_global_batch=2,
        batch_dice=False,
        num_task_datasets=2,
    )
    configuration_manager = SimpleNamespace(
        network_arch_class_name=architecture['network_class_name'],
        network_arch_init_kwargs=architecture['arch_kwargs'],
        network_arch_init_kwargs_req_import=architecture['_kw_requires_import'],
    )

    network = nnUNetTrainer.build_network_architecture(
        plans_manager=None,
        configuration_manager=configuration_manager,
        num_input_channels=1,
        num_output_channels=2,
        enable_deep_supervision=True,
    ).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = network(
            torch.randn(1, 1, 8, 8, 8, device='cuda'),
            torch.tensor([0], device='cuda'),
        )

    assert isinstance(network, UniversalResidualEncoderUNet)
    assert isinstance(outputs, list)
    assert all(output.shape[1] == 5 for output in outputs)

    targets = UniversalPartialLabelBatch(
        [torch.randint(0, 2, (1, 5, 8, 8, 8), device='cuda').float()],
        [torch.arange(5, device='cuda')],
    )
    with torch.autocast('cuda', dtype=torch.bfloat16):
        joint_outputs, loss = network(
            torch.randn(1, 1, 8, 8, 8, device='cuda'),
            torch.tensor([1], device='cuda'),
            targets,
        )
    assert isinstance(joint_outputs, list)
    assert loss.ndim == 0
    assert loss.isfinite()


def test_corpus_grid_ddp_keeps_lowest_deep_supervision_head_in_graph(
    monkeypatch,
):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
    network = UniversalResidualEncoderUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=2,
        num_samples_per_global_batch=12,
        batch_dice=False,
        num_task_datasets=2,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
        num_active_regions_per_global_batch=2,
        tab_tokens_per_dataset=2,
        tab_dim=8,
        tab_depth=1,
        tab_heads=2,
        tab_mlp_dim=16,
        tab_attention_downsample_rate=2,
        tab_fourier_scale=1.0,
        tab_fourier_seed=1,
        n_stages=2,
        features_per_stage=(4, 8),
        conv_op=torch.nn.Conv3d,
        kernel_sizes=((3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2)),
        n_blocks_per_stage=(1, 1),
        n_conv_per_stage_decoder=(1,),
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
    )

    assert network.partial_label_loss.keep_last_for_ddp is True
    assert network.partial_label_loss.ddp_world_size == 4
    assert (
        network.partial_label_loss.loss_normalization
        == REGION_BALANCED_LOSS_NORMALIZATION
    )
    assert network.partial_label_loss.global_active_region_count == 2


def test_omitting_the_region_count_selects_the_dynamic_divisor(monkeypatch):
    """A batch smaller than the suite cannot know its active-region total up front."""
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
    network = UniversalResidualEncoderUNet(
        input_channels=1,
        num_classes=1,
        num_canonical_regions=2,
        num_samples_per_global_batch=4,
        batch_dice=False,
        num_task_datasets=2,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
        num_active_regions_per_global_batch=None,
        tab_tokens_per_dataset=2,
        tab_dim=8,
        tab_depth=1,
        tab_heads=2,
        tab_mlp_dim=16,
        tab_attention_downsample_rate=2,
        tab_fourier_scale=1.0,
        tab_fourier_seed=1,
        n_stages=2,
        features_per_stage=(4, 8),
        conv_op=torch.nn.Conv3d,
        kernel_sizes=((3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2)),
        n_blocks_per_stage=(1, 1),
        n_conv_per_stage_decoder=(1,),
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
    )
    loss_module = network.partial_label_loss

    assert loss_module.global_active_region_count is None
    assert loss_module.keep_last_for_ddp is True
    # The dynamic divisor reads world size from the collective, not from construction.
    assert loss_module.ddp_world_size == 1


def test_joint_train_step_accepts_partial_label_batch():
    loss_module = UniversalPartialLabelLoss(batch_dice=False)

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 3, kernel_size=1)

        def forward(self, data, dataset_indices, targets=None):
            logits = self.conv(data)
            if targets is None:
                return logits
            return logits, loss_module(logits, targets)

    trainer = object.__new__(CorpusGridUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = Network()
    trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.1)
    trainer.grad_scaler = None
    trainer.loss = loss_module
    trainer.dataset_index_by_id = {'503': 0, '506': 1}
    before = trainer.network.conv.weight.detach().clone()
    batch = {
        'data': torch.randn(2, 1, 4, 4, 4),
        'dataset_ids': ('503', '506'),
        'target': UniversalPartialLabelBatch(
            [
                torch.randint(0, 2, (1, 2, 4, 4, 4)).float(),
                torch.randint(0, 2, (1, 1, 4, 4, 4)).float(),
            ],
            [torch.tensor([0, 1]), torch.tensor([2])],
        ),
    }

    result = trainer.train_step(batch)

    assert result['loss'].ndim == 0
    assert not torch.equal(before, trainer.network.conv.weight)


def test_corpus_grid_validation_scatter_maps_metrics_to_canonical_rows():
    registry = CanonicalRegionRegistry(
        {
            '503': {'labels': {'background': 0, 'a': 1}},
            '506': {'labels': {'background': 0, 'b': 1}},
        },
        shared_regions={},
    )

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss = UniversalPartialLabelLoss(batch_dice=False)

        def forward(self, data, dataset_indices, targets=None):
            logits = torch.full((2, 2, *data.shape[2:]), -10.0)
            logits[0, 0] = 10
            logits[1, 1] = 10
            if targets is not None:
                return logits, self.loss(logits, targets)
            return logits

    trainer = object.__new__(CorpusGridUniversalTrainer)
    trainer.device = torch.device('cpu')
    trainer.network = Network()
    trainer.loss = trainer.network.loss
    trainer.registry = registry
    trainer.dataset_index_by_id = {'503': 0, '506': 1}
    batch = {
        'data': torch.zeros(2, 1, 2, 2, 2),
        'dataset_ids': ('503', '506'),
        'target': UniversalPartialLabelBatch(
            [
                torch.ones(1, 1, 2, 2, 2),
                torch.zeros(1, 1, 2, 2, 2),
            ],
            [registry.indices('503'), registry.indices('506')],
        ),
    }

    result = trainer.validation_step(batch)

    assert result['tp_hard'].tolist() == [8, 0]
    assert result['fp_hard'].tolist() == [0, 8]
    assert result['fn_hard'].tolist() == [0, 0]


def test_validation_batches_follow_the_stream_order_and_cycle():
    patch_size = (2, 2, 2)
    dataset_jsons = {
        '503': {'labels': {'background': 0, 'a': 1}},
        '506': {'labels': {'background': 0, 'b': 1}},
        '507': {'labels': {'background': 0, 'c': 1}},
        '508': {'labels': {'background': 0, 'd': 1}},
    }
    registry = CanonicalRegionRegistry(dataset_jsons, shared_regions={})

    class Loader:
        def __init__(self, dataset_id: str):
            self.dataset_id = dataset_id
            self.force_fg = None

        def set_next_force_fg(self, force_fg: bool):
            self.force_fg = force_fg

        def __next__(self):
            assert self.force_fg is not None
            self.force_fg = None
            return {
                'data': torch.zeros(1, 1, *patch_size),
                'target': torch.zeros(1, 1, *patch_size),
                'keys': [f'{self.dataset_id}-case'],
            }

    loader = UniversalValidationBatchLoader(
        _FlatReader(
            _group_records((
                ('503', '506', '507', '508'),
                ('508', '506', '503', '507'),
            )),
            logical_batch_size=4,
        ),
        {dataset_id: Loader(dataset_id) for dataset_id in dataset_jsons},
        {
            dataset_id: registry.indices(dataset_id)
            for dataset_id in dataset_jsons
        },
        global_batch_size=4,
        local_batch_size=2,
        record_offset=0,
        collate_fn=_keep_universal_samples,
    )

    def next_ids():
        return tuple(sample.dataset_id for sample in next(loader)['samples'])

    assert next_ids() == ('503', '506')
    assert next_ids() == ('508', '506')
    assert next_ids() == ('503', '506')


def test_replay_batch_prescribes_the_cases_named_by_the_flat_stream():
    patch_size = (2, 2, 2)
    registry = CanonicalRegionRegistry(
        {
            '503': {'labels': {'background': 0, 'a': 1}},
            '506': {'labels': {'background': 0, 'b': 1}},
        },
        shared_regions={},
    )
    reader = _FlatReader(
        [
            record
            for step in range(8)
            for record in (
                ReplayRecord('503', f'case-a{step}', False),
                ReplayRecord('506', f'case-b{step}', True),
            )
        ],
        logical_batch_size=2,
    )

    class Loader(ReplayDataLoader):
        def __init__(self, expected_force_fg: bool):
            self.expected_force_fg = expected_force_fg
            self.next_case = None
            self.force_fg = None
            self.consumed_cases = []

        def set_next_case(self, case_id: str):
            self.next_case = case_id

        def set_next_force_fg(self, force_fg: bool):
            self.force_fg = force_fg

        def __next__(self):
            assert self.force_fg is self.expected_force_fg
            case_id = self.next_case
            self.next_case = None
            self.force_fg = None
            self.consumed_cases.append(case_id)
            return {
                'data': torch.zeros(1, 1, *patch_size),
                'target': torch.zeros(1, 1, *patch_size),
                'keys': [case_id],
            }

    native_loaders = {
        '503': Loader(False),
        '506': Loader(True),
    }
    loader = UniversalReplayBatchLoader(
        reader,
        native_loaders,
        {
            dataset_id: registry.indices(dataset_id)
            for dataset_id in native_loaders
        },
        global_batch_size=2,
        local_batch_size=2,
        record_offset=0,
        start_step=7,
        collate_fn=collate_corpus_grid_samples,
    )

    batch = next(loader)
    assert set(batch) == {'data', 'dataset_ids', 'target'}
    assert batch['dataset_ids'] == ('503', '506')
    assert reader.reads == [14]
    assert native_loaders['503'].consumed_cases == ['case-a7']
    assert native_loaders['506'].consumed_cases == ['case-b7']


def test_each_ddp_rank_consumes_its_own_slice_of_the_permuted_group():
    patch_size = (2, 2, 2)
    dataset_ids = ('503', '506', '507', '508')
    registry = CanonicalRegionRegistry(
        {
            dataset_id: {'labels': {'background': 0, dataset_id: 1}}
            for dataset_id in dataset_ids
        },
        shared_regions={},
    )
    # The final group is permuted, so neither rank owns a fixed pair of datasets.
    groups = (*((dataset_ids,) * 7), ('507', '503', '508', '506'))
    reader = _FlatReader(
        _group_records(groups, frozenset({'508'})),
        logical_batch_size=4,
    )

    class Loader(ReplayDataLoader):
        def __init__(self):
            self.next_case = None
            self.force_fg = None
            self.consumed_cases = []

        def set_next_case(self, case_id: str):
            self.next_case = case_id

        def set_next_force_fg(self, force_fg: bool):
            self.force_fg = force_fg

        def __next__(self):
            assert self.force_fg is not None
            self.force_fg = None
            self.consumed_cases.append(self.next_case)
            return {
                'data': torch.zeros(1, 1, *patch_size),
                'target': torch.zeros(1, 1, *patch_size),
                'keys': [self.next_case],
            }

    rank_loaders = [
        {dataset_id: Loader() for dataset_id in dataset_ids}
        for _ in range(2)
    ]
    rank_batches = [
        next(UniversalReplayBatchLoader(
            reader,
            rank_loaders[rank],
            {
                dataset_id: registry.indices(dataset_id)
                for dataset_id in dataset_ids
            },
            global_batch_size=4,
            local_batch_size=2,
            record_offset=2 * rank,
            start_step=7,
            collate_fn=collate_corpus_grid_samples,
        ))
        for rank in range(2)
    ]
    assert all(
        set(batch) == {'data', 'dataset_ids', 'target'}
        for batch in rank_batches
    )
    assert reader.reads == [28, 30]
    assert rank_batches[0]['dataset_ids'] == ('507', '503')
    assert rank_batches[1]['dataset_ids'] == ('508', '506')
    assert {
        dataset_id
        for batch in rank_batches
        for dataset_id in batch['dataset_ids']
    } == set(dataset_ids)


def test_validation_batch_partitions_global_slots_across_ddp_ranks():
    patch_size = (2, 2, 2)
    dataset_jsons = {
        dataset_id: {'labels': {'background': 0, dataset_id: 1}}
        for dataset_id in ('503', '506', '507', '508')
    }
    registry = CanonicalRegionRegistry(dataset_jsons, shared_regions={})

    class Loader:
        def __init__(self, dataset_id: str):
            self.dataset_id = dataset_id
            self.force_fg = None

        def set_next_force_fg(self, force_fg: bool):
            self.force_fg = force_fg

        def __next__(self):
            assert self.force_fg is not None
            self.force_fg = None
            return {
                'data': torch.zeros(1, 1, *patch_size),
                'target': torch.zeros(1, 1, *patch_size),
                'keys': [f'{self.dataset_id}-case'],
            }

    reader = _FlatReader(
        _group_records((('503', '506', '507', '508'),)),
        logical_batch_size=4,
    )
    rank_loaders = [
        UniversalValidationBatchLoader(
            reader,
            {
                dataset_id: Loader(dataset_id)
                for dataset_id in dataset_jsons
            },
            {
                dataset_id: registry.indices(dataset_id)
                for dataset_id in dataset_jsons
            },
            global_batch_size=4,
            local_batch_size=2,
            record_offset=2 * rank,
            collate_fn=_keep_universal_samples,
        )
        for rank in range(2)
    ]
    def sample_ids(batch):
        return tuple(sample.dataset_id for sample in batch['samples'])

    # A single-group stream cycles, so each rank keeps returning its own slice.
    assert sample_ids(next(rank_loaders[0])) == ('503', '506')
    assert sample_ids(next(rank_loaders[1])) == ('507', '508')
    assert sample_ids(next(rank_loaders[0])) == ('503', '506')
    assert sample_ids(next(rank_loaders[1])) == ('507', '508')


def test_validation_batches_tile_each_replay_group_across_ranks():
    patch_size = (2, 2, 2)
    dataset_ids = tuple(f'{dataset_id:03d}' for dataset_id in range(12))
    dataset_jsons = {
        dataset_id: {'labels': {'background': 0, dataset_id: 1}}
        for dataset_id in dataset_ids
    }
    registry = CanonicalRegionRegistry(dataset_jsons, shared_regions={})

    class Loader:
        def __init__(self, dataset_id: str):
            self.dataset_id = dataset_id
            self.force_fg = None

        def set_next_force_fg(self, force_fg: bool):
            self.force_fg = force_fg

        def __next__(self):
            assert self.force_fg is not None
            self.force_fg = None
            return {
                'data': torch.zeros(1, 1, *patch_size),
                'target': torch.zeros(1, 1, *patch_size),
                'keys': [f'{self.dataset_id}-case'],
            }

    foreground = frozenset(dataset_ids[6:])
    reader = _FlatReader(
        _group_records((dataset_ids,), foreground),
        logical_batch_size=12,
    )
    # Global batch 4 over 2 ranks: one replay group spans three validation steps.
    rank_loaders = [
        UniversalValidationBatchLoader(
            reader,
            {
                dataset_id: Loader(dataset_id)
                for dataset_id in dataset_ids
            },
            {
                dataset_id: registry.indices(dataset_id)
                for dataset_id in dataset_ids
            },
            global_batch_size=4,
            local_batch_size=2,
            record_offset=2 * rank,
            collate_fn=_keep_universal_samples,
        )
        for rank in range(2)
    ]
    group_counts = Counter()
    foreground_counts = Counter()
    for _ in range(3):
        for loader in rank_loaders:
            batch = next(loader)
            for sample in batch['samples']:
                group_counts[sample.dataset_id] += 1
                if sample.dataset_id in foreground:
                    foreground_counts[sample.dataset_id] += 1

    assert group_counts == Counter({dataset_id: 1 for dataset_id in dataset_ids})
    assert foreground_counts == Counter({dataset_id: 1 for dataset_id in foreground})


def test_prescribed_foreground_flag_is_consumed_exactly_once():
    loader = object.__new__(PrescribedForegroundDataLoader)
    loader._prescribed_force_fg = None

    loader.set_next_force_fg(True)
    assert loader._get_prescribed_force_fg(0) is True
    with pytest.raises(RuntimeError, match='set_next_force_fg'):
        loader._get_prescribed_force_fg(0)

    loader.set_next_force_fg(False)
    with pytest.raises(RuntimeError, match='not consumed'):
        loader.set_next_force_fg(True)


def test_get_dataloaders_reuses_native_augmentation_contract(monkeypatch):
    patch_size = (128, 192, 192)
    initial_patch_size = (243, 308, 270)
    rotation = (-0.5, 0.5)
    mirror_axes = (0, 1, 2)
    registry = CanonicalRegionRegistry(
        {'503': {'labels': {'background': 0, 'liver': 1}}},
        shared_regions={},
    )
    trainer = object.__new__(CorpusGridUniversalTrainer)
    trainer.batch_size = 1
    trainer.configuration_manager = SimpleNamespace(batch_size=1)
    trainer.datasets = {'503': SimpleNamespace(patch_size=patch_size)}
    trainer.registry = registry
    trainer.current_epoch = 0
    trainer.num_iterations_per_epoch = 250
    trainer.num_val_iterations_per_epoch = 50
    trainer.replay_reader = SimpleNamespace(
        logical_batch_size=1,
        dataset_ids=('503',),
    )
    trainer.configuration_name = '3d_fullres'
    trainer.oversample_foreground_percent = 0.5
    trainer.device = torch.device('cpu')
    trainer.is_ddp = False
    calls = []

    def build_training_loader(
        dataset,
        configuration_name,
        initial_patch_size,
        rotation_for_da,
        mirror_axes,
        do_dummy_2d_data_aug,
        foreground_oversample_probability,
    ):
        calls.append((
            'train',
            initial_patch_size,
            rotation_for_da,
            mirror_axes,
            do_dummy_2d_data_aug,
            foreground_oversample_probability,
        ))
        return object()

    def build_validation_loader(
        dataset,
        configuration_name,
        foreground_oversample_probability,
    ):
        calls.append(('validation', foreground_oversample_probability))
        return object()

    monkeypatch.setattr(replay_module, 'get_allowed_n_proc_DA', lambda: 0)
    monkeypatch.setattr(
        replay_module,
        'native_augmentation_parameters',
        lambda patch_size: (rotation, False, initial_patch_size, mirror_axes),
    )
    monkeypatch.setattr(replay_module, 'build_training_loader', build_training_loader)
    monkeypatch.setattr(replay_module, 'build_validation_loader', build_validation_loader)

    trainer.get_dataloaders()

    assert trainer.inference_allowed_mirroring_axes == mirror_axes
    assert calls == [
        (
            'train',
            initial_patch_size,
            rotation,
            mirror_axes,
            False,
            0.5,
        ),
        ('validation', 0.5),
    ]


def test_native_augmentation_contract_for_corpus_grid_patch():
    rotation, dummy_2d, initial_patch_size, mirror_axes = (
        native_augmentation_parameters((128, 192, 192))
    )

    assert rotation == pytest.approx((-np.pi / 6, np.pi / 6))
    assert dummy_2d is False
    assert tuple(initial_patch_size) == (243, 308, 270)
    assert mirror_axes == (0, 1, 2)


def test_replay_workers_partition_steps_without_changing_global_order():
    patch_size = (2, 2, 2)
    registry = CanonicalRegionRegistry(
        {'503': {'labels': {'background': 0, 'a': 1}}},
        shared_regions={},
    )

    class Loader(ReplayDataLoader):
        def __init__(self):
            self.next_case = None
            self.force_fg = None

        def set_next_case(self, case_id: str):
            self.next_case = case_id

        def set_next_force_fg(self, force_fg: bool):
            self.force_fg = force_fg

        def __next__(self):
            assert self.force_fg is False
            self.force_fg = None
            return {
                'data': torch.zeros(1, 1, *patch_size),
                'target': torch.zeros(1, 1, *patch_size),
                'keys': [self.next_case],
            }

    def make_worker(thread_id: int):
        reader = _FlatReader(_group_records((('503',),) * 20), logical_batch_size=1)
        worker = UniversalReplayBatchLoader(
            reader,
            {'503': Loader()},
            {'503': registry.indices('503')},
            global_batch_size=1,
            local_batch_size=1,
            record_offset=0,
            start_step=10,
            collate_fn=collate_corpus_grid_samples,
        )
        worker.configure_worker_pool(3)
        worker.set_thread_id(thread_id)
        return worker, reader

    workers_and_readers = [make_worker(thread_id) for thread_id in range(3)]
    for _ in range(2):
        for worker, _ in workers_and_readers:
            next(worker)
    assert [
        start
        for _, reader in workers_and_readers
        for start in reader.reads
    ] == [10, 13, 11, 14, 12, 15]


def test_ordered_augmenter_stops_cleanly_after_finite_replay():
    patch_size = (2, 2, 2)
    reader = _FlatReader(
        [ReplayRecord('503', f'case-{step}', False) for step in range(4)],
        logical_batch_size=1,
    )

    class Loader(ReplayDataLoader):
        def __init__(self):
            self.next_case = None

        def set_next_case(self, case_id: str):
            self.next_case = case_id

        def set_next_force_fg(self, force_fg: bool):
            pass

        def __next__(self):
            value = float(self.next_case.rsplit('-', 1)[1])
            return {
                'data': torch.full((1, 1, *patch_size), value),
                'target': torch.zeros(1, 1, *patch_size),
            }

    source = UniversalReplayBatchLoader(
        reader,
        {'503': Loader()},
        {'503': torch.tensor([0])},
        global_batch_size=1,
        local_batch_size=1,
        record_offset=0,
        start_step=0,
        collate_fn=collate_corpus_grid_samples,
    )
    source.configure_worker_pool(2)
    augmenter = MultiThreadedAugmenter(
        source,
        None,
        num_processes=2,
        num_cached_per_queue=1,
        seeds=None,
        pin_memory=False,
        wait_time=0.002,
    )
    try:
        values = [
            float(next(augmenter)['data'][0, 0, 0, 0, 0])
            for _ in range(4)
        ]
        with pytest.raises(StopIteration):
            next(augmenter)
    finally:
        augmenter._finish(timeout=1, force=True)

    assert values == [0, 1, 2, 3]


def test_task046_split_inherits_task017_and_task062(tmp_path: Path):
    experiment = UniversalExperimentManifest.load(
        Path('configs/downstream/spad_unet/multitalent_ct_universal.json')
    )
    task017_identifiers = [f'img{i:04d}' for i in range(5)]
    task017_splits = [
        {
            'train': [identifier for identifier in task017_identifiers if identifier != task017_identifiers[fold]],
            'val': [task017_identifiers[fold]],
        }
        for fold in range(5)
    ]
    task017_base = tmp_path / 'Dataset517_BTCV'
    task017_base.mkdir()
    (task017_base / 'splits_final.json').write_text(json.dumps(task017_splits))
    pancreas_identifiers = [f'PANCREAS_{index:04d}' for index in range(5)]
    task062_identifiers = [f'Pancreas-CT_{index:04d}' for index in range(8)]
    task062_base = tmp_path / 'Dataset562_NIHPancreas'
    data_identifier = f'{CORPUS_GRID_SOURCE_PLANS_IDENTIFIER}_3d_fullres'
    task062_data = task062_base / data_identifier
    task062_data.mkdir(parents=True)
    plans = {
        'configurations': {
            '3d_fullres': {
                'data_identifier': data_identifier,
            },
        },
    }
    (task062_base / f'{CORPUS_GRID_SOURCE_PLANS_IDENTIFIER}.json').write_text(json.dumps(plans))
    for identifier in task062_identifiers:
        (task062_data / f'{identifier}.b2nd').touch()

    splits = expected_universal_splits(
        experiment,
        experiment.datasets['546'],
        [*task017_identifiers, *pancreas_identifiers],
        tmp_path,
        plans_name=CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
        configuration_name=CORPUS_GRID_CONFIGURATION_NAME,
    )

    task062_splits = generate_crossval_split(task062_identifiers)
    for fold, split in enumerate(splits):
        task062_val = set(task062_splits[fold]['val'])
        expected_pancreas_val = [
            identifier
            for identifier in pancreas_identifiers
            if f'Pancreas-CT_{identifier.removeprefix("PANCREAS_")}' in task062_val
        ]
        assert split['val'] == [task017_identifiers[fold], *expected_pancreas_val]
        assert set(split['train']) | set(split['val']) == set(task017_identifiers + pancreas_identifiers)
        assert not set(split['train']) & set(split['val'])


def test_batched_forward_with_loss_module():
    """One batched forward + UniversalPartialLabelLoss produces valid loss and gradients."""
    torch.manual_seed(0)
    B, C_full, S = 4, 10, 8

    # Synthetic registry: 3 datasets with 3, 4, 3 channels (total 10)
    dataset_jsons = {
        '001': {'labels': {'background': 0, 'a': 1, 'b': 2, 'c': 3}},
        '002': {'labels': {'background': 0, 'x': 1, 'y': 2, 'z': 3, 'w': 4}},
        '003': {'labels': {'background': 0, 'p': 1, 'q': 2, 'r': 3}},
    }
    registry = CanonicalRegionRegistry(dataset_jsons, shared_regions={})
    assert len(registry) == C_full

    # Minimal network mock: identity-like module returning DS outputs
    class MockNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, C_full, 1)

        def forward(self, x):
            out = self.conv(x)
            return [out, out[:, :, ::2, ::2, ::2]]

    model = MockNetwork()
    loss_module = UniversalPartialLabelLoss(batch_dice=False)

    # Simulate 4 samples from datasets: 001, 002, 003, 001
    dataset_ids = ('001', '002', '003', '001')
    images = [torch.randn(1, 1, S, S, S) for _ in range(B)]
    targets = []
    region_indices_list = []
    for did in dataset_ids:
        n_ch = len(registry.dataset_mappings[did].local_names)
        targets.append(torch.randint(0, 2, (1, n_ch, S, S, S)).float())
        region_indices_list.append(registry.indices(did))

    stacked = torch.cat(images, dim=0)
    canonical_logits = model(stacked)
    loss = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, region_indices_list),
    )

    assert loss.ndim == 0
    assert loss.isfinite()
    loss.backward()
    assert model.conv.weight.grad is not None
    assert model.conv.weight.grad.abs().sum() > 0
