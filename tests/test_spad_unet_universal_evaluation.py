"""Tests for shared Universal full-volume evaluation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from pumit.spad_unet import evaluation as evaluation_module
from pumit.spad_unet.experiments.corpus_grid import (
    CORPUS_GRID_1P5_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_CONFIGURATION_NAME,
    CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
    validate_corpus_grid_configurations,
)
from pumit.spad_unet.data import (
    UniversalExperimentManifest,
    build_region_registry,
    get_preprocessed_root,
    load_universal_datasets,
)
from pumit.spad_unet.evaluation import (
    ActiveRegionNetwork,
    binary_region_metrics,
    partition_validation_jobs,
    predict_source_grid_probabilities,
    region_masks_to_segmentation,
    restore_region_probabilities,
    selected_regions_class_order,
    summarize_universal_metrics,
    validate_source_and_ground_truth_geometry,
)
from pumit.spad_unet.experiments import corpus_grid as corpus_grid_module
from pumit.spad_unet.experiments.corpus_grid import CorpusGridUniversalTrainer
from pumit.spad_unet.universal import CanonicalRegionRegistry


def test_fold0_evaluation_contract_has_12_datasets_285_cases_41_rows():
    experiment = UniversalExperimentManifest.load(
        Path('configs/downstream/spad_unet/multitalent_ct_universal.json')
    )
    datasets = load_universal_datasets(
        experiment,
        get_preprocessed_root(),
        plans_name=CORPUS_GRID_1P5_SOURCE_PLANS_IDENTIFIER,
        configuration_name=CORPUS_GRID_CONFIGURATION_NAME,
    )
    validate_corpus_grid_configurations({
        dataset_id: dataset.plans['configurations'][CORPUS_GRID_CONFIGURATION_NAME]
        for dataset_id, dataset in datasets.items()
    })
    registry = build_region_registry(experiment, datasets)

    assert len(datasets) == 12
    assert sum(len(dataset.validation_identifiers) for dataset in datasets.values()) == 285
    assert len(registry) == 41
    with pytest.raises(ValueError, match='complete ordered Universal dataset suite'):
        build_region_registry(experiment, {'503': datasets['503']})


def test_v2_fold0_evaluation_contract_has_405_cases_and_52_rows():
    experiment = UniversalExperimentManifest.load(
        Path('configs/downstream/spad_unet/spad_ct_universal_v2.json')
    )
    datasets = load_universal_datasets(
        experiment,
        get_preprocessed_root(),
        plans_name=CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
        configuration_name=CORPUS_GRID_CONFIGURATION_NAME,
    )
    registry = build_region_registry(experiment, datasets)

    assert len(datasets) == 12
    assert sum(
        len(dataset.validation_identifiers) for dataset in datasets.values()
    ) == 405
    assert len(registry) == 52


def test_active_region_network_selects_rows_from_universal_logits():
    class Network(torch.nn.Module):
        def forward(self, x, dataset_indices):
            assert dataset_indices.tolist() == [2, 2]
            return torch.arange(5, dtype=x.dtype).view(1, 5, 1, 1, 1).expand(
                x.shape[0],
                5,
                *x.shape[2:],
            )

    network = ActiveRegionNetwork(Network(), torch.tensor([3, 1]), dataset_index=2)
    output = network(torch.zeros(2, 1, 2, 2, 2))

    assert output.shape == (2, 2, 2, 2, 2)
    assert torch.all(output[:, 0] == 3)
    assert torch.all(output[:, 1] == 1)


def test_active_region_network_rejects_deep_supervision_outputs():
    class Network(torch.nn.Module):
        def forward(self, x, dataset_indices):
            return [x]

    network = ActiveRegionNetwork(
        Network(),
        torch.tensor([0]),
        dataset_index=0,
    )

    with pytest.raises(TypeError, match='deep supervision'):
        network(torch.zeros(1, 1, 1, 1, 1))


def test_native_predictor_mirroring_uses_all_flip_combinations():
    class CountingIdentity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x

    predictor = nnUNetPredictor(
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device('cpu'),
    )
    predictor.network = CountingIdentity()
    predictor.allowed_mirroring_axes = (0, 1, 2)
    x = torch.randn(1, 1, 2, 3, 4)

    torch.testing.assert_close(predictor._internal_maybe_mirror_and_predict(x), x)
    assert predictor.network.calls == 8


def test_native_predictor_batches_sliding_windows_without_changing_output():
    class RecordingIdentity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, x):
            self.batch_sizes.append(x.shape[0])
            return x

    data = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    outputs = []
    network_batch_sizes = {}
    for batch_size in (1, 8):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device('cpu'),
            allow_tqdm=False,
            sliding_window_batch_size=batch_size,
        )
        predictor.network = RecordingIdentity()
        predictor.configuration_manager = SimpleNamespace(
            patch_size=(2, 2, 2),
        )
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=1)
        slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
        outputs.append(
            predictor._internal_predict_sliding_window_return_logits(
                data,
                slicers,
                do_on_device=False,
            )
        )
        network_batch_sizes[batch_size] = predictor.network.batch_sizes

    assert outputs[0].shape == data.shape
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert network_batch_sizes[8] == [8, 8, 8, 3]


def test_native_predictor_propagates_window_producer_errors(monkeypatch):
    predictor = nnUNetPredictor(
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device('cpu'),
        allow_tqdm=False,
        sliding_window_batch_size=2,
    )
    predictor.network = torch.nn.Identity()
    predictor.configuration_manager = SimpleNamespace(patch_size=(2, 2, 2))
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=1)
    data = torch.zeros(1, 4, 4, 4)
    slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
    monkeypatch.setattr(
        torch,
        'stack',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError('producer failed')
        ),
    )

    with pytest.raises(RuntimeError, match='producer failed'):
        predictor._internal_predict_sliding_window_return_logits(
            data,
            slicers,
            do_on_device=False,
        )


def test_predictor_manual_initialization_can_skip_outer_compile(monkeypatch):
    monkeypatch.setenv('nnUNet_compile', 'true')
    compile_calls = []
    compiled_network = torch.nn.Identity()
    monkeypatch.setattr(
        torch,
        'compile',
        lambda network: compile_calls.append(network) or compiled_network,
    )
    plans_manager = SimpleNamespace(
        get_label_manager=lambda dataset_json: SimpleNamespace(),
    )
    network = torch.nn.Identity()
    args = (
        network,
        plans_manager,
        SimpleNamespace(),
        None,
        {},
        'Trainer',
        None,
    )

    predictor = nnUNetPredictor(
        perform_everything_on_device=False,
        device=torch.device('cpu'),
    )
    predictor.manual_initialization(*args, allow_compile=False)
    assert predictor.network is network
    assert compile_calls == []

    predictor = nnUNetPredictor(
        perform_everything_on_device=False,
        device=torch.device('cpu'),
    )
    predictor.manual_initialization(*args)
    assert predictor.network is compiled_network
    assert compile_calls == [network]


def test_universal_evaluator_disables_predictor_outer_compile(monkeypatch, tmp_path):
    class InitializationObserved(Exception):
        pass

    captured = {}

    class Predictor:
        def __init__(self, **kwargs):
            captured['predictor'] = kwargs

        def manual_initialization(self, *args, **kwargs):
            captured['allowed_mirroring_axes'] = args[6]
            captured['initialization'] = kwargs
            raise InitializationObserved

    monkeypatch.setattr(evaluation_module, 'nnUNetPredictor', Predictor)
    dataset = SimpleNamespace(dataset_json={}, region_names=('organ',))

    with pytest.raises(InitializationObserved):
        predict_source_grid_probabilities(
            inference_networks=(torch.nn.Identity(),),
            data=torch.zeros(1, 1, 2, 2, 2),
            properties={},
            dataset=dataset,
            device=torch.device('cpu'),
            plans_manager=SimpleNamespace(),
            configuration_manager=SimpleNamespace(),
            tile_step_size=0.4,
            sliding_window_batch_size=8,
        )

    assert captured['predictor']['tile_step_size'] == 0.4
    assert captured['predictor']['sliding_window_batch_size'] == 8
    assert captured['predictor']['use_mirroring'] is False
    assert captured['allowed_mirroring_axes'] is None
    assert captured['initialization'] == {'allow_compile': False}


def test_universal_evaluator_enables_mirroring_when_axes_are_given(monkeypatch):
    class InitializationObserved(Exception):
        pass

    captured = {}

    class Predictor:
        def __init__(self, **kwargs):
            captured['predictor'] = kwargs

        def manual_initialization(self, *args, **kwargs):
            captured['allowed_mirroring_axes'] = args[6]
            raise InitializationObserved

    monkeypatch.setattr(evaluation_module, 'nnUNetPredictor', Predictor)
    dataset = SimpleNamespace(dataset_json={}, region_names=('organ',))

    with pytest.raises(InitializationObserved):
        predict_source_grid_probabilities(
            inference_networks=(torch.nn.Identity(),),
            data=torch.zeros(1, 1, 2, 2, 2),
            properties={},
            dataset=dataset,
            device=torch.device('cpu'),
            plans_manager=SimpleNamespace(),
            configuration_manager=SimpleNamespace(),
            mirroring_axes=(0, 1, 2),
        )

    assert captured['predictor']['use_mirroring'] is True
    assert captured['allowed_mirroring_axes'] == (0, 1, 2)


def test_restore_region_probabilities_restores_bbox_and_original_shape():
    plans_manager = SimpleNamespace(
        transpose_forward=(0, 1, 2),
        transpose_backward=(0, 1, 2),
    )

    class Configuration:
        spacing = (1.0, 1.0, 1.0)

        @staticmethod
        def resampling_fn_probabilities(logits, target_shape, current_spacing, spacing):
            assert tuple(logits.shape[1:]) == tuple(target_shape)
            assert tuple(current_spacing) == tuple(spacing)
            return logits

    properties = {
        'spacing': (1.0, 1.0, 1.0),
        'shape_after_cropping_and_before_resampling': (2, 2, 2),
        'shape_before_cropping': (4, 4, 4),
        'bbox_used_for_cropping': [[1, 3], [1, 3], [1, 3]],
    }
    restored = restore_region_probabilities(
        torch.zeros(2, 2, 2, 2),
        properties,
        plans_manager,
        Configuration(),
    )

    assert restored.shape == (2, 4, 4, 4)
    assert np.all(restored[:, 1:3, 1:3, 1:3] == 0.5)
    assert np.all(restored[:, 0] == 0)


def test_restore_region_probabilities_skips_copy_for_full_bbox(monkeypatch):
    plans_manager = SimpleNamespace(
        transpose_forward=(0, 1, 2),
        transpose_backward=(0, 1, 2),
    )

    class Configuration:
        spacing = (1.0, 1.0, 1.0)

        @staticmethod
        def resampling_fn_probabilities(logits, *args):
            return logits

    monkeypatch.setattr(
        evaluation_module,
        'insert_crop_into_image',
        lambda *args: pytest.fail('full bbox must not insert into a new image'),
    )
    properties = {
        'spacing': (1.0, 1.0, 1.0),
        'shape_after_cropping_and_before_resampling': (2, 2, 2),
        'shape_before_cropping': (2, 2, 2),
        'bbox_used_for_cropping': [[0, 2], [0, 2], [0, 2]],
    }

    restored = restore_region_probabilities(
        torch.zeros(2, 2, 2, 2),
        properties,
        plans_manager,
        Configuration(),
    )

    assert restored.shape == (2, 2, 2, 2)
    assert np.all(restored == 0.5)


def test_torch_restore_matches_scipy_restore_semantics():
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape,
    )
    from nnunetv2.preprocessing.resampling.resample_torch import (
        resample_torch_fornnunet,
    )

    rng = np.random.default_rng(0)
    cases = [
        # FixedIso-style separate-z: iso 1 mm grid back to a 5 mm-z source grid.
        ((1.0, 1.0, 1.0), (5.0, 0.9, 0.9), (40, 37, 37), (8, 41, 41)),
        # Near-isotropic: plain trilinear path.
        ((1.0, 1.0, 1.0), (0.8, 0.8, 0.8), (16, 20, 20), (20, 25, 25)),
    ]
    for current_spacing, target_spacing, current_shape, target_shape in cases:
        logits = rng.standard_normal((3, *current_shape)).astype(np.float32)
        scipy_result = resample_data_or_seg_to_shape(
            logits,
            target_shape,
            current_spacing,
            target_spacing,
            is_seg=False,
            order=1,
            order_z=0,
            force_separate_z=None,
        )
        torch_result = resample_torch_fornnunet(
            torch.from_numpy(logits),
            target_shape,
            current_spacing,
            target_spacing,
            is_seg=False,
            force_separate_z=None,
        ).numpy()
        # The fp16 probability cache resolves ~1e-3; implementation drift must stay far below it.
        np.testing.assert_allclose(torch_result, scipy_result, atol=1e-4)


def test_gpu_restore_eligibility_matches_frozen_plans_configuration():
    eligible = SimpleNamespace(
        configuration={
            'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
            'resampling_fn_probabilities_kwargs': {
                'is_seg': False,
                'order': 1,
                'order_z': 0,
                'force_separate_z': None,
            },
        }
    )
    assert evaluation_module._gpu_restore_eligible(eligible)
    other_fn = SimpleNamespace(
        configuration={
            **eligible.configuration,
            'resampling_fn_probabilities': 'resample_torch_fornnunet',
        }
    )
    assert not evaluation_module._gpu_restore_eligible(other_fn)
    other_kwargs = SimpleNamespace(
        configuration={
            **eligible.configuration,
            'resampling_fn_probabilities_kwargs': {
                'is_seg': False,
                'order': 3,
                'order_z': 0,
                'force_separate_z': None,
            },
        }
    )
    assert not evaluation_module._gpu_restore_eligible(other_kwargs)
    assert not evaluation_module._gpu_restore_eligible(SimpleNamespace())


def test_region_export_preserves_nested_region_order():
    whole = np.ones((2, 2, 2), dtype=bool)
    tumor = np.zeros_like(whole)
    tumor[0, 0, 0] = True

    segmentation = region_masks_to_segmentation(
        np.stack([whole, tumor]),
        [1, 2],
    )

    assert segmentation[1, 1, 1] == 1
    assert segmentation[0, 0, 0] == 2


def test_selected_class_order_follows_region_subset_and_order():
    dataset = SimpleNamespace(
        dataset_id='001',
        dataset_json={
            'labels': {'background': 0, 'a': [1], 'b': [2], 'c': [3]},
            'regions_class_order': [1, 2, 3],
        },
        region_names=('c', 'a'),
    )

    assert selected_regions_class_order(dataset) == [3, 1]


def test_source_and_ground_truth_geometry_must_match():
    properties = {
        'sitk_stuff': {
            'spacing': (1.0, 1.0, 2.0),
            'origin': (0.0, 0.0, 0.0),
            'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        },
    }
    validate_source_and_ground_truth_geometry(properties, properties)
    mismatched = {
        'sitk_stuff': {
            **properties['sitk_stuff'],
            'origin': (1.0, 0.0, 0.0),
        },
    }
    with pytest.raises(ValueError, match='origin'):
        validate_source_and_ground_truth_geometry(properties, mismatched)


def test_binary_metrics_handle_identical_shifted_and_empty_masks():
    reference = np.zeros((5, 5, 5), dtype=bool)
    reference[1:4, 1:4, 1:4] = True
    identical = binary_region_metrics(reference, reference, (1.0, 1.0, 1.0))
    assert identical == {'dsc': 1.0, 'nsd': 1.0}

    shifted = np.roll(reference, 1, axis=0)
    shifted_metrics = binary_region_metrics(
        reference,
        shifted,
        (3.0, 1.0, 1.0),
    )
    assert shifted_metrics['dsc'] < 1
    assert shifted_metrics['nsd'] < 1

    empty = np.zeros_like(reference)
    assert binary_region_metrics(empty, empty, (1.0, 1.0, 1.0)) == {
        'dsc': None,
        'nsd': None,
    }
    assert binary_region_metrics(reference, empty, (1.0, 1.0, 1.0)) == {
        'dsc': 0.0,
        'nsd': 0.0,
    }


def test_ddp_job_partition_visits_each_validation_case_once():
    datasets = {
        '001': SimpleNamespace(validation_identifiers=('a', 'b', 'c')),
        '002': SimpleNamespace(validation_identifiers=('d', 'e')),
    }
    partitions = [
        partition_validation_jobs(datasets, rank, 2)
        for rank in range(2)
    ]

    assert set(partitions[0]).isdisjoint(partitions[1])
    assert set(partitions[0]) | set(partitions[1]) == {
        ('001', 'a'),
        ('001', 'b'),
        ('001', 'c'),
        ('002', 'd'),
        ('002', 'e'),
    }


def test_summary_requires_exact_cases_and_aggregates_source_regions():
    datasets = {
        '001': SimpleNamespace(
            name='one',
            validation_identifiers=('a', 'b'),
            region_values=(1,),
        ),
    }
    registry = CanonicalRegionRegistry(
        {'001': {'labels': {'background': 0, 'organ': 1}}},
        shared_regions={},
    )
    records = [
        {
            'dataset_id': '001',
            'case_id': 'a',
            'regions': [{'dsc': 0.5, 'nsd': 0.75}],
        },
        {
            'dataset_id': '001',
            'case_id': 'b',
            'regions': [{'dsc': 1.0, 'nsd': 1.0}],
        },
    ]

    summary = summarize_universal_metrics(records, datasets, registry)
    assert summary['corpus_macro'] == pytest.approx({
        'dsc': 0.75,
        'nsd': 0.875,
    })
    with pytest.raises(RuntimeError, match='coverage mismatch'):
        summarize_universal_metrics(records[:1], datasets, registry)


def _resumable_validation_trainer(tmp_path, epoch=1000):
    """Minimal trainer surface for driving the orchestrator without a model."""
    return SimpleNamespace(
        datasets={
            '001': SimpleNamespace(
                name='one',
                validation_identifiers=('a', 'b', 'c'),
                region_values=(1,),
            ),
        },
        registry=CanonicalRegionRegistry(
            {'001': {'labels': {'background': 0, 'organ': 1}}},
            shared_regions={},
        ),
        output_folder=str(tmp_path),
        current_epoch=epoch,
        is_ddp=False,
        network=SimpleNamespace(eval=lambda: None),
        logger=SimpleNamespace(log_summary=lambda *args: None),
        set_deep_supervision_enabled=lambda enabled: None,
        print_to_log_file=lambda *args, **kwargs: None,
    )


def _recording_case_predictor(calls, fail_on=None):
    def predict(dataset_id, case_id, output_folder):
        calls.append(case_id)
        if case_id == fail_on:
            raise RuntimeError('predictor stopped')
        return {
            'dataset_id': dataset_id,
            'case_id': case_id,
            'regions': [{'dsc': 0.5, 'nsd': 0.75}],
        }

    return predict


def test_validation_reuses_case_records_and_skips_prediction_on_rerun(tmp_path):
    trainer = _resumable_validation_trainer(tmp_path)
    first_calls = []
    first = evaluation_module.perform_universal_full_volume_validation(
        trainer,
        case_predictor=_recording_case_predictor(first_calls),
    )
    assert first_calls == ['a', 'b', 'c']

    second_calls = []
    second = evaluation_module.perform_universal_full_volume_validation(
        trainer,
        case_predictor=_recording_case_predictor(second_calls),
    )

    assert second_calls == []
    assert second == first


def test_validation_uses_configured_summary_log_prefix(tmp_path):
    trainer = _resumable_validation_trainer(tmp_path)
    logged = []
    trainer.logger = SimpleNamespace(
        log_summary=lambda key, value: logged.append((key, value))
    )

    evaluation_module.perform_universal_full_volume_validation(
        trainer,
        summary_log_prefix='final_val/model_ema',
        case_predictor=_recording_case_predictor([]),
    )

    assert [key for key, _ in logged] == [
        'final_val/model_ema/foreground_dice',
        'final_val/model_ema/foreground_nsd',
    ]


def test_validation_collects_async_case_records(tmp_path):
    trainer = _resumable_validation_trainer(tmp_path)
    calls = []

    def record(dataset_id, case_id):
        return {
            'dataset_id': dataset_id,
            'case_id': case_id,
            'regions': [{'dsc': 0.5, 'nsd': 0.75}],
        }

    with ThreadPoolExecutor(max_workers=1) as executor:
        def predict(dataset_id, case_id, output_folder):
            calls.append(case_id)
            return executor.submit(record, dataset_id, case_id)

        summary = evaluation_module.perform_universal_full_volume_validation(
            trainer,
            case_predictor=predict,
        )

    assert calls == ['a', 'b', 'c']
    assert summary['num_cases'] == 3


def test_validation_recomputes_every_case_when_the_epoch_changes(tmp_path):
    trainer = _resumable_validation_trainer(tmp_path)
    evaluation_module.perform_universal_full_volume_validation(
        trainer,
        case_predictor=_recording_case_predictor([]),
    )

    trainer.current_epoch = 1001
    calls = []
    evaluation_module.perform_universal_full_volume_validation(
        trainer,
        case_predictor=_recording_case_predictor(calls),
    )

    assert calls == ['a', 'b', 'c']


def test_validation_resume_recomputes_only_the_cases_without_records(tmp_path):
    trainer = _resumable_validation_trainer(tmp_path)
    first_calls = []
    with pytest.raises(RuntimeError, match='predictor stopped'):
        evaluation_module.perform_universal_full_volume_validation(
            trainer,
            case_predictor=_recording_case_predictor(first_calls, fail_on='b'),
        )
    assert first_calls == ['a', 'b']

    second_calls = []
    summary = evaluation_module.perform_universal_full_volume_validation(
        trainer,
        case_predictor=_recording_case_predictor(second_calls),
    )

    assert second_calls == ['b', 'c']
    assert summary['num_cases'] == 3


def test_validation_rejects_a_malformed_case_record(tmp_path):
    trainer = _resumable_validation_trainer(tmp_path)
    record_path = (
        Path(trainer.output_folder) / 'validation' / 'dataset-001' / 'a.record.json'
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_bytes(b'{"unexpected": 1}')

    with pytest.raises(ValueError, match='malformed case record'):
        evaluation_module.perform_universal_full_volume_validation(
            trainer,
            case_predictor=_recording_case_predictor([]),
        )


def test_corpus_grid_actual_validation_only_delegates(monkeypatch):
    trainer = object.__new__(CorpusGridUniversalTrainer)
    calls = []
    monkeypatch.setattr(
        corpus_grid_module,
        'perform_universal_full_volume_validation',
        lambda received, save: calls.append((received, save)) or {'ok': True},
    )

    assert trainer.perform_actual_validation(True) == {'ok': True}
    assert calls == [(trainer, True)]


def test_corpus_grid_inference_factory_selects_active_rows():
    trainer = object.__new__(CorpusGridUniversalTrainer)
    trainer.network = torch.nn.Conv3d(1, 5, kernel_size=1)
    trainer.device = torch.device('cpu')
    trainer.registry = SimpleNamespace(
        indices=lambda dataset_id, device: torch.tensor([4, 2], device=device)
    )
    trainer.dataset_index_by_id = {'001': 3}

    adapter = trainer.build_dataset_inference_network('001')

    assert isinstance(adapter, ActiveRegionNetwork)
    assert adapter.network is trainer.network
    assert adapter.canonical_indices.tolist() == [4, 2]
    assert adapter.dataset_index.item() == 3
