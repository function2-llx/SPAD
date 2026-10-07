"""Corpus-grid Universal experiment contract and nnU-Net integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import autocast, nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.numa import pin_to_gpu_numa
from pumit.spad_unet.data import (
    UniversalExperimentManifest,
    build_region_registry,
    get_preprocessed_root,
    load_universal_datasets,
)
from pumit.spad_unet.evaluation import (
    ActiveRegionNetwork,
    aggregate_canonical_region_confusion,
    perform_universal_full_volume_validation,
    unwrap_inference_network,
)
from pumit.spad_unet.plan_contract import _num_epochs_for_updates
from pumit.spad_unet.loss import (
    SUPPORTED_LOSS_NORMALIZATIONS,
    UniversalPartialLabelBatch,
)
from pumit.spad_unet.replay import (
    COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    UniversalSample,
    build_universal_replay_dataloaders,
    open_grouped_replay_reader,
)
from pumit.spad_unet.universal import UniversalResidualEncoderUNet

CORPUS_GRID_CONFIGURATION_NAME = '3d_fullres'
CORPUS_GRID_SOURCE_PLANS_IDENTIFIER = 'nnUNetResEncUNetLPlans1x1x1FOV192'
CORPUS_GRID_1P0_P224_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans1x1x1P224'
)
CORPUS_GRID_1P5_SOURCE_PLANS_IDENTIFIER = 'nnUNetResEncUNetLPlans1p5x1x1FOV192'
CORPUS_GRID_0P9_P224_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans0p9x0p9x0p9P224'
)
CORPUS_GRID_SUPPORTED_SOURCE_PLANS_IDENTIFIERS = (
    CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_1P0_P224_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_1P5_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_0P9_P224_SOURCE_PLANS_IDENTIFIER,
)


def validate_corpus_grid_replay_metadata(
    metadata: Mapping[str, Any],
    world_size: int,
) -> None:
    """Validate the replay contract a Corpus-grid plan declares."""
    if metadata.get('samples_per_dataset') != 1:
        raise ValueError(
            'Corpus-grid complementary replay requires samples_per_dataset=1'
        )
    if metadata.get('sampling_rule') != COMPLEMENTARY_FOREGROUND_REPLAY_RULE:
        raise ValueError(
            'Corpus-grid requires the complementary-foreground replay rule'
        )
    if metadata.get('loss_normalization') not in SUPPORTED_LOSS_NORMALIZATIONS:
        raise ValueError(
            'Corpus-grid requires an explicit supported loss normalization'
        )
    global_batch_size = metadata.get('global_batch_size')
    if world_size < 1 or global_batch_size % world_size:
        raise ValueError(
            f'global batch size {global_batch_size} is not divisible by '
            f'world size {world_size}'
        )

def comparable_corpus_grid_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the one dataset-specific statistic from a Corpus-grid configuration."""
    return {
        key: value
        for key, value in configuration.items()
        if key != 'median_image_size_in_voxels'
    }

def validate_corpus_grid_configurations(
    configurations: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require every dataset to use the same complete Corpus-grid configuration."""
    if not configurations:
        raise ValueError('Corpus-grid validation requires at least one dataset configuration')
    reference_id, reference = next(iter(configurations.items()))
    comparable_reference = comparable_corpus_grid_configuration(reference)
    for dataset_id, configuration in configurations.items():
        if comparable_corpus_grid_configuration(configuration) != comparable_reference:
            raise ValueError(
                f'dataset {dataset_id} does not share the frozen Corpus-grid configuration '
                f'of dataset {reference_id}',
            )

def collate_corpus_grid_samples(
    samples: tuple[UniversalSample, ...],
) -> dict[str, object]:
    """Stack shared-grid samples for one native batched network forward."""
    if not samples:
        raise ValueError('cannot collate an empty Corpus-grid batch')
    spatial_shapes = {tuple(sample.data.shape[2:]) for sample in samples}
    if len(spatial_shapes) != 1:
        raise ValueError(f'Corpus-grid samples disagree on spatial shape: {spatial_shapes}')
    return {
        'data': torch.cat([sample.data for sample in samples]),
        'dataset_ids': tuple(sample.dataset_id for sample in samples),
        'target': UniversalPartialLabelBatch(
            [sample.target for sample in samples],
            [sample.region_indices for sample in samples],
        ),
    }

class CorpusGridUniversalTrainer(RetainPeriodicCheckpointsMixin, nnUNetTrainer):
    """Thin nnU-Net trainer adapter for replayed Corpus-grid Universal batches."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        # DDPOptimizer subgraphs can never hit AOTAutogradCache (unconditional fakify_first_call bypass), which
        # forces a full AOT re-derivation of every warmup path on each restart. Unsplit graphs keep restarts on
        # the warm-cache path; the lost allreduce/backward overlap is negligible at this node count.
        torch._dynamo.config.optimize_ddp = False
        self.numa_affinity = (
            pin_to_gpu_numa() if dist.is_initialized() else None
        )
        metadata = plans.get('pumit_spad_unet')
        if not isinstance(metadata, dict):
            raise ValueError('plans are missing pumit_spad_unet experiment metadata')
        experiment = UniversalExperimentManifest.from_dict(metadata['experiment'])
        source_plans_identifier = metadata['source_plans_identifier']
        if source_plans_identifier not in CORPUS_GRID_SUPPORTED_SOURCE_PLANS_IDENTIFIERS:
            raise ValueError(
                f'Corpus-grid source plans must be one of '
                f'{CORPUS_GRID_SUPPORTED_SOURCE_PLANS_IDENTIFIERS}, '
                f'got {source_plans_identifier}'
            )
        seed = metadata['seed']
        global_batch_size = metadata['global_batch_size']
        replay_total_groups = metadata.get(
            'replay_total_groups',
            experiment.num_updates,
        )
        if (
            isinstance(replay_total_groups, bool)
            or not isinstance(replay_total_groups, int)
            or replay_total_groups < 1
        ):
            raise ValueError(
                f'replay_total_groups must be a positive integer, '
                f'got {replay_total_groups!r}'
            )
        foreground_samples_per_batch = metadata['foreground_samples_per_batch']
        samples_per_dataset = metadata['samples_per_dataset']
        sampling_rule = metadata['sampling_rule']
        replay_directory = metadata['replay_directory']

        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.device.type == 'cuda':
            torch.set_autocast_dtype('cuda', torch.bfloat16)
            self.grad_scaler = None
        if experiment.fold != fold:
            raise ValueError(
                f'plan experiment requests fold {experiment.fold}, but CLI selected fold {fold}',
            )
        if configuration != CORPUS_GRID_CONFIGURATION_NAME:
            raise ValueError(
                f'Corpus-grid requires {CORPUS_GRID_CONFIGURATION_NAME}, '
                f'but CLI selected {configuration}',
            )
        actual_world_size = dist.get_world_size() if self.is_ddp else 1
        validate_corpus_grid_replay_metadata(metadata, actual_world_size)

        self.datasets = load_universal_datasets(
            experiment,
            get_preprocessed_root(),
            plans_name=source_plans_identifier,
            configuration_name=CORPUS_GRID_CONFIGURATION_NAME,
        )
        validate_corpus_grid_configurations(
            {
                dataset_id: dataset.plans['configurations'][CORPUS_GRID_CONFIGURATION_NAME]
                for dataset_id, dataset in self.datasets.items()
            },
        )
        if self.configuration_manager.batch_size != global_batch_size:
            raise ValueError(
                f'Corpus-grid planned batch size '
                f'{self.configuration_manager.batch_size} does not match '
                f'global batch size {global_batch_size}',
            )
        self.registry = build_region_registry(experiment, self.datasets)
        self.task_dataset_ids = tuple(experiment.dataset_ids)
        self.dataset_index_by_id = {
            dataset_id: index
            for index, dataset_id in enumerate(self.task_dataset_ids)
        }
        planned_regions = tuple(self.plans_manager.plans['pumit_canonical_regions'])
        if self.registry.canonical_names != planned_regions:
            raise ValueError(
                'canonical region registry does not match the region bank stored in plans',
            )
        planned_num_regions = self.configuration_manager.network_arch_init_kwargs.get(
            'num_canonical_regions',
        )
        if planned_num_regions != len(self.registry):
            raise ValueError(
                f'plans architecture requests {planned_num_regions} canonical regions, '
                f'but the registry defines {len(self.registry)}',
            )
        planned_num_task_datasets = (
            self.configuration_manager.network_arch_init_kwargs.get(
                'num_task_datasets'
            )
        )
        if planned_num_task_datasets != len(self.task_dataset_ids):
            raise ValueError(
                f'plans architecture requests {planned_num_task_datasets} task datasets, '
                f'but the experiment defines {len(self.task_dataset_ids)}'
            )
        self.num_epochs = _num_epochs_for_updates(
            experiment.num_updates,
            self.num_iterations_per_epoch,
        )

        training_identifiers = {
            dataset_id: dataset.training_identifiers
            for dataset_id, dataset in self.datasets.items()
        }
        self.replay_reader = open_grouped_replay_reader(
            Path(self.preprocessed_dataset_folder_base) / replay_directory,
            {
                'seed': seed,
                'replay_total_groups': replay_total_groups,
                'samples_per_dataset': samples_per_dataset,
                'foreground_samples_per_batch': foreground_samples_per_batch,
                'sampling_rule': sampling_rule,
            },
            training_identifiers,
        )
        self.replay_reader.validate()
        required_records = experiment.num_updates * global_batch_size
        if self.replay_reader.total_records < required_records:
            raise ValueError(
                f'replay has {self.replay_reader.total_records} records, but '
                f'{experiment.num_updates} optimizer steps at global batch '
                f'{global_batch_size} require {required_records}'
            )

    def _build_loss(self) -> nn.Module:
        network = self.network
        if isinstance(network, DistributedDataParallel):
            network = network.module
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        if not isinstance(network, UniversalResidualEncoderUNet):
            raise TypeError(
                'Corpus-grid training requires UniversalResidualEncoderUNet'
            )
        expected_normalization = self.plans_manager.plans[
            'pumit_spad_unet'
        ]['loss_normalization']
        if network.partial_label_loss.loss_normalization != expected_normalization:
            raise ValueError(
                'Corpus-grid architecture and experiment metadata disagree on '
                'loss normalization'
            )
        return network.partial_label_loss

    def _compile_network(self, network: nn.Module) -> nn.Module:
        """Compile the shape-static components; the partial-label loss stays eager.

        Per-sample loss shapes follow batch composition, which now varies per step; a compiled
        loss specialized per region count, desynchronizing DDP collectives across ranks.
        """
        network.encoder.compile(dynamic=False, mode='default')
        network.task_aware_bottleneck.compile(dynamic=False, mode='default')
        network.decoder.compile(dynamic=False, mode='default')
        return network

    def perform_actual_validation(self, save_probabilities: bool = False):
        return perform_universal_full_volume_validation(self, save_probabilities)

    def build_dataset_inference_network(self, dataset_id: str) -> nn.Module:
        """Expose one source dataset's active rows to the shared evaluator."""
        return ActiveRegionNetwork(
            unwrap_inference_network(self.network),
            self.registry.indices(dataset_id, self.device),
            self.dataset_index_by_id[dataset_id],
        )

    def _batch_dataset_indices(self, batch: dict) -> torch.Tensor:
        return torch.tensor(
            [self.dataset_index_by_id[dataset_id] for dataset_id in batch['dataset_ids']],
            dtype=torch.long,
            device=self.device,
        )

    def get_dataloaders(self):
        train_loader, val_loader, mirror_axes = build_universal_replay_dataloaders(
            self,
            self.datasets,
            self.registry,
            self.replay_reader,
            collate_corpus_grid_samples,
        )
        self.inference_allowed_mirroring_axes = mirror_axes
        return train_loader, val_loader

    def train_step(self, batch: dict) -> dict[str, np.ndarray]:
        data = batch['data'].to(self.device, non_blocking=True)
        dataset_indices = self._batch_dataset_indices(batch)
        target_batch = batch['target'].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            logits = self.network(data, dataset_indices)
            loss = self.loss(logits, target_batch)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict[str, np.ndarray]:
        data = batch['data'].to(self.device, non_blocking=True)
        dataset_indices = self._batch_dataset_indices(batch)
        target_batch = batch['target'].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            outputs = self.network(data, dataset_indices)
            loss = self.loss(outputs, target_batch)

        logits = outputs[0] if isinstance(outputs, list) else outputs
        tp, fp, fn = aggregate_canonical_region_confusion(
            tuple(
                (
                    logits[sample_index: sample_index + 1].index_select(1, indices),
                    target,
                    indices,
                )
                for sample_index, (target, indices) in enumerate(
                    zip(
                        target_batch.targets,
                        target_batch.region_indices,
                        strict=True,
                    ),
                )
            ),
            len(self.registry),
        )
        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp.cpu().numpy(),
            'fp_hard': fp.cpu().numpy(),
            'fn_hard': fn.cpu().numpy(),
        }
