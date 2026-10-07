"""SPAD Universal experiment integration for nnU-Net."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

from pumit.nnunet.compile_cache import CompileCacheMixin
from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.nnunet.model_ema import ModelEmaMixin, validate_model_ema_decay
from pumit.numa import pin_to_gpu_numa
from pumit.spad_unet.architecture import (
    UNIVERSAL_SPAD_MIN_BOTTLENECK,
    UNIVERSAL_SPAD_N_STAGES,
    resolve_fgc_return_downsample_mode,
    resolve_fgc_return_mode,
    resolve_fgc_return_prefilter,
    universal_spad_architecture_profile,
)
from pumit.spad_unet.geometry import (
    compute_continuous_da,
    compute_fgc_geometry,
    decompose_continuous_da,
    select_da_pair,
)
from pumit.spad_unet.data import (
    UniversalExperimentManifest,
    build_region_registry,
    get_preprocessed_root,
    load_universal_datasets,
)
from pumit.spad_unet.evaluation import (
    DEFAULT_SLIDING_WINDOW_BATCH_SIZE,
    DEFAULT_SLIDING_WINDOW_TILE_STEP_SIZE,
    aggregate_canonical_region_confusion,
    evaluate_universal_dataset_case,
    load_universal_dataset_case,
    predict_source_grid_probabilities,
    perform_universal_full_volume_validation,
    unwrap_inference_network,
    validate_sliding_window_batch_size,
    validate_sliding_window_tile_step_size,
)
from pumit.spad_unet.loss import (
    REGION_BALANCED_LOSS_NORMALIZATION,
    SAMPLE_MEAN_LOSS_NORMALIZATION,
    SIGMOID_REGIONS,
    UniversalPartialLabelLoss,
)
from pumit.spad_unet.plan_contract import (
    SPAD_UNIVERSAL_CONFIGURATION_NAME,
    SPAD_UNIVERSAL_DA_DISCRETIZATIONS,
    SPAD_UNIVERSAL_DATASET_OBJECTIVES,
    SPAD_UNIVERSAL_DATASET_SAMPLINGS,
    SPAD_UNIVERSAL_GLOBAL_BATCH_SIZE,
    SPAD_UNIVERSAL_INFERENCE_MODES,
    SPAD_UNIVERSAL_LEGACY_NATIVE_Z_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_LEGACY_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_INPLANE_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_LKR_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_PLANNED_XY_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIERS,
    SPAD_UNIVERSAL_SAMPLING_MATCHED_OBJECTIVE,
    SPAD_UNIVERSAL_SEVEN_STAGE_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SOURCE_PLANS_IDENTIFIER,
    SPAD_UNIVERSAL_SQRT_DATASET_SAMPLING,
    SPAD_UNIVERSAL_SUPPORTED_SOURCE_PLANS_IDENTIFIERS,
    SPAD_UNIVERSAL_UNIFORM_DATASET_OBJECTIVE,
    SPAD_UNIVERSAL_UNIFORM_REGION_OBJECTIVE,
    _num_epochs_for_updates,
    resolve_spad_universal_dataset_objective,
    resolve_sqrt_replay_dataset_probabilities,
    spad_universal_dataset_loss_weights,
    validate_spad_universal_dataset_sampling,
    validate_spad_universal_inference_mode,
    validate_spad_universal_replay_metadata,
)
from pumit.spad_unet.replay import (
    COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    ReplayStreamReader,
    SQRT_DATASET_REPLAY_RULE,
    UniversalSample,
    build_universal_replay_dataloaders,
    open_grouped_replay_reader,
)
from pumit.spad_unet.universal import build_dataset_output_contract


SPAD_UNIVERSAL_INFERENCE_COMPONENTS = ('ff', 'fc', 'cf', 'cc')
SPAD_UNIVERSAL_TILE_STEP_SIZE_ENV = 'SPAD_UNIVERSAL_TILE_STEP_SIZE'
SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE_ENV = (
    'SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE'
)
SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS_ENV = (
    'SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS'
)
SPAD_UNIVERSAL_DEFAULT_CASE_EVALUATION_WORKERS = 4
SPAD_UNIVERSAL_VALIDATION_WEIGHTS_ENV = 'SPAD_UNIVERSAL_VALIDATION_WEIGHTS'
SPAD_UNIVERSAL_VALIDATION_WEIGHTS_CHOICES = ('both', 'raw', 'model_ema')
SPAD_UNIVERSAL_TTA_MIRRORING_ENV = 'SPAD_UNIVERSAL_TTA_MIRRORING'


def _resolve_inference_tile_step_size(raw_value: str | None) -> float:
    if raw_value is None:
        return DEFAULT_SLIDING_WINDOW_TILE_STEP_SIZE
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f'{SPAD_UNIVERSAL_TILE_STEP_SIZE_ENV} must be numeric, '
            f'got {raw_value!r}'
        ) from error
    return validate_sliding_window_tile_step_size(value)


def _resolve_sliding_window_batch_size(raw_value: str | None) -> int:
    if raw_value is None:
        return DEFAULT_SLIDING_WINDOW_BATCH_SIZE
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f'{SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE_ENV} must be an '
            f'integer, got {raw_value!r}'
        ) from error
    return validate_sliding_window_batch_size(value)


def _resolve_case_evaluation_workers(raw_value: str | None) -> int:
    if raw_value is None:
        return SPAD_UNIVERSAL_DEFAULT_CASE_EVALUATION_WORKERS
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f'{SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS_ENV} must be an '
            f'integer, got {raw_value!r}'
        ) from error
    return validate_sliding_window_batch_size(value)


def _resolve_validation_weights(raw_value: str | None) -> str:
    if raw_value is None:
        return 'both'
    if raw_value not in SPAD_UNIVERSAL_VALIDATION_WEIGHTS_CHOICES:
        raise ValueError(
            f'{SPAD_UNIVERSAL_VALIDATION_WEIGHTS_ENV} must be one of '
            f'{SPAD_UNIVERSAL_VALIDATION_WEIGHTS_CHOICES}, got {raw_value!r}'
        )
    return raw_value


def _resolve_tta_mirroring(raw_value: str | None) -> bool:
    if raw_value is None:
        return False
    if raw_value not in ('0', '1'):
        raise ValueError(
            f'{SPAD_UNIVERSAL_TTA_MIRRORING_ENV} must be 0 or 1, got {raw_value!r}'
        )
    return raw_value == '1'


def _inference_variant_name(
    inference_mode: str,
    tile_step_size: float,
    sliding_window_batch_size: int = DEFAULT_SLIDING_WINDOW_BATCH_SIZE,
    tta_mirroring: bool = False,
) -> str:
    tile_step_size = validate_sliding_window_tile_step_size(tile_step_size)
    sliding_window_batch_size = validate_sliding_window_batch_size(
        sliding_window_batch_size
    )
    variant = inference_mode
    if tile_step_size != DEFAULT_SLIDING_WINDOW_TILE_STEP_SIZE:
        variant += f'_step{tile_step_size!r}'
    if sliding_window_batch_size != DEFAULT_SLIDING_WINDOW_BATCH_SIZE:
        variant += f'_batch{sliding_window_batch_size}'
    if tta_mirroring:
        variant += '_mirror'
    return variant


@dataclass(frozen=True)
class _SPADCompileWarmupPath:
    """One reachable graph and a dataset that can instantiate it."""

    patch_size: tuple[int, int, int]
    da_encoder: int
    da_decoder: int
    canonical_shape: tuple[int, int, int] | None
    dataset_id: str
    continuous_da: float
    exposure: float


def collate_spad_universal_samples(
    samples: tuple[UniversalSample, ...],
) -> dict[str, object]:
    """Preserve dataset-planned-grid samples for sequential forwarding."""
    if not samples:
        raise ValueError('cannot collate an empty SPAD Universal batch')
    return {'samples': samples}


class _SPADActiveRegionNetwork(nn.Module):
    """Bind one dataset's inference DA and active output rows."""

    def __init__(
        self,
        network: nn.Module,
        canonical_indices: torch.Tensor,
        da_encoder: int,
        da_decoder: int,
        dataset_index: int,
        canonical_shape: tuple[int, int, int] | None = None,
        packed_output_rows: torch.Tensor | None = None,
    ):
        super().__init__()
        self.network = network
        self.da_encoder = da_encoder
        self.da_decoder = da_decoder
        self.canonical_shape = canonical_shape
        self.register_buffer(
            'canonical_indices',
            canonical_indices.detach().clone().long(),
            persistent=False,
        )
        if packed_output_rows is None:
            self.packed_output_rows = None
        else:
            self.register_buffer(
                'packed_output_rows',
                packed_output_rows.detach().clone().long(),
                persistent=False,
            )
        self.register_buffer(
            'dataset_index',
            torch.tensor(
                dataset_index,
                dtype=torch.long,
                device=canonical_indices.device,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_args = (
            x,
            self.da_encoder,
            self.da_decoder,
            self.dataset_index.expand(x.shape[0]),
        )
        if self.packed_output_rows is None:
            logits = (
                self.network.forward_sample(*sample_args)
                if self.canonical_shape is None
                else self.network.forward_sample(
                    *sample_args,
                    self.canonical_shape,
                )
            )
        else:
            logits = self.network.forward_sample(
                *sample_args,
                self.canonical_shape,
                self.packed_output_rows,
            )
        if not isinstance(logits, torch.Tensor):
            raise TypeError('full-volume inference requires deep supervision to be disabled')
        if self.packed_output_rows is None:
            return logits.index_select(1, self.canonical_indices)
        return logits


class SPADUniversalTrainer(
    CompileCacheMixin,
    RetainPeriodicCheckpointsMixin,
    ModelEmaMixin,
    nnUNetTrainer,
):
    """Thin nnU-Net adapter for replayed dataset-planned-grid SPAD training."""

    inference_tile_step_size = DEFAULT_SLIDING_WINDOW_TILE_STEP_SIZE
    inference_sliding_window_batch_size = DEFAULT_SLIDING_WINDOW_BATCH_SIZE
    inference_case_evaluation_workers = (
        SPAD_UNIVERSAL_DEFAULT_CASE_EVALUATION_WORKERS
    )
    inference_validation_weights = 'both'
    inference_tta_mirroring = False

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
        if metadata.get('condition') != 'spad_universal':
            raise ValueError('SPADUniversalTrainer requires condition=spad_universal')
        self.model_ema_decay = validate_model_ema_decay(
            metadata.get('model_ema_decay')
        )
        experiment = UniversalExperimentManifest.from_dict(metadata['experiment'])
        source_plans_identifier = metadata['source_plans_identifier']
        if (
            source_plans_identifier
            not in SPAD_UNIVERSAL_SUPPORTED_SOURCE_PLANS_IDENTIFIERS
        ):
            raise ValueError(
                f'SPAD Universal source plans must be one of '
                f'{SPAD_UNIVERSAL_SUPPORTED_SOURCE_PLANS_IDENTIFIERS}, '
                f'got {source_plans_identifier}'
            )
        global_batch_size = metadata['global_batch_size']
        replay_stream = metadata.get('replay_stream')
        if replay_stream is None:
            seed = metadata['seed']
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
            foreground_samples_per_batch = metadata[
                'foreground_samples_per_batch'
            ]
            samples_per_dataset = metadata['samples_per_dataset']
            sampling_rule = metadata['sampling_rule']
            replay_directory = metadata['replay_directory']
        else:
            if not isinstance(replay_stream, dict):
                raise TypeError('replay_stream must be a mapping')
            expected_replay_stream_keys = {
                'directory',
                'stream_fingerprint',
                'num_records',
            }
            if set(replay_stream) != expected_replay_stream_keys:
                raise ValueError(
                    f'replay_stream requires exactly '
                    f'{sorted(expected_replay_stream_keys)}, got '
                    f'{sorted(replay_stream)}'
                )
            if (
                not isinstance(replay_stream['directory'], str)
                or not replay_stream['directory']
            ):
                raise ValueError('replay_stream.directory must be non-empty')
            if (
                not isinstance(replay_stream['stream_fingerprint'], str)
                or len(replay_stream['stream_fingerprint']) != 64
            ):
                raise ValueError(
                    'replay_stream.stream_fingerprint must be a full SHA-256 '
                    'digest'
                )
            if (
                isinstance(replay_stream['num_records'], bool)
                or not isinstance(replay_stream['num_records'], int)
                or replay_stream['num_records'] < 1
            ):
                raise ValueError(
                    'replay_stream.num_records must be a positive integer'
                )
        cross_da = metadata['cross_da']
        da_discretization = metadata['da_discretization']
        feature_grid_canonicalization_stage = metadata.get(
            'feature_grid_canonicalization_stage'
        )
        feature_grid_canonicalization_return = metadata.get(
            'feature_grid_canonicalization_return'
        )
        feature_grid_canonicalization_return_downsample_mode = metadata.get(
            'feature_grid_canonicalization_return_downsample_mode'
        )
        feature_grid_canonicalization_return_prefilter = metadata.get(
            'feature_grid_canonicalization_return_prefilter',
            False,
        )
        sample_native_z = metadata.get('sample_native_z', False)
        if not isinstance(cross_da, bool):
            raise TypeError('cross_da must be boolean')
        if not isinstance(sample_native_z, bool):
            raise TypeError('sample_native_z must be boolean')
        if da_discretization not in SPAD_UNIVERSAL_DA_DISCRETIZATIONS:
            raise ValueError(
                f'unsupported DA discretization {da_discretization!r}; expected '
                f'one of {SPAD_UNIVERSAL_DA_DISCRETIZATIONS}'
            )
        if cross_da and da_discretization != 'stochastic':
            raise ValueError('cross DA requires stochastic DA')
        if sample_native_z != (
            source_plans_identifier
            in SPAD_UNIVERSAL_SAMPLE_NATIVE_Z_SOURCE_PLANS_IDENTIFIERS
        ):
            raise ValueError(
                'sample_native_z metadata must match the Sample-Native-Z '
                'source plans identifier'
            )

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _num_epochs_for_updates(
            experiment.num_updates,
            self.num_iterations_per_epoch,
        )
        if self.device.type == 'cuda':
            torch.set_autocast_dtype('cuda', torch.bfloat16)
            self.grad_scaler = None
        if experiment.fold != fold:
            raise ValueError(
                f'plan experiment requests fold {experiment.fold}, but CLI selected fold {fold}'
            )
        if configuration != SPAD_UNIVERSAL_CONFIGURATION_NAME:
            raise ValueError(
                f'SPAD Universal requires {SPAD_UNIVERSAL_CONFIGURATION_NAME}, '
                f'but CLI selected {configuration}'
            )
        actual_world_size = dist.get_world_size() if self.is_ddp else 1
        self.datasets = load_universal_datasets(
            experiment,
            get_preprocessed_root(),
            plans_name=source_plans_identifier,
            configuration_name=SPAD_UNIVERSAL_CONFIGURATION_NAME,
        )
        self.sample_native_z = sample_native_z
        if self.sample_native_z and any(
            dataset.case_geometries is None for dataset in self.datasets.values()
        ):
            raise ValueError(
                'Sample-Native-Z training requires case geometry for every dataset'
            )
        validate_spad_universal_replay_metadata(metadata, actual_world_size)
        if self.configuration_manager.batch_size != global_batch_size:
            raise ValueError(
                f'SPAD Universal planned batch size '
                f'{self.configuration_manager.batch_size} does not match '
                f'global batch size {global_batch_size}'
            )
        if self.configuration_manager.batch_dice:
            raise ValueError('SPAD Universal derived plans must set batch_dice=false')
        self.registry = build_region_registry(experiment, self.datasets)
        self.loss_normalization = metadata['loss_normalization']
        self.dataset_objective = resolve_spad_universal_dataset_objective(
            metadata.get('dataset_objective'),
            self.loss_normalization,
        )
        configured_output_contract = metadata.get('dataset_output_contract')
        if configured_output_contract is None:
            self.dataset_output_contract = None
            self.dataset_prediction_modes = {}
            self.packed_output_rows_by_dataset = {}
        else:
            expected_output_contract = build_dataset_output_contract(
                self.registry,
                self.datasets,
            )
            if configured_output_contract != expected_output_contract:
                raise ValueError(
                    'dataset output contract does not match the source '
                    f'LabelManagers and canonical registry: expected '
                    f'{expected_output_contract}, got '
                    f'{configured_output_contract}'
                )
            if self.loss_normalization != SAMPLE_MEAN_LOSS_NORMALIZATION:
                raise ValueError(
                    'dataset-native output requires sample-level loss '
                    'normalization'
                )
            self.dataset_output_contract = expected_output_contract
            self.dataset_prediction_modes = {
                dataset_id: contract['prediction_mode']
                for dataset_id, contract in expected_output_contract[
                    'datasets'
                ].items()
            }
            self.packed_output_rows_by_dataset = {
                dataset_id: torch.tensor(
                    contract['packed_output_rows'],
                    dtype=torch.long,
                    device=self.device,
                )
                for dataset_id, contract in expected_output_contract[
                    'datasets'
                ].items()
            }
        self.global_batch_size = global_batch_size
        if replay_stream is not None:
            self.num_active_regions_per_global_batch = None
        else:
            records_per_replay_group = len(self.datasets) * samples_per_dataset
            if 'num_active_regions_per_global_batch' in metadata:
                self.num_active_regions_per_global_batch = metadata[
                    'num_active_regions_per_global_batch'
                ]
            elif global_batch_size == records_per_replay_group:
                # Compatibility with the original batch-12 plans, whose every step
                # contains all 52 task-specific regions exactly once.
                self.num_active_regions_per_global_batch = len(self.registry)
            else:
                raise ValueError(
                    'plans with a partial replay group per optimizer step must '
                    'declare num_active_regions_per_global_batch'
                )
        if (
            self.num_active_regions_per_global_batch is not None
            and (
                not isinstance(
                    self.num_active_regions_per_global_batch,
                    int,
                )
                or isinstance(
                    self.num_active_regions_per_global_batch,
                    bool,
                )
                or self.num_active_regions_per_global_batch < 1
            )
        ):
            raise ValueError(
                'num_active_regions_per_global_batch must be a positive '
                'integer or null'
            )
        self.task_dataset_ids = tuple(experiment.dataset_ids)
        self.dataset_index_by_id = {
            dataset_id: index
            for index, dataset_id in enumerate(self.task_dataset_ids)
        }
        planned_regions = tuple(self.plans_manager.plans['pumit_canonical_regions'])
        if self.registry.canonical_names != planned_regions:
            raise ValueError(
                'canonical region registry does not match the region bank stored in plans'
            )
        planned_num_regions = self.configuration_manager.network_arch_init_kwargs.get(
            'num_canonical_regions'
        )
        if planned_num_regions != len(self.registry):
            raise ValueError(
                f'plans architecture requests {planned_num_regions} canonical regions, '
                f'but the registry defines {len(self.registry)}'
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
        architecture_kwargs = self.configuration_manager.network_arch_init_kwargs
        planned_packed_output_channels = architecture_kwargs.get(
            'num_packed_output_channels'
        )
        expected_packed_output_channels = (
            None
            if self.dataset_output_contract is None
            else self.dataset_output_contract['num_packed_output_channels']
        )
        if planned_packed_output_channels != expected_packed_output_channels:
            raise ValueError(
                f'plans architecture requests '
                f'{planned_packed_output_channels} packed output channels, '
                f'but the dataset output contract requires '
                f'{expected_packed_output_channels}'
            )
        planned_fgc_stage = architecture_kwargs.get(
            'feature_grid_canonicalization_stage'
        )
        if planned_fgc_stage != feature_grid_canonicalization_stage:
            raise ValueError(
                'FGC stage metadata must match the architecture configuration'
            )
        planned_fgc_return = architecture_kwargs.get(
            'feature_grid_canonicalization_return'
        )
        if planned_fgc_return != feature_grid_canonicalization_return:
            raise ValueError(
                'FGC return metadata must match the architecture configuration'
            )
        planned_fgc_return_downsample_mode = architecture_kwargs.get(
            'feature_grid_canonicalization_return_downsample_mode'
        )
        if (
            planned_fgc_return_downsample_mode
            != feature_grid_canonicalization_return_downsample_mode
        ):
            raise ValueError(
                'FGC return downsample metadata must match the architecture '
                'configuration'
            )
        planned_fgc_return_prefilter = architecture_kwargs.get(
            'feature_grid_canonicalization_return_prefilter',
            False,
        )
        if (
            planned_fgc_return_prefilter
            != feature_grid_canonicalization_return_prefilter
        ):
            raise ValueError(
                'FGC return prefilter metadata must match the architecture '
                'configuration'
            )
        resolved_fgc_return = resolve_fgc_return_mode(
            feature_grid_canonicalization_stage,
            feature_grid_canonicalization_return,
        )
        resolved_fgc_return_downsample_mode = (
            resolve_fgc_return_downsample_mode(
                feature_grid_canonicalization_stage,
                feature_grid_canonicalization_return_downsample_mode,
            )
        )
        resolved_fgc_return_prefilter = resolve_fgc_return_prefilter(
            feature_grid_canonicalization_stage,
            resolved_fgc_return,
            feature_grid_canonicalization_return_prefilter,
        )
        planned_n_stages = architecture_kwargs.get('n_stages')
        if isinstance(planned_n_stages, bool) or not isinstance(planned_n_stages, int):
            raise ValueError(
                f'SPAD Universal architecture requires an integer n_stages, '
                f'got {planned_n_stages!r}'
            )
        expected_features, expected_blocks, expected_decoder_convs = (
            universal_spad_architecture_profile(planned_n_stages)
        )
        self.spad_n_stages = planned_n_stages
        self.spad_features_per_stage = expected_features
        architecture_contract = {
            'n_stages': planned_n_stages,
            'features_per_stage': expected_features,
            'n_blocks_per_stage': expected_blocks,
            'n_conv_per_stage_decoder': expected_decoder_convs,
            'min_bottleneck': UNIVERSAL_SPAD_MIN_BOTTLENECK,
        }
        mismatched_architecture = {}
        for key, expected in architecture_contract.items():
            actual = architecture_kwargs.get(key)
            comparable_actual = (
                tuple(actual) if isinstance(actual, (list, tuple)) else actual
            )
            if comparable_actual != expected:
                mismatched_architecture[key] = {
                    'expected': expected,
                    'actual': actual,
                }
        if mismatched_architecture:
            raise ValueError(
                f'SPAD Universal architecture contract mismatch: '
                f'{mismatched_architecture}'
            )
        if feature_grid_canonicalization_stage is not None:
            fgc_contract = {
                'sample_native_z': sample_native_z,
                'da_discretization': da_discretization,
                'cross_da': cross_da,
                'n_stages': planned_n_stages,
                'learnable_kernel_reduction': architecture_kwargs.get(
                    'learnable_kernel_reduction'
                ),
                'feature_grid_canonicalization_stage': (
                    feature_grid_canonicalization_stage
                ),
                'feature_grid_canonicalization_return': resolved_fgc_return,
                'feature_grid_canonicalization_return_downsample_mode': (
                    resolved_fgc_return_downsample_mode
                ),
                'feature_grid_canonicalization_return_prefilter': (
                    resolved_fgc_return_prefilter
                ),
                'patch_policy': metadata.get('patch_policy'),
                'target': metadata.get(
                    'feature_grid_canonicalization_target'
                ),
                'shape_closure': metadata.get(
                    'feature_grid_canonicalization_shape_closure'
                ),
            }
            if da_discretization not in ('floor', 'stochastic'):
                raise ValueError(
                    'FGC requires floor or stochastic DA discretization'
                )
            expected_fgc_contract = {
                'sample_native_z': True,
                'da_discretization': da_discretization,
                'cross_da': False,
                'n_stages': UNIVERSAL_SPAD_N_STAGES,
                'learnable_kernel_reduction': False,
                'feature_grid_canonicalization_stage': (
                    feature_grid_canonicalization_stage
                ),
                'feature_grid_canonicalization_return': resolved_fgc_return,
                'feature_grid_canonicalization_return_downsample_mode': (
                    resolved_fgc_return_downsample_mode
                ),
                'feature_grid_canonicalization_return_prefilter': (
                    resolved_fgc_return_prefilter
                ),
                'patch_policy': 'common_full_route',
                'target': 'floor_dyadic',
                'shape_closure': 'nearest_compatible_half_up',
            }
            if fgc_contract != expected_fgc_contract:
                raise ValueError(
                    f'FGC plan contract mismatch: expected '
                    f'{expected_fgc_contract}, got {fgc_contract}'
                )
        self.continuous_da = {
            dataset_id: compute_continuous_da(dataset.spacing)
            for dataset_id, dataset in self.datasets.items()
        }
        self.cross_da = cross_da
        self.da_discretization = da_discretization
        self.feature_grid_canonicalization_stage = (
            feature_grid_canonicalization_stage
        )
        self.feature_grid_canonicalization_return = resolved_fgc_return
        self.feature_grid_canonicalization_return_downsample_mode = (
            resolved_fgc_return_downsample_mode
        )
        self.feature_grid_canonicalization_return_prefilter = (
            resolved_fgc_return_prefilter
        )
        self.inference_mode = os.environ.get(
            'SPAD_UNIVERSAL_INFERENCE_MODE',
            metadata.get('inference_mode', 'cross'),
        )
        self.inference_tile_step_size = (
            _resolve_inference_tile_step_size(
                os.environ.get(SPAD_UNIVERSAL_TILE_STEP_SIZE_ENV)
            )
        )
        self.inference_sliding_window_batch_size = (
            _resolve_sliding_window_batch_size(
                os.environ.get(
                    SPAD_UNIVERSAL_SLIDING_WINDOW_BATCH_SIZE_ENV
                )
            )
        )
        self.inference_case_evaluation_workers = (
            _resolve_case_evaluation_workers(
                os.environ.get(
                    SPAD_UNIVERSAL_CASE_EVALUATION_WORKERS_ENV
                )
            )
        )
        self.inference_validation_weights = _resolve_validation_weights(
            os.environ.get(SPAD_UNIVERSAL_VALIDATION_WEIGHTS_ENV)
        )
        self.inference_tta_mirroring = _resolve_tta_mirroring(
            os.environ.get(SPAD_UNIVERSAL_TTA_MIRRORING_ENV)
        )
        validate_spad_universal_inference_mode(
            self.inference_mode,
            da_discretization,
        )

        training_identifiers = {
            dataset_id: dataset.training_identifiers
            for dataset_id, dataset in self.datasets.items()
        }
        if replay_stream is None:
            self.replay_reader = open_grouped_replay_reader(
                Path(self.preprocessed_dataset_folder_base)
                / replay_directory,
                {
                    'seed': seed,
                    'replay_total_groups': replay_total_groups,
                    'samples_per_dataset': samples_per_dataset,
                    'foreground_samples_per_batch': (
                        foreground_samples_per_batch
                    ),
                    'sampling_rule': sampling_rule,
                },
                training_identifiers,
            )
        else:
            replay_stream_dir = (
                Path(self.preprocessed_dataset_folder_base)
                / replay_stream['directory']
            )
            self.replay_reader = ReplayStreamReader(
                replay_stream_dir,
                expected_stream_fingerprint=(
                    replay_stream['stream_fingerprint']
                ),
                valid_identifiers=training_identifiers,
            )
            if not self.replay_reader.has_exact_stream_fingerprint:
                raise ValueError(
                    'replay_stream requires an exact-content replay artifact'
                )
        if self.replay_reader.dataset_ids != tuple(self.datasets):
            raise ValueError(
                f'replay datasets {self.replay_reader.dataset_ids} do not '
                f'match the experiment suite {tuple(self.datasets)}'
            )
        self.replay_reader.validate()
        if (
            replay_stream is not None
            and self.replay_reader.total_records
            != replay_stream['num_records']
        ):
            raise ValueError(
                f'replay has {self.replay_reader.total_records} records, but '
                f'the plan declares {replay_stream["num_records"]}'
            )
        required_records = experiment.num_updates * global_batch_size
        if self.replay_reader.total_records < required_records:
            raise ValueError(
                f'replay has {self.replay_reader.total_records} records, but '
                f'{experiment.num_updates} optimizer steps at global batch '
                f'{global_batch_size} require {required_records}'
            )
        replay_sampling_probabilities = self.replay_reader.meta.get(
            'dataset_sampling_probabilities'
        )
        validate_spad_universal_dataset_sampling(
            metadata.get('dataset_sampling'),
            self.replay_reader.dataset_ids,
            self.replay_reader.sampling_rule,
            replay_sampling_probabilities,
        )
        self.dataset_loss_weights = spad_universal_dataset_loss_weights(
            self.dataset_objective,
            self.replay_reader.dataset_ids,
            self.replay_reader.sampling_rule,
            replay_sampling_probabilities,
            loss_normalization=self.loss_normalization,
            dataset_region_counts={
                dataset_id: len(dataset.region_names)
                for dataset_id, dataset in self.datasets.items()
            },
        )

    def _get_deep_supervision_scales(self):
        return None

    def _compile_network(self, network: nn.Module) -> nn.Module:
        """Compile the stable encoder and decoder components."""
        network.encoder.compile(dynamic=False, mode='default')
        network.decoder.compile(dynamic=False, mode='default')
        return network

    def _static_component_paths(self) -> tuple[tuple, ...]:
        paths = set()
        for dataset_id, dataset in self.datasets.items():
            case_geometries = getattr(dataset, 'case_geometries', None)
            if case_geometries is None:
                geometries = ((
                    tuple(int(value) for value in dataset.patch_size),
                    self.continuous_da[dataset_id],
                ),)
            else:
                geometries = {
                    (geometry.patch_size, geometry.continuous_da)
                    for geometry in case_geometries.values()
                }
            for patch_size, continuous_da in geometries:
                if getattr(
                    self,
                    'feature_grid_canonicalization_stage',
                    None,
                ) is None:
                    paths.update(
                        (patch_size, da)
                        for da in self._possible_da_values_for(continuous_da)
                    )
                else:
                    for route_da in self._fgc_route_das(continuous_da):
                        geometry = compute_fgc_geometry(
                            continuous_da,
                            patch_size,
                            stage=self.feature_grid_canonicalization_stage,
                            n_stages=self.spad_n_stages,
                            min_bottleneck=UNIVERSAL_SPAD_MIN_BOTTLENECK,
                            route_da=route_da,
                        )
                        paths.add((
                            patch_size,
                            geometry.route_da,
                            geometry.canonical_shape,
                        ))
        return tuple(sorted(paths))

    def _fgc_route_das(self, continuous_da: float) -> tuple[int, ...]:
        """List the integer FGC routes executed during training or inference."""
        floor_da, ceil_da, _ = decompose_continuous_da(continuous_da)
        if (
            getattr(self, 'da_discretization', None) != 'stochastic'
            and getattr(self, 'inference_mode', None) != 'endpoints'
            or ceil_da == floor_da
        ):
            return (floor_da,)
        return (floor_da, ceil_da)

    def _possible_da_values_for(self, continuous_da: float) -> tuple[int, ...]:
        da_discretization = self.da_discretization
        if da_discretization == 'floor':
            return (math.floor(continuous_da),)
        return tuple(sorted({
            math.floor(continuous_da),
            math.ceil(continuous_da),
        }))

    def _compile_warmup_route_exposures(
        self,
        continuous_da: float,
    ) -> tuple[tuple[int, int, float], ...]:
        """Return the exact training probability of each encoder/decoder route."""
        da_discretization = self.da_discretization
        floor_da, ceil_da, ceil_weight = decompose_continuous_da(continuous_da)
        if da_discretization == 'floor':
            return ((floor_da, floor_da, 1.0),)
        if floor_da == ceil_da:
            return ((floor_da, floor_da, 1.0),)

        endpoint_weights = (
            (floor_da, 1.0 - ceil_weight),
            (ceil_da, ceil_weight),
        )
        if not getattr(self, 'cross_da', False):
            return tuple(
                (route_da, route_da, weight)
                for route_da, weight in endpoint_weights
            )
        return tuple(
            (encoder_da, decoder_da, encoder_weight * decoder_weight)
            for encoder_da, encoder_weight in endpoint_weights
            for decoder_da, decoder_weight in endpoint_weights
        )

    def _ranked_compile_warmup_paths(
        self,
        *,
        validation: bool = False,
    ) -> tuple[_SPADCompileWarmupPath, ...]:
        """Rank graph paths by dataset-balanced case and route exposure."""
        if not self.datasets:
            raise ValueError('compile warmup requires at least one dataset')

        path_exposure: dict[tuple, float] = defaultdict(float)
        representative_exposure: dict[tuple, dict[tuple[str, float], float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        dataset_order = {
            dataset_id: index
            for index, dataset_id in enumerate(self.datasets)
        }
        dataset_weight = 1.0 / len(self.datasets)
        for dataset_id, dataset in self.datasets.items():
            identifiers = (
                dataset.validation_identifiers
                if validation
                else dataset.training_identifiers
            )
            if not identifiers:
                raise ValueError(
                    f'dataset {dataset_id} has no '
                    f'{"validation" if validation else "training"} cases for '
                    f'compile warmup'
                )
            case_weight = dataset_weight / len(identifiers)
            for case_id in identifiers:
                geometry = dataset.geometry_for_case(case_id)
                patch_size = tuple(int(value) for value in geometry.patch_size)
                continuous_da = geometry.continuous_da
                route_exposures = (
                    (*self._validation_da_pair(continuous_da), 1.0),
                ) if validation else self._compile_warmup_route_exposures(
                    continuous_da
                )
                for da_encoder, da_decoder, route_weight in (
                    route_exposures
                ):
                    if route_weight == 0:
                        continue
                    if (
                        getattr(
                            self,
                            'feature_grid_canonicalization_stage',
                            None,
                        )
                        is not None
                        and da_encoder != da_decoder
                    ):
                        raise ValueError('FGC compile warmup requires tied routes')
                    canonical_shape = self._fgc_canonical_shape(
                        continuous_da,
                        patch_size,
                        route_da=da_encoder,
                    )
                    path_key = (
                        patch_size,
                        da_encoder,
                        da_decoder,
                        canonical_shape,
                    )
                    exposure = case_weight * route_weight
                    path_exposure[path_key] += exposure
                    representative_exposure[path_key][
                        (dataset_id, continuous_da)
                    ] += exposure

        def path_sort_key(item: tuple[tuple, float]) -> tuple:
            path_key, exposure = item
            patch_size, da_encoder, da_decoder, canonical_shape = path_key
            return (
                -exposure,
                patch_size,
                da_encoder,
                da_decoder,
                () if canonical_shape is None else canonical_shape,
            )

        paths = []
        for path_key, exposure in sorted(
            path_exposure.items(),
            key=path_sort_key,
        ):
            representatives = representative_exposure[path_key]
            dataset_id, continuous_da = min(
                representatives,
                key=lambda representative: (
                    -representatives[representative],
                    dataset_order[representative[0]],
                    representative[1],
                ),
            )
            patch_size, da_encoder, da_decoder, canonical_shape = path_key
            paths.append(_SPADCompileWarmupPath(
                patch_size=patch_size,
                da_encoder=da_encoder,
                da_decoder=da_decoder,
                canonical_shape=canonical_shape,
                dataset_id=dataset_id,
                continuous_da=continuous_da,
                exposure=exposure,
            ))
        return tuple(paths)

    @staticmethod
    def _node_local_compile_warmup_indices(
        num_paths: int,
        local_rank: int,
        local_world_size: int,
    ) -> tuple[int, ...]:
        """Rotate paths per node so every rank prepares every process-local graph."""
        if num_paths < 1:
            raise ValueError('compile warmup requires at least one path')
        if local_world_size < 1:
            raise ValueError('local world size must be positive')
        if not 0 <= local_rank < local_world_size:
            raise ValueError(
                f'local rank {local_rank} is outside local world size '
                f'{local_world_size}'
            )
        return tuple(
            (round_index + local_rank) % num_paths
            for round_index in range(num_paths)
        )

    def _compile_warmup_local_context(self) -> tuple[int, int]:
        if not dist.is_initialized():
            return 0, 1
        local_rank = int(
            os.environ.get(
                'LOCAL_RANK',
                getattr(self, 'local_rank', dist.get_rank()),
            )
        )
        local_world_size = int(
            os.environ.get('LOCAL_WORLD_SIZE', dist.get_world_size())
        )
        if dist.get_world_size() % local_world_size:
            raise ValueError(
                f'global world size {dist.get_world_size()} is not divisible '
                f'by local world size {local_world_size}'
            )
        if dist.get_rank() % local_world_size != local_rank:
            raise ValueError(
                f'global rank {dist.get_rank()} is inconsistent with local '
                f'rank {local_rank} and local world size {local_world_size}'
            )
        return local_rank, local_world_size

    def _synthetic_compile_warmup_sample(
        self,
        path: _SPADCompileWarmupPath,
    ) -> UniversalSample:
        region_indices = self.registry.indices(path.dataset_id, self.device)
        return UniversalSample(
            dataset_id=path.dataset_id,
            data=torch.zeros(
                1,
                self.num_input_channels,
                *path.patch_size,
                device=self.device,
            ),
            target=torch.zeros(
                1,
                int(region_indices.numel()),
                *path.patch_size,
                device=self.device,
            ),
            region_indices=region_indices,
            continuous_da=path.continuous_da,
        )

    def _warmup_compile_paths(
        self,
        paths: tuple[_SPADCompileWarmupPath, ...],
    ) -> None:
        """Compile frequent full training graphs without synchronizing gradients."""
        local_rank, local_world_size = self._compile_warmup_local_context()
        path_indices = self._node_local_compile_warmup_indices(
            len(paths),
            local_rank,
            local_world_size,
        )
        if getattr(self, 'global_rank', 0) == 0:
            self.print_to_log_file(
                f'Warming all {len(paths)} reachable SPAD Universal training '
                f'paths on every rank, rotated across {local_world_size} local '
                f'ranks in {len(path_indices)} rounds.'
            )

        self.optimizer.zero_grad(set_to_none=True)
        for path_index in path_indices:
            path = paths[path_index]
            sample = self._synthetic_compile_warmup_sample(path)
            no_sync = (
                self.network.no_sync()
                if self.is_ddp
                else dummy_context()
            )
            autocast = (
                torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=True,
                )
                if self.device.type == 'cuda'
                else dummy_context()
            )
            with no_sync:
                with autocast:
                    (outputs,) = self.network((
                        self._network_input_for_sample(
                            sample,
                            (path.da_encoder, path.da_decoder),
                        ),
                    ))
                    active_outputs = self._active_sample_outputs(
                        outputs,
                        sample,
                    )
                    sample_loss = self.loss.sample_loss(
                        active_outputs,
                        sample.target,
                        self._dataset_prediction_mode(sample.dataset_id),
                    )
                    loss = self._add_lkr_zero_gradient_anchor(sample_loss)
                loss.backward()
            del active_outputs, loss, outputs, sample, sample_loss
            if dist.is_initialized():
                dist.barrier()
        self.optimizer.zero_grad(set_to_none=True)

    def _warmup_validation_compile_paths(
        self,
        paths: tuple[_SPADCompileWarmupPath, ...],
    ) -> None:
        """Compile frequent no-gradient validation forwards per node."""
        local_rank, local_world_size = self._compile_warmup_local_context()
        path_indices = self._node_local_compile_warmup_indices(
            len(paths),
            local_rank,
            local_world_size,
        )
        if getattr(self, 'global_rank', 0) == 0:
            self.print_to_log_file(
                f'Warming all {len(paths)} reachable validation paths on every '
                f'rank, rotated across {local_world_size} local ranks in '
                f'{len(path_indices)} rounds.'
            )

        was_training = self.network.training
        self.network.eval()
        try:
            for path_index in path_indices:
                path = paths[path_index]
                sample = self._synthetic_compile_warmup_sample(path)
                autocast = (
                    torch.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=True,
                    )
                    if self.device.type == 'cuda'
                    else dummy_context()
                )
                with torch.no_grad(), autocast:
                    (outputs,) = self.network((
                        self._network_input_for_sample(
                            sample,
                            (path.da_encoder, path.da_decoder),
                        ),
                    ))
                    active_outputs = self._active_sample_outputs(
                        outputs,
                        sample,
                    )
                del active_outputs, outputs, sample
                if dist.is_initialized():
                    dist.barrier()
        finally:
            self.network.train(was_training)

    def _select_da_pair(self, continuous_da: float) -> tuple[int, int]:
        da_discretization = self.da_discretization
        if da_discretization == 'floor':
            floor_da, _, _ = decompose_continuous_da(continuous_da)
            return floor_da, floor_da
        return select_da_pair(
            continuous_da,
            cross=self.cross_da,
        )

    def _validation_da_pair(self, continuous_da: float) -> tuple[int, int]:
        floor_da, _, _ = decompose_continuous_da(continuous_da)
        return floor_da, floor_da

    def _fgc_canonical_shape(
        self,
        continuous_da: float,
        patch_size: tuple[int, int, int],
        *,
        route_da: int | None = None,
    ) -> tuple[int, int, int] | None:
        if getattr(
            self,
            'feature_grid_canonicalization_stage',
            None,
        ) is None:
            return None
        return compute_fgc_geometry(
            continuous_da,
            patch_size,
            stage=self.feature_grid_canonicalization_stage,
            n_stages=self.spad_n_stages,
            min_bottleneck=UNIVERSAL_SPAD_MIN_BOTTLENECK,
            route_da=route_da,
        ).canonical_shape

    def _compile_cache_paths(self) -> tuple[Path, Path]:
        archive = Path(
            os.environ.get(
                'SPAD_UNIVERSAL_COMPILE_CACHE_ARCHIVE',
                Path(self.output_folder) / 'torchinductor-runtime-cache.tar.zst',
            )
        )
        archive_key = hashlib.sha256(str(archive.resolve()).encode()).hexdigest()[:16]
        cache_root = Path(
            os.environ.get(
                'SPAD_UNIVERSAL_COMPILE_CACHE_ROOT',
                '/dev/shm/spad_unet_compile_cache',
            )
        )
        return archive, cache_root / archive_key

    def initialize(self) -> None:
        super().initialize()
        if not self._do_i_compile():
            return

        cache_was_missing = getattr(
            self,
            '_compile_cache_needs_archive',
            False,
        )
        paths = self._static_component_paths()
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit,
            2 * len(paths) + 8,
        )
        training_paths = self._ranked_compile_warmup_paths()
        validation_paths = self._ranked_compile_warmup_paths(
            validation=True,
        )
        self._warmup_compile_paths(training_paths)
        self._warmup_validation_compile_paths(validation_paths)
        self.print_to_log_file(
            f'Warmed all {len(training_paths)} reachable training and '
            f'{len(validation_paths)} validation paths on every rank '
            f'({len(paths)} static training geometries).'
        )
        if cache_was_missing:
            self.archive_compile_cache_now()

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch
        super().on_epoch_end()
        if completed_epoch == 0 and self._do_i_compile():
            self.archive_compile_cache_now(force=True)

    def _get_ddp_kwargs(self) -> dict:
        return {'find_unused_parameters': False}

    def _add_lkr_zero_gradient_anchor(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Give every optional LKR tensor a defined gradient on every step."""
        anchor_terms = [
            # One indexed value is enough to create a dense zero gradient for
            # the whole tensor without reducing every LKR coefficient.
            parameter.reshape(-1)[0]
            for name, parameter in self.network.named_parameters()
            if parameter.requires_grad
            and name.endswith('kernel_reduction_delta')
        ]
        if not anchor_terms:
            return loss
        return loss + torch.stack(anchor_terms).sum() * 0

    def _build_loss(self) -> nn.Module:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        loss_normalization = getattr(
            self,
            'loss_normalization',
            SAMPLE_MEAN_LOSS_NORMALIZATION,
        )
        if loss_normalization == REGION_BALANCED_LOSS_NORMALIZATION:
            active_region_count = getattr(
                self,
                'num_active_regions_per_global_batch',
                len(self.registry),
            )
            normalization_count = (
                {}
                if active_region_count is None
                else {'global_active_region_count': active_region_count}
            )
            # The dynamic path obtains world size together with the live count
            # in mean_sample_loss's collective.
            loss_world_size = world_size if active_region_count is not None else 1
        else:
            normalization_count = {
                'global_sample_count': getattr(
                    self,
                    'global_batch_size',
                    SPAD_UNIVERSAL_GLOBAL_BATCH_SIZE,
                )
            }
            loss_world_size = world_size
        return UniversalPartialLabelLoss(
            batch_dice=False,
            # Compilation is component-scoped, so preserve the existing DDP
            # connection to the lowest-resolution segmentation head.
            keep_last_for_ddp=self.is_ddp,
            loss_normalization=loss_normalization,
            ddp_world_size=loss_world_size,
            **normalization_count,
        )

    def _tta_mirroring_axes(self) -> tuple[int, ...] | None:
        """Training-DA mirror axes when TTA is enabled; training mirrored without label swap, so
        mirrored inference uses the same image-space label convention."""
        if not self.inference_tta_mirroring:
            return None
        axes = getattr(self, 'inference_allowed_mirroring_axes', None)
        if not axes:
            raise ValueError(
                f'{SPAD_UNIVERSAL_TTA_MIRRORING_ENV}=1 requires the training mirror axes, '
                'but inference_allowed_mirroring_axes is unset'
            )
        return tuple(axes)

    def _perform_actual_validation(
        self,
        save_probabilities: bool,
        *,
        model_weights: str,
    ):
        inference_mode = self.inference_mode
        tile_step_size = self.inference_tile_step_size
        sliding_window_batch_size = self.inference_sliding_window_batch_size
        case_evaluation_workers = self.inference_case_evaluation_workers
        inference_variant = _inference_variant_name(
            inference_mode,
            tile_step_size,
            sliding_window_batch_size,
            self.inference_tta_mirroring,
        )
        output_folder_name = f'validation_{inference_variant}'
        if model_weights != 'raw':
            output_folder_name += f'_{model_weights}'
        self._inference_model_weights = model_weights
        with ThreadPoolExecutor(
            max_workers=case_evaluation_workers,
            thread_name_prefix='spad-case-evaluation',
        ) as evaluation_executor:
            return perform_universal_full_volume_validation(
                self,
                save_probabilities,
                output_folder_name=output_folder_name,
                mirroring_axes=self._tta_mirroring_axes(),
                summary_metadata={
                    'inference_mode': inference_mode,
                    'tile_step_size': tile_step_size,
                    'sliding_window_batch_size': sliding_window_batch_size,
                    'case_evaluation_workers': case_evaluation_workers,
                    'model_weights': model_weights,
                },
                summary_log_prefix=(
                    'final_val'
                    if model_weights == 'raw'
                    else f'final_val/{model_weights}'
                ),
                case_predictor=partial(
                    self._predict_inference_case,
                    save_probabilities=save_probabilities,
                    evaluation_executor=evaluation_executor,
                ),
                max_pending_case_predictions=case_evaluation_workers,
            )

    def perform_actual_validation(self, save_probabilities: bool = False):
        weights = self.inference_validation_weights
        if weights == 'model_ema' and self.model_ema_decay is None:
            raise ValueError(
                f'{SPAD_UNIVERSAL_VALIDATION_WEIGHTS_ENV}=model_ema requires '
                'plans with a model EMA decay'
            )
        if weights != 'model_ema':
            raw_result = self._perform_actual_validation(
                save_probabilities,
                model_weights='raw',
            )
            if self.model_ema_decay is None or weights == 'raw':
                return raw_result
        self.print_to_log_file(
            f'Evaluating rank-zero model EMA with decay '
            f'{self.model_ema_decay}.'
        )
        try:
            with self.model_ema_weights():
                ema_result = self._perform_actual_validation(
                    save_probabilities,
                    model_weights='model_ema',
                )
        finally:
            self._inference_model_weights = 'raw'
        if weights == 'model_ema':
            return ema_result
        return {'raw': raw_result, 'model_ema': ema_result}

    def _inference_da_pair(self, dataset_id: str) -> tuple[int, int]:
        floor_da, ceil_da, _ = decompose_continuous_da(
            self.continuous_da[dataset_id]
        )
        if self.inference_mode == 'floor':
            return floor_da, floor_da
        if self.inference_mode == 'ceil':
            return ceil_da, ceil_da
        raise ValueError(
            f'{self.inference_mode} inference uses explicit route components and '
            f'cannot construct a single dataset inference network'
        )

    def build_dataset_inference_network(
        self,
        dataset_id: str,
        *,
        da_encoder: int | None = None,
        da_decoder: int | None = None,
        canonical_shape: tuple[int, int, int] | None = None,
    ) -> nn.Module:
        if da_encoder is None or da_decoder is None:
            if da_encoder is not None or da_decoder is not None:
                raise ValueError('da_encoder and da_decoder must be provided together')
            da_encoder, da_decoder = self._inference_da_pair(dataset_id)
        if (canonical_shape is None) != (
            getattr(
                self,
                'feature_grid_canonicalization_stage',
                None,
            ) is None
        ):
            raise ValueError(
                'canonical shape must be provided exactly for FGC inference'
            )
        packed_output_rows = getattr(
            self,
            'packed_output_rows_by_dataset',
            {},
        ).get(dataset_id)
        return _SPADActiveRegionNetwork(
            unwrap_inference_network(self.network),
            self.registry.indices(dataset_id, self.device),
            da_encoder,
            da_decoder,
            self.dataset_index_by_id[dataset_id],
            canonical_shape,
            packed_output_rows,
        )

    def _dataset_prediction_mode(self, dataset_id: str) -> str:
        return getattr(self, 'dataset_prediction_modes', {}).get(
            dataset_id,
            SIGMOID_REGIONS,
        )

    def _apply_dataset_objective_weight(
        self,
        sample_loss: torch.Tensor,
        dataset_id: str,
    ) -> torch.Tensor:
        """Weight one dataset-local loss toward the plan's dataset objective."""
        weights = getattr(self, 'dataset_loss_weights', None)
        if not weights:
            return sample_loss
        return sample_loss * weights[dataset_id]

    def _active_sample_outputs(
        self,
        outputs: torch.Tensor | list[torch.Tensor],
        sample: UniversalSample,
    ) -> list[torch.Tensor]:
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if getattr(self, 'dataset_output_contract', None) is not None:
            return outputs
        return [
            output.index_select(1, sample.region_indices)
            for output in outputs
        ]

    def _sample_dataset_indices(self, sample: UniversalSample) -> torch.Tensor:
        return torch.full(
            (sample.data.shape[0],),
            self.dataset_index_by_id[sample.dataset_id],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_continuous_da(self, sample: UniversalSample) -> float:
        if sample.continuous_da is not None:
            return sample.continuous_da
        if getattr(self, 'sample_native_z', False):
            raise ValueError(
                'Sample-Native-Z loader did not attach case-specific continuous DA'
            )
        return self.continuous_da[sample.dataset_id]

    def _network_input_for_sample(
        self,
        sample: UniversalSample,
        da_pair: tuple[int, int],
    ) -> tuple:
        sample_input = (
            sample.data,
            *da_pair,
            self._sample_dataset_indices(sample),
        )
        if (
            getattr(self, 'feature_grid_canonicalization_stage', None)
            is not None
            and da_pair[0] != da_pair[1]
        ):
            raise ValueError('FGC training requires tied encoder and decoder DA')
        canonical_shape = self._fgc_canonical_shape(
            self._sample_continuous_da(sample),
            tuple(int(value) for value in sample.data.shape[2:]),
            route_da=da_pair[0],
        )
        packed_output_rows = getattr(
            self,
            'packed_output_rows_by_dataset',
            {},
        ).get(sample.dataset_id)
        if packed_output_rows is not None:
            return (*sample_input, canonical_shape, packed_output_rows)
        if canonical_shape is None:
            return sample_input
        return (*sample_input, canonical_shape)

    def _inference_component_pair(
        self,
        dataset_id: str,
        component: str,
        continuous_da: float | None = None,
    ) -> tuple[int, int]:
        if component not in SPAD_UNIVERSAL_INFERENCE_COMPONENTS:
            raise ValueError(f'unsupported inference component {component!r}')
        if continuous_da is None:
            continuous_da = self.continuous_da[dataset_id]
        floor_da, ceil_da, _ = decompose_continuous_da(
            continuous_da
        )
        return (
            floor_da if component[0] == 'f' else ceil_da,
            floor_da if component[1] == 'f' else ceil_da,
        )

    def _case_inference_geometry(
        self,
        dataset_id: str,
        case_id: str,
        configuration_manager: ConfigurationManager,
    ) -> tuple[float, ConfigurationManager]:
        if not getattr(self, 'sample_native_z', False):
            return self.continuous_da[dataset_id], configuration_manager

        geometry = self.datasets[dataset_id].geometry_for_case(case_id)
        case_configuration = deepcopy(configuration_manager.configuration)
        case_configuration['spacing'] = list(geometry.spacing)
        case_configuration['patch_size'] = list(geometry.patch_size)
        return (
            geometry.continuous_da,
            ConfigurationManager(case_configuration),
        )

    def _inference_probability_cache_path(
        self,
        component: str,
        dataset_id: str,
        case_id: str,
    ) -> Path:
        if component not in SPAD_UNIVERSAL_INFERENCE_COMPONENTS:
            raise ValueError(f'unsupported inference component {component!r}')
        cache_root = (
            Path(self.output_folder)
            / 'inference_probability_cache'
        )
        model_weights = getattr(self, '_inference_model_weights', 'raw')
        if model_weights != 'raw':
            cache_root /= model_weights
        cache_variant = _inference_variant_name(
            '',
            self.inference_tile_step_size,
            self.inference_sliding_window_batch_size,
            self.inference_tta_mirroring,
        ).lstrip('_')
        if cache_variant:
            cache_root /= cache_variant
        return (
            cache_root
            / component
            / f'dataset-{dataset_id}'
            / f'{case_id}.npz'
        )

    @staticmethod
    def _load_inference_probabilities(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as artifact:
            if 'probabilities' not in artifact:
                raise ValueError(f'cached artifact has no probabilities: {path}')
            return np.asarray(artifact['probabilities'], dtype=np.float32)

    @staticmethod
    def _cached_probabilities_match(path: Path, epoch: int) -> bool:
        """Read only the epoch member; the NPZ is a zip, so the probabilities stay untouched."""
        with np.load(path, allow_pickle=False) as artifact:
            return int(artifact['epoch']) == epoch

    @staticmethod
    def _save_inference_probabilities(
        path: Path,
        probabilities: np.ndarray,
        epoch: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w+b',
                prefix=f'.{path.name}.',
                suffix='.tmp',
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.savez(
                    temporary_file,
                    probabilities=np.asarray(probabilities, dtype=np.float16),
                    epoch=np.asarray(epoch, dtype=np.int64),
                )
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _ensure_inference_component_probability_cache(
        self,
        dataset_id: str,
        case_id: str,
        component: str,
        *,
        continuous_da: float,
        data: torch.Tensor,
        properties: dict[str, object],
        plans_manager,
        configuration_manager,
    ) -> tuple[Path, np.ndarray | None]:
        """Return the component cache path, plus freshly predicted fp16 probabilities when the
        cache is stale; the evaluation worker persists the payload to the path before use."""
        floor_da, ceil_da, _ = decompose_continuous_da(
            continuous_da
        )
        if floor_da == ceil_da:
            component = 'ff'
        cache_path = self._inference_probability_cache_path(
            component,
            dataset_id,
            case_id,
        )
        if (
            cache_path.is_file()
            and self._cached_probabilities_match(cache_path, self.current_epoch)
        ):
            return cache_path, None

        da_encoder, da_decoder = self._inference_component_pair(
            dataset_id,
            component,
            continuous_da,
        )
        fgc_enabled = getattr(
            self,
            'feature_grid_canonicalization_stage',
            None,
        ) is not None
        if fgc_enabled and da_encoder != da_decoder:
            raise ValueError(
                f'FGC requires tied encoder and decoder DA, so inference '
                f'component {component!r} has no canonical geometry'
            )
        canonical_shape = (
            # Each route canonicalizes its own native bridge, so the shape follows its DA.
            self._fgc_canonical_shape(
                continuous_da,
                tuple(int(value) for value in configuration_manager.patch_size),
                route_da=da_encoder,
            )
            if fgc_enabled
            else None
        )
        inference_network = (
            self.build_dataset_inference_network(
                dataset_id,
                da_encoder=da_encoder,
                da_decoder=da_decoder,
            )
            if canonical_shape is None
            else self.build_dataset_inference_network(
                dataset_id,
                da_encoder=da_encoder,
                da_decoder=da_decoder,
                canonical_shape=canonical_shape,
            )
        )
        (probabilities,) = predict_source_grid_probabilities(
            inference_networks=(inference_network,),
            data=data,
            properties=properties,
            dataset=self.datasets[dataset_id],
            device=self.device,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            prediction_mode=self._dataset_prediction_mode(dataset_id),
            tile_step_size=self.inference_tile_step_size,
            sliding_window_batch_size=(
                self.inference_sliding_window_batch_size
            ),
            mirroring_axes=self._tta_mirroring_axes(),
            cast_to_fp16=True,
        )
        # Hand the fp16 payload to the evaluation worker, which persists it to cache_path before
        # evaluating; keeping the multi-GB write off this thread is what keeps the GPU fed.
        return cache_path, np.asarray(probabilities, dtype=np.float16)

    def _predict_inference_case(
        self,
        dataset_id: str,
        case_id: str,
        output_folder: Path,
        *,
        save_probabilities: bool,
        evaluation_executor: Executor | None = None,
    ) -> dict[str, object] | Future[dict[str, object]]:
        dataset = self.datasets[dataset_id]
        data, properties, plans_manager, configuration_manager = (
            load_universal_dataset_case(
                dataset=dataset,
                configuration_name=self.configuration_name,
                case_id=case_id,
            )
        )
        continuous_da, configuration_manager = self._case_inference_geometry(
            dataset_id,
            case_id,
            configuration_manager,
        )
        if self.inference_mode == 'floor':
            weighted_components = (('ff', 1.0),)
        elif self.inference_mode == 'ceil':
            weighted_components = (('cc', 1.0),)
        elif self.inference_mode == 'cross':
            floor_da, ceil_da, ceil_weight = decompose_continuous_da(
                continuous_da
            )
            if floor_da == ceil_da:
                weighted_components = (('ff', 1.0),)
            else:
                floor_weight = 1.0 - ceil_weight
                weighted_components = (
                    ('ff', floor_weight * floor_weight),
                    ('fc', floor_weight * ceil_weight),
                    ('cf', ceil_weight * floor_weight),
                    ('cc', ceil_weight * ceil_weight),
                )
        elif self.inference_mode == 'endpoints':
            floor_da, ceil_da, ceil_weight = decompose_continuous_da(
                continuous_da
            )
            if floor_da == ceil_da:
                weighted_components = (('ff', 1.0),)
            else:
                weighted_components = (
                    ('ff', 1.0 - ceil_weight),
                    ('cc', ceil_weight),
                )
        else:
            raise ValueError(f'unsupported inference mode {self.inference_mode!r}')

        component_payloads = {
            component: self._ensure_inference_component_probability_cache(
                dataset_id,
                case_id,
                component,
                continuous_da=continuous_da,
                data=data,
                properties=properties,
                plans_manager=plans_manager,
                configuration_manager=configuration_manager,
            )
            for component, _ in weighted_components
        }
        component_cache_paths = {
            component: path for component, (path, _) in component_payloads.items()
        }
        # Pending futures hold at most max_pending fp16 volumes per rank while awaiting a worker.
        component_probabilities = {
            component: probabilities
            for component, (_, probabilities) in component_payloads.items()
            if probabilities is not None
        }
        del data

        evaluate = partial(
            self._evaluate_inference_component_caches,
            component_cache_paths=component_cache_paths,
            component_probabilities=component_probabilities,
            weighted_components=weighted_components,
            properties=properties,
            plans_manager=plans_manager,
            dataset=dataset,
            case_id=case_id,
            output_folder=output_folder,
            save_probabilities=save_probabilities,
        )
        if evaluation_executor is None:
            return evaluate()
        return evaluation_executor.submit(evaluate)

    def _evaluate_inference_component_caches(
        self,
        *,
        component_cache_paths: Mapping[str, Path],
        weighted_components: tuple[tuple[str, float], ...],
        properties: dict[str, object],
        plans_manager,
        dataset,
        case_id: str,
        output_folder: Path,
        save_probabilities: bool,
        component_probabilities: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, object]:
        """Blend cached route probabilities and evaluate one case on a CPU worker."""

        probabilities = None
        for component, weight in weighted_components:
            fresh = (component_probabilities or {}).get(component)
            if fresh is not None:
                self._save_inference_probabilities(
                    component_cache_paths[component],
                    fresh,
                    self.current_epoch,
                )
                component_values = np.asarray(fresh, dtype=np.float32)
            else:
                component_values = self._load_inference_probabilities(
                    component_cache_paths[component]
                )
            if weight != 1.0:
                np.multiply(
                    component_values,
                    weight,
                    out=component_values,
                )
            if probabilities is None:
                probabilities = component_values
            else:
                np.add(
                    probabilities,
                    component_values,
                    out=probabilities,
                )
            del component_values
        if probabilities is None:
            raise RuntimeError('inference requires at least one component')
        return evaluate_universal_dataset_case(
            probabilities=probabilities,
            properties=properties,
            plans_manager=plans_manager,
            dataset=dataset,
            case_id=case_id,
            output_folder=output_folder,
            save_probabilities=save_probabilities,
            nsd_tolerance_mm=2.0,
            prediction_mode=self._dataset_prediction_mode(
                dataset.dataset_id
            ),
        )

    def get_dataloaders(self):
        train_loader, val_loader, mirror_axes = build_universal_replay_dataloaders(
            self,
            self.datasets,
            self.registry,
            self.replay_reader,
            collate_spad_universal_samples,
        )
        self.inference_allowed_mirroring_axes = mirror_axes
        return train_loader, val_loader

    def train_step(self, batch: dict) -> dict[str, np.ndarray]:
        samples = tuple(
            sample.to(self.device, non_blocking=True)
            for sample in batch['samples']
        )
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == 'cuda'
            else dummy_context()
        ):
            outputs_per_sample = self.network(tuple(
                self._network_input_for_sample(
                    sample,
                    self._select_da_pair(self._sample_continuous_da(sample)),
                )
                for sample in samples
            ))
            if len(outputs_per_sample) != len(samples):
                raise ValueError('Universal SPAD network output count does not match samples')
            sample_losses = []
            for sample, outputs in zip(samples, outputs_per_sample, strict=True):
                active_outputs = self._active_sample_outputs(outputs, sample)
                sample_losses.append(
                    self._apply_dataset_objective_weight(
                        self.loss.sample_loss(
                            active_outputs,
                            sample.target,
                            self._dataset_prediction_mode(sample.dataset_id),
                        ),
                        sample.dataset_id,
                    )
                )
            loss = self.loss.mean_sample_loss(
                sample_losses,
                [int(sample.region_indices.numel()) for sample in samples],
            )
            loss = self._add_lkr_zero_gradient_anchor(loss)

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
        self._update_model_ema()
        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict[str, np.ndarray]:
        samples = tuple(
            sample.to(self.device, non_blocking=True)
            for sample in batch['samples']
        )
        sample_losses = []
        metric_samples = []
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == 'cuda'
            else dummy_context()
        ):
            outputs_per_sample = self.network(tuple(
                self._network_input_for_sample(
                    sample,
                    self._validation_da_pair(
                        self._sample_continuous_da(sample)
                    ),
                )
                for sample in samples
            ))
            if len(outputs_per_sample) != len(samples):
                raise ValueError('Universal SPAD network output count does not match samples')
            for sample, outputs in zip(samples, outputs_per_sample, strict=True):
                active_outputs = self._active_sample_outputs(outputs, sample)
                sample_losses.append(
                    self._apply_dataset_objective_weight(
                        self.loss.sample_loss(
                            active_outputs,
                            sample.target,
                            self._dataset_prediction_mode(sample.dataset_id),
                        ),
                        sample.dataset_id,
                    )
                )
                metric_samples.append(
                    (
                        active_outputs[0],
                        sample.target,
                        sample.region_indices,
                        self._dataset_prediction_mode(sample.dataset_id),
                    )
                )

        tp, fp, fn = aggregate_canonical_region_confusion(
            tuple(metric_samples),
            len(self.registry),
        )
        return {
            'loss': self.loss.mean_sample_loss(
                sample_losses,
                [int(sample.region_indices.numel()) for sample in samples],
            ).detach().cpu().numpy(),
            'tp_hard': tp.cpu().numpy(),
            'fp_hard': fp.cpu().numpy(),
            'fn_hard': fn.cpu().numpy(),
        }
