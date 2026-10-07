"""Shared semantic output space for Universal SPAD U-Net systems."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from pumit.spad_unet.loss import (
    REGION_BALANCED_LOSS_NORMALIZATION,
    SAMPLE_MEAN_LOSS_NORMALIZATION,
    SIGMOID_REGIONS,
    SOFTMAX_LABELS,
    UniversalPartialLabelBatch,
    UniversalPartialLabelLoss,
)
from pumit.spad_unet.task_aware_bottleneck import (
    TAB_ATTENTION_DOWNSAMPLE_RATE,
    TAB_DEPTH,
    TAB_DIM,
    TAB_FOURIER_SCALE,
    TAB_FOURIER_SEED,
    TAB_HEADS,
    TAB_MLP_DIM,
    TAB_MIN_INPLANE_DOWNSAMPLE,
    TAB_TOKENS_PER_DATASET,
    MultiScaleTaskAwareBottleneck,
    TaskAwareBottleneck,
)

CORPUS_GRID_UNIVERSAL_PLANS_IDENTIFIER = (
    'SPADUNetCorpusGrid1x1x1FOV192ComplementaryFG6B1MultiScaleTABT16RBPlans'
)
UNIVERSAL_RESENC_NETWORK_CLASS_NAME = (
    'pumit.spad_unet.universal.UniversalResidualEncoderUNet'
)
_RESENC_NETWORK_CLASS_NAME = (
    'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
)
DATASET_NATIVE_OUTPUT_MODE = 'dataset_native'


class UniversalResidualEncoderUNet(ResidualEncoderUNet):
    """ResEnc U-Net whose canonical output bank is declared by the plans."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        num_canonical_regions: int,
        num_samples_per_global_batch: int,
        batch_dice: bool,
        num_task_datasets: int,
        loss_normalization: str = SAMPLE_MEAN_LOSS_NORMALIZATION,
        num_active_regions_per_global_batch: int | None = None,
        tab_tokens_per_dataset: int,
        tab_dim: int,
        tab_depth: int,
        tab_heads: int,
        tab_mlp_dim: int,
        tab_attention_downsample_rate: int,
        tab_fourier_scale: float,
        tab_fourier_seed: int,
        tab_feature_level_indices: tuple[int, ...] | list[int] | None = None,
        **architecture_kwargs: Any,
    ):
        if num_classes < 1:
            raise ValueError(
                'the nnU-Net dataset label manager must expose at least one output'
            )
        if num_canonical_regions < 1:
            raise ValueError('num_canonical_regions must be positive')
        if num_samples_per_global_batch < 1:
            raise ValueError('num_samples_per_global_batch must be positive')
        super().__init__(
            input_channels=input_channels,
            num_classes=num_canonical_regions,
            **architecture_kwargs,
        )
        self.tab_feature_level_indices = (
            None
            if tab_feature_level_indices is None
            else tuple(int(index) for index in tab_feature_level_indices)
        )
        common_kwargs = {
            'num_datasets': num_task_datasets,
            'tokens_per_dataset': tab_tokens_per_dataset,
            'embedding_dim': tab_dim,
            'depth': tab_depth,
            'num_heads': tab_heads,
            'mlp_dim': tab_mlp_dim,
            'attention_downsample_rate': tab_attention_downsample_rate,
            'fourier_scale': tab_fourier_scale,
            'fourier_seed': tab_fourier_seed,
        }
        if self.tab_feature_level_indices is None:
            self.task_aware_bottleneck = TaskAwareBottleneck(
                bottleneck_channels=self.encoder.output_channels[-1],
                **common_kwargs,
            )
        else:
            output_channels = tuple(self.encoder.output_channels)
            if (
                not self.tab_feature_level_indices
                or tuple(sorted(set(self.tab_feature_level_indices)))
                != self.tab_feature_level_indices
                or self.tab_feature_level_indices[-1] >= len(output_channels)
            ):
                raise ValueError(
                    'tab_feature_level_indices must be sorted, unique, and valid'
                )
            selected_channels = tuple(
                output_channels[index]
                for index in self.tab_feature_level_indices
            )
            self.task_aware_bottleneck = MultiScaleTaskAwareBottleneck(
                feature_channels=selected_channels,
                **common_kwargs,
            )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        normalization_count = (
            {
                'global_active_region_count': (
                    num_active_regions_per_global_batch
                )
            }
            if loss_normalization == REGION_BALANCED_LOSS_NORMALIZATION
            else {'global_sample_count': num_samples_per_global_batch}
        )
        fixed_global_count = next(iter(normalization_count.values()))
        self.partial_label_loss = UniversalPartialLabelLoss(
            batch_dice=batch_dice,
            keep_last_for_ddp=world_size > 1,
            loss_normalization=loss_normalization,
            # Omitting the count selects the dynamic divisor, which all-reduces the live active-region
            # total and therefore reads world size at runtime.
            ddp_world_size=world_size if fixed_global_count is not None else 1,
            **normalization_count,
        )

    def forward(
        self,
        x: torch.Tensor,
        dataset_indices: torch.Tensor,
        targets: UniversalPartialLabelBatch | None = None,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[
        torch.Tensor | list[torch.Tensor],
        torch.Tensor,
    ]:
        """Return logits for inference or logits and loss for training."""
        skips = self.encoder(x)
        if self.tab_feature_level_indices is None:
            skips[-1] = self.task_aware_bottleneck(skips[-1], dataset_indices)
        else:
            selected_features = [
                skips[index] for index in self.tab_feature_level_indices
            ]
            conditioned_features = self.task_aware_bottleneck(
                selected_features,
                dataset_indices,
            )
            for index, feature in zip(
                self.tab_feature_level_indices,
                conditioned_features,
                strict=True,
            ):
                skips[index] = feature
        logits = self.decoder(skips)
        if targets is None:
            return logits
        return logits, self.partial_label_loss(logits, targets)


def build_universal_resenc_architecture(
    source_architecture: Mapping[str, Any],
    num_canonical_regions: int,
    num_samples_per_global_batch: int,
    batch_dice: bool,
    num_task_datasets: int,
    loss_normalization: str = SAMPLE_MEAN_LOSS_NORMALIZATION,
    num_active_regions_per_global_batch: int | None = None,
) -> dict[str, Any]:
    """Derive a native nnU-Net architecture schema for a canonical region bank."""
    if source_architecture.get('network_class_name') != _RESENC_NETWORK_CLASS_NAME:
        raise ValueError(
            'Corpus-grid Universal requires the planned ResidualEncoderUNet architecture'
        )
    source_kwargs = source_architecture.get('arch_kwargs')
    if not isinstance(source_kwargs, Mapping):
        raise TypeError('source architecture arch_kwargs must be a mapping')
    if num_canonical_regions < 1:
        raise ValueError('num_canonical_regions must be positive')
    if num_samples_per_global_batch < 1:
        raise ValueError('num_samples_per_global_batch must be positive')

    strides = source_kwargs.get('strides')
    if not isinstance(strides, (list, tuple)) or not strides:
        raise ValueError('Corpus-grid architecture must define its stride schedule')
    inplane_downsample = 1
    tab_feature_level_indices = []
    for level_index, stride in enumerate(strides):
        if len(stride) != 3:
            raise ValueError('Corpus-grid Universal requires 3D strides')
        if stride[1] != stride[2]:
            raise ValueError('Corpus-grid Universal requires matched in-plane strides')
        inplane_downsample *= int(stride[1])
        if inplane_downsample >= TAB_MIN_INPLANE_DOWNSAMPLE:
            tab_feature_level_indices.append(level_index)
    if not tab_feature_level_indices:
        raise ValueError(
            'Corpus-grid architecture has no feature level at or below /16'
        )

    architecture = deepcopy(dict(source_architecture))
    architecture['network_class_name'] = UNIVERSAL_RESENC_NETWORK_CLASS_NAME
    architecture['arch_kwargs'] = {
        **source_kwargs,
        'num_canonical_regions': num_canonical_regions,
        'num_samples_per_global_batch': num_samples_per_global_batch,
        'batch_dice': batch_dice,
        'num_task_datasets': num_task_datasets,
        'loss_normalization': loss_normalization,
        'num_active_regions_per_global_batch': (
            num_active_regions_per_global_batch
        ),
        'tab_tokens_per_dataset': TAB_TOKENS_PER_DATASET,
        'tab_dim': TAB_DIM,
        'tab_depth': TAB_DEPTH,
        'tab_heads': TAB_HEADS,
        'tab_mlp_dim': TAB_MLP_DIM,
        'tab_attention_downsample_rate': TAB_ATTENTION_DOWNSAMPLE_RATE,
        'tab_fourier_scale': TAB_FOURIER_SCALE,
        'tab_fourier_seed': TAB_FOURIER_SEED,
        'tab_feature_level_indices': tab_feature_level_indices,
    }
    return architecture


def _is_background_region(value: Any) -> bool:
    if isinstance(value, int):
        return value == 0
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(item == 0 for item in value)
    raise TypeError(f'unsupported label value {value!r}')


def foreground_region_names(dataset_json: dict[str, Any]) -> tuple[str, ...]:
    """Return region names in nnU-Net's region-channel order."""
    names = []
    for name, value in dataset_json['labels'].items():
        if name == 'ignore':
            raise ValueError(
                'ignore labels are not supported by the Universal region bank'
            )
        if not _is_background_region(value):
            names.append(name)
    if not names:
        raise ValueError('dataset has no foreground regions')
    return tuple(names)


def foreground_region_values(
    dataset_json: dict[str, Any],
) -> tuple[int | tuple[int, ...], ...]:
    """Return region definitions in the same order as :func:`foreground_region_names`."""
    values = []
    for name, value in dataset_json['labels'].items():
        if name == 'ignore':
            raise ValueError(
                'ignore labels are not supported by the Universal region bank'
            )
        if _is_background_region(value):
            continue
        if isinstance(value, list):
            value = tuple(value)
        values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class DatasetRegionMapping:
    """Map one dataset's target channels into the canonical region bank.

    Output-space view only; the manifest-level suite contract is
    `pumit.spad_unet.data.UniversalDatasetSpec`.
    """

    local_names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    canonical_indices: tuple[int, ...]


class CanonicalRegionRegistry:
    """Auditable hybrid ontology with shared compatible regions and dataset-local fallbacks."""

    def __init__(
        self,
        dataset_jsons: dict[str, dict[str, Any]],
        shared_regions: dict[str, dict[str, str]],
    ):
        if not dataset_jsons:
            raise ValueError('at least one dataset is required')

        local_names = {
            dataset_id: foreground_region_names(dataset_json)
            for dataset_id, dataset_json in dataset_jsons.items()
        }
        shared_lookup: dict[tuple[str, str], str] = {}
        for canonical_name, references in shared_regions.items():
            if '/' not in canonical_name:
                raise ValueError(
                    f'shared canonical name must be namespaced: {canonical_name!r}'
                )
            if len(references) < 2:
                raise ValueError(
                    f'shared region {canonical_name!r} must reference at least two datasets'
                )
            for dataset_id, region_name in references.items():
                if dataset_id not in local_names:
                    raise ValueError(
                        f'shared region {canonical_name!r} references unknown dataset {dataset_id!r}'
                    )
                if region_name not in local_names[dataset_id]:
                    raise ValueError(
                        f'shared region {canonical_name!r} references unknown region '
                        f'{dataset_id}:{region_name}'
                    )
                reference = (dataset_id, region_name)
                if reference in shared_lookup:
                    raise ValueError(
                        f'{dataset_id}:{region_name} belongs to both '
                        f'{shared_lookup[reference]!r} and {canonical_name!r}'
                    )
                shared_lookup[reference] = canonical_name

        canonical_names: list[str] = []
        canonical_index: dict[str, int] = {}
        mappings: dict[str, DatasetRegionMapping] = {}
        for dataset_id, names in local_names.items():
            mapped_names = []
            mapped_indices = []
            for local_name in names:
                name = shared_lookup.get(
                    (dataset_id, local_name),
                    f'dataset-{dataset_id}/{local_name}',
                )
                if name not in canonical_index:
                    canonical_index[name] = len(canonical_names)
                    canonical_names.append(name)
                mapped_names.append(name)
                mapped_indices.append(canonical_index[name])
            if len(set(mapped_indices)) != len(mapped_indices):
                raise ValueError(
                    f'dataset {dataset_id} maps multiple local regions to one canonical region'
                )
            mappings[dataset_id] = DatasetRegionMapping(
                local_names=names,
                canonical_names=tuple(mapped_names),
                canonical_indices=tuple(mapped_indices),
            )

        missing_shared = set(shared_regions) - set(canonical_names)
        if missing_shared:
            raise ValueError(
                f'unused shared canonical regions: {sorted(missing_shared)}'
            )

        self.canonical_names = tuple(canonical_names)
        self.dataset_mappings = mappings

    def __len__(self) -> int:
        return len(self.canonical_names)

    def indices(
        self, dataset_id: str, device: torch.device | None = None
    ) -> torch.Tensor:
        mapping = self.dataset_mappings[dataset_id]
        return torch.tensor(mapping.canonical_indices, device=device, dtype=torch.long)


def build_region_registry(
    manifest: 'UniversalExperimentManifest',
    datasets: Mapping[str, Any],
) -> CanonicalRegionRegistry:
    """Build the canonical registry from the manifest's ordered dataset suite."""
    missing = set(manifest.dataset_ids) - set(datasets)
    unknown = set(datasets) - set(manifest.dataset_ids)
    if tuple(datasets) != manifest.dataset_ids:
        raise ValueError(
            'canonical region registry requires the complete ordered Universal dataset suite; '
            f'missing={sorted(missing)}, unknown={sorted(unknown)}, '
            f'expected_order={manifest.dataset_ids}, actual_order={tuple(datasets)}'
        )
    dataset_jsons = {
        dataset_id: {
            'labels': {
                'background': 0,
                **{
                    name: dataset.dataset_json['labels'][name]
                    for name in dataset.region_names
                },
            },
        }
        for dataset_id, dataset in datasets.items()
    }
    shared_regions = {
        name: {
            dataset_id: region_name
            for dataset_id, region_name in references.items()
            if dataset_id in datasets
        }
        for name, references in manifest.shared_regions.items()
        if sum(dataset_id in datasets for dataset_id in references) >= 2
    }
    return CanonicalRegionRegistry(dataset_jsons, shared_regions)


def build_dataset_output_contract(
    registry: CanonicalRegionRegistry,
    datasets: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the packed dataset-native output contract from source plans."""
    if tuple(datasets) != tuple(registry.dataset_mappings):
        raise ValueError(
            'dataset-native output requires datasets in canonical registry order'
        )

    dataset_contracts = {}
    next_background_row = len(registry)
    for dataset_id, dataset in datasets.items():
        label_manager = PlansManager(dataset.plans).get_label_manager(
            dataset.dataset_json
        )
        canonical_foreground_rows = list(
            registry.dataset_mappings[dataset_id].canonical_indices
        )
        if label_manager.has_regions:
            prediction_mode = SIGMOID_REGIONS
            packed_output_rows = canonical_foreground_rows
        else:
            selected_labels = tuple(
                value
                if isinstance(value, int)
                else value[0]
                if isinstance(value, (tuple, list)) and len(value) == 1
                else value
                for value in dataset.region_values
            )
            if any(not isinstance(value, int) for value in selected_labels):
                raise ValueError(
                    f'dataset {dataset_id} uses label prediction but has '
                    f'non-scalar foreground definitions {dataset.region_values}'
                )
            if selected_labels != tuple(label_manager.foreground_labels):
                raise ValueError(
                    f'dataset {dataset_id} native softmax contract requires all '
                    f'source foreground labels in class order; '
                    f'selected={selected_labels}, '
                    f'available={label_manager.foreground_labels}'
                )
            prediction_mode = SOFTMAX_LABELS
            packed_output_rows = [
                next_background_row,
                *canonical_foreground_rows,
            ]
            next_background_row += 1
        dataset_contracts[dataset_id] = {
            'prediction_mode': prediction_mode,
            'canonical_foreground_rows': canonical_foreground_rows,
            'packed_output_rows': packed_output_rows,
        }

    return {
        'mode': DATASET_NATIVE_OUTPUT_MODE,
        'num_packed_output_channels': next_background_row,
        'datasets': dataset_contracts,
    }
