"""Full-volume evaluation shared by Universal SPAD U-Net systems."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson
import torch
import torch.distributed as dist
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from surface_distance import metrics as surface_metrics
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from pumit.spad_unet.loss import SIGMOID_REGIONS, SOFTMAX_LABELS

if TYPE_CHECKING:
    from pumit.spad_unet.data import UniversalDataset
    from pumit.spad_unet.universal import CanonicalRegionRegistry


NSD_TOLERANCE_MM = 2.0
DEFAULT_SLIDING_WINDOW_TILE_STEP_SIZE = 0.5
DEFAULT_SLIDING_WINDOW_BATCH_SIZE = 1
SPAD_UNIVERSAL_GPU_RESTORE_ENV = 'SPAD_UNIVERSAL_GPU_RESTORE'
# resample_torch_fornnunet reproduces these exact semantics (same separate-z decision, half-pixel
# order-1 in-plane, half-pixel nearest along the anisotropic axis), so only this configuration may
# take the device fast path.
GPU_RESTORE_ELIGIBLE_RESAMPLING = (
    'resample_data_or_seg_to_shape',
    {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None},
)


def gpu_restore_enabled() -> bool:
    return os.environ.get(SPAD_UNIVERSAL_GPU_RESTORE_ENV, '1') != '0'


def validate_sliding_window_tile_step_size(value: object) -> float:
    """Return a finite nnU-Net tile step in the interval (0, 1]."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(
            f'tile_step_size must be a finite real in (0, 1], got {value!r}'
        )
    return float(value)


def validate_sliding_window_batch_size(value: object) -> int:
    """Return a positive number of sliding-window tiles per network forward."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(
            f'sliding_window_batch_size must be a positive integer, got {value!r}'
        )
    return int(value)


def aggregate_canonical_region_confusion(
    samples: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        ...,
    ],
    num_regions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate active-region TP/FP/FN into the canonical output bank."""
    if not samples:
        raise ValueError('canonical confusion aggregation requires at least one sample')
    if num_regions < 1:
        raise ValueError('num_regions must be positive')
    device = samples[0][0].device
    tp = torch.zeros(num_regions, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    for sample in samples:
        if len(sample) == 3:
            active_logits, target, indices = sample
            prediction_mode = SIGMOID_REGIONS
        else:
            active_logits, target, indices, prediction_mode = sample
        expected_channels = target.shape[1] + (
            1 if prediction_mode == SOFTMAX_LABELS else 0
        )
        if (
            active_logits.shape[1] != expected_channels
            or len(indices) != target.shape[1]
        ):
            raise ValueError('active logits, target channels, and canonical indices must align')
        if prediction_mode == SIGMOID_REGIONS:
            prediction = (torch.sigmoid(active_logits) > 0.5).long()
        elif prediction_mode == SOFTMAX_LABELS:
            labels = active_logits.argmax(dim=1, keepdim=True)
            prediction = torch.cat(
                [
                    labels == local_label
                    for local_label in range(1, target.shape[1] + 1)
                ],
                dim=1,
            ).long()
        else:
            raise ValueError(f'unsupported prediction mode {prediction_mode!r}')
        sample_tp, sample_fp, sample_fn, _ = get_tp_fp_fn_tn(
            prediction,
            target,
            axes=(0, *range(2, prediction.ndim)),
        )
        tp.index_add_(0, indices, sample_tp)
        fp.index_add_(0, indices, sample_fp)
        fn.index_add_(0, indices, sample_fn)
    return tp, fp, fn


class ActiveRegionNetwork(nn.Module):
    """Expose one dataset's active rows from a Universal segmentation network."""

    def __init__(
        self,
        network: nn.Module,
        canonical_indices: torch.Tensor,
        dataset_index: int,
    ):
        super().__init__()
        self.network = network
        self.register_buffer(
            'canonical_indices',
            canonical_indices.detach().clone().long(),
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
        logits = self.network(x, self.dataset_index.expand(x.shape[0]))
        if not isinstance(logits, torch.Tensor):
            raise TypeError('full-volume inference requires deep supervision to be disabled')
        return logits.index_select(1, self.canonical_indices)


@dataclass(frozen=True)
class _HeadCount:
    num_segmentation_heads: int


def unwrap_inference_network(network: nn.Module) -> nn.Module:
    """Remove DDP while preserving a compiled module for uneven inference workloads."""
    if isinstance(network, DistributedDataParallel):
        return network.module
    return network


def _gpu_restore_eligible(configuration_manager: Any) -> bool:
    configuration = getattr(configuration_manager, 'configuration', None)
    if not isinstance(configuration, dict):
        return False
    fn_name, fn_kwargs = GPU_RESTORE_ELIGIBLE_RESAMPLING
    return (
        configuration.get('resampling_fn_probabilities') == fn_name
        and configuration.get('resampling_fn_probabilities_kwargs') == fn_kwargs
    )


def restore_region_probabilities(
    predicted_logits: torch.Tensor | np.ndarray,
    properties: dict[str, Any],
    plans_manager: PlansManager,
    configuration_manager: Any,
    prediction_mode: str = SIGMOID_REGIONS,
    *,
    device: torch.device | None = None,
    cast_to_fp16: bool = False,
) -> np.ndarray:
    """Restore dataset-local probabilities to the source image geometry.

    A CUDA ``device`` runs the resample and the nonlinearity on that device, but only under the
    standard probability-resampling configuration (``GPU_RESTORE_ELIGIBLE_RESAMPLING``); any other
    configuration is authoritative and keeps the plans-configured CPU path. ``cast_to_fp16``
    returns float16 probabilities, cast on the device when one is active: numpy's software half
    conversion of a multi-GB volume takes tens of seconds, the device cast is free.
    """
    spacing_transposed = [
        properties['spacing'][axis]
        for axis in plans_manager.transpose_forward
    ]
    shape_after_cropping = properties['shape_after_cropping_and_before_resampling']
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(shape_after_cropping)
        else [spacing_transposed[0], *configuration_manager.spacing]
    )
    if (
        device is not None
        and device.type == 'cuda'
        and _gpu_restore_eligible(configuration_manager)
    ):
        logits = resample_torch_fornnunet(
            torch.as_tensor(predicted_logits, device=device),
            shape_after_cropping,
            current_spacing,
            spacing_transposed,
            is_seg=False,
            device=device,
            force_separate_z=None,
        ).float()
    else:
        if (
            torch.is_tensor(predicted_logits)
            and predicted_logits.device.type != 'cpu'
        ):
            predicted_logits = predicted_logits.cpu()
        logits = configuration_manager.resampling_fn_probabilities(
            predicted_logits,
            shape_after_cropping,
            current_spacing,
            spacing_transposed,
        )
        logits = torch.as_tensor(logits).float()
    if prediction_mode == SIGMOID_REGIONS:
        probabilities = torch.sigmoid(logits)
    elif prediction_mode == SOFTMAX_LABELS:
        probabilities = torch.softmax(logits, dim=0)
    else:
        raise ValueError(f'unsupported prediction mode {prediction_mode!r}')
    if cast_to_fp16:
        probabilities = probabilities.half()
    probabilities = probabilities.cpu().numpy()
    source_shape = tuple(properties['shape_before_cropping'])
    bbox = properties['bbox_used_for_cropping']
    full_bbox = all(
        int(lower) == 0 and int(upper) == int(size)
        for (lower, upper), size in zip(bbox, source_shape, strict=True)
    )
    if full_bbox:
        if tuple(probabilities.shape[1:]) != source_shape:
            raise ValueError(
                f'full-image probabilities have shape {probabilities.shape[1:]}, '
                f'expected {source_shape}'
            )
        restored = probabilities
    else:
        restored = np.zeros(
            (probabilities.shape[0], *source_shape),
            dtype=probabilities.dtype,
        )
        if prediction_mode == SOFTMAX_LABELS:
            restored[0].fill(1)
        insert_crop_into_image(restored, probabilities, bbox)
    return restored.transpose([
        0,
        *(axis + 1 for axis in plans_manager.transpose_backward),
    ])


def region_masks_to_segmentation(
    masks: np.ndarray,
    regions_class_order: list[int],
) -> np.ndarray:
    """Convert independent region masks to nnU-Net's source label-map encoding."""
    if masks.ndim < 2 or masks.shape[0] != len(regions_class_order):
        raise ValueError('region masks and regions_class_order must have matching channels')
    max_label = max(regions_class_order)
    segmentation = np.zeros(
        masks.shape[1:],
        dtype=np.uint8 if max_label < 255 else np.uint16,
    )
    for mask, label in zip(masks, regions_class_order, strict=True):
        segmentation[mask] = label
    return segmentation


def selected_regions_class_order(dataset: UniversalDataset) -> list[int]:
    """Map selected local regions to their source label-map class order."""
    full_order = dataset.dataset_json.get('regions_class_order')
    if not isinstance(full_order, list):
        raise ValueError(
            f'dataset {dataset.dataset_id} is missing regions_class_order'
        )
    from pumit.spad_unet.universal import foreground_region_names

    available_names = foreground_region_names(dataset.dataset_json)
    if len(available_names) != len(full_order):
        raise ValueError(
            f'dataset {dataset.dataset_id} region names and class order disagree'
        )
    order_by_name = dict(zip(available_names, full_order, strict=True))
    return [order_by_name[name] for name in dataset.region_names]


# Each nnU-Net reader records geometry under its own key; both sides of a comparison share one reader.
_GEOMETRY_BLOCKS = {
    'sitk_stuff': ('spacing', 'origin', 'direction'),
    'nibabel_stuff': ('original_affine', 'reoriented_affine'),
}


def validate_source_and_ground_truth_geometry(
    source_properties: dict[str, Any],
    ground_truth_properties: dict[str, Any],
) -> None:
    """Require source image and GT to describe the same physical grid."""
    block = next(
        (
            name for name in _GEOMETRY_BLOCKS
            if name in source_properties and name in ground_truth_properties
        ),
        None,
    )
    if block is None:
        raise ValueError(
            f'source and ground-truth properties share no known geometry block; '
            f'expected one of {sorted(_GEOMETRY_BLOCKS)}, source has '
            f'{sorted(source_properties)}, ground truth has {sorted(ground_truth_properties)}'
        )
    for field in _GEOMETRY_BLOCKS[block]:
        source_value = source_properties[block][field]
        ground_truth_value = ground_truth_properties[block][field]
        if not np.allclose(source_value, ground_truth_value, rtol=0, atol=1e-6):
            raise ValueError(
                f'source and ground-truth {field} differ: '
                f'{source_value} != {ground_truth_value}'
            )


def binary_region_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_mm: tuple[float, ...] | list[float],
    nsd_tolerance_mm: float = NSD_TOLERANCE_MM,
) -> dict[str, float | None]:
    """Compute DSC and surface-area-weighted NSD with explicit empty-mask semantics."""
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape:
        raise ValueError(
            f'reference shape {reference.shape} does not match prediction {prediction.shape}'
        )
    reference_size = int(reference.sum())
    prediction_size = int(prediction.sum())
    if reference_size == 0 and prediction_size == 0:
        return {'dsc': None, 'nsd': None}
    if reference_size == 0 or prediction_size == 0:
        return {'dsc': 0.0, 'nsd': 0.0}
    dsc = 2 * np.logical_and(reference, prediction).sum() / (
        reference_size + prediction_size
    )
    distances = surface_metrics.compute_surface_distances(
        reference,
        prediction,
        spacing_mm,
    )
    nsd = surface_metrics.compute_surface_dice_at_tolerance(
        distances,
        nsd_tolerance_mm,
    )
    return {'dsc': float(dsc), 'nsd': float(nsd)}


def case_record_path(output_folder: Path, case_id: str) -> Path:
    """Locate one case's persisted metric record beside its prediction artifacts."""
    return output_folder / f'{case_id}.record.json'


def load_case_record(path: Path, epoch: int) -> dict[str, Any] | None:
    """Return a persisted case record, or None when it is absent or belongs to another epoch."""
    if not path.is_file():
        return None
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.keys() != {'epoch', 'record'}:
        raise ValueError(f'malformed case record: {path}')
    if payload['epoch'] != epoch:
        return None
    return payload['record']


def save_case_record(path: Path, record: dict[str, Any], epoch: int) -> None:
    """Persist a case record atomically, so its presence implies the case artifacts are complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f'{path.suffix}.tmp')
    temporary_path.write_bytes(
        orjson.dumps({'epoch': epoch, 'record': record}, option=orjson.OPT_INDENT_2)
    )
    os.replace(temporary_path, path)


def partition_validation_jobs(
    datasets: dict[str, UniversalDataset],
    rank: int,
    world_size: int,
) -> tuple[tuple[str, str], ...]:
    """Shard the frozen dataset/case order without duplication."""
    if not 0 <= rank < world_size:
        raise ValueError(f'rank {rank} is outside world size {world_size}')
    jobs = tuple(
        (dataset_id, case_id)
        for dataset_id, dataset in datasets.items()
        for case_id in dataset.validation_identifiers
    )
    return jobs[rank::world_size]


def summarize_universal_metrics(
    records: list[dict[str, Any]],
    datasets: dict[str, UniversalDataset],
    registry: CanonicalRegionRegistry,
    nsd_tolerance_mm: float = NSD_TOLERANCE_MM,
) -> dict[str, Any]:
    """Validate complete case coverage and aggregate by source dataset and local region."""
    expected = Counter(
        (dataset_id, case_id)
        for dataset_id, dataset in datasets.items()
        for case_id in dataset.validation_identifiers
    )
    observed = Counter((record['dataset_id'], record['case_id']) for record in records)
    if observed != expected:
        missing = list((expected - observed).elements())
        extra = list((observed - expected).elements())
        raise RuntimeError(
            f'full-volume evaluation coverage mismatch, missing={missing}, extra={extra}'
        )

    dataset_summaries = {}
    all_region_means: dict[str, list[float]] = {'dsc': [], 'nsd': []}
    for dataset_id, dataset in datasets.items():
        mapping = registry.dataset_mappings[dataset_id]
        case_records = sorted(
            (
                record for record in records
                if record['dataset_id'] == dataset_id
            ),
            key=lambda record: record['case_id'],
        )
        region_summaries = {}
        dataset_region_means: dict[str, list[float]] = {'dsc': [], 'nsd': []}
        for local_index, (
            local_name,
            canonical_name,
            canonical_index,
            label_value,
        ) in enumerate(zip(
            mapping.local_names,
            mapping.canonical_names,
            mapping.canonical_indices,
            dataset.region_values,
            strict=True,
        )):
            metric_lists = {
                metric: [
                    record['regions'][local_index][metric]
                    for record in case_records
                    if record['regions'][local_index][metric] is not None
                ]
                for metric in ('dsc', 'nsd')
            }
            if any(not values for values in metric_lists.values()):
                raise RuntimeError(
                    f'dataset {dataset_id} region {local_name!r} has no evaluable cases'
                )
            means = {
                metric: float(np.mean(values))
                for metric, values in metric_lists.items()
            }
            region_summaries[local_name] = {
                'canonical_name': canonical_name,
                'canonical_index': canonical_index,
                'label_value': label_value,
                'mean': means,
            }
            for metric, value in means.items():
                dataset_region_means[metric].append(value)
                all_region_means[metric].append(value)
        macro = {
            metric: float(np.mean(values))
            for metric, values in dataset_region_means.items()
        }
        dataset_summaries[dataset_id] = {
            'name': dataset.name,
            'num_cases': len(case_records),
            'regions': region_summaries,
            'macro': macro,
            'cases': case_records,
        }

    corpus_macro = {
        metric: float(np.mean([
            summary['macro'][metric]
            for summary in dataset_summaries.values()
        ]))
        for metric in ('dsc', 'nsd')
    }
    region_macro = {
        metric: float(np.mean(values))
        for metric, values in all_region_means.items()
    }
    return {
        'fold': 0,
        'num_datasets': len(datasets),
        'num_cases': len(records),
        'num_canonical_regions': len(registry),
        'nsd_tolerance_mm': nsd_tolerance_mm,
        'datasets': dataset_summaries,
        'corpus_macro': corpus_macro,
        'region_macro': region_macro,
    }


def load_universal_dataset_case(
    *,
    dataset: UniversalDataset,
    configuration_name: str,
    case_id: str,
) -> tuple[torch.Tensor, dict[str, Any], PlansManager, Any]:
    """Load one preprocessed case and the geometry needed for source-grid restoration."""
    plans_manager = PlansManager(dataset.plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    dataset_class = infer_dataset_class(str(dataset.data_folder))
    preprocessed = dataset_class(str(dataset.data_folder), identifiers=[case_id])
    data, _, seg_prev, properties = preprocessed.load_case(case_id)
    if seg_prev is not None:
        raise ValueError('cascaded inputs are not supported by Universal full-volume evaluation')
    return torch.from_numpy(data[:]), properties, plans_manager, configuration_manager


def predict_source_grid_probabilities(
    *,
    inference_networks: tuple[nn.Module, ...],
    data: torch.Tensor,
    properties: dict[str, Any],
    dataset: UniversalDataset,
    device: torch.device,
    plans_manager: PlansManager,
    configuration_manager: Any,
    prediction_mode: str = SIGMOID_REGIONS,
    tile_step_size: float = DEFAULT_SLIDING_WINDOW_TILE_STEP_SIZE,
    sliding_window_batch_size: int = DEFAULT_SLIDING_WINDOW_BATCH_SIZE,
    cast_to_fp16: bool = False,
    mirroring_axes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    """Predict one loaded case under one or more network paths; ``mirroring_axes`` enables mirror
    TTA over those axes, otherwise inference runs without TTA."""
    tile_step_size = validate_sliding_window_tile_step_size(tile_step_size)
    sliding_window_batch_size = validate_sliding_window_batch_size(
        sliding_window_batch_size
    )
    probabilities = []
    for inference_network in inference_networks:
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=mirroring_axes is not None,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
            sliding_window_batch_size=sliding_window_batch_size,
        )
        predictor.manual_initialization(
            inference_network,
            plans_manager,
            configuration_manager,
            None,
            dataset.dataset_json,
            'UniversalFullVolumeEvaluator',
            mirroring_axes,
            allow_compile=False,
        )
        num_output_channels = len(dataset.region_names) + (
            1 if prediction_mode == SOFTMAX_LABELS else 0
        )
        predictor.label_manager = _HeadCount(num_output_channels)
        logits = predictor.predict_sliding_window_return_logits(data)
        restore_device = (
            logits.device
            if gpu_restore_enabled() and logits.device.type == 'cuda'
            else None
        )
        if restore_device is None:
            logits = logits.cpu()
        probabilities.append(restore_region_probabilities(
            logits,
            properties,
            plans_manager,
            configuration_manager,
            prediction_mode,
            device=restore_device,
            cast_to_fp16=cast_to_fp16,
        ))
    return tuple(probabilities)


def evaluate_universal_dataset_case(
    *,
    probabilities: np.ndarray,
    properties: dict[str, Any],
    plans_manager: PlansManager,
    dataset: UniversalDataset,
    case_id: str,
    output_folder: Path,
    save_probabilities: bool,
    nsd_tolerance_mm: float,
    prediction_mode: str = SIGMOID_REGIONS,
) -> dict[str, Any]:
    """Export and score source-grid probabilities under one dataset contract."""
    expected_channels = len(dataset.region_names) + (
        1 if prediction_mode == SOFTMAX_LABELS else 0
    )
    if probabilities.ndim != 4 or probabilities.shape[0] != expected_channels:
        raise ValueError(
            f'{dataset.dataset_id}:{case_id} probability shape {probabilities.shape} '
            f'does not match {expected_channels} local outputs'
        )
    if prediction_mode == SIGMOID_REGIONS:
        prediction_masks = probabilities > 0.5
    elif prediction_mode == SOFTMAX_LABELS:
        prediction_labels = probabilities.argmax(axis=0)
        prediction_masks = np.stack(
            [
                prediction_labels == local_label
                for local_label in range(1, len(dataset.region_names) + 1)
            ]
        )
    else:
        raise ValueError(f'unsupported prediction mode {prediction_mode!r}')

    reader_writer = plans_manager.image_reader_writer_class()
    file_ending = dataset.dataset_json['file_ending']
    gt_path = (
        dataset.data_folder.parent
        / 'gt_segmentations'
        / f'{case_id}{file_ending}'
    )
    ground_truth, ground_truth_properties = reader_writer.read_seg(str(gt_path))
    ground_truth = ground_truth[0]
    validate_source_and_ground_truth_geometry(properties, ground_truth_properties)
    if tuple(prediction_masks.shape[1:]) != tuple(ground_truth.shape):
        raise ValueError(
            f'{dataset.dataset_id}:{case_id} restored prediction shape '
            f'{prediction_masks.shape[1:]} does not match GT {ground_truth.shape}'
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    regions_class_order = selected_regions_class_order(dataset)
    segmentation = region_masks_to_segmentation(
        prediction_masks,
        regions_class_order,
    )
    reader_writer.write_seg(
        segmentation,
        str(output_folder / f'{case_id}{file_ending}'),
        properties,
    )
    prediction_artifact = {'region_masks': prediction_masks}
    if save_probabilities:
        prediction_artifact['probabilities'] = probabilities
    save_npz = np.savez if save_probabilities else np.savez_compressed
    save_npz(output_folder / f'{case_id}.npz', **prediction_artifact)

    spacing = tuple(float(value) for value in properties['spacing'])
    region_metrics = [
        binary_region_metrics(
            region_or_label_to_mask(ground_truth, region_value),
            prediction_masks[region_index],
            spacing,
            nsd_tolerance_mm,
        )
        for region_index, region_value in enumerate(dataset.region_values)
    ]
    return {
        'dataset_id': dataset.dataset_id,
        'case_id': case_id,
        'regions': region_metrics,
    }


def predict_universal_dataset_case(
    *,
    inference_network: nn.Module,
    dataset: UniversalDataset,
    configuration_name: str,
    device: torch.device,
    case_id: str,
    output_folder: Path,
    save_probabilities: bool,
    nsd_tolerance_mm: float,
    prediction_mode: str = SIGMOID_REGIONS,
) -> dict[str, Any]:
    data, properties, plans_manager, configuration_manager = (
        load_universal_dataset_case(
            dataset=dataset,
            configuration_name=configuration_name,
            case_id=case_id,
        )
    )
    (probabilities,) = predict_source_grid_probabilities(
        inference_networks=(inference_network,),
        data=data,
        properties=properties,
        dataset=dataset,
        device=device,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        prediction_mode=prediction_mode,
    )
    return evaluate_universal_dataset_case(
        probabilities=probabilities,
        properties=properties,
        plans_manager=plans_manager,
        dataset=dataset,
        case_id=case_id,
        output_folder=output_folder,
        save_probabilities=save_probabilities,
        nsd_tolerance_mm=nsd_tolerance_mm,
        prediction_mode=prediction_mode,
    )


def perform_universal_full_volume_validation(
    trainer: Any,
    save_probabilities: bool = False,
    nsd_tolerance_mm: float = NSD_TOLERANCE_MM,
    *,
    output_folder_name: str = 'validation',
    summary_metadata: dict[str, Any] | None = None,
    summary_log_prefix: str = 'final_val',
    case_predictor: Callable[
        [str, str, Path],
        dict[str, Any] | Future[dict[str, Any]],
    ] | None = None,
    max_pending_case_predictions: int = 1,
    mirroring_axes: tuple[int, ...] | None = None,
) -> dict[str, Any] | None:
    """Run complete source-grid validation for one Universal trainer."""
    if not output_folder_name or Path(output_folder_name).name != output_folder_name:
        raise ValueError(
            f'output_folder_name must be one path component, got {output_folder_name!r}'
        )
    if (
        isinstance(max_pending_case_predictions, bool)
        or not isinstance(max_pending_case_predictions, Integral)
        or max_pending_case_predictions < 1
    ):
        raise ValueError(
            f'max_pending_case_predictions must be a positive integer, '
            f'got {max_pending_case_predictions!r}'
        )
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    try:
        rank = dist.get_rank() if trainer.is_ddp else 0
        world_size = dist.get_world_size() if trainer.is_ddp else 1
        output_root = Path(trainer.output_folder) / output_folder_name
        if case_predictor is None:
            inference_networks = {
                dataset_id: trainer.build_dataset_inference_network(dataset_id)
                for dataset_id in trainer.datasets
            }

            def predict_case(
                dataset_id: str,
                case_id: str,
                output_folder: Path,
            ) -> dict[str, Any]:
                return predict_universal_dataset_case(
                    inference_network=inference_networks[dataset_id],
                    dataset=trainer.datasets[dataset_id],
                    configuration_name=trainer.configuration_name,
                    device=trainer.device,
                    case_id=case_id,
                    output_folder=output_folder,
                    save_probabilities=save_probabilities,
                    nsd_tolerance_mm=nsd_tolerance_mm,
                )
        else:
            predict_case = case_predictor

        local_records = []
        pending_records: deque[
            tuple[Path, Future[dict[str, Any]]]
        ] = deque()

        def collect_oldest_pending_record() -> None:
            record_path, future = pending_records.popleft()
            record = future.result()
            save_case_record(record_path, record, trainer.current_epoch)
            local_records.append(record)

        for dataset_id, case_id in partition_validation_jobs(
            trainer.datasets,
            rank,
            world_size,
        ):
            output_folder = output_root / f'dataset-{dataset_id}'
            record_path = case_record_path(output_folder, case_id)
            cached_record = load_case_record(record_path, trainer.current_epoch)
            if cached_record is not None:
                trainer.print_to_log_file(
                    f'full-volume validation {dataset_id}:{case_id}, rank {rank}, cached'
                )
                local_records.append(cached_record)
                continue
            trainer.print_to_log_file(
                f'full-volume validation {dataset_id}:{case_id}, rank {rank}'
            )
            record_or_future = predict_case(dataset_id, case_id, output_folder)
            if isinstance(record_or_future, Future):
                if len(pending_records) >= max_pending_case_predictions:
                    collect_oldest_pending_record()
                pending_records.append((record_path, record_or_future))
            else:
                save_case_record(
                    record_path,
                    record_or_future,
                    trainer.current_epoch,
                )
                local_records.append(record_or_future)

        while pending_records:
            collect_oldest_pending_record()

        if trainer.is_ddp:
            gathered: list[list[dict[str, Any]] | None] = [None] * world_size
            dist.all_gather_object(gathered, local_records)
            records = [
                record
                for rank_records in gathered
                if rank_records is not None
                for record in rank_records
            ]
        else:
            records = local_records

        if rank != 0:
            return None
        summary = summarize_universal_metrics(
            records,
            trainer.datasets,
            trainer.registry,
            nsd_tolerance_mm,
        )
        summary['mirroring_axes'] = (
            list(mirroring_axes) if mirroring_axes is not None else None
        )
        if getattr(trainer, 'dataset_output_contract', None) is None:
            summary['prediction_contract'] = (
                'dataset-local independent region masks in each case NPZ are '
                'authoritative; NIfTI label maps use source '
                'regions_class_order for visualization'
            )
        else:
            summary['prediction_contract'] = (
                'dataset-native softmax labels or independent sigmoid regions; '
                'case NPZ region_masks contain the evaluated foreground masks'
            )
        if summary_metadata is not None:
            overlap = summary.keys() & summary_metadata.keys()
            if overlap:
                raise ValueError(
                    f'summary metadata collides with evaluator fields: {sorted(overlap)}'
                )
            summary.update(summary_metadata)
        (output_root / 'summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n'
        )
        trainer.logger.log_summary(
            f'{summary_log_prefix}/foreground_dice',
            summary['corpus_macro']['dsc'],
        )
        trainer.logger.log_summary(
            f'{summary_log_prefix}/foreground_nsd',
            summary['corpus_macro']['nsd'],
        )
        trainer.print_to_log_file(
            'Universal full-volume validation complete, '
            f"corpus-macro DSC={summary['corpus_macro']['dsc']:.4f}, "
            f"NSD@{nsd_tolerance_mm:g}mm={summary['corpus_macro']['nsd']:.4f}",
            also_print_to_console=True,
        )
        return summary
    finally:
        trainer.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
