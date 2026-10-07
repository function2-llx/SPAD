"""Shared dataset contracts for SPAD U-Net Universal experiments."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from pumit.spad_unet.geometry import compute_continuous_da
from pumit.spad_unet.universal import (
    build_region_registry,
    foreground_region_names,
    foreground_region_values,
)

UNIVERSAL_NNUNET_DATASET_ID = 590
UNIVERSAL_NNUNET_DATASET_NAME = 'Dataset590_SPADUniversal'


@dataclass(frozen=True)
class UniversalCaseGeometry:
    """Training geometry derived from one preprocessed case."""

    spacing: tuple[float, float, float]
    patch_size: tuple[int, int, int]
    continuous_da: float


@dataclass(frozen=True)
class UniversalDataset:
    """One source dataset resolved against an nnU-Net preprocessing plan."""

    dataset_id: str
    name: str
    dataset_json: dict[str, Any]
    plans: dict[str, Any]
    data_folder: Path
    num_cases: int
    training_identifiers: tuple[str, ...]
    validation_identifiers: tuple[str, ...]
    spacing: tuple[float, float, float]
    patch_size: tuple[int, int, int]
    batch_size: int
    batch_dice: bool
    region_names: tuple[str, ...]
    region_values: tuple[int | tuple[int, ...], ...]
    case_geometries: dict[str, UniversalCaseGeometry] | None = None

    def geometry_for_case(self, case_id: str) -> UniversalCaseGeometry:
        """Return case-specific geometry, or the dataset plan geometry."""
        if self.case_geometries is None:
            return UniversalCaseGeometry(
                spacing=self.spacing,
                patch_size=self.patch_size,
                continuous_da=compute_continuous_da(self.spacing),
            )
        try:
            return self.case_geometries[case_id]
        except KeyError as error:
            raise KeyError(
                f'dataset {self.dataset_id} has no geometry for case {case_id!r}'
            ) from error


def derive_sample_native_patch_size(
    spacing: tuple[float, float, float],
    *,
    target_fov_mm: float,
    n_stages: int,
) -> tuple[int, int, int]:
    """Derive the per-case six-stage patch from effective preprocessed spacing."""
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in spacing
    ):
        raise ValueError(
            f'spacing must contain three finite positive values, got {spacing}'
        )
    if not math.isfinite(target_fov_mm) or target_fov_mm <= 0:
        raise ValueError(
            f'target_fov_mm must be finite and positive, got {target_fov_mm!r}'
        )
    if isinstance(n_stages, bool) or not isinstance(n_stages, int) or n_stages < 1:
        raise ValueError(f'n_stages must be a positive integer, got {n_stages!r}')
    inplane_divisibility = 2 ** max(0, n_stages - 1)
    inplane_patch = int(
        math.ceil(
            (target_fov_mm / min(spacing[1:]) - 1e-8)
            / inplane_divisibility
        )
        * inplane_divisibility
    )
    floor_da = math.floor(compute_continuous_da(spacing))
    depth_divisibility = 2 ** max(0, n_stages - 1 - floor_da)
    depth_patch = int(
        math.ceil(
            (target_fov_mm / spacing[0] - 1e-8) / depth_divisibility
        )
        * depth_divisibility
    )
    return depth_patch, inplane_patch, inplane_patch


def derive_sample_native_max_bottleneck_patch_size(
    spacing: tuple[float, float, float],
    image_shape: tuple[int, int, int],
    *,
    inplane_patch_size: int,
    n_stages: int,
    floor_bottleneck_min: int,
    floor_bottleneck_max: int,
) -> tuple[int, int, int]:
    """Derive an image-driven patch whose floor bottleneck is bounded."""
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in spacing
    ):
        raise ValueError(
            f'spacing must contain three finite positive values, got {spacing}'
        )
    if len(image_shape) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in image_shape
    ):
        raise ValueError(
            f'image_shape must contain three positive integers, got {image_shape}'
        )
    if (
        isinstance(inplane_patch_size, bool)
        or not isinstance(inplane_patch_size, int)
        or inplane_patch_size < 1
    ):
        raise ValueError(
            f'inplane_patch_size must be a positive integer, got '
            f'{inplane_patch_size!r}'
        )
    if isinstance(n_stages, bool) or not isinstance(n_stages, int) or n_stages < 1:
        raise ValueError(f'n_stages must be a positive integer, got {n_stages!r}')
    if not (
        isinstance(floor_bottleneck_min, int)
        and not isinstance(floor_bottleneck_min, bool)
        and isinstance(floor_bottleneck_max, int)
        and not isinstance(floor_bottleneck_max, bool)
        and 1 <= floor_bottleneck_min <= floor_bottleneck_max
    ):
        raise ValueError(
            f'floor bottleneck bounds must be positive ordered integers, got '
            f'{floor_bottleneck_min!r}, {floor_bottleneck_max!r}'
        )
    floor_da = math.floor(compute_continuous_da(spacing))
    depth_divisibility = 2 ** max(0, n_stages - 1 - floor_da)
    floor_bottleneck = min(
        max(
            math.ceil(image_shape[0] / depth_divisibility),
            floor_bottleneck_min,
        ),
        floor_bottleneck_max,
    )
    return (
        floor_bottleneck * depth_divisibility,
        inplane_patch_size,
        inplane_patch_size,
    )


def _load_sample_native_case_geometries(
    data_folder: Path,
    identifiers: list[str],
    *,
    spacing_property: str,
    expected_inplane_spacing: tuple[float, float],
    minimum_z_spacing_mm: float,
    transpose_forward: tuple[int, int, int],
    target_fov_mm: float,
    n_stages: int,
) -> dict[str, UniversalCaseGeometry]:
    geometries = {}
    for case_id in identifiers:
        properties_path = data_folder / f'{case_id}.pkl'
        if not properties_path.is_file():
            raise FileNotFoundError(properties_path)
        properties = load_pickle(properties_path)
        raw_spacing = properties.get(spacing_property)
        if not isinstance(raw_spacing, (list, tuple)) or len(raw_spacing) != 3:
            raise ValueError(
                f'{properties_path} must contain three-value '
                f'{spacing_property!r}, got {raw_spacing!r}'
            )
        spacing = tuple(float(value) for value in raw_spacing)
        original_spacing = properties.get('spacing')
        if (
            not isinstance(original_spacing, (list, tuple))
            or len(original_spacing) != 3
        ):
            raise ValueError(
                f'{properties_path} must contain three-value original '
                f'spacing, got {original_spacing!r}'
            )
        expected_z_spacing = max(
            float(original_spacing[transpose_forward[0]]),
            minimum_z_spacing_mm,
        )
        if not math.isclose(
            spacing[0],
            expected_z_spacing,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f'{properties_path} Sample-Native-Z spacing must use '
                f'z=max(native, {minimum_z_spacing_mm}), got {spacing[0]} '
                f'instead of {expected_z_spacing}'
            )
        if any(
            not math.isclose(actual, expected)
            for actual, expected in zip(
                spacing[1:],
                expected_inplane_spacing,
                strict=True,
            )
        ):
            raise ValueError(
                f'{properties_path} Sample-Native-Z in-plane spacing must be '
                f'{expected_inplane_spacing}, got {spacing[1:]}'
            )
        patch_size = derive_sample_native_patch_size(
            spacing,
            target_fov_mm=target_fov_mm,
            n_stages=n_stages,
        )
        geometries[case_id] = UniversalCaseGeometry(
            spacing=spacing,
            patch_size=patch_size,
            continuous_da=compute_continuous_da(spacing),
        )
    return geometries


def _load_sample_native_max_bottleneck_case_geometries(
    data_folder: Path,
    identifiers: list[str],
    *,
    spacing_property: str,
    shape_property: str,
    expected_inplane_spacing: tuple[float, float],
    inplane_patch_size: int,
    n_stages: int,
    floor_bottleneck_min: int,
    floor_bottleneck_max: int,
) -> dict[str, UniversalCaseGeometry]:
    geometries = {}
    for case_id in identifiers:
        properties_path = data_folder / f'{case_id}.pkl'
        if not properties_path.is_file():
            raise FileNotFoundError(properties_path)
        properties = load_pickle(properties_path)
        raw_spacing = properties.get(spacing_property)
        if not isinstance(raw_spacing, (list, tuple)) or len(raw_spacing) != 3:
            raise ValueError(
                f'{properties_path} must contain three-value '
                f'{spacing_property!r}, got {raw_spacing!r}'
            )
        spacing = tuple(float(value) for value in raw_spacing)
        if not all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
            for actual, expected in zip(
                spacing[1:],
                expected_inplane_spacing,
                strict=True,
            )
        ):
            raise ValueError(
                f'{properties_path} Sample-Native-Z planned XY spacing '
                f'{spacing[1:]} does not match {expected_inplane_spacing}'
            )
        raw_shape = properties.get(shape_property)
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
            raise ValueError(
                f'{properties_path} must contain three-value '
                f'{shape_property!r}, got {raw_shape!r}'
            )
        image_shape = tuple(int(value) for value in raw_shape)
        patch_size = derive_sample_native_max_bottleneck_patch_size(
            spacing,
            image_shape,
            inplane_patch_size=inplane_patch_size,
            n_stages=n_stages,
            floor_bottleneck_min=floor_bottleneck_min,
            floor_bottleneck_max=floor_bottleneck_max,
        )
        geometries[case_id] = UniversalCaseGeometry(
            spacing=spacing,
            patch_size=patch_size,
            continuous_da=compute_continuous_da(spacing),
        )
    return geometries


@dataclass(frozen=True)
class UniversalDatasetSpec:
    """Frozen dataset-level protocol shared by all Universal conditions.

    Manifest-level suite contract; the output-space channel view is
    `pumit.spad_unet.universal.DatasetRegionMapping`.
    """

    source_task: str
    expected_num_training: int
    regions: tuple[str, ...]
    split_policy: Literal['nnunet_seeded', 'multitalent_task046']


@dataclass(frozen=True)
class UniversalExperimentManifest:
    """Typed view of the frozen multi-dataset Universal experiment manifest."""

    ontology_status: str
    reference: str
    fold: int
    num_updates: int
    datasets: dict[str, UniversalDatasetSpec]
    shared_regions: dict[str, dict[str, str]]
    universal_dataset_id: int = UNIVERSAL_NNUNET_DATASET_ID
    universal_dataset_name: str = UNIVERSAL_NNUNET_DATASET_NAME

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError('the experiment must contain at least one dataset')
        if self.fold != 0:
            raise ValueError('the publication protocol requires fold 0 only')
        if self.num_updates < 1:
            raise ValueError('num_updates must be positive')
        expected_prefix = f'Dataset{self.universal_dataset_id:03d}_'
        if not self.universal_dataset_name.startswith(expected_prefix):
            raise ValueError(
                f'universal dataset name must start with {expected_prefix!r}, '
                f'got {self.universal_dataset_name!r}'
            )
        if self.ontology_status == 'frozen' and self.shared_regions:
            raise ValueError('the frozen MultiTalent label bank must not merge regions across datasets')
        task046_specs = [
            spec for spec in self.datasets.values()
            if spec.source_task == 'Task046_AbdOrgSegm2'
        ]
        special_splits = [
            spec for spec in self.datasets.values()
            if spec.split_policy == 'multitalent_task046'
        ]
        if len(task046_specs) > 1:
            raise ValueError('the experiment must not contain Task046 more than once')
        if special_splits != task046_specs:
            raise ValueError('Task046 alone must use the MultiTalent special split')

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(self.datasets)

    @classmethod
    def load(cls, path: Path) -> 'UniversalExperimentManifest':
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'UniversalExperimentManifest':
        manifest = cls(
            ontology_status=raw['ontology_status'],
            reference=raw['reference'],
            fold=raw['fold'],
            num_updates=raw['num_updates'],
            datasets={
                dataset_id: UniversalDatasetSpec(
                    source_task=spec['source_task'],
                    expected_num_training=spec['expected_num_training'],
                    regions=tuple(spec['regions']),
                    split_policy=spec.get('split_policy', 'nnunet_seeded'),
                )
                for dataset_id, spec in raw['datasets'].items()
            },
            shared_regions=raw['shared_regions'],
            universal_dataset_id=raw.get(
                'universal_dataset_id',
                UNIVERSAL_NNUNET_DATASET_ID,
            ),
            universal_dataset_name=raw.get(
                'universal_dataset_name',
                UNIVERSAL_NNUNET_DATASET_NAME,
            ),
        )
        return manifest


def resolve_dataset_folder(preprocessed_root: Path, dataset_id: str) -> Path:
    matches = sorted(preprocessed_root.glob(f'Dataset{dataset_id}_*'))
    if len(matches) != 1:
        raise RuntimeError(
            f'expected exactly one preprocessed folder for dataset {dataset_id}, got {matches}'
        )
    return matches[0]


def _load_splits(path: Path) -> list[dict[str, list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    splits = json.loads(path.read_text())
    if len(splits) != 5:
        raise ValueError(f'{path} must define nnU-Net v2 five-fold splits, got {len(splits)}')
    return splits


def _expected_task046_splits(
    manifest: UniversalExperimentManifest,
    identifiers: list[str],
    preprocessed_root: Path,
    plans_name: str,
    configuration_name: str,
) -> list[dict[str, list[str]]]:
    task017_ids = [
        dataset_id
        for dataset_id, spec in manifest.datasets.items()
        if spec.source_task == 'Task017_AbdominalOrganSegmentation'
    ]
    if len(task017_ids) != 1:
        raise ValueError(f'expected exactly one Task017 source, got {task017_ids}')
    task017_base = resolve_dataset_folder(preprocessed_root, task017_ids[0])
    task017_splits = _load_splits(task017_base / 'splits_final.json')
    task017_identifiers = set(task017_splits[0]['train']) | set(task017_splits[0]['val'])
    identifier_set = set(identifiers)
    if not task017_identifiers <= identifier_set:
        raise ValueError(
            'Task046 must contain every Task017 training case with the same identifier; '
            f'missing={sorted(task017_identifiers - identifier_set)}'
        )
    pancreas_identifiers = sorted(identifier for identifier in identifiers if identifier.startswith('PAN'))
    if identifier_set != task017_identifiers | set(pancreas_identifiers):
        raise ValueError(
            'Task046 must contain only Task017 training cases and PANCREAS-CT cases; '
            f'unknown={sorted(identifier_set - task017_identifiers - set(pancreas_identifiers))}'
        )

    task062_ids = [
        dataset_id
        for dataset_id, spec in manifest.datasets.items()
        if spec.source_task == 'Task062_NIHPancreas'
    ]
    if len(task062_ids) != 1:
        raise ValueError(f'expected exactly one Task062 source, got {task062_ids}')
    task062_base = resolve_dataset_folder(preprocessed_root, task062_ids[0])
    task062_plans = json.loads((task062_base / f'{plans_name}.json').read_text())
    task062_data_identifier = task062_plans['configurations'][configuration_name]['data_identifier']
    task062_data_folder = task062_base / task062_data_identifier
    task062_identifiers = sorted(
        infer_dataset_class(str(task062_data_folder)).get_identifiers(str(task062_data_folder))
    )
    task062_splits = generate_crossval_split(task062_identifiers)
    task046_to_task062 = {
        identifier: f'Pancreas-CT_{identifier.removeprefix("PANCREAS_")}'
        for identifier in pancreas_identifiers
    }
    unknown_pancreas = set(task046_to_task062.values()) - set(task062_identifiers)
    if unknown_pancreas:
        raise ValueError(
            'Task046 Pancreas-CT cases must be a subset of Task062; '
            f'unknown={sorted(unknown_pancreas)}'
        )

    splits = []
    for task017_split, task062_split in zip(task017_splits, task062_splits, strict=True):
        task062_train = set(task062_split['train'])
        task062_val = set(task062_split['val'])
        pancreas_train = [
            identifier
            for identifier in pancreas_identifiers
            if task046_to_task062[identifier] in task062_train
        ]
        pancreas_val = [
            identifier
            for identifier in pancreas_identifiers
            if task046_to_task062[identifier] in task062_val
        ]
        splits.append({
            'train': [*task017_split['train'], *pancreas_train],
            'val': [*task017_split['val'], *pancreas_val],
        })
    return splits


def expected_universal_splits(
    manifest: UniversalExperimentManifest,
    spec: UniversalDatasetSpec,
    identifiers: list[str],
    preprocessed_root: Path,
    *,
    plans_name: str,
    configuration_name: str,
) -> list[dict[str, list[str]]]:
    if spec.split_policy == 'nnunet_seeded':
        return generate_crossval_split(sorted(identifiers))
    if spec.split_policy == 'multitalent_task046':
        return _expected_task046_splits(
            manifest,
            identifiers,
            preprocessed_root,
            plans_name,
            configuration_name,
        )
    raise ValueError(f'unknown split policy {spec.split_policy!r}')


def materialize_universal_splits(
    manifest: UniversalExperimentManifest,
    preprocessed_root: Path,
    dataset_ids: tuple[str, ...],
    plans_name: str,
    configuration_name: str,
) -> dict[str, tuple[bool, int, int]]:
    """Write each dataset's splits_final.json unless an identical file already exists.

    Returns:
        Per dataset ID, (created, fold-0 train count, fold-0 validation count).
    """
    outcomes = {}
    for dataset_id in dataset_ids:
        spec = manifest.datasets[dataset_id]
        base = resolve_dataset_folder(preprocessed_root, dataset_id)
        plans = PlansManager(base / f'{plans_name}.json')
        configuration = plans.get_configuration(configuration_name)
        data_folder = base / configuration.data_identifier
        identifiers = sorted(
            infer_dataset_class(str(data_folder)).get_identifiers(str(data_folder))
        )
        splits = expected_universal_splits(
            manifest,
            spec,
            identifiers,
            preprocessed_root,
            plans_name=plans_name,
            configuration_name=configuration_name,
        )
        splits_path = base / 'splits_final.json'
        created = not splits_path.exists()
        if not created:
            if json.loads(splits_path.read_text()) != splits:
                raise ValueError(
                    f'{splits_path} conflicts with the frozen split policy'
                )
        else:
            splits_path.write_text(json.dumps(splits, indent=4) + '\n')
        outcomes[dataset_id] = (created, len(splits[0]['train']), len(splits[0]['val']))
    return outcomes


def _validate_splits(
    manifest: UniversalExperimentManifest,
    spec: UniversalDatasetSpec,
    identifiers: list[str],
    splits: list[dict[str, list[str]]],
    preprocessed_root: Path,
    plans_name: str,
    configuration_name: str,
) -> None:
    expected = expected_universal_splits(
        manifest,
        spec,
        identifiers,
        preprocessed_root,
        plans_name=plans_name,
        configuration_name=configuration_name,
    )
    if splits != expected:
        raise ValueError(
            f'{spec.source_task} splits do not match the frozen {spec.split_policy} policy'
        )


def load_universal_datasets(
    manifest: UniversalExperimentManifest,
    preprocessed_root: Path,
    selected_ids: tuple[str, ...] | None = None,
    *,
    plans_name: str,
    configuration_name: str,
) -> dict[str, UniversalDataset]:
    """Resolve the Universal dataset suite against explicit nnU-Net plans."""
    fold = manifest.fold
    dataset_ids = manifest.dataset_ids if selected_ids is None else selected_ids
    unknown = set(dataset_ids) - set(manifest.dataset_ids)
    if unknown:
        raise ValueError(f'dataset IDs are absent from the experiment config: {sorted(unknown)}')

    datasets = {}
    for dataset_id in dataset_ids:
        spec = manifest.datasets[dataset_id]
        base = resolve_dataset_folder(preprocessed_root, dataset_id)
        dataset_json = json.loads((base / 'dataset.json').read_text())
        plans_path = base / f'{plans_name}.json'
        plans = json.loads(plans_path.read_text())
        configuration = PlansManager(plans).get_configuration(configuration_name)

        data_folder = base / configuration.data_identifier
        if not data_folder.is_dir():
            raise FileNotFoundError(data_folder)
        identifiers = infer_dataset_class(str(data_folder)).get_identifiers(str(data_folder))
        if len(identifiers) != dataset_json['numTraining']:
            raise ValueError(
                f'dataset {dataset_id} has {len(identifiers)} preprocessed cases '
                f'but dataset.json declares {dataset_json["numTraining"]}'
            )
        if len(identifiers) != spec.expected_num_training:
            raise ValueError(
                f'dataset {dataset_id} has {len(identifiers)} preprocessed cases, '
                f'but the frozen protocol requires {spec.expected_num_training}'
            )
        splits = _load_splits(base / 'splits_final.json')
        _validate_splits(
            manifest,
            spec,
            identifiers,
            splits,
            preprocessed_root,
            plans_name,
            configuration_name,
        )
        training_identifiers = tuple(splits[fold]['train'])
        validation_identifiers = tuple(splits[fold]['val'])
        identifier_set = set(identifiers)
        training_set = set(training_identifiers)
        validation_set = set(validation_identifiers)
        if training_set & validation_set:
            raise ValueError(f'dataset {dataset_id} fold {fold} has overlapping train and validation cases')
        if training_set | validation_set != identifier_set:
            missing = identifier_set - training_set - validation_set
            unknown = (training_set | validation_set) - identifier_set
            raise ValueError(
                f'dataset {dataset_id} fold {fold} does not partition the preprocessed cases; '
                f'missing={sorted(missing)}, unknown={sorted(unknown)}'
            )
        available_names = foreground_region_names(dataset_json)
        if len(set(spec.regions)) != len(spec.regions):
            raise ValueError(f'dataset {dataset_id} config contains duplicate region names')
        unknown_regions = set(spec.regions) - set(available_names)
        if unknown_regions:
            raise ValueError(
                f'dataset {dataset_id} config references unknown regions: {sorted(unknown_regions)}'
            )
        region_value_by_name = dict(zip(
            available_names,
            foreground_region_values(dataset_json),
            strict=True,
        ))
        adaptation = plans.get('pumit_spad_unet_source_adaptation')
        case_geometries = None
        if isinstance(adaptation, dict) and adaptation.get('sample_native_z') is True:
            spacing_property = adaptation.get('runtime_spacing_property')
            if not isinstance(spacing_property, str) or not spacing_property:
                raise ValueError(
                    f'dataset {dataset_id} Sample-Native-Z source plan must '
                    f'declare runtime_spacing_property'
                )
            architecture_kwargs = plans['configurations'][configuration_name][
                'architecture'
            ]['arch_kwargs']
            n_stages = architecture_kwargs.get('n_stages')
            if isinstance(n_stages, bool) or not isinstance(n_stages, int):
                raise ValueError(
                    f'dataset {dataset_id} Sample-Native-Z architecture must '
                    f'declare integer n_stages, got {n_stages!r}'
                )
            runtime_patch_policy = adaptation.get(
                'runtime_patch_policy',
                'minimum_physical_fov',
            )
            if runtime_patch_policy == 'minimum_physical_fov':
                target_fov_mm = adaptation.get('target_fov_mm')
                if isinstance(target_fov_mm, bool) or not isinstance(
                    target_fov_mm,
                    (int, float),
                ):
                    raise ValueError(
                        f'dataset {dataset_id} Sample-Native-Z source plan '
                        f'must declare numeric target_fov_mm'
                    )
                minimum_z_spacing_mm = adaptation.get(
                    'minimum_z_spacing_mm'
                )
                if (
                    isinstance(minimum_z_spacing_mm, bool)
                    or not isinstance(minimum_z_spacing_mm, (int, float))
                    or not math.isfinite(float(minimum_z_spacing_mm))
                    or minimum_z_spacing_mm <= 0
                ):
                    raise ValueError(
                        f'dataset {dataset_id} Sample-Native-Z source plan '
                        f'must declare positive numeric minimum_z_spacing_mm'
                    )
                case_geometries = _load_sample_native_case_geometries(
                    data_folder,
                    identifiers,
                    spacing_property=spacing_property,
                    expected_inplane_spacing=tuple(
                        float(value) for value in configuration.spacing[1:]
                    ),
                    minimum_z_spacing_mm=float(minimum_z_spacing_mm),
                    transpose_forward=tuple(plans['transpose_forward']),
                    target_fov_mm=float(target_fov_mm),
                    n_stages=n_stages,
                )
            elif runtime_patch_policy == 'image_depth_floor_bottleneck_cap':
                shape_property = adaptation.get('runtime_shape_property')
                inplane_patch_size = adaptation.get('inplane_patch_size')
                floor_bottleneck_min = adaptation.get('floor_bottleneck_min')
                floor_bottleneck_max = adaptation.get('floor_bottleneck_max')
                if not isinstance(shape_property, str) or not shape_property:
                    raise ValueError(
                        f'dataset {dataset_id} planned-XY source plan must '
                        f'declare runtime_shape_property'
                    )
                if (
                    not isinstance(inplane_patch_size, list)
                    or len(inplane_patch_size) != 2
                    or inplane_patch_size[0] != inplane_patch_size[1]
                ):
                    raise ValueError(
                        f'dataset {dataset_id} planned-XY source plan must '
                        f'declare equal two-value inplane_patch_size'
                    )
                case_geometries = (
                    _load_sample_native_max_bottleneck_case_geometries(
                        data_folder,
                        identifiers,
                        spacing_property=spacing_property,
                        shape_property=shape_property,
                        expected_inplane_spacing=tuple(
                            float(value) for value in configuration.spacing[1:]
                        ),
                        inplane_patch_size=int(inplane_patch_size[0]),
                        n_stages=n_stages,
                        floor_bottleneck_min=floor_bottleneck_min,
                        floor_bottleneck_max=floor_bottleneck_max,
                    )
                )
            else:
                raise ValueError(
                    f'dataset {dataset_id} has unsupported Sample-Native-Z '
                    f'runtime patch policy {runtime_patch_policy!r}'
                )
        datasets[dataset_id] = UniversalDataset(
            dataset_id=dataset_id,
            name=base.name,
            dataset_json=dataset_json,
            plans=plans,
            data_folder=data_folder,
            num_cases=len(identifiers),
            training_identifiers=training_identifiers,
            validation_identifiers=validation_identifiers,
            spacing=tuple(configuration.spacing),
            patch_size=tuple(configuration.patch_size),
            batch_size=configuration.batch_size,
            batch_dice=configuration.batch_dice,
            region_names=spec.regions,
            region_values=tuple(region_value_by_name[name] for name in spec.regions),
            case_geometries=case_geometries,
        )
    return datasets


def prepare_universal_nnunet_namespace(
    preprocessed_root: Path,
    source_data_folder: Path,
    canonical_names: tuple[str, ...],
    source_dataset_ids: tuple[str, ...],
    *,
    dataset_name: str = UNIVERSAL_NNUNET_DATASET_NAME,
) -> Path:
    """Create the virtual nnU-Net dataset that owns Universal plans and results."""
    source_data_folder = source_data_folder.resolve(strict=True)
    namespace = preprocessed_root / dataset_name
    namespace.mkdir(parents=False, exist_ok=True)

    dataset_json = {
        'channel_names': {'0': 'CT'},
        'labels': {
            'background': 0,
            **{
                name: [index]
                for index, name in enumerate(canonical_names, start=1)
            },
        },
        'regions_class_order': list(range(1, len(canonical_names) + 1)),
        'numTraining': 0,
        'file_ending': '.nii.gz',
        'pumit_spad_unet': {
            'virtual_dataset': True,
            'source_dataset_ids': list(source_dataset_ids),
        },
    }
    fingerprint = {
        'pumit_spad_unet': {
            'virtual_dataset': True,
            'source_dataset_ids': list(source_dataset_ids),
        },
    }
    for path, expected in (
        (namespace / 'dataset.json', dataset_json),
        (namespace / 'dataset_fingerprint.json', fingerprint),
    ):
        if path.exists():
            actual = json.loads(path.read_text())
            if actual != expected:
                raise ValueError(f'{path} does not match the Universal namespace contract')
        else:
            path.write_text(json.dumps(expected, indent=2) + '\n')

    data_link = namespace / source_data_folder.name
    if data_link.is_symlink():
        if data_link.resolve(strict=True) != source_data_folder:
            raise ValueError(
                f'{data_link} points to {data_link.resolve()}, expected {source_data_folder}'
            )
    elif data_link.exists():
        raise FileExistsError(
            f'{data_link} must be a symlink to the carrier data folder'
        )
    else:
        relative_target = Path(
            os.path.relpath(source_data_folder, namespace.resolve(strict=True))
        )
        data_link.symlink_to(relative_target, target_is_directory=True)
    return namespace


def get_preprocessed_root() -> Path:
    value = os.environ.get('nnUNet_preprocessed')
    if value is None:
        raise RuntimeError('nnUNet_preprocessed is not set; run through the nnunet pixi environment')
    return Path(value)
