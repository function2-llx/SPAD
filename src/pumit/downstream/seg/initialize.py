"""Create nnU-Net initialization weights from an existing segmentation plan."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .registry import BACKBONES

INITIALIZATION_SEED = 233


def _build_initialization_checkpoint(
    plans: Mapping[str, object],
    dataset_json: Mapping[str, object],
    configuration: str,
    backbone_name: str,
    weights: Path | None,
) -> dict[str, dict[str, torch.Tensor]]:
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INITIALIZATION_SEED)
        network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            determine_num_input_channels(
                plans_manager,
                configuration_manager,
                dataset_json,
            ),
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=True,
        )
        network.load_pretrained(weights)

    return {
        'network_weights': {
            key: value
            for key, value in network.state_dict().items()
            if '.seg_layers.' not in key
        }
    }


def _save_initialization_checkpoint(
    checkpoint: Mapping[str, object],
    initialization_path: Path,
) -> None:
    if initialization_path.exists():
        raise FileExistsError(f'refusing to overwrite initialization: {initialization_path}')
    staging_path = initialization_path.with_name(
        f'.{initialization_path.name}.{os.getpid()}.tmp'
    )
    staging_path.unlink(missing_ok=True)
    try:
        torch.save(checkpoint, staging_path)
        os.link(staging_path, initialization_path)
    finally:
        staging_path.unlink(missing_ok=True)


def initialize_from_plan(
    dataset_name_or_id: str,
    configuration: str,
    plans_identifier: str,
    *,
    weights: Path | None,
) -> Path:
    """Create nnU-Net initialization weights from one existing manual plan."""
    if '__' in plans_identifier:
        raise ValueError('plans_identifier cannot contain nnU-Net reserved separator "__"')

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    dataset_folder = Path(nnUNet_preprocessed) / dataset_name
    plans_path = dataset_folder / f'{plans_identifier}.json'
    plans = load_json(plans_path)
    dataset_json = load_json(dataset_folder / 'dataset.json')
    if plans.get('plans_name') != plans_identifier:
        raise ValueError(
            f'plan identifier mismatch: requested {plans_identifier!r}, '
            f'file declares {plans.get("plans_name")!r}'
        )

    configurations = plans.get('configurations')
    if not isinstance(configurations, Mapping) or configuration not in configurations:
        raise KeyError(f'plan {plans_identifier!r} has no configuration {configuration!r}')
    configuration_dict = configurations[configuration]
    if not isinstance(configuration_dict, Mapping):
        raise TypeError(f'configuration {configuration!r} must be a mapping')
    architecture = configuration_dict.get('architecture')
    if not isinstance(architecture, Mapping):
        raise TypeError(f'configuration {configuration!r} architecture must be a mapping')
    architecture_kwargs = architecture.get('arch_kwargs')
    if not isinstance(architecture_kwargs, Mapping):
        raise TypeError(f'configuration {configuration!r} architecture kwargs must be a mapping')
    backbone_name = architecture_kwargs.get('backbone_name')
    backbone_config = architecture_kwargs.get('backbone_config')
    if not isinstance(backbone_name, str) or backbone_name not in BACKBONES:
        raise ValueError(f'unsupported plan backbone: {backbone_name!r}')
    if not isinstance(backbone_config, Mapping):
        raise TypeError('plan backbone_config must be a mapping')

    initialization_path = (
        dataset_folder / f'{plans_identifier}_{configuration}_initialization.pth'
    )
    if initialization_path.exists():
        raise FileExistsError(f'refusing to overwrite initialization: {initialization_path}')
    checkpoint = _build_initialization_checkpoint(
        plans,
        dataset_json,
        configuration,
        backbone_name,
        weights,
    )
    _save_initialization_checkpoint(checkpoint, initialization_path)
    return initialization_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id')
    parser.add_argument('configuration')
    parser.add_argument('plans_identifier')
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()

    initialization_path = initialize_from_plan(
        args.dataset_name_or_id,
        args.configuration,
        args.plans_identifier,
        weights=args.weights,
    )
    print(f'prepared initialization: {initialization_path}')


if __name__ == '__main__':
    main()
