"""Prepare a plans-defined SPAD U-Net experiment for the native nnU-Net CLI."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from .architecture import build_spad_architecture
from .geometry import compute_continuous_da, decompose_continuous_da

SPAD_PLANS_IDENTIFIER = 'SPADUNetPlans'


def prepare_spad_plans(
    dataset_name_or_id: str,
    configuration: str,
    *,
    source_plans_identifier: str,
    derived_plans_identifier: str = SPAD_PLANS_IDENTIFIER,
    min_bottleneck: int = 4,
) -> Path:
    """Write the derived plan consumed by ``nnUNetv2_train``."""
    for name, value in (
        ('source_plans_identifier', source_plans_identifier),
        ('derived_plans_identifier', derived_plans_identifier),
    ):
        if '__' in value:
            raise ValueError(f'{name} cannot contain nnU-Net reserved separator "__"')

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    dataset_folder = Path(nnUNet_preprocessed) / dataset_name
    plans = load_json(dataset_folder / f'{source_plans_identifier}.json')
    plan_path = dataset_folder / f'{derived_plans_identifier}.json'
    if plan_path.exists():
        raise FileExistsError(plan_path)

    configuration_dict = plans['configurations'][configuration]
    spacing = tuple(float(value) for value in configuration_dict['spacing'])
    inference_da, _, _ = decompose_continuous_da(compute_continuous_da(spacing))
    configuration_dict['architecture'] = build_spad_architecture(
        configuration_dict['architecture'],
        inference_da=inference_da,
        min_bottleneck=min_bottleneck,
    )
    plans['plans_name'] = derived_plans_identifier
    plans['pumit_spad_unet'] = {
        'source_plans_identifier': source_plans_identifier,
        'configuration': configuration,
        'inference_da': inference_da,
        'min_bottleneck': min_bottleneck,
    }
    save_json(plans, plan_path, sort_keys=False)
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id')
    parser.add_argument('configuration')
    parser.add_argument('--source-plans', default='nnUNetResEncUNetLPlans')
    parser.add_argument('--derived-plans', default=SPAD_PLANS_IDENTIFIER)
    parser.add_argument('--min-bottleneck', type=int, default=4)
    args = parser.parse_args()

    plan_path = prepare_spad_plans(
        args.dataset_name_or_id,
        args.configuration,
        source_plans_identifier=args.source_plans,
        derived_plans_identifier=args.derived_plans,
        min_bottleneck=args.min_bottleneck,
    )
    command = [
        'nnUNetv2_train',
        args.dataset_name_or_id,
        args.configuration,
        '<FOLD>',
        '-tr',
        'SPADUNetTrainer',
        '-p',
        args.derived_plans,
    ]
    print(f'prepared plans: {plan_path}')
    print(f'run: {shlex.join(command)}')


if __name__ == '__main__':
    main()
