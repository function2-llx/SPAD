"""Prepare an nnU-Net plan for one dense encoder baseline."""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .adapters.fixed_patch_vit import validate_fixed_patch_size
from .adapters.pyramid import PYRAMID_BRANCHES
from .registry import BACKBONES

NETWORK_CLASS_NAMES = {
    'suprem': 'pumit.downstream.seg.suprem.SupremSegmentationNetwork',
    'unet': 'pumit.downstream.seg.network.PlanAlignedSegmentationNetwork',
    'unet-no-refiner': (
        'pumit.downstream.seg.network.'
        'PlanAlignedUNetWithoutRefinerSegmentationNetwork'
    ),
    'mask2former': (
        'pumit.downstream.seg.mask2former.PlanAlignedMask2FormerSegmentationNetwork'
    ),
}
_ARCHITECTURE_KEYS = (
    'features_per_stage',
    'kernel_sizes',
    'strides',
    'n_blocks_per_stage',
    'n_conv_per_stage_decoder',
    'conv_op',
    'conv_bias',
    'norm_op',
    'norm_op_kwargs',
    'dropout_op',
    'dropout_op_kwargs',
    'nonlin',
    'nonlin_kwargs',
)


def _configure_vit_adapter(
    backbone_name: str,
    backbone_config: Mapping[str, object],
    adapter_dim: int | None,
    deform_attention_dim: int | None,
    deform_num_heads: int | None,
    conv_ffn_hidden_dim: int | None,
    spatial_prior_channels: Sequence[int] | None,
    output_fusion: str | None,
) -> dict[str, object]:
    """Apply explicit ViT-Adapter architecture overrides to one generated plan."""
    config = dict(backbone_config)
    overrides = {
        key: value
        for key, value in (
            ('adapter_dim', adapter_dim),
            ('deform_attention_dim', deform_attention_dim),
            ('deform_num_heads', deform_num_heads),
            ('conv_ffn_hidden_dim', conv_ffn_hidden_dim),
            ('output_fusion', output_fusion),
        )
        if value is not None
    }
    if spatial_prior_channels is not None:
        channels = tuple(int(value) for value in spatial_prior_channels)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError(
                f'spatial_prior_channels must contain four positive values, got '
                f'{channels}'
            )
        overrides['spatial_prior_channels'] = list(channels)
    if not overrides:
        return config
    if not BACKBONES[backbone_name].is_vit_adapter:
        raise ValueError(f'{backbone_name} has no ViT-Adapter architecture to override')
    for key in (
        'adapter_dim',
        'deform_attention_dim',
        'deform_num_heads',
        'conv_ffn_hidden_dim',
    ):
        value = overrides.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f'{key} must be positive, got {value}')
    config.update(overrides)
    return config


def _configure_features_per_stage(
    source_architecture: Mapping[str, object],
    features_per_stage: Sequence[int] | None,
) -> dict[str, object]:
    architecture = dict(source_architecture)
    source_kwargs = architecture.get('arch_kwargs')
    if not isinstance(source_kwargs, Mapping):
        raise TypeError('source architecture arch_kwargs must be a mapping')
    if features_per_stage is None:
        return architecture

    source_channels = source_kwargs.get('features_per_stage')
    if not isinstance(source_channels, Sequence):
        raise TypeError('source architecture features_per_stage must be a sequence')
    channels = tuple(int(value) for value in features_per_stage)
    if len(channels) != len(source_channels) or any(value <= 0 for value in channels):
        raise ValueError(
            f'features_per_stage must contain {len(source_channels)} positive values, '
            f'got {channels}'
        )
    architecture['arch_kwargs'] = {
        **source_kwargs,
        'features_per_stage': list(channels),
    }
    return architecture


def _truncate_architecture_stages(
    source_architecture: Mapping[str, object],
    num_stages: int | None,
) -> dict[str, object]:
    """Keep only the shallowest stages of the source plan for a native-depth encoder."""
    architecture = dict(source_architecture)
    if num_stages is None:
        return architecture
    source_kwargs = architecture.get('arch_kwargs')
    if not isinstance(source_kwargs, Mapping):
        raise TypeError('source architecture arch_kwargs must be a mapping')
    stages = len(source_kwargs['features_per_stage'])
    if not 2 <= int(num_stages) <= stages:
        raise ValueError(f'num_stages must be in [2, {stages}], got {num_stages}')
    architecture['arch_kwargs'] = {
        **source_kwargs,
        **({'n_stages': int(num_stages)} if 'n_stages' in source_kwargs else {}),
        **{
            key: list(source_kwargs[key][: int(num_stages)])
            for key in ('features_per_stage', 'kernel_sizes', 'strides', 'n_blocks_per_stage')
        },
        'n_conv_per_stage_decoder': list(
            source_kwargs['n_conv_per_stage_decoder'][: int(num_stages) - 1]
        ),
    }
    return architecture


def _configure_vit_patch_size(
    backbone_name: str,
    backbone_config: Mapping[str, object],
    vit_patch_size: Sequence[int] | None,
    input_patch_size: object,
) -> dict[str, object]:
    if not BACKBONES[backbone_name].requires_vit_patch_size:
        if vit_patch_size is not None:
            raise ValueError(f'{backbone_name} does not accept vit_patch_size')
        return dict(backbone_config)
    if vit_patch_size is None:
        raise ValueError(f'{backbone_name} requires an explicit vit_patch_size')
    fixed_patch_size = validate_fixed_patch_size(
        vit_patch_size,
        in_plane_patch_size=16,
    )
    if not isinstance(input_patch_size, Sequence) or len(input_patch_size) != 3:
        raise ValueError(
            f'nnU-Net configuration patch_size must contain three values, got '
            f'{input_patch_size!r}'
        )
    input_patch_size = tuple(int(value) for value in input_patch_size)
    if any(
        size % step
        for size, step in zip(input_patch_size, fixed_patch_size, strict=True)
    ):
        raise ValueError(
            f'nnU-Net input patch size {input_patch_size} must be divisible by '
            f'fixed ViT patch size {fixed_patch_size}'
        )
    return {
        **backbone_config,
        'vit_patch_size': list(fixed_patch_size),
    }


def _configure_vit_adapter_paths(
    backbone_name: str,
    backbone_config: Mapping[str, object],
    *,
    readout: str,
    spatial_prior_input: str | None,
) -> dict[str, object]:
    if spatial_prior_input not in {None, 'raw', 'p1'}:
        raise ValueError(f'unsupported spatial-prior input: {spatial_prior_input!r}')
    backbone = BACKBONES[backbone_name]
    with_high_resolution_stem = readout == 'unet'
    if spatial_prior_input is None:
        spatial_prior_input = 'p1' if backbone.is_vit_adapter and with_high_resolution_stem else 'raw'
    if not backbone.is_vit_adapter:
        if spatial_prior_input != 'raw':
            raise ValueError(f'{backbone_name} has no ViT-Adapter spatial prior')
        if backbone.can_drop_stem and not with_high_resolution_stem:
            return {**backbone_config, 'with_high_resolution_stem': False}
        return dict(backbone_config)

    if spatial_prior_input == 'p1' and not with_high_resolution_stem:
        raise ValueError(
            f'P1 spatial-prior input requires the U-Net high-resolution refiner, '
            f'got readout {readout!r}'
        )
    return {
        **backbone_config,
        'spatial_prior_input': spatial_prior_input,
        'with_high_resolution_stem': with_high_resolution_stem,
    }


def build_native_architecture(
    source_architecture: Mapping[str, object],
    backbone_name: str,
    backbone_config: Mapping[str, object],
    readout: str = 'unet',
    readout_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Replace nnU-Net's full-network specification with the downstream architecture."""
    source_kwargs = source_architecture.get('arch_kwargs')
    if not isinstance(source_kwargs, Mapping):
        raise TypeError('source architecture arch_kwargs must be a mapping')
    architecture_keys = set(_ARCHITECTURE_KEYS)
    missing = architecture_keys - source_kwargs.keys()
    if missing:
        raise KeyError(f'source architecture is missing required kwargs: {sorted(missing)}')

    source_import_keys = source_architecture.get('_kw_requires_import')
    if not isinstance(source_import_keys, list):
        raise TypeError('source architecture _kw_requires_import must be a list')
    import_keys = [key for key in source_import_keys if key in architecture_keys]
    unresolved_imports = {
        key
        for key in _ARCHITECTURE_KEYS
        if isinstance(source_kwargs[key], str) and key not in import_keys
    }
    if unresolved_imports:
        raise ValueError(
            f'source architecture leaves importable kwargs unresolved: {sorted(unresolved_imports)}'
        )

    if readout not in NETWORK_CLASS_NAMES:
        raise ValueError(f'unsupported readout: {readout!r}')
    if readout == 'suprem' and backbone_name != 'suprem-unet':
        raise ValueError('suprem readout requires the suprem-unet backbone')
    backbone = BACKBONES[backbone_name]
    if readout == 'mask2former' and not backbone.is_vit_adapter:
        raise ValueError(f'{readout} requires a ViT-Adapter backbone')
    if readout == 'unet-no-refiner' and not backbone.can_drop_stem:
        raise ValueError(f'{readout} requires a backbone that can drop its high-resolution stem')
    if readout_kwargs and readout != 'mask2former':
        raise ValueError(f'{readout} does not accept readout-specific architecture options')
    return {
        'network_class_name': NETWORK_CLASS_NAMES[readout],
        'arch_kwargs': {
            **{key: source_kwargs[key] for key in _ARCHITECTURE_KEYS},
            'backbone_name': backbone_name,
            'backbone_config': dict(backbone_config),
            **dict(readout_kwargs or {}),
        },
        '_kw_requires_import': import_keys,
    }


def prepare_experiment(
    dataset_name_or_id: str,
    configuration: str,
    *,
    source_plans_identifier: str,
    backbone_name: str,
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
    optimization: Mapping[str, object],
    oversample_foreground_percent: float | None = None,
    batch_dice: bool | None = None,
    num_epochs: int | None = None,
    deep_supervision: bool | None = None,
    pyramid_branch: str | None = None,
    drop_path_rate: float | None = None,
    output_plans_identifier: str | None = None,
    readout: str = 'unet',
    vit_patch_size: Sequence[int] | None = None,
    adapter_dim: int | None = None,
    deform_attention_dim: int | None = None,
    deform_num_heads: int | None = None,
    conv_ffn_hidden_dim: int | None = None,
    spatial_prior_channels: Sequence[int] | None = None,
    output_fusion: str | None = None,
    features_per_stage: Sequence[int] | None = None,
    num_stages: int | None = None,
    spatial_prior_input: str | None = None,
    mask2former_feature_schedule: str = 'concat',
) -> str:
    """Create a downstream plan."""
    if (
        oversample_foreground_percent is not None
        and (
            isinstance(oversample_foreground_percent, bool)
            or not isinstance(oversample_foreground_percent, int | float)
            or not 0 <= oversample_foreground_percent <= 1
        )
    ):
        raise ValueError(
            'oversample_foreground_percent must be a number in [0, 1], got '
            f'{oversample_foreground_percent!r}'
        )
    if '__' in source_plans_identifier:
        raise ValueError('source_plans_identifier cannot contain nnU-Net reserved separator "__"')
    if mask2former_feature_schedule not in {'concat', 'cycle'}:
        raise ValueError(
            f'unsupported Mask2Former feature schedule: '
            f'{mask2former_feature_schedule!r}'
        )
    if readout != 'mask2former' and mask2former_feature_schedule != 'concat':
        raise ValueError(
            f'{readout} does not use a Mask2Former feature schedule'
        )

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    dataset_folder = Path(nnUNet_preprocessed) / dataset_name
    plans = load_json(dataset_folder / f'{source_plans_identifier}.json')
    dataset_json = load_json(dataset_folder / 'dataset.json')

    derived_plans_identifier = (
        output_plans_identifier
        if output_plans_identifier is not None
        else (
            f'{source_plans_identifier}_{backbone_name}'
            if readout == 'unet'
            else f'{source_plans_identifier}_{backbone_name}_{readout}'
        )
    )
    if '__' in derived_plans_identifier:
        raise ValueError('output_plans_identifier cannot contain nnU-Net reserved separator "__"')
    plans_path = dataset_folder / f'{derived_plans_identifier}.json'
    initialization_path = (
        dataset_folder / f'{derived_plans_identifier}_{configuration}_initialization.pth'
    )
    existing_paths = [path for path in (plans_path, initialization_path) if path.exists()]
    if existing_paths:
        raise FileExistsError(
            f'refusing to overwrite existing preparation artifacts: '
            f'{[str(path) for path in existing_paths]}'
        )

    backbone = BACKBONES[backbone_name]
    backbone_config = backbone.prepare_config(
        weights,
        checkpoint_format,
        gradient_checkpointing,
    )
    if pyramid_branch is not None:
        if 'pyramid_branch' not in backbone.optional_config_keys:
            raise ValueError(f'{backbone_name} has no pyramid branch to select')
        backbone_config = {**backbone_config, 'pyramid_branch': pyramid_branch}
    elif readout == 'unet-no-refiner' and 'pyramid_branch' in backbone.optional_config_keys:
        # The trunk-only readout is defined by its branch; a plan must record which one it uses.
        raise ValueError(f'{readout} requires --pyramid-branch for {backbone_name}')
    if drop_path_rate is not None:
        if (
            isinstance(drop_path_rate, bool)
            or not isinstance(drop_path_rate, int | float)
            or not 0 <= drop_path_rate < 1
        ):
            raise ValueError(f'drop_path_rate must be a number in [0, 1), got {drop_path_rate!r}')
        architecture = backbone_config.get('architecture')
        if not isinstance(architecture, Mapping) or 'drop_path_rate' not in architecture:
            raise ValueError(f'{backbone_name} has no serialized ViT drop_path_rate to set')
        backbone_config = {
            **backbone_config,
            'architecture': {**architecture, 'drop_path_rate': float(drop_path_rate)},
        }
    plans['plans_name'] = derived_plans_identifier
    configuration_dict = plans['configurations'][configuration]
    backbone_config = _configure_vit_patch_size(
        backbone_name,
        backbone_config,
        vit_patch_size,
        configuration_dict.get('patch_size'),
    )
    backbone_config = _configure_vit_adapter(
        backbone_name,
        backbone_config,
        adapter_dim,
        deform_attention_dim,
        deform_num_heads,
        conv_ffn_hidden_dim,
        spatial_prior_channels,
        output_fusion,
    )
    backbone_config = _configure_vit_adapter_paths(
        backbone_name,
        backbone_config,
        readout=readout,
        spatial_prior_input=spatial_prior_input,
    )
    backbone.validate_config(backbone_config)
    source_architecture = _configure_features_per_stage(
        _truncate_architecture_stages(
            configuration_dict['architecture'],
            num_stages,
        ),
        features_per_stage,
    )
    configuration_dict['architecture'] = build_native_architecture(
        source_architecture,
        backbone_name,
        backbone_config,
        readout,
        (
            {
                'mask_attention_mode': (
                    'regions'
                    if PlansManager(plans).get_label_manager(dataset_json).has_regions
                    else 'classes'
                ),
                'query_feature_schedule': mask2former_feature_schedule,
            }
            if readout == 'mask2former'
            else None
        ),
    )
    configuration_dict['optimization'] = dict(optimization)
    if oversample_foreground_percent is not None:
        configuration_dict['oversample_foreground_percent'] = float(
            oversample_foreground_percent
        )
    if batch_dice is not None:
        configuration_dict['batch_dice'] = bool(batch_dice)
    if num_epochs is not None:
        configuration_dict['num_epochs'] = num_epochs
    if deep_supervision is not None:
        configuration_dict['deep_supervision'] = bool(deep_supervision)

    save_json(
        plans,
        plans_path,
        sort_keys=False,
    )
    return derived_plans_identifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id')
    parser.add_argument('configuration')
    parser.add_argument('--plans', default='nnUNetResEncUNetLPlans')
    parser.add_argument('--output-plans')
    parser.add_argument('--backbone', required=True, choices=sorted(BACKBONES))
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--checkpoint-format')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--readout', choices=sorted(NETWORK_CLASS_NAMES), default='unet')
    parser.add_argument(
        '--mask2former-feature-schedule',
        choices=('concat', 'cycle'),
        default='concat',
        help='query cross-attention schedule; only used by the Mask2Former readout',
    )
    parser.add_argument(
        '--vit-patch-size',
        type=int,
        nargs=3,
        metavar=('D', 'H', 'W'),
        help='explicit fixed ViT patch size; H=W=16 and D must be one of 1,2,4,8,16',
    )
    parser.add_argument(
        '--adapter-dim',
        type=int,
        help='ViT-Adapter working width; defaults to the backbone\'s pinned value',
    )
    parser.add_argument('--deform-attention-dim', type=int)
    parser.add_argument('--deform-num-heads', type=int)
    parser.add_argument('--conv-ffn-hidden-dim', type=int)
    parser.add_argument(
        '--spatial-prior-channels',
        type=int,
        nargs=4,
        metavar=('P2', 'P3', 'P4', 'P5'),
    )
    parser.add_argument(
        '--output-fusion',
        choices=('add-then-project', 'project-then-add'),
    )
    parser.add_argument(
        '--features-per-stage',
        type=int,
        nargs='+',
        help='override the source plan channel schedule, including P0 and P1',
    )
    parser.add_argument(
        '--num-stages',
        type=int,
        help='truncate the source plan to its shallowest N stages for a native-depth encoder',
    )
    parser.add_argument(
        '--spatial-prior-input',
        choices=('raw', 'p1'),
        default=None,
        help=(
            'ViT-Adapter convolutional-prior input; defaults to shared p1 for the full U-Net readout and raw without its high-resolution stem'
        ),
    )
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument(
        '--backbone-lr',
        type=float,
        help='top backbone LR before layer-wise decay; defaults to --learning-rate',
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='freeze the pretrained ViT trunk and train only the adapter and decoder',
    )
    parser.add_argument('--layer-decay', type=float, required=True)
    parser.add_argument('--warmup-epochs', type=int, default=50)
    parser.add_argument(
        '--backbone-freeze-epochs',
        type=int,
        help='hold the backbone LR at zero for this many epochs; the trunk stays in the optimizer, '
        'so DDP buckets and the compiled graph are unchanged and its Adam moments accumulate',
    )
    parser.add_argument(
        '--backbone-unfreeze-ramp-epochs',
        type=int,
        help='linearly ramp the backbone LR from zero after the freeze; 0 steps straight to the schedule',
    )
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument(
        '--backbone-wd',
        type=float,
        help='weight decay for the backbone scope; defaults to --weight-decay',
    )
    parser.add_argument(
        '--weight-decay-policy',
        choices=('all', 'vit_standard'),
        default='vit_standard',
    )
    parser.add_argument(
        '--amsgrad',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--poly-exponent', type=float, default=0.9)
    parser.add_argument(
        '--oversample-foreground-percent',
        type=float,
        help=(
            'fraction of each global batch forced to center a foreground voxel; '
            'defaults to the nnU-Net setting'
        ),
    )
    parser.add_argument(
        '--batch-dice',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='aggregate the Dice loss over the batch instead of per sample; defaults to the source plan',
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='schedule length for the derived plan; defaults to the trainer\'s 1000',
    )
    parser.add_argument(
        '--deep-supervision',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='nnU-Net deep supervision; defaults to the trainer\'s on',
    )
    parser.add_argument(
        '--pyramid-branch',
        choices=PYRAMID_BRANCHES,
        help='branch mapping trunk features onto each pyramid level; defaults to the deconvolution SimpleFPN',
    )
    parser.add_argument(
        '--drop-path',
        type=float,
        help='ViT stochastic depth, ramped linearly from 0 at the first block to this rate at the last; '
        'defaults to the backbone\'s pinned architecture (0)',
    )
    parser.add_argument(
        '--model-ema-decay',
        type=float,
        default=0.9998,
        help='maintain a rank-zero FP32 model EMA with this decay and validate it separately; '
        'applies to every new plan by default, pass 0 to disable',
    )
    args = parser.parse_args()

    optimization = {
        'optimizer': 'adamw',
        'learning_rate': args.learning_rate,
        'layer_decay': args.layer_decay,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
        'weight_decay_policy': args.weight_decay_policy,
        'amsgrad': args.amsgrad,
        'lr_scheduler': 'poly',
        'poly_exponent': args.poly_exponent,
    }
    if args.backbone_lr is not None:
        optimization['backbone_lr'] = args.backbone_lr
    if args.backbone_freeze_epochs is not None:
        optimization['backbone_freeze_epochs'] = args.backbone_freeze_epochs
    if args.backbone_unfreeze_ramp_epochs is not None:
        optimization['backbone_unfreeze_ramp_epochs'] = args.backbone_unfreeze_ramp_epochs
    if args.backbone_wd is not None:
        optimization['backbone_wd'] = args.backbone_wd
    if args.freeze_backbone:
        optimization['freeze_backbone'] = True
    if args.model_ema_decay is not None and args.model_ema_decay > 0:
        optimization['model_ema_decay'] = args.model_ema_decay

    plans_identifier = prepare_experiment(
        args.dataset_name_or_id,
        args.configuration,
        source_plans_identifier=args.plans,
        backbone_name=args.backbone,
        weights=args.weights,
        checkpoint_format=args.checkpoint_format,
        gradient_checkpointing=args.gradient_checkpointing,
        optimization=optimization,
        oversample_foreground_percent=args.oversample_foreground_percent,
        batch_dice=args.batch_dice,
        num_epochs=args.num_epochs,
        deep_supervision=args.deep_supervision,
        pyramid_branch=args.pyramid_branch,
        drop_path_rate=args.drop_path,
        output_plans_identifier=args.output_plans,
        readout=args.readout,
        vit_patch_size=args.vit_patch_size,
        adapter_dim=args.adapter_dim,
        deform_attention_dim=args.deform_attention_dim,
        deform_num_heads=args.deform_num_heads,
        conv_ffn_hidden_dim=args.conv_ffn_hidden_dim,
        spatial_prior_channels=args.spatial_prior_channels,
        output_fusion=args.output_fusion,
        features_per_stage=args.features_per_stage,
        num_stages=args.num_stages,
        spatial_prior_input=args.spatial_prior_input,
        mask2former_feature_schedule=args.mask2former_feature_schedule,
    )
    command = [
        'scripts/downstream/seg/launch_one.sh',
        maybe_convert_to_dataset_name(args.dataset_name_or_id),
        plans_identifier,
    ]
    if args.weights is not None:
        command.extend(('--weights', str(args.weights)))
    print(f'prepared plans: {plans_identifier}')
    print(f'run: {shlex.join(command)}')


if __name__ == '__main__':
    main()
