"""Optimizer protocol for downstream dense segmentation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
import json
import os
from pathlib import Path

import torch
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.nnunet.compile_cache import CompileCacheMixin
from pumit.nnunet.model_ema import ModelEmaMixin, validate_model_ema_decay

from .mask2former import MASK2FORMER_NUM_OUTPUTS

_NO_WEIGHT_DECAY_NAME_PARTS = (
    'cls_token',
    'level_embeddings',
    'mask_token',
    'pos_embed',
    'position_embed',
    'query_features',
    'query_positions',
    'rel_pos',
    'relative_position',
    'register_token',
)
_REQUIRED_OPTIMIZATION_KEYS = {
    'optimizer',
    'learning_rate',
    'layer_decay',
    'warmup_epochs',
    'weight_decay',
    'weight_decay_policy',
    'amsgrad',
    'lr_scheduler',
    'poly_exponent',
}
_OPTIONAL_OPTIMIZATION_KEYS = {
    'backbone_lr',
    'backbone_wd',
    'freeze_backbone',
    'backbone_freeze_epochs',
    'backbone_unfreeze_ramp_epochs',
    'model_ema_decay',
}
_MASK2FORMER_NETWORK_CLASS = (
    'pumit.downstream.seg.mask2former.PlanAlignedMask2FormerSegmentationNetwork'
)


def _keep_adamw_fused_after_resume(optimizer: Optimizer) -> None:
    # Older checkpoints store fused=None in each parameter group and would otherwise override the new optimizer default.
    for group in optimizer.param_groups:
        group['fused'] = True


def _apply_plan_foreground_oversampling(trainer: nnUNetTrainer) -> None:
    value = trainer.configuration_manager.configuration.get(
        'oversample_foreground_percent'
    )
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            'oversample_foreground_percent must be a number in [0, 1], got '
            f'{value!r}'
        )
    trainer.oversample_foreground_percent = float(value)


def _apply_plan_num_epochs(trainer: nnUNetTrainer) -> None:
    value = trainer.configuration_manager.configuration.get('num_epochs')
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'num_epochs must be a positive integer, got {value!r}')
    trainer.num_epochs = value


def _apply_plan_deep_supervision(trainer: nnUNetTrainer) -> None:
    value = trainer.configuration_manager.configuration.get('deep_supervision')
    if value is None:
        return
    if not isinstance(value, bool):
        raise TypeError(f'deep_supervision must be a bool, got {value!r}')
    trainer.enable_deep_supervision = value


class _CUDAGraphStepPredictor(nnUNetPredictor):
    """Keep one patch and all of its mirrored forwards in the same CUDA Graph step."""

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        return super()._internal_maybe_mirror_and_predict(x)


@contextmanager
def _use_cudagraph_step_predictor() -> Iterator[None]:
    trainer_module = import_module(
        'nnunetv2.training.nnUNetTrainer.nnUNetTrainer'
    )
    original = trainer_module.nnUNetPredictor
    trainer_module.nnUNetPredictor = _CUDAGraphStepPredictor
    try:
        yield
    finally:
        trainer_module.nnUNetPredictor = original


class _ReduceOverheadValidationMixin:
    """Declare sliding-window patch boundaries for reduce-overhead CUDA Graphs."""

    def perform_actual_validation(self, save_probabilities: bool = False):
        if self.device.type != 'cuda' or self.compile_mode != 'reduce-overhead':
            return super().perform_actual_validation(save_probabilities)
        with _use_cudagraph_step_predictor():
            return super().perform_actual_validation(save_probabilities)


class _NamedFoldSplitMixin:
    """Resolve string folds through the ``name`` key of ``splits_final.json`` entries.

    Named folds keep label-fraction splits self-describing end to end (manifest ``fold`` -> split entry ->
    ``fold_<name>`` result dir). Integer folds keep stock semantics except for stock nnU-Net's out-of-range
    fallback, which silently substitutes a seeded random 80:20 split; that now raises instead.
    """

    def do_split(self):
        splits_file = Path(self.preprocessed_dataset_folder_base) / 'splits_final.json'
        if isinstance(self.fold, str) and self.fold != 'all':
            splits = json.loads(splits_file.read_text())
            matches = [split for split in splits if split.get('name') == self.fold]
            if len(matches) != 1:
                raise ValueError(
                    f'expected exactly one split named {self.fold!r} in {splits_file}, '
                    f'found {len(matches)}'
                )
            tr_keys, val_keys = matches[0]['train'], matches[0]['val']
            overlap = set(tr_keys) & set(val_keys)
            if overlap:
                raise ValueError(
                    f'split {self.fold!r} has {len(overlap)} cases in both train and val'
                )
            self.print_to_log_file(
                f'Using named split {self.fold!r} from {splits_file}: '
                f'{len(tr_keys)} training and {len(val_keys)} validation cases.'
            )
            return tr_keys, val_keys
        if isinstance(self.fold, int) and splits_file.is_file():
            num_splits = len(json.loads(splits_file.read_text()))
            if self.fold >= num_splits:
                raise ValueError(
                    f'fold {self.fold} is out of range for the {num_splits} splits in {splits_file}; '
                    f'stock nnU-Net would silently substitute a random 80:20 split here'
                )
        return super().do_split()


def _init_plan_trainer(trainer: nnUNetTrainer, owner: type, *args: object) -> None:
    """Run the next constructor after ``owner`` in the MRO, then apply the plan-level training keys.

    Both trainers spell out nnU-Net's constructor signature (it is read back via ``inspect.signature``), so the
    shared body lives here.
    """
    # DDPOptimizer subgraphs can never hit AOTAutogradCache (unconditional fakify_first_call bypass), so
    # multi-GPU restarts re-derive every subgraph. Unsplit graphs keep restarts on the warm-cache path;
    # the lost allreduce/backward overlap is negligible single-node.
    torch._dynamo.config.optimize_ddp = False
    super(owner, trainer).__init__(*args)
    _apply_plan_foreground_oversampling(trainer)
    _apply_plan_num_epochs(trainer)
    _apply_plan_deep_supervision(trainer)


class nnUNetTrainerRetainCheckpoints(
    _ReduceOverheadValidationMixin,
    CompileCacheMixin,
    RetainPeriodicCheckpointsMixin,
    _NamedFoldSplitMixin,
    nnUNetTrainer,
):
    """Use the official nnU-Net trainer while retaining every periodic checkpoint."""

    # nnUNetTrainer records its constructor arguments via inspect.signature(self.__init__). Keep the native
    # signature explicit because inheriting the mixin's *args/**kwargs signature breaks that reflection.
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        _init_plan_trainer(self, nnUNetTrainerRetainCheckpoints, plans, configuration, fold, dataset_json, device)


class _WarmupPolyLRScheduler(LRScheduler):
    """Apply linear warmup followed by Poly decay while preserving group LR ratios.

    Backbone groups can additionally be held at LR 0 for the first ``backbone_freeze_steps`` steps and
    ramped back linearly over ``backbone_ramp_steps``. Their parameters keep ``requires_grad`` and stay in
    the optimizer throughout, which is what makes delayed unfreezing safe under DDP and torch.compile: the
    graph and reducer buckets never change, and the accumulated second moments precondition the first real
    update at unfreeze.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        *,
        warmup_steps: int,
        exponent: float = 0.9,
        backbone_freeze_steps: int = 0,
        backbone_ramp_steps: int = 0,
    ):
        if not 0 <= warmup_steps < max_steps:
            raise ValueError(
                f'warmup_steps must be in [0, {max_steps}), got {warmup_steps}'
            )
        if not 0 <= backbone_freeze_steps < max_steps:
            raise ValueError(
                f'backbone_freeze_steps must be in [0, {max_steps}), got {backbone_freeze_steps}'
            )
        if backbone_ramp_steps < 0:
            raise ValueError(f'backbone_ramp_steps must be non-negative, got {backbone_ramp_steps}')
        self.initial_lrs = tuple(group['lr'] for group in optimizer.param_groups)
        self.backbone_group = tuple(
            str(group.get('name', '')).startswith('backbone_layer_')
            for group in optimizer.param_groups
        )
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        self.backbone_freeze_steps = backbone_freeze_steps
        self.backbone_ramp_steps = backbone_ramp_steps
        self.ctr = 0
        super().__init__(optimizer)

    def _backbone_factor(self, current_step: int) -> float:
        if current_step < self.backbone_freeze_steps:
            return 0.0
        if self.backbone_ramp_steps == 0:
            return 1.0
        progress = (current_step - self.backbone_freeze_steps + 1) / self.backbone_ramp_steps
        return min(1.0, progress)

    def step(self, current_step: int | None = None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        if not 0 <= current_step < self.max_steps:
            raise ValueError(
                f'current_step must be in [0, {self.max_steps}), got {current_step}'
            )
        if current_step < self.warmup_steps:
            factor = (current_step + 1) / self.warmup_steps
        else:
            decay_steps = self.max_steps - self.warmup_steps
            factor = (
                1 - (current_step - self.warmup_steps) / decay_steps
            ) ** self.exponent
        backbone_factor = self._backbone_factor(current_step)
        for initial_lr, is_backbone, param_group in zip(
            self.initial_lrs, self.backbone_group, self.optimizer.param_groups
        ):
            group_factor = factor * backbone_factor if is_backbone else factor
            param_group['lr'] = initial_lr * group_factor
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class _DDPDeepSupervisionWrapper(DeepSupervisionWrapper):
    """Keep zero-weight outputs in autograd so every DDP output head has a gradient."""

    def forward(self, *args):
        loss = super().forward(*args)
        for prediction, weight in zip(args[0], self.weight_factors):
            if weight == 0:
                loss = loss + prediction.float().mean() * 0
        return loss


class _DownstreamModelEmaMixin(ModelEmaMixin):
    """EMA step hook and dual final validation, kept out of DownstreamSegTrainer's own dict so the
    stock-loop guard (test_trainer_preserves_nnunet_training_loop) keeps holding."""

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        self._update_model_ema()
        return result

    def perform_actual_validation(self, save_probabilities: bool = False):
        # EMA first: the queue's completion marker is the raw validation/summary.json, so writing it
        # last guarantees a job marked done also carries its EMA summary.
        if self.model_ema_decay is not None:
            self.print_to_log_file(
                f'Evaluating rank-zero model EMA with decay {self.model_ema_decay}.'
            )
            # Redirect nnU-Net's fixed `<output_folder>/validation` target for the EMA pass.
            raw_output_folder = self.output_folder
            try:
                with self.model_ema_weights():
                    self.output_folder = os.path.join(raw_output_folder, 'model_ema')
                    os.makedirs(self.output_folder, exist_ok=True)
                    super().perform_actual_validation(save_probabilities)
            finally:
                self.output_folder = raw_output_folder
        super().perform_actual_validation(save_probabilities)


class DownstreamSegTrainer(
    _DownstreamModelEmaMixin,
    _ReduceOverheadValidationMixin,
    CompileCacheMixin,
    RetainPeriodicCheckpointsMixin,
    _NamedFoldSplitMixin,
    nnUNetTrainer,
):
    """Use BF16 AMP and the optimization protocol declared by the nnU-Net plan."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        _init_plan_trainer(self, DownstreamSegTrainer, plans, configuration, fold, dataset_json, device)
        optimization = self._get_optimization_config()
        self.initial_lr = float(optimization['learning_rate'])
        self.weight_decay = float(optimization['weight_decay'])
        self.model_ema_decay = validate_model_ema_decay(optimization.get('model_ema_decay'))
        if self.device.type == 'cuda':
            torch.set_autocast_dtype('cuda', torch.bfloat16)
            self.grad_scaler = None

    def _get_optimization_config(self) -> dict[str, object]:
        config = self.configuration_manager.configuration.get('optimization')
        if not isinstance(config, dict):
            raise TypeError(f'optimization must be a mapping, got {type(config).__name__}')
        missing = _REQUIRED_OPTIMIZATION_KEYS - config.keys()
        unexpected = (
            config.keys()
            - _REQUIRED_OPTIMIZATION_KEYS
            - _OPTIONAL_OPTIMIZATION_KEYS
        )
        if missing or unexpected:
            raise ValueError(
                f'optimization keys mismatch: '
                f'missing={sorted(missing)}, unexpected={sorted(unexpected)}'
            )
        if config['optimizer'] != 'adamw':
            raise ValueError(f"unsupported optimizer: {config['optimizer']!r}")
        if config['lr_scheduler'] != 'poly':
            raise ValueError(f"unsupported lr_scheduler: {config['lr_scheduler']!r}")
        if config['weight_decay_policy'] not in {'all', 'vit_standard'}:
            raise ValueError(
                f"unsupported weight_decay_policy: {config['weight_decay_policy']!r}"
            )
        if not isinstance(config['amsgrad'], bool):
            raise TypeError('optimization.amsgrad must be a bool')
        if config.get('freeze_backbone') is not None and not isinstance(config['freeze_backbone'], bool):
            raise TypeError('optimization.freeze_backbone must be a bool')
        for key in ('learning_rate', 'backbone_lr', 'poly_exponent'):
            value = config.get(key)
            if value is not None and (
                not isinstance(value, int | float) or value <= 0
            ):
                raise ValueError(f'optimization.{key} must be positive')
        if (
            not isinstance(config['layer_decay'], int | float)
            or not 0 < config['layer_decay'] <= 1
        ):
            raise ValueError('optimization.layer_decay must be in (0, 1]')
        if (
            not isinstance(config['warmup_epochs'], int)
            or not 0 <= config['warmup_epochs'] < self.num_epochs
        ):
            raise ValueError(
                f'optimization.warmup_epochs must be an integer in [0, {self.num_epochs})'
            )
        if not isinstance(config['weight_decay'], int | float) or config['weight_decay'] < 0:
            raise ValueError('optimization.weight_decay must be non-negative')
        backbone_wd = config.get('backbone_wd')
        if backbone_wd is not None and (
            not isinstance(backbone_wd, int | float) or backbone_wd < 0
        ):
            raise ValueError('optimization.backbone_wd must be non-negative')
        return dict(config)

    def _uses_mask2former_readout(self) -> bool:
        return (
            self.configuration_manager.network_arch_class_name
            == _MASK2FORMER_NETWORK_CLASS
        )

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision and self._uses_mask2former_readout():
            return [[1.0, 1.0, 1.0] for _ in range(MASK2FORMER_NUM_OUTPUTS)]
        return super()._get_deep_supervision_scales()

    def _build_loss(self):
        loss = super()._build_loss()
        if not self.enable_deep_supervision:
            return loss
        if not isinstance(loss, DeepSupervisionWrapper):
            raise TypeError(
                f'nnU-Net deep-supervision loss must be DeepSupervisionWrapper, '
                f'got {type(loss).__name__}'
            )
        if self._uses_mask2former_readout():
            weights = [1.0] * MASK2FORMER_NUM_OUTPUTS
            return DeepSupervisionWrapper(loss.loss, weights)
        if self.is_ddp and any(weight == 0 for weight in loss.weight_factors):
            return _DDPDeepSupervisionWrapper(loss.loss, loss.weight_factors)
        return loss

    @staticmethod
    def _backbone_parameter_layer_ids(
        backbone: torch.nn.Module,
        layer_decay: float,
        *,
        frozen: bool,
    ) -> tuple[set[int], ...]:
        trainable_ids = {
            id(parameter)
            for parameter in backbone.parameters()
            if parameter.requires_grad
        }
        if frozen:
            return ()
        if layer_decay == 1:
            return (trainable_ids,)

        grouping = getattr(backbone, 'parameter_layers', None)
        if not callable(grouping):
            raise TypeError(
                f'{type(backbone).__name__} must define parameter_layers() '
                f'when layer_decay={layer_decay}'
            )
        parameter_layers = tuple(
            tuple(
                id(parameter)
                for parameter in parameters
                if parameter.requires_grad
            )
            for parameters in grouping()
        )
        if not parameter_layers or any(not layer for layer in parameter_layers):
            raise RuntimeError('backbone parameter layers must be non-empty')
        flattened = [
            parameter_id
            for layer in parameter_layers
            for parameter_id in layer
        ]
        if len(flattened) != len(set(flattened)):
            raise RuntimeError('backbone parameter layers contain duplicates')
        if set(flattened) != trainable_ids:
            missing = trainable_ids - set(flattened)
            unexpected = set(flattened) - trainable_ids
            raise RuntimeError(
                f'backbone parameter layers do not cover trainable parameters: '
                f'missing={len(missing)}, unexpected={len(unexpected)}'
            )
        return tuple(set(layer) for layer in parameter_layers)

    def configure_optimizers(self):
        config = self._get_optimization_config()
        network = self.network._orig_mod if isinstance(self.network, OptimizedModule) else self.network
        freeze_backbone = bool(config.get('freeze_backbone', False))
        if freeze_backbone:
            # Freeze before the first forward so the lazy compile trace never sees them.
            for parameter in network.encoder.backbone.parameters():
                parameter.requires_grad = False
        layer_decay = float(config['layer_decay'])
        learning_rate = float(config['learning_rate'])
        backbone_learning_rate = float(config.get('backbone_lr', learning_rate))
        weight_decay = float(config['weight_decay'])
        backbone_weight_decay = float(config.get('backbone_wd', weight_decay))
        backbone_layers = self._backbone_parameter_layer_ids(
            network.encoder.backbone,
            layer_decay,
            frozen=freeze_backbone,
        )
        backbone_parameter_ids = set().union(*backbone_layers)
        backbone_layer_by_parameter = {
            parameter_id: layer
            for layer, parameter_ids in enumerate(backbone_layers)
            for parameter_id in parameter_ids
        }

        parameter_groups: dict[str, list[torch.nn.Parameter]] = {}
        trainable_parameter_ids = set()
        for name, parameter in network.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            trainable_parameter_ids.add(parameter_id)
            if parameter_id in backbone_parameter_ids:
                scope = f'backbone_layer_{backbone_layer_by_parameter[parameter_id]:02d}'
            else:
                scope = 'scratch'
            if config['weight_decay_policy'] == 'vit_standard':
                normalized_name = name.lower()
                no_decay = parameter.ndim <= 1 or any(
                    part in normalized_name
                    for part in _NO_WEIGHT_DECAY_NAME_PARTS
                )
                group_name = f'{scope}_no_decay' if no_decay else f'{scope}_decay'
            else:
                group_name = scope
            parameter_groups.setdefault(group_name, []).append(parameter)

        if not freeze_backbone and (
            not backbone_parameter_ids or backbone_parameter_ids - trainable_parameter_ids
        ):
            raise RuntimeError('failed to identify all trainable backbone parameters')
        scopes = (
            'scratch',
            *(f'backbone_layer_{layer:02d}' for layer in range(len(backbone_layers))),
        )
        scope_lrs = {
            scope: (
                learning_rate
                if scope == 'scratch'
                else backbone_learning_rate
                * layer_decay
                ** (
                    len(backbone_layers)
                    - 1
                    - int(scope.removeprefix('backbone_layer_'))
                )
            )
            for scope in scopes
        }
        suffixes = (
            ('decay', 'no_decay')
            if config['weight_decay_policy'] == 'vit_standard'
            else (None,)
        )
        group_specs = tuple(
            (f'{scope}_{suffix}' if suffix is not None else scope, scope_lrs[scope])
            for scope in scopes
            for suffix in suffixes
            if (f'{scope}_{suffix}' if suffix is not None else scope) in parameter_groups
        )
        if not group_specs or not any(
            name.startswith('scratch') for name, _ in group_specs
        ):
            raise RuntimeError('optimizer contains no trainable scratch parameters')

        optimizer_groups = []
        for name, group_lr in group_specs:
            if name.endswith('no_decay'):
                group_weight_decay = 0.0
            elif name.startswith('backbone_layer_'):
                group_weight_decay = backbone_weight_decay
            else:
                group_weight_decay = weight_decay
            optimizer_groups.append(
                {
                    'params': parameter_groups[name],
                    'lr': group_lr,
                    'weight_decay': group_weight_decay,
                    'name': name,
                }
            )

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=float(config['weight_decay']),
            amsgrad=bool(config['amsgrad']),
            fused=True,
        )
        optimizer.register_load_state_dict_post_hook(_keep_adamw_fused_after_resume)
        backbone_freeze_epochs = int(config.get('backbone_freeze_epochs', 0))
        if freeze_backbone and backbone_freeze_epochs:
            raise ValueError(
                'freeze_backbone and backbone_freeze_epochs are mutually exclusive: the first removes '
                'the trunk from the optimizer, the second holds its LR at zero'
            )
        return optimizer, _WarmupPolyLRScheduler(
            optimizer,
            self.num_epochs,
            warmup_steps=int(config['warmup_epochs']),
            exponent=float(config['poly_exponent']),
            backbone_freeze_steps=backbone_freeze_epochs,
            backbone_ramp_steps=int(config.get('backbone_unfreeze_ramp_epochs', 0)),
        )
