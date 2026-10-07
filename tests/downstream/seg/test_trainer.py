from importlib import import_module
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

import pumit.nnunet.compile_cache as compile_cache_module
from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.nnunet.compile_cache import CompileCacheMixin
from pumit.downstream.seg.trainer import (
    _CUDAGraphStepPredictor,
    _NamedFoldSplitMixin,
    _use_cudagraph_step_predictor,
    DownstreamSegTrainer,
    nnUNetTrainerRetainCheckpoints,
)
from pumit.downstream.seg.adapters.vit import make_parameter_layers


class _FakeLayeredBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)

    def parameter_layers(self):
        return make_parameter_layers(
            self.patch_embed.parameters(),
            self.blocks,
            self.norm.parameters(),
        )


class _FakeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = _FakeLayeredBackbone()
        self.encoder.high_resolution_stem = nn.Linear(4, 4)
        self.encoder.simple_fpn = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 2)


_ORIGINAL_OPTIMIZATION = {
    'optimizer': 'adamw',
    'learning_rate': 3e-4,
    'layer_decay': 0.5,
    'warmup_epochs': 10,
    'weight_decay': 0.05,
    'weight_decay_policy': 'vit_standard',
    'amsgrad': False,
    'lr_scheduler': 'poly',
    'poly_exponent': 0.9,
}


class _StockSplit:
    def do_split(self):
        return ['stock-train'], ['stock-val']


class _SplitProbe(_NamedFoldSplitMixin, _StockSplit):
    def __init__(self, base, fold):
        self.preprocessed_dataset_folder_base = str(base)
        self.fold = fold

    def print_to_log_file(self, *args, **kwargs):
        pass


def _write_splits(tmp_path, splits):
    (tmp_path / 'splits_final.json').write_text(json.dumps(splits))


def test_named_fold_resolves_the_matching_split_entry(tmp_path):
    _write_splits(tmp_path, [
        {'train': ['a', 'b'], 'val': ['c']},
        {'name': 'le-5pct-a', 'train': ['a'], 'val': ['c']},
    ])

    tr_keys, val_keys = _SplitProbe(tmp_path, 'le-5pct-a').do_split()

    assert (tr_keys, val_keys) == (['a'], ['c'])


@pytest.mark.parametrize('splits, match', [
    ([{'train': ['a'], 'val': ['c']}], 'found 0'),
    (
        [
            {'name': 'le-5pct-a', 'train': ['a'], 'val': ['c']},
            {'name': 'le-5pct-a', 'train': ['b'], 'val': ['c']},
        ],
        'found 2',
    ),
    ([{'name': 'le-5pct-a', 'train': ['a', 'c'], 'val': ['c']}], 'both train and val'),
])
def test_named_fold_lookup_fails_loudly(tmp_path, splits, match):
    _write_splits(tmp_path, splits)

    with pytest.raises(ValueError, match=match):
        _SplitProbe(tmp_path, 'le-5pct-a').do_split()


def test_integer_folds_delegate_to_stock_within_range(tmp_path):
    _write_splits(tmp_path, [{'train': ['a'], 'val': ['c']}])

    assert _SplitProbe(tmp_path, 0).do_split() == (['stock-train'], ['stock-val'])
    assert _SplitProbe(tmp_path, 'all').do_split() == (['stock-train'], ['stock-val'])


def test_out_of_range_integer_fold_raises_instead_of_stock_random_split(tmp_path):
    _write_splits(tmp_path, [{'train': ['a'], 'val': ['c']}])

    with pytest.raises(ValueError, match='out of range'):
        _SplitProbe(tmp_path, 1).do_split()


def test_trainer_preserves_nnunet_training_loop():
    inherited_methods = {
        'initialize',
        'get_dataloaders',
        'train_step',
        'validation_step',
        'run_training',
        'load_checkpoint',
    }

    assert inherited_methods.isdisjoint(DownstreamSegTrainer.__dict__)
    assert set(DownstreamSegTrainer.__dict__) & {
        '__init__',
        'configure_optimizers',
        '_get_deep_supervision_scales',
        '_build_loss',
    } == {
        '__init__',
        'configure_optimizers',
        '_get_deep_supervision_scales',
        '_build_loss',
    }
    assert DownstreamSegTrainer.save_checkpoint is RetainPeriodicCheckpointsMixin.save_checkpoint


def test_cudagraph_predictor_replacement_is_scoped():
    trainer_module = import_module(
        'nnunetv2.training.nnUNetTrainer.nnUNetTrainer'
    )
    original = trainer_module.nnUNetPredictor

    with _use_cudagraph_step_predictor():
        assert trainer_module.nnUNetPredictor is _CUDAGraphStepPredictor

    assert trainer_module.nnUNetPredictor is original


def test_cudagraph_predictor_marks_one_step_for_all_mirrored_forwards(monkeypatch):
    predictor = object.__new__(_CUDAGraphStepPredictor)
    predictor.allowed_mirroring_axes = (0, 1)
    predictor.use_mirroring = True
    calls = []
    marks = []

    def network(x):
        calls.append(x)
        return x.clone()

    predictor.network = network
    monkeypatch.setattr(
        torch.compiler,
        'cudagraph_mark_step_begin',
        lambda: marks.append(None),
    )
    x = torch.randn(1, 1, 2, 3, 4)

    prediction = predictor._internal_maybe_mirror_and_predict(x)

    assert torch.equal(prediction, x)
    assert len(calls) == 4
    assert len(marks) == 1


def test_resenc_retention_trainer_keeps_official_optimizer():
    assert nnUNetTrainerRetainCheckpoints.configure_optimizers is nnUNetTrainer.configure_optimizers


def test_resenc_retention_trainer_exposes_native_init_signature():
    trainer = object.__new__(nnUNetTrainerRetainCheckpoints)

    assert tuple(inspect.signature(trainer.__init__).parameters) == (
        'plans',
        'configuration',
        'fold',
        'dataset_json',
        'device',
    )


@pytest.mark.parametrize(
    ('configuration_dict', 'expected'),
    [({}, 0.33), ({'oversample_foreground_percent': 0.25}, 0.25)],
)
def test_resenc_retention_trainer_reads_foreground_oversampling_from_plan(
    monkeypatch,
    configuration_dict,
    expected,
):
    def fake_init(self, plans, configuration, fold, dataset_json, device):
        self.oversample_foreground_percent = 0.33
        self.configuration_manager = SimpleNamespace(
            configuration=configuration_dict
        )

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init)
    trainer = nnUNetTrainerRetainCheckpoints(
        {}, '3d_fullres', 0, {}, torch.device('cpu')
    )

    assert trainer.oversample_foreground_percent == expected


def test_compile_cache_supports_installed_nnunet_rank_contract(
    tmp_path,
    monkeypatch,
):
    trainer = object.__new__(nnUNetTrainerRetainCheckpoints)
    trainer.was_initialized = False
    trainer.local_rank = 0
    trainer.fold = 0
    trainer.output_folder_base = str(tmp_path)
    trainer.compile_mode = 'default'
    trainer.compile_dynamic = False
    trainer._do_i_compile = lambda: True
    trainer.print_to_log_file = lambda *args, **kwargs: None

    monkeypatch.setattr(
        compile_cache_module,
        'extract_compile_cache',
        lambda archive, cache_dir, **kwargs: (cache_dir, True),
    )
    monkeypatch.setattr(
        nnUNetTrainer,
        'initialize',
        lambda self: setattr(self, 'was_initialized', True),
    )

    CompileCacheMixin.initialize(trainer)

    assert trainer.was_initialized
    assert trainer._compile_cache_needs_archive


def test_trainer_uses_bf16_autocast_without_grad_scaling(monkeypatch):
    grad_scaler = object()

    def fake_init(self, plans, configuration, fold, dataset_json, device):
        self.device = device
        self.grad_scaler = grad_scaler
        self.num_epochs = 100
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.configuration_manager = SimpleNamespace(
            configuration={
                'optimization': _ORIGINAL_OPTIMIZATION,
                'oversample_foreground_percent': 0.25,
            }
        )

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init)
    previous_dtype = torch.get_autocast_dtype('cuda')
    try:
        trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cuda'))
        assert torch.get_autocast_dtype('cuda') is torch.bfloat16
        assert trainer.grad_scaler is None
        assert trainer.initial_lr == 3e-4
        assert trainer.weight_decay == 0.05
        assert trainer.oversample_foreground_percent == 0.25
    finally:
        torch.set_autocast_dtype('cuda', previous_dtype)


def _fake_adam_init(optimization):
    def fake_init(self, plans, configuration, fold, dataset_json, device):
        self.device = device
        self.grad_scaler = None
        self.num_epochs = 100
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.configuration_manager = SimpleNamespace(
            configuration={'optimization': optimization}
        )

    return fake_init


def test_trainer_reads_model_ema_decay_from_plan(monkeypatch):
    monkeypatch.setattr(
        nnUNetTrainer,
        '__init__',
        _fake_adam_init({**_ORIGINAL_OPTIMIZATION, 'model_ema_decay': 0.9998}),
    )
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))
    assert trainer.model_ema_decay == 0.9998

    monkeypatch.setattr(
        nnUNetTrainer,
        '__init__',
        _fake_adam_init(_ORIGINAL_OPTIMIZATION),
    )
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))
    assert trainer.model_ema_decay is None

    monkeypatch.setattr(
        nnUNetTrainer,
        '__init__',
        _fake_adam_init({**_ORIGINAL_OPTIMIZATION, 'model_ema_decay': 1.5}),
    )
    with pytest.raises(ValueError, match='between 0 and 1'):
        DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))


def test_trainer_reads_num_epochs_from_plan(monkeypatch):
    def fake_init_with_num_epochs(num_epochs):
        fake_init = _fake_adam_init(_ORIGINAL_OPTIMIZATION)

        def init(self, plans, configuration, fold, dataset_json, device):
            fake_init(self, plans, configuration, fold, dataset_json, device)
            self.configuration_manager.configuration['num_epochs'] = num_epochs

        return init

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init_with_num_epochs(200))
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))
    assert trainer.num_epochs == 200

    monkeypatch.setattr(nnUNetTrainer, '__init__', _fake_adam_init(_ORIGINAL_OPTIMIZATION))
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))
    assert trainer.num_epochs == 100

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init_with_num_epochs(0))
    with pytest.raises(ValueError, match='num_epochs must be a positive integer'):
        DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))

    # The warmup bound is checked against the plan's schedule length.
    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init_with_num_epochs(10))
    with pytest.raises(ValueError, match='warmup_epochs must be an integer in \\[0, 10\\)'):
        DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))


def test_trainer_reads_deep_supervision_from_plan(monkeypatch):
    def fake_init_with_deep_supervision(value):
        fake_init = _fake_adam_init(_ORIGINAL_OPTIMIZATION)

        def init(self, plans, configuration, fold, dataset_json, device):
            fake_init(self, plans, configuration, fold, dataset_json, device)
            self.enable_deep_supervision = True
            self.configuration_manager.configuration['deep_supervision'] = value

        return init

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init_with_deep_supervision(False))
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))
    assert trainer.enable_deep_supervision is False

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init_with_deep_supervision(None))
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))
    assert trainer.enable_deep_supervision is True

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init_with_deep_supervision('no'))
    with pytest.raises(TypeError, match='deep_supervision must be a bool'):
        DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))


def test_ema_validation_runs_first_so_the_raw_summary_marks_completion(monkeypatch, tmp_path):
    from contextlib import contextmanager

    import pumit.downstream.seg.trainer as trainer_module

    monkeypatch.setattr(
        nnUNetTrainer,
        '__init__',
        _fake_adam_init({**_ORIGINAL_OPTIMIZATION, 'model_ema_decay': 0.9998}),
    )
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))

    calls = []

    def fake_validation(self, save_probabilities=False):
        calls.append((self.output_folder, save_probabilities))

    monkeypatch.setattr(
        trainer_module._ReduceOverheadValidationMixin,
        'perform_actual_validation',
        fake_validation,
    )

    @contextmanager
    def fake_weights():
        yield

    monkeypatch.setattr(trainer, 'model_ema_weights', fake_weights)
    monkeypatch.setattr(trainer, 'print_to_log_file', lambda *args, **kwargs: None)
    trainer.output_folder = str(tmp_path)

    trainer.perform_actual_validation(save_probabilities=False)

    assert calls == [
        (str(tmp_path / 'model_ema'), False),
        (str(tmp_path), False),
    ]
    assert (tmp_path / 'model_ema').is_dir()
    assert trainer.output_folder == str(tmp_path)


def test_trainer_preserves_default_foreground_oversampling(monkeypatch):
    def fake_init(self, plans, configuration, fold, dataset_json, device):
        self.device = device
        self.grad_scaler = None
        self.num_epochs = 100
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.configuration_manager = SimpleNamespace(
            configuration={'optimization': _ORIGINAL_OPTIMIZATION}
        )

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init)
    trainer = DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))

    assert trainer.oversample_foreground_percent == 0.33


@pytest.mark.parametrize('value', [-0.1, 1.1, True, '0.25'])
def test_trainer_rejects_invalid_foreground_oversampling(monkeypatch, value):
    def fake_init(self, plans, configuration, fold, dataset_json, device):
        self.device = device
        self.grad_scaler = None
        self.num_epochs = 100
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.configuration_manager = SimpleNamespace(
            configuration={
                'optimization': _ORIGINAL_OPTIMIZATION,
                'oversample_foreground_percent': value,
            }
        )

    monkeypatch.setattr(nnUNetTrainer, '__init__', fake_init)
    with pytest.raises(ValueError, match='must be a number in'):
        DownstreamSegTrainer({}, '3d_fullres', 0, {}, torch.device('cpu'))


def test_foreground_oversampling_is_partitioned_from_the_global_batch(monkeypatch):
    local_percentages = []
    for rank in range(4):
        trainer = object.__new__(DownstreamSegTrainer)
        trainer.is_ddp = True
        trainer.oversample_foreground_percent = 0.25
        trainer.configuration_manager = SimpleNamespace(batch_size=8)
        monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda rank=rank: rank)

        trainer._set_batch_size_and_oversample()

        assert trainer.batch_size == 2
        local_percentages.append(trainer.oversample_foreground_percent)

    assert local_percentages == [0.0, 0.0, 0.0, 1.0]


def test_mask2former_uses_full_resolution_targets_and_unit_layer_weights():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.enable_deep_supervision = True
    trainer.configuration_manager = SimpleNamespace(
        network_arch_class_name=(
            'pumit.downstream.seg.mask2former.'
            'PlanAlignedMask2FormerSegmentationNetwork'
        ),
        batch_dice=False,
    )
    trainer.label_manager = SimpleNamespace(
        has_regions=False,
        ignore_label=None,
    )
    trainer.is_ddp = False
    trainer._do_i_compile = lambda: False

    scales = trainer._get_deep_supervision_scales()
    loss = trainer._build_loss()

    assert scales == [[1.0, 1.0, 1.0]] * 10
    assert isinstance(loss, DeepSupervisionWrapper)
    assert loss.weight_factors == pytest.approx([1.0] * 10)

    segmentation = torch.randint(0, 4, (1, 3, 4, 5), dtype=torch.int16)
    targets = DownsampleSegForDSTransform(scales)._apply_to_segmentation(
        segmentation
    )
    predictions = [
        torch.randn(1, 4, 3, 4, 5, requires_grad=True)
        for _ in range(10)
    ]
    batched_targets = [target.unsqueeze(0) for target in targets]
    value = loss(predictions, batched_targets)
    value.backward()

    assert [tuple(target.shape) for target in targets] == [(1, 3, 4, 5)] * 10
    assert all(prediction.grad is not None for prediction in predictions)


def test_ddp_zero_weight_output_preserves_loss_and_active_gradients():
    from pumit.downstream.seg.trainer import _DDPDeepSupervisionWrapper

    class SquaredError(nn.Module):
        def forward(self, prediction, target):
            return (prediction - target).square().mean()

    predictions = [torch.randn(2, 3, 4, requires_grad=True) for _ in range(3)]
    copies = [value.detach().clone().requires_grad_() for value in predictions]
    targets = [torch.randn_like(value) for value in predictions]
    weights = (2 / 3, 1 / 3, 0)
    original = DeepSupervisionWrapper(SquaredError(), weights)(predictions, targets)
    adapted = _DDPDeepSupervisionWrapper(SquaredError(), weights)(copies, targets)
    torch.testing.assert_close(adapted, original, atol=0, rtol=0)
    original.backward()
    adapted.backward()
    for left, right in zip(predictions[:2], copies[:2]):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert predictions[-1].grad is None
    assert copies[-1].grad is not None
    assert torch.count_nonzero(copies[-1].grad) == 0


def test_optimizer_partitions_backbone_from_scratch_parameters():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = _FakeNetwork()
    trainer.num_epochs = 100
    trainer.weight_decay = 3e-5
    trainer.configuration_manager = SimpleNamespace(
        configuration={'optimization': _ORIGINAL_OPTIMIZATION}
    )

    optimizer, scheduler = trainer.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['fused']
    assert not optimizer.defaults['amsgrad']
    assert optimizer.defaults['weight_decay'] == 0.05
    assert [group['name'] for group in optimizer.param_groups] == [
        'scratch_decay',
        'scratch_no_decay',
        'backbone_layer_00_decay',
        'backbone_layer_00_no_decay',
        'backbone_layer_01_decay',
        'backbone_layer_01_no_decay',
        'backbone_layer_02_decay',
        'backbone_layer_02_no_decay',
    ]
    target_lrs = [
        3e-4,
        3e-4,
        7.5e-5,
        7.5e-5,
        1.5e-4,
        1.5e-4,
        3e-4,
        3e-4,
    ]
    assert scheduler.initial_lrs == pytest.approx(target_lrs)
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx(
        [lr / 10 for lr in target_lrs]
    )

    backbone_ids = {id(parameter) for parameter in trainer.network.encoder.backbone.parameters()}
    assert {
        id(parameter)
        for group in optimizer.param_groups
        if group['name'].startswith('backbone')
        for parameter in group['params']
    } == backbone_ids
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    } == {id(parameter) for parameter in trainer.network.parameters()}

    scheduler.step(10)
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx(
        target_lrs
    )


def test_backbone_parameter_layers_reject_duplicates_within_one_layer():
    backbone = _FakeLayeredBackbone()
    parameter = next(backbone.parameters())
    backbone.parameter_layers = lambda: ((parameter, parameter),)

    with pytest.raises(RuntimeError, match='contain duplicates'):
        DownstreamSegTrainer._backbone_parameter_layer_ids(backbone, 0.5, frozen=False)


def test_scheduler_holds_and_ramps_the_backbone_groups():
    from pumit.downstream.seg.trainer import _WarmupPolyLRScheduler

    scratch = torch.nn.Parameter(torch.zeros(1))
    trunk = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [
            {'params': [scratch], 'lr': 1e-4, 'name': 'scratch_decay'},
            {'params': [trunk], 'lr': 5e-6, 'name': 'backbone_layer_00_decay'},
        ]
    )
    scheduler = _WarmupPolyLRScheduler(
        optimizer,
        1000,
        warmup_steps=50,
        backbone_freeze_steps=250,
        backbone_ramp_steps=25,
    )

    def poly(step: int) -> float:
        if step < 50:
            return (step + 1) / 50
        return (1 - (step - 50) / 950) ** 0.9

    scheduler.step(0)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4 * poly(0))
    assert optimizer.param_groups[1]['lr'] == 0.0
    scheduler.step(249)
    assert optimizer.param_groups[1]['lr'] == 0.0
    scheduler.step(250)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(5e-6 * poly(250) / 25)
    scheduler.step(274)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(5e-6 * poly(274))
    scheduler.step(500)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4 * poly(500))
    assert optimizer.param_groups[1]['lr'] == pytest.approx(5e-6 * poly(500))


def test_scheduler_defaults_preserve_group_ratios():
    from pumit.downstream.seg.trainer import _WarmupPolyLRScheduler

    scratch = torch.nn.Parameter(torch.zeros(1))
    trunk = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [
            {'params': [scratch], 'lr': 1e-4, 'name': 'scratch_decay'},
            {'params': [trunk], 'lr': 5e-6, 'name': 'backbone_layer_00_decay'},
        ]
    )
    scheduler = _WarmupPolyLRScheduler(optimizer, 1000, warmup_steps=50)

    for step in (0, 49, 500, 999):
        scheduler.step(step)
        ratio = optimizer.param_groups[1]['lr'] / optimizer.param_groups[0]['lr']
        assert ratio == pytest.approx(5e-6 / 1e-4)


def test_delayed_unfreeze_rejects_a_frozen_backbone():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = _FakeNetwork()
    trainer.num_epochs = 100
    trainer.weight_decay = 3e-5
    trainer.configuration_manager = SimpleNamespace(
        configuration={
            'optimization': {
                **_ORIGINAL_OPTIMIZATION,
                'freeze_backbone': True,
                'backbone_freeze_epochs': 25,
            }
        }
    )

    with pytest.raises(ValueError, match='mutually exclusive'):
        trainer.configure_optimizers()


def test_scheduler_resume_is_reconstructed_from_current_epoch():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = _FakeNetwork()
    trainer.num_epochs = 100
    trainer.weight_decay = 3e-5
    trainer.configuration_manager = SimpleNamespace(
        configuration={'optimization': _ORIGINAL_OPTIMIZATION}
    )
    optimizer, scheduler = trainer.configure_optimizers()
    scheduler.step(35)

    restored_trainer = object.__new__(DownstreamSegTrainer)
    restored_trainer.network = _FakeNetwork()
    restored_trainer.num_epochs = 100
    restored_trainer.weight_decay = 3e-5
    restored_trainer.configuration_manager = SimpleNamespace(
        configuration={'optimization': _ORIGINAL_OPTIMIZATION}
    )
    restored_optimizer, restored_scheduler = restored_trainer.configure_optimizers()
    restored_scheduler.step(35)

    assert [group['lr'] for group in restored_optimizer.param_groups] == pytest.approx(
        [group['lr'] for group in optimizer.param_groups]
    )


def test_optimizer_keeps_fused_adamw_when_loading_an_old_state():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = _FakeNetwork()
    trainer.num_epochs = 100
    trainer.configuration_manager = SimpleNamespace(
        configuration={'optimization': _ORIGINAL_OPTIMIZATION}
    )
    optimizer, _ = trainer.configure_optimizers()
    old_state = optimizer.state_dict()
    for group in old_state['param_groups']:
        group['fused'] = None

    optimizer.load_state_dict(old_state)

    assert all(group['fused'] is True for group in optimizer.param_groups)


def test_optimizer_reads_lr_and_weight_decay_groups_from_plan():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = _FakeNetwork()
    trainer.num_epochs = 100
    trainer.configuration_manager = SimpleNamespace(
        configuration={
            'optimization': {
                'optimizer': 'adamw',
                'learning_rate': 3e-4,
                'layer_decay': 0.5,
                'warmup_epochs': 10,
                'weight_decay': 0.05,
                'weight_decay_policy': 'vit_standard',
                'amsgrad': False,
                'lr_scheduler': 'poly',
                'poly_exponent': 0.9,
            }
        }
    )

    optimizer, scheduler = trainer.configure_optimizers()

    assert not optimizer.defaults['amsgrad']
    assert [group['name'] for group in optimizer.param_groups] == [
        'scratch_decay',
        'scratch_no_decay',
        'backbone_layer_00_decay',
        'backbone_layer_00_no_decay',
        'backbone_layer_01_decay',
        'backbone_layer_01_no_decay',
        'backbone_layer_02_decay',
        'backbone_layer_02_no_decay',
    ]
    assert scheduler.initial_lrs == pytest.approx(
        [3e-4, 3e-4, 7.5e-5, 7.5e-5, 1.5e-4, 1.5e-4, 3e-4, 3e-4]
    )
    assert [group['weight_decay'] for group in optimizer.param_groups] == [
        0.05,
        0.0,
        0.05,
        0.0,
        0.05,
        0.0,
        0.05,
        0.0,
    ]

    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {
        id(parameter)
        for parameter in trainer.network.parameters()
    }

    scheduler.step(50)
    factor = (1 - 40 / 90) ** 0.9
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx(
        [
            3e-4 * factor,
            3e-4 * factor,
            7.5e-5 * factor,
            7.5e-5 * factor,
            1.5e-4 * factor,
            1.5e-4 * factor,
            3e-4 * factor,
            3e-4 * factor,
        ]
    )


def test_optimizer_applies_separate_backbone_learning_rate_before_layer_decay():
    trainer = object.__new__(DownstreamSegTrainer)
    trainer.network = _FakeNetwork()
    trainer.num_epochs = 100
    trainer.configuration_manager = SimpleNamespace(
        configuration={
            'optimization': {
                **_ORIGINAL_OPTIMIZATION,
                'backbone_lr': 1e-4,
            }
        }
    )

    optimizer, scheduler = trainer.configure_optimizers()

    assert [group['name'] for group in optimizer.param_groups] == [
        'scratch_decay',
        'scratch_no_decay',
        'backbone_layer_00_decay',
        'backbone_layer_00_no_decay',
        'backbone_layer_01_decay',
        'backbone_layer_01_no_decay',
        'backbone_layer_02_decay',
        'backbone_layer_02_no_decay',
    ]
    assert scheduler.initial_lrs == pytest.approx(
        [3e-4, 3e-4, 2.5e-5, 2.5e-5, 5e-5, 5e-5, 1e-4, 1e-4]
    )
