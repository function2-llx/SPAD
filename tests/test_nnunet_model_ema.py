import pytest
import torch

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin
from pumit.nnunet.model_ema import ModelEmaMixin, validate_model_ema_decay


class _BaseTrainer:
    global_rank = 0
    device = torch.device('cpu')
    disable_checkpointing = False

    def __init__(self):
        self.network = torch.nn.Linear(2, 1)
        self.was_initialized = False

    def initialize(self):
        self.was_initialized = True

    def load_checkpoint(self, filename_or_checkpoint):
        if not self.was_initialized:
            self.initialize()
        checkpoint = (
            torch.load(filename_or_checkpoint, weights_only=False)
            if isinstance(filename_or_checkpoint, str)
            else filename_or_checkpoint
        )
        self.network.load_state_dict(checkpoint['network_weights'])

    def save_checkpoint(self, filename):
        torch.save({'network_weights': self.network.state_dict()}, filename)


class _Trainer(ModelEmaMixin, _BaseTrainer):
    model_ema_decay = 0.5


@pytest.mark.parametrize('decay', [True, '0.9'])
def test_model_ema_decay_requires_a_number(decay):
    with pytest.raises(TypeError, match='must be a number'):
        validate_model_ema_decay(decay)


@pytest.mark.parametrize('decay', [-0.1, 0, 1, 1.1])
def test_model_ema_decay_requires_open_unit_interval(decay):
    with pytest.raises(ValueError, match='between 0 and 1'):
        validate_model_ema_decay(decay)


def test_rank_zero_ema_updates_and_temporarily_replaces_live_weights():
    trainer = _Trainer()
    trainer.initialize()

    trainer.network.weight.data.fill_(1)
    trainer.network.bias.data.fill_(1)
    trainer._update_model_ema()
    trainer.network.weight.data.fill_(3)
    trainer.network.bias.data.fill_(3)
    trainer._update_model_ema()

    assert trainer._model_ema_num_updates == 2
    assert torch.equal(
        trainer._model_ema_state['weight'],
        torch.full_like(trainer.network.weight, 2),
    )

    trainer.network.weight.data.fill_(7)
    trainer.network.bias.data.fill_(7)
    with trainer.model_ema_weights():
        assert torch.equal(
            trainer.network.weight,
            torch.full_like(trainer.network.weight, 2),
        )
        assert torch.equal(
            trainer.network.bias,
            torch.full_like(trainer.network.bias, 2),
        )
    assert torch.equal(
        trainer.network.weight,
        torch.full_like(trainer.network.weight, 7),
    )
    assert torch.equal(
        trainer.network.bias,
        torch.full_like(trainer.network.bias, 7),
    )


def test_non_owner_uses_broadcast_ema_without_a_local_shadow(monkeypatch):
    trainer = _Trainer()
    trainer.global_rank = 1
    trainer.initialize()
    trainer.network.weight.data.fill_(7)
    trainer.network.bias.data.fill_(7)
    assert trainer._model_ema_state is None
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        'broadcast',
        lambda tensor, src: tensor.fill_(2),
    )

    with trainer.model_ema_weights():
        assert torch.equal(
            trainer.network.weight,
            torch.full_like(trainer.network.weight, 2),
        )
        assert torch.equal(
            trainer.network.bias,
            torch.full_like(trainer.network.bias, 2),
        )
    assert torch.equal(
        trainer.network.weight,
        torch.full_like(trainer.network.weight, 7),
    )
    assert torch.equal(
        trainer.network.bias,
        torch.full_like(trainer.network.bias, 7),
    )


def test_model_ema_checkpoint_round_trip(tmp_path):
    trainer = _Trainer()
    trainer.initialize()
    trainer.network.weight.data.fill_(4)
    trainer.network.bias.data.fill_(5)
    trainer._update_model_ema()
    payload = trainer._augment_checkpoint_payload(
        {
            'network_weights': trainer.network.state_dict(),
        }
    )
    checkpoint_path = tmp_path / 'checkpoint.pth'
    torch.save(payload, checkpoint_path)

    restored = _Trainer()
    restored.load_checkpoint(str(checkpoint_path))

    assert restored._model_ema_num_updates == 1
    assert restored.model_ema_decay == pytest.approx(0.5)
    assert torch.equal(
        restored._model_ema_state['weight'],
        torch.full_like(restored.network.weight, 4),
    )
    assert torch.equal(
        restored._model_ema_state['bias'],
        torch.full_like(restored.network.bias, 5),
    )


def test_retained_checkpoint_contains_model_ema(tmp_path):
    class _CheckpointTrainer(
        RetainPeriodicCheckpointsMixin,
        ModelEmaMixin,
        _BaseTrainer,
    ):
        model_ema_decay = 0.5
        output_folder = str(tmp_path)

        def __init__(self):
            super().__init__()
            self.current_epoch = 0

    trainer = _CheckpointTrainer()
    trainer.initialize()
    trainer.network.weight.data.fill_(2)
    trainer._update_model_ema()
    checkpoint_path = tmp_path / 'checkpoint_final.pth'
    trainer.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    assert checkpoint['model_ema_decay'] == pytest.approx(0.5)
    assert checkpoint['model_ema_num_updates'] == 1
    assert torch.equal(
        checkpoint['model_ema_state_dict']['weight'],
        torch.full_like(trainer.network.weight, 2),
    )


def test_ema_enabled_trainer_rejects_checkpoint_without_ema():
    trainer = _Trainer()
    checkpoint = {'network_weights': trainer.network.state_dict()}

    with pytest.raises(ValueError, match='checkpoint missing'):
        trainer.load_checkpoint(checkpoint)
