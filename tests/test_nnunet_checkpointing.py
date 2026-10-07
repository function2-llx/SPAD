from pathlib import Path

import numpy as np
import pytest
import torch

from pumit.nnunet.checkpointing import RetainPeriodicCheckpointsMixin


class _Logger:
    def __init__(self, ema_history):
        self.ema_history = ema_history

    def get_value(self, key, step):
        assert key == 'ema_fg_dice'
        assert step is None
        return self.ema_history


class _BaseTrainer:
    global_rank = 0
    local_rank = 0
    disable_checkpointing = False

    def __init__(self):
        self.current_epoch = 49
        self.saved_epochs = []
        self._best_ema = 0.5
        self.logger = _Logger([0.5])

    def save_checkpoint(self, filename):
        self.saved_epochs.append(self.current_epoch + 1)
        torch.save(
            {
                'current_epoch': self.current_epoch + 1,
                '_best_ema': self._best_ema,
                'logging': {'ema_fg_dice': self.logger.ema_history},
            },
            filename,
        )

    def load_checkpoint(self, filename_or_checkpoint):
        checkpoint = (
            torch.load(filename_or_checkpoint, weights_only=False)
            if isinstance(filename_or_checkpoint, str)
            else filename_or_checkpoint
        )
        self.current_epoch = checkpoint['current_epoch']
        self._best_ema = checkpoint['_best_ema']
        self.logger = _Logger(checkpoint['logging']['ema_fg_dice'])


def _finish_write(trainer):
    trainer._join_checkpoint_writer()


def _load(path):
    return torch.load(path, weights_only=False)


def test_periodic_checkpoint_is_retained(tmp_path):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    trainer = _Trainer()
    latest_path = tmp_path / 'checkpoint_latest.pth'
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)

    assert _load(latest_path)['current_epoch'] == 50
    assert _load(tmp_path / 'checkpoint_epoch_0050.pth')['current_epoch'] == 50


def test_checkpoint_payload_extension_runs_before_snapshot(tmp_path):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

        def _augment_checkpoint_payload(self, payload):
            payload['extra_state'] = torch.tensor(3.0)
            return payload

    trainer = _Trainer()
    latest_path = tmp_path / 'checkpoint_latest.pth'
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)

    assert _load(latest_path)['extra_state'].item() == 3.0


def test_repeated_save_atomically_replaces_periodic_checkpoint(tmp_path):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    latest_path = tmp_path / 'checkpoint_latest.pth'
    retained_path = tmp_path / 'checkpoint_epoch_0050.pth'
    torch.save({'current_epoch': -1}, latest_path)
    torch.save({'current_epoch': -1}, retained_path)

    trainer = _Trainer()
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)

    assert _load(latest_path)['current_epoch'] == 50
    assert _load(retained_path)['current_epoch'] == 50


def test_latest_checkpoint_rotates_at_configured_cadence(tmp_path, monkeypatch):
    monkeypatch.setenv('PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY', '5')

    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    trainer = _Trainer()
    latest_path = tmp_path / 'checkpoint_latest.pth'
    assert trainer.save_every == 5

    trainer.current_epoch = 3
    trainer.save_checkpoint(str(latest_path))
    assert not latest_path.exists()

    trainer.current_epoch = 4
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)
    assert _load(latest_path)['current_epoch'] == 5
    assert not (tmp_path / 'checkpoint_epoch_0005.pth').exists()

    trainer.current_epoch = 9
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)
    assert _load(latest_path)['current_epoch'] == 10
    assert list(tmp_path.glob('checkpoint_latest*.pth')) == [latest_path]


def test_nnunet_epoch_end_dispatches_both_checkpoint_tiers(tmp_path):
    class _EpochEndBaseTrainer(_BaseTrainer):
        def __init__(self):
            super().__init__()
            self.current_epoch = 0
            self.save_every = 50

        def on_epoch_end(self):
            if (self.current_epoch + 1) % self.save_every == 0:
                self.save_checkpoint(str(tmp_path / 'checkpoint_latest.pth'))
            self.current_epoch += 1

    class _Trainer(RetainPeriodicCheckpointsMixin, _EpochEndBaseTrainer):
        output_folder = str(tmp_path)

    trainer = _Trainer()
    for _ in range(50):
        trainer.on_epoch_end()
    _finish_write(trainer)

    assert trainer.saved_epochs == list(range(5, 51, 5))
    assert _load(tmp_path / 'checkpoint_latest.pth')['current_epoch'] == 50
    assert _load(tmp_path / 'checkpoint_epoch_0050.pth')['current_epoch'] == 50


def test_latest_and_retained_cadences_need_not_divide(tmp_path, monkeypatch):
    monkeypatch.setenv('PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY', '7')

    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    trainer = _Trainer()
    latest_path = tmp_path / 'checkpoint_latest.pth'
    assert trainer.save_every == 1

    trainer.current_epoch = 6
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)
    assert latest_path.exists()

    trainer.current_epoch = 49
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)
    assert (tmp_path / 'checkpoint_epoch_0050.pth').exists()


@pytest.mark.parametrize('value', ['0', '-1', 'invalid'])
def test_latest_checkpoint_interval_must_be_positive(value, monkeypatch):
    monkeypatch.setenv('PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY', value)

    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        pass

    with pytest.raises(ValueError, match='must be a positive integer'):
        _Trainer()


def test_interrupted_checkpoint_write_preserves_previous_latest(tmp_path, monkeypatch):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    latest_path = tmp_path / 'checkpoint_latest.pth'
    torch.save({'current_epoch': -1}, latest_path)
    original_save = torch.save

    def interrupted_save(payload, filename, *args, **kwargs):
        Path(filename).write_bytes(b'incomplete')
        raise OSError('interrupted write')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    trainer = _Trainer()
    trainer.save_checkpoint(str(latest_path))
    with pytest.raises(OSError, match='interrupted write'):
        _finish_write(trainer)
    monkeypatch.setattr(torch, 'save', original_save)

    assert _load(latest_path)['current_epoch'] == -1
    assert not (tmp_path / '.checkpoint_latest.pth.tmp').exists()


def test_interrupted_retained_link_preserves_previous_files(tmp_path, monkeypatch):
    import pumit.nnunet.checkpointing as checkpointing

    def interrupted_link(source, destination):
        raise OSError('interrupted link')

    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    latest_path = tmp_path / 'checkpoint_latest.pth'
    retained_path = tmp_path / 'checkpoint_epoch_0050.pth'
    torch.save({'current_epoch': -1}, latest_path)
    torch.save({'current_epoch': -2}, retained_path)
    monkeypatch.setattr(checkpointing.os, 'link', interrupted_link)

    trainer = _Trainer()
    trainer.save_checkpoint(str(latest_path))
    with pytest.raises(OSError, match='interrupted link'):
        _finish_write(trainer)

    assert _load(latest_path)['current_epoch'] == -1
    assert _load(retained_path)['current_epoch'] == -2
    assert not (tmp_path / '.checkpoint_epoch_0050.pth.tmp').exists()


@pytest.mark.parametrize('ema_history', [[0.6, 0.9, 0.8], np.array([0.6, 0.9, 0.8])])
def test_latest_payload_best_ema_includes_complete_logger_history(tmp_path, ema_history):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        output_folder = str(tmp_path)

    trainer = _Trainer()
    trainer.current_epoch = 4
    trainer._best_ema = 0.6
    trainer.logger = _Logger(ema_history)
    latest_path = tmp_path / 'checkpoint_latest.pth'
    trainer.save_checkpoint(str(latest_path))
    _finish_write(trainer)

    assert _load(latest_path)['_best_ema'] == pytest.approx(0.9)


@pytest.mark.parametrize('ema_history', [[0.6, 0.9, 0.8], np.array([0.6, 0.9, 0.8])])
def test_resume_rebuilds_stale_best_ema_from_logger_history(ema_history):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        pass

    trainer = _Trainer()
    trainer.load_checkpoint(
        {
            'current_epoch': 5,
            '_best_ema': 0.6,
            'logging': {'ema_fg_dice': ema_history},
        }
    )

    assert trainer._best_ema == pytest.approx(0.9)


@pytest.mark.parametrize(
    ('ema_history', 'error', 'message'),
    [
        ((0.5, 0.6), TypeError, 'must be a list or numpy.ndarray'),
        (np.array([[0.5, 0.6]]), ValueError, 'must be one-dimensional'),
        (['invalid'], TypeError, 'must contain numeric values'),
    ],
)
def test_resume_rejects_invalid_ema_history_schema(ema_history, error, message):
    class _Trainer(RetainPeriodicCheckpointsMixin, _BaseTrainer):
        pass

    trainer = _Trainer()
    with pytest.raises(error, match=message):
        trainer.load_checkpoint(
            {
                'current_epoch': 5,
                '_best_ema': 0.6,
                'logging': {'ema_fg_dice': ema_history},
            }
        )
