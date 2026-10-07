"""End-to-end test: SpikeMonitor window dump produces valid checkpoint files."""

import tempfile
from collections import OrderedDict
from pathlib import Path

import torch

from pumit.codec.spike_monitor import SpikeMonitor


def test_dump_produces_loadable_checkpoints():
    mon = SpikeMonitor(monitor_after=0, abs_threshold=0.10, ema_factor=3.0, cooldown=100, window_size=5)

    # Simulate 5 steps of normal training
    for i in range(1, 6):
        state = {
            'model': OrderedDict({'w': torch.randn(32, 32)}),
            'ema': OrderedDict({'w': torch.randn(32, 32)}),
            'optimizer': {'state': {}, 'param_groups': [{'lr': 1e-4}]},
            'scheduler': {'last_epoch': i},
            'wandb_id': None,
            'step': i,
            'config': {'lr': 1e-4},
        }
        mon.push_state(state)
        mon.check(0.04, optim_step=i)

    # Step 6: spike
    assert mon.check(0.50, optim_step=6)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = mon.dump_window(Path(tmpdir))
        assert len(paths) == 5
        for p in paths:
            assert p.exists()
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            assert 'model' in ckpt
            assert 'step' in ckpt
        steps = [torch.load(p, map_location='cpu', weights_only=False)['step'] for p in paths]
        assert steps == [1, 2, 3, 4, 5]


def test_dump_partial_window_on_early_spike():
    mon = SpikeMonitor(monitor_after=0, abs_threshold=0.10, ema_factor=3.0, cooldown=100, window_size=5)

    # Only 2 steps before spike
    for i in range(1, 3):
        mon.push_state({
            'model': OrderedDict({'w': torch.randn(8, 8)}),
            'ema': OrderedDict({'w': torch.randn(8, 8)}),
            'optimizer': {'state': {}, 'param_groups': []},
            'scheduler': {'last_epoch': i},
            'wandb_id': None,
            'step': i,
            'config': {},
        })
        mon.check(0.04, optim_step=i)

    assert mon.check(0.50, optim_step=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = mon.dump_window(Path(tmpdir))
        assert len(paths) == 2  # only 2 in window
