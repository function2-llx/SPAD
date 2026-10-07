"""Tests for SpikeMonitor."""

from collections import OrderedDict

import torch

from pumit.codec.spike_monitor import SpikeMonitor


def _make_fake_state(step: int) -> dict:
    """Create a minimal fake checkpoint state."""
    return {
        'model': OrderedDict({'w': torch.randn(4, 4)}),
        'ema': OrderedDict({'w': torch.randn(4, 4)}),
        'optimizer': {'state': {0: {'exp_avg': torch.randn(4, 4), 'step': torch.tensor(step)}}, 'param_groups': []},
        'scheduler': {'last_epoch': step},
        'wandb_id': None,
        'step': step,
        'config': {},
    }


class TestSpikeMonitor:
    def test_no_trigger_below_threshold(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=0.10, ema_factor=3.0, cooldown=100)
        assert not mon.check(0.05, optim_step=1)

    def test_trigger_d_absolute(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=0.10, ema_factor=3.0, cooldown=100)
        assert mon.check(0.15, optim_step=1)

    def test_trigger_b_ema_relative(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100)
        # Feed 20 steps of low loss to stabilize EMA
        for i in range(1, 21):
            mon.check(0.04, optim_step=i)
        # EMA ~0.04, threshold = 3.0 * 0.04 = 0.12
        assert mon.check(0.13, optim_step=21)

    def test_monitor_after_gate(self):
        mon = SpikeMonitor(monitor_after=10, abs_threshold=0.10, ema_factor=3.0, cooldown=100)
        assert not mon.check(0.50, optim_step=5)  # before gate
        assert mon.check(0.50, optim_step=11)  # after gate

    def test_cooldown(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=0.10, ema_factor=3.0, cooldown=10)
        assert mon.check(0.50, optim_step=1)  # fires
        assert not mon.check(0.50, optim_step=5)  # in cooldown
        assert mon.check(0.50, optim_step=12)  # cooldown expired

    def test_ema_initialized_from_first_step(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100)
        mon.check(0.05, optim_step=1)
        assert abs(mon.ema - 0.05) < 1e-6

    def test_ema_updated_after_check(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100)
        mon.check(0.10, optim_step=1)  # EMA = 0.10
        # Check with 0.31: trigger check uses EMA=0.10, threshold=3.0*0.10=0.30, 0.31 >= 0.30 fires
        fired = mon.check(0.31, optim_step=2)
        assert fired
        # EMA updated AFTER check: 0.95 * 0.10 + 0.05 * 0.31 = 0.1105
        assert abs(mon.ema - 0.1105) < 1e-6

    def test_window_stores_deepcopies(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100, window_size=3)
        state = _make_fake_state(1)
        mon.push_state(state)
        mon._wait_bg()
        # Mutate the original
        state['model']['w'].fill_(999.0)
        # Window should have the old value
        assert mon.window[0]['model']['w'].max().item() != 999.0

    def test_window_circular(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100, window_size=3)
        for i in range(5):
            mon.push_state(_make_fake_state(i))
        mon._wait_bg()
        assert len(mon.window) == 3
        steps = [s['step'] for s in mon.window]
        assert steps == [2, 3, 4]

    def test_window_empty_on_init(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=0.10, ema_factor=3.0, cooldown=100)
        assert len(mon.window) == 0

    def test_window_carries_scaler_state(self):
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100, window_size=3)
        state = _make_fake_state(1)
        state['scaler'] = {'scale': 8192.0, '_growth_tracker': 3}
        mon.push_state(state)
        mon._wait_bg()
        assert mon.window[0]['scaler'] == {'scale': 8192.0, '_growth_tracker': 3}

    def test_window_scaler_optional(self):
        # A state without a 'scaler' key must still push without raising.
        mon = SpikeMonitor(monitor_after=0, abs_threshold=1.0, ema_factor=3.0, cooldown=100, window_size=3)
        mon.push_state(_make_fake_state(1))  # _make_fake_state has no 'scaler'
        mon._wait_bg()
        assert mon.window[0].get('scaler') is None
