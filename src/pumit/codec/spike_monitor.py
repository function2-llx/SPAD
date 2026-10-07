"""Spike detection monitor with sliding window of checkpoint states."""

from __future__ import annotations

import copy
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch


class SpikeMonitor:
    def __init__(
        self,
        *,
        monitor_after: int = 20,
        abs_threshold: float = 0.10,
        ema_factor: float = 3.0,
        cooldown: int = 100,
        window_size: int = 5,
    ):
        self.monitor_after = monitor_after
        self.abs_threshold = abs_threshold
        self.ema_factor = ema_factor
        self.cooldown = cooldown
        self.window_size = window_size

        self.ema: float | None = None
        self._cooldown_until: int = -1
        self.window: deque[dict] = deque(maxlen=window_size)
        self._bg_executor = ThreadPoolExecutor(max_workers=1)
        self._bg_future: Future | None = None

    def check(self, step_loss: float, optim_step: int) -> bool:
        """Check whether step_loss constitutes a spike.

        Returns True if a spike is detected (and not suppressed by cooldown).
        EMA is always updated after the trigger check.
        """
        if self.ema is None:
            self.ema = step_loss

        if optim_step <= self.monitor_after:
            self.ema = 0.95 * self.ema + 0.05 * step_loss
            return False

        trigger_d = step_loss >= self.abs_threshold
        trigger_b = step_loss >= self.ema_factor * self.ema
        fired = (trigger_d or trigger_b) and optim_step > self._cooldown_until

        self.ema = 0.95 * self.ema + 0.05 * step_loss

        if fired:
            self._cooldown_until = optim_step + self.cooldown

        return fired

    def push_state(self, state: dict):
        """Push a checkpoint state into the sliding window (background deepcopy)."""
        self._wait_bg()
        self._bg_future = self._bg_executor.submit(self._deepcopy_push, state)

    def _wait_bg(self):
        if self._bg_future is not None:
            self._bg_future.result()
            self._bg_future = None

    def _deepcopy_push(self, state: dict):
        copied = {
            'model': OrderedDict({k: v.cpu().clone() for k, v in state['model'].items()}),
            'ema': OrderedDict({k: v.cpu().clone() for k, v in state['ema'].items()}),
            'optimizer': copy.deepcopy(state['optimizer']),
            'scheduler': copy.deepcopy(state['scheduler']),
            'wandb_id': state.get('wandb_id'),
            'step': state['step'],
            'config': state.get('config', {}),
        }
        if 'scaler' in state:
            copied['scaler'] = copy.deepcopy(state['scaler'])
        self.window.append(copied)

    def dump_window(self, save_dir: Path) -> list[Path]:
        """Save all checkpoint states in the window to disk (parallel torch.save)."""
        self._wait_bg()
        if not self.window:
            return []
        ckpt_dir = save_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        entries = list(self.window)
        paths = [ckpt_dir / f'checkpoint-{e["step"]}.pt' for e in entries]
        with ThreadPoolExecutor(max_workers=len(entries)) as ex:
            futures = [ex.submit(torch.save, e, p) for e, p in zip(entries, paths)]
            for f in futures:
                f.result()
        return paths
