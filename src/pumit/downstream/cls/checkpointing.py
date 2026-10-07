"""Atomic latest checkpoints and per-rank random state for classification training."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state(generator: torch.Generator, device: torch.device) -> dict:
    """Capture only this rank's CUDA generator alongside CPU and sampler randomness."""
    device = torch.device(device)
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].astype(np.int64)),
            *numpy_state[2:],
        ),
        'torch': torch.get_rng_state(),
        'shuffle_generator': generator.get_state(),
    }
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict, generator: torch.Generator, device: torch.device) -> None:
    """Restore random streams after model construction and checkpoint loading."""
    device = torch.device(device)
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state(
        (numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]),
    )
    torch.set_rng_state(state['torch'])
    generator.set_state(state['shuffle_generator'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def cpu_snapshot(value):
    """Copy tensor state so later optimizer updates cannot mutate a saved snapshot."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Replace the latest checkpoint only after its complete serialization succeeds."""
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path: str | Path) -> dict:
    """Load CPU state without concealing missing or corrupt checkpoint errors."""
    return torch.load(path, map_location='cpu', weights_only=True)
