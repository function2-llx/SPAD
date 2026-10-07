"""Classification checkpoint persistence and random-stream restoration contracts."""

import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from pumit.downstream.cls.checkpointing import (
    capture_rng_state,
    cpu_snapshot,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def test_rng_roundtrip_through_weights_only_checkpoint(tmp_path):
    generator = torch.Generator().manual_seed(17)
    device = torch.device('cpu')
    original = capture_rng_state(generator, device)
    try:
        random.seed(23)
        np.random.seed(24)
        torch.manual_seed(25)
        np.random.normal()
        path = tmp_path / 'latest.pt'
        save_checkpoint(path, {'rng': capture_rng_state(generator, device)})
        expected = (
            random.random(), np.random.normal(size=8), torch.rand(8),
            torch.randperm(19, generator=generator),
        )
        restore_rng_state(load_checkpoint(path)['rng'], generator, device)
        assert random.random() == expected[0]
        np.testing.assert_array_equal(np.random.normal(size=8), expected[1])
        torch.testing.assert_close(torch.rand(8), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(
            torch.randperm(19, generator=generator), expected[3], rtol=0, atol=0,
        )
    finally:
        restore_rng_state(original, generator, device)


def test_cpu_rng_does_not_touch_cuda(monkeypatch):
    get_cuda = Mock(side_effect=AssertionError('CPU checkpoint must not access CUDA'))
    set_cuda = Mock(side_effect=AssertionError('CPU checkpoint must not access CUDA'))
    monkeypatch.setattr(torch.cuda, 'get_rng_state', get_cuda)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', set_cuda)
    generator = torch.Generator()
    state = capture_rng_state(generator, torch.device('cpu'))
    restore_rng_state(state, generator, torch.device('cpu'))
    assert 'cuda' not in state
    get_cuda.assert_not_called()
    set_cuda.assert_not_called()


def test_cuda_rng_only_accesses_requested_rank(monkeypatch):
    cuda_state = torch.tensor([1, 2], dtype=torch.uint8)
    get_cuda = Mock(return_value=cuda_state)
    set_cuda = Mock()
    monkeypatch.setattr(torch.cuda, 'get_rng_state', get_cuda)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', set_cuda)
    generator = torch.Generator()
    device = torch.device('cuda:2')
    state = capture_rng_state(generator, device)
    restore_rng_state(state, generator, device)
    get_cuda.assert_called_once_with(device)
    set_cuda.assert_called_once_with(cuda_state, device)


def test_atomic_save_replaces_latest(tmp_path):
    path = tmp_path / 'latest.pt'
    save_checkpoint(path, {'step': 10})
    save_checkpoint(path, {'step': 20})
    assert load_checkpoint(path) == {'step': 20}
    assert not path.with_suffix('.pt.tmp').exists()


def test_failed_serialization_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'latest.pt'
    save_checkpoint(path, {'step': 10})

    def interrupted_save(payload, destination):
        destination.write_bytes(b'incomplete')
        raise OSError('interrupted write')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='interrupted write'):
        save_checkpoint(path, {'step': 20})
    assert load_checkpoint(path) == {'step': 10}


def test_corrupt_load_does_not_fall_back_to_temporary_file(tmp_path):
    path = tmp_path / 'latest.pt'
    save_checkpoint(path, {'step': 10})
    path.rename(path.with_suffix('.pt.tmp'))
    path.write_bytes(b'corrupt checkpoint')
    with pytest.raises(Exception):
        load_checkpoint(path)


def test_missing_checkpoint_is_visible(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / 'missing.pt')


def test_cpu_snapshot_detaches_and_copies_nested_tensors():
    tensor = torch.tensor([2.0], requires_grad=True)
    value = {'model': tensor, 'optimizer': [{'moments': (tensor, 3)}]}
    snapshot = cpu_snapshot(value)
    with torch.no_grad():
        tensor.add_(7)
    assert snapshot['model'].item() == 2.0
    assert snapshot['optimizer'][0]['moments'][0].item() == 2.0
    assert snapshot['optimizer'][0]['moments'][1] == 3
    assert snapshot['model'].device.type == 'cpu'
    assert not snapshot['model'].requires_grad
