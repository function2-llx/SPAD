"""Tests for BackgroundCheckpointer — two-tier background checkpointing."""
import threading
from pathlib import Path

import pytest
import torch

from pumit.train_utils import BackgroundCheckpointer, _snapshot_to_cpu, find_latest_checkpoint


def _state(step: int) -> dict:
    return {'model': {'w': torch.full((4,), float(step))}, 'step': step}


def _ckpts(save_dir: Path) -> set[str]:
    return {p.name for p in (save_dir / 'checkpoints').glob('checkpoint-*.pt')}


def test_snapshot_decouples_from_live_tensors():
    live = {'w': torch.zeros(3), 'nested': [torch.ones(2)], 'step': 1}
    snap = _snapshot_to_cpu(live)
    live['w'].add_(99)          # simulate the next optimizer step mutating in place
    live['nested'][0].add_(99)
    assert torch.equal(snap['w'], torch.zeros(3)), 'snapshot must not alias live tensors'
    assert torch.equal(snap['nested'][0], torch.ones(2))
    assert snap['step'] == 1


def test_rotation_keeps_newest_two(tmp_path: Path):
    c = BackgroundCheckpointer(tmp_path, keep=2)
    for step in (100, 200, 300, 400):
        c.save(_state(step), rotating=True)
    c.close()
    assert _ckpts(tmp_path) == {'checkpoint-300.pt', 'checkpoint-400.pt'}
    # latest symlink points at the newest write
    latest = (tmp_path / 'checkpoint-latest.pt').resolve()
    assert latest.name == 'checkpoint-400.pt'


def test_rotation_recovers_only_retained_history_without_cleanup(tmp_path: Path):
    import pumit.train_utils as tu

    for step in (100, 200, 300, 400, 500, 600, 700):
        tu.save_checkpoint(_state(step), tmp_path)
    before = _ckpts(tmp_path)

    c = BackgroundCheckpointer(tmp_path, keep=2, permanent_every=500)
    assert _ckpts(tmp_path) == before

    c.save(_state(800), rotating=True)
    c.close()
    assert _ckpts(tmp_path) == {
        'checkpoint-100.pt',
        'checkpoint-200.pt',
        'checkpoint-300.pt',
        'checkpoint-400.pt',
        'checkpoint-500.pt',
        'checkpoint-700.pt',
        'checkpoint-800.pt',
    }


def test_permanent_never_rotated(tmp_path: Path):
    c = BackgroundCheckpointer(tmp_path, keep=2)
    c.save(_state(500), rotating=True)
    c.save(_state(1000), rotating=False)   # permanent
    c.save(_state(1500), rotating=True)
    c.save(_state(2000), rotating=True)
    c.save(_state(2500), rotating=True)    # rotates 500 and 1500 out
    c.close()
    names = _ckpts(tmp_path)
    assert 'checkpoint-1000.pt' in names, 'permanent checkpoint was deleted by rotation'
    assert names == {'checkpoint-1000.pt', 'checkpoint-2000.pt', 'checkpoint-2500.pt'}


def test_same_path_saved_as_permanent_then_rotating_survives(tmp_path: Path):
    # Pathological cadence overlap across a resume: step saved permanent once,
    # later enqueued again as rotating. The permanent record must protect it.
    c = BackgroundCheckpointer(tmp_path, keep=1)
    c.save(_state(5000), rotating=False)
    c.save(_state(5000), rotating=True)    # same filename re-enters rotating history
    c.save(_state(5500), rotating=True)    # would rotate 5000 out
    c.close()
    assert 'checkpoint-5000.pt' in _ckpts(tmp_path)


def test_writer_error_surfaces_on_close(tmp_path: Path, monkeypatch):
    c = BackgroundCheckpointer(tmp_path, keep=2)
    import pumit.train_utils as tu
    def boom(*a, **k):
        raise OSError('disk full')
    monkeypatch.setattr(tu, 'save_checkpoint', boom)
    c.save(_state(100), rotating=True)
    with pytest.raises(RuntimeError, match='checkpoint writer failed'):
        c.close()


def test_writer_error_can_be_polled_between_saves(tmp_path: Path, monkeypatch):
    c = BackgroundCheckpointer(tmp_path, keep=2)
    import pumit.train_utils as tu

    def boom(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(tu, 'save_checkpoint', boom)
    c.save(_state(100), rotating=True)
    c._queue.join()
    with pytest.raises(RuntimeError, match='checkpoint writer failed'):
        c.raise_if_failed()
    with pytest.raises(RuntimeError, match='checkpoint writer failed'):
        c.close()


def test_close_does_not_deadlock_with_queued_item_after_writer_failure(tmp_path: Path, monkeypatch):
    c = BackgroundCheckpointer(tmp_path, keep=2)
    import pumit.train_utils as tu

    writer_started = threading.Event()
    fail_write = threading.Event()

    def boom(*args, **kwargs):
        writer_started.set()
        assert fail_write.wait(timeout=2)
        raise OSError('disk full')

    monkeypatch.setattr(tu, 'save_checkpoint', boom)
    c.save(_state(100), rotating=True)
    assert writer_started.wait(timeout=2)
    c.save(_state(200), rotating=True)
    fail_write.set()

    errors = []

    def close():
        try:
            c.close()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), 'close() deadlocked with an item queued behind the failed write'
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert 'checkpoint writer failed' in str(errors[0])


def test_resume_finds_rotating_checkpoint(tmp_path: Path):
    # find_latest_checkpoint resolves base/latest-run/checkpoint-latest.pt,
    # regardless of which tier wrote last (train.py layout: base/latest-run -> run dir).
    run_dir = tmp_path / 'run-xyz'
    run_dir.mkdir()
    (tmp_path / 'latest-run').symlink_to(run_dir.name)
    c = BackgroundCheckpointer(run_dir, keep=2)
    c.save(_state(5000), rotating=False)
    c.save(_state(5500), rotating=True)
    c.close()
    found = find_latest_checkpoint(tmp_path)
    assert found is not None and found.name == 'checkpoint-5500.pt'
