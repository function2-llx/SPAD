import os
from pathlib import Path

import pytest
import torch
import yaml

from pumit.ucpt.train.state import RunLock, prepare_run


def _config(save_dir: Path) -> dict:
    return {
        'save_dir': str(save_dir),
        'steps': 100,
        'betas': (0.9, 0.999),
        'wandb_entity': None,
        'wandb_project': 'pumit-ucpt',
    }


def _commit_checkpoint(save_dir: Path, step: int) -> Path:
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / f'checkpoint-{step}.pt'
    torch.save({'step': step}, checkpoint)
    (save_dir / 'checkpoint-latest.pt').symlink_to(Path('checkpoints') / checkpoint.name)
    return checkpoint


def test_fresh_run_writes_canonical_manifest(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        state = prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
    finally:
        lock.close()

    assert state.new_run
    assert state.wandb_resume == 'never'
    assert state.checkpoint is None
    assert state.checkpoint_step == 0
    raw = yaml.safe_load((save_dir / 'run.yaml').read_text())
    assert raw['wandb_id'] == 'wandb123'
    assert raw['world_size'] == 8
    assert raw['config']['betas'] == [0.9, 0.999]
    assert raw['run_id']


def test_existing_manifest_without_checkpoint_restarts_at_zero(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        first = prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
        second = prepare_run(save_dir, _config(save_dir), 8, wandb_id='unused')
    finally:
        lock.close()

    assert not second.new_run
    assert second.manifest == first.manifest
    assert second.checkpoint is None
    assert second.checkpoint_step == 0
    assert second.wandb_resume == 'allow'


def test_existing_manifest_resumes_only_committed_latest(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
        committed = _commit_checkpoint(save_dir, 40)
        torch.save({'step': 60}, save_dir / 'checkpoints' / 'checkpoint-60.pt')
        state = prepare_run(save_dir, _config(save_dir), 8, wandb_id='unused')
    finally:
        lock.close()

    assert state.checkpoint == committed.resolve()
    assert state.checkpoint_step == 40
    assert state.wandb_resume == 'must'


def test_existing_manifest_rejects_config_mismatch(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
        changed = _config(save_dir)
        changed['steps'] = 101
        with pytest.raises(ValueError, match=r"run config mismatch.*steps"):
            prepare_run(save_dir, changed, 8, wandb_id='unused')
    finally:
        lock.close()


def test_existing_manifest_allows_world_size_change(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        first = prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
        resumed = prepare_run(save_dir, _config(save_dir), 4, wandb_id='unused')
    finally:
        lock.close()

    assert resumed.manifest == first.manifest
    assert resumed.manifest.world_size == 8
    assert not resumed.new_run


def test_nonempty_legacy_directory_is_not_guessed(tmp_path: Path):
    save_dir = tmp_path / 'run'
    save_dir.mkdir()
    (save_dir / 'latest-run').symlink_to('run-old')
    lock = RunLock.acquire(save_dir)
    try:
        with pytest.raises(RuntimeError, match='has no run.yaml but is not empty'):
            prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
    finally:
        lock.close()


def test_uncommitted_manifest_temp_does_not_define_a_run(tmp_path: Path):
    save_dir = tmp_path / 'run'
    save_dir.mkdir()
    (save_dir / '.run.yaml.123.deadbeef.tmp').write_text('partial')
    lock = RunLock.acquire(save_dir)
    try:
        state = prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
    finally:
        lock.close()

    assert state.new_run
    assert (save_dir / 'run.yaml').exists()


def test_broken_latest_is_fatal(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
        (save_dir / 'checkpoint-latest.pt').symlink_to('checkpoints/checkpoint-40.pt')
        with pytest.raises(FileNotFoundError):
            prepare_run(save_dir, _config(save_dir), 8, wandb_id='unused')
    finally:
        lock.close()


def test_latest_must_target_direct_checkpoint_child(tmp_path: Path):
    save_dir = tmp_path / 'run'
    lock = RunLock.acquire(save_dir)
    try:
        prepare_run(save_dir, _config(save_dir), 8, wandb_id='wandb123')
        outside = tmp_path / 'checkpoint-40.pt'
        torch.save({'step': 40}, outside)
        (save_dir / 'checkpoint-latest.pt').symlink_to(Path('..') / outside.name)
        with pytest.raises(ValueError, match='invalid latest checkpoint target'):
            prepare_run(save_dir, _config(save_dir), 8, wandb_id='unused')
    finally:
        lock.close()


def test_run_lock_rejects_concurrent_owner(tmp_path: Path):
    save_dir = tmp_path / 'run'
    first = RunLock.acquire(save_dir)
    try:
        with pytest.raises(RuntimeError, match='another process'):
            RunLock.acquire(save_dir)
    finally:
        first.close()

    second = RunLock.acquire(save_dir)
    second.close()
    assert os.path.exists(save_dir / '.run.lock')
