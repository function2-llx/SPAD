"""Unit tests for pumit.train_utils — run-dir artifact helpers."""
import os
import sys
from io import StringIO
from pathlib import Path

import pytest
import torch
import yaml

from pumit.train_utils import (
    _TeeStream,
    setup_logging,
    dump_config,
    setup_run_dir,
    save_checkpoint,
    find_latest_checkpoint,
)


# ---------------------------------------------------------------------------
# _TeeStream
# ---------------------------------------------------------------------------

class TestTeeStream:
    def test_writes_to_both(self):
        file = StringIO()
        console = StringIO()
        tee = _TeeStream(file, console)
        tee.write('hello')
        assert file.getvalue() == 'hello'
        assert console.getvalue() == 'hello'

    def test_flush_calls_both(self):
        file = StringIO()
        console = StringIO()
        tee = _TeeStream(file, console)
        tee.write('x')
        tee.flush()
        # No assertion needed — flush must not raise; the StringIO flush
        # is a no-op but _TeeStream must call both.

    def test_isatty_delegates_to_stream(self):
        file = StringIO()
        console = StringIO()
        tee = _TeeStream(file, console)
        assert tee.isatty() == console.isatty()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_stdout_appears_in_log_file(self, tmp_path: Path):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            setup_logging(tmp_path)
            print('hello from test', flush=True)
            log_content = (tmp_path / 'train.log').read_text()
            assert 'hello from test' in log_content
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def test_replaces_stdout_and_stderr(self, tmp_path: Path):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            setup_logging(tmp_path)
            assert isinstance(sys.stdout, _TeeStream)
            assert isinstance(sys.stderr, _TeeStream)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# dump_config
# ---------------------------------------------------------------------------

class TestDumpConfig:
    def test_writes_config_yaml(self, tmp_path: Path):
        cfg = {'lr': 1.5e-4, 'steps': 100000, 'seed': 42}
        dump_config(cfg, tmp_path)
        config_path = tmp_path / 'config.yaml'
        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text())
        assert loaded == cfg

    def test_overwrites_existing(self, tmp_path: Path):
        (tmp_path / 'config.yaml').write_text('old: data')
        dump_config({'new': 'value'}, tmp_path)
        loaded = yaml.safe_load((tmp_path / 'config.yaml').read_text())
        assert loaded == {'new': 'value'}


# ---------------------------------------------------------------------------
# setup_run_dir
# ---------------------------------------------------------------------------

class TestSetupRunDir:
    def test_creates_nested_dir(self, tmp_path: Path):
        base = tmp_path / 'outputs'
        run_dir = setup_run_dir(base, 'run-001')
        assert run_dir == base / 'run-001'
        assert run_dir.is_dir()

    def test_creates_latest_run_symlink(self, tmp_path: Path):
        base = tmp_path / 'outputs'
        run_dir = setup_run_dir(base, 'run-001')
        latest = base / 'latest-run'
        assert latest.is_symlink()
        assert latest.resolve() == run_dir.resolve()

    def test_relative_symlink_target(self, tmp_path: Path):
        """Symlink target is the bare directory name, not an absolute path."""
        base = tmp_path / 'outputs'
        setup_run_dir(base, 'run-001')
        latest = base / 'latest-run'
        # os.readlink gives the raw symlink target
        target = os.readlink(str(latest))
        assert target == 'run-001'

    def test_second_call_repoints_symlink(self, tmp_path: Path):
        base = tmp_path / 'outputs'
        run_a = setup_run_dir(base, 'run-A')
        run_b = setup_run_dir(base, 'run-B')
        latest = base / 'latest-run'
        # Must point at the NEW run, not the old one
        assert latest.resolve() == run_b.resolve()
        # Old run directory still exists
        assert run_a.is_dir()


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_writes_to_checkpoints_subdir(self, tmp_path: Path):
        state = {'step': 5000, 'model': {'a': torch.zeros(1)}}
        ckpt_path = save_checkpoint(state, tmp_path)
        expected = tmp_path / 'checkpoints' / 'checkpoint-5000.pt'
        assert ckpt_path == expected
        assert ckpt_path.exists()

    def test_creates_latest_symlink(self, tmp_path: Path):
        state = {'step': 5000, 'model': {'a': torch.zeros(1)}}
        save_checkpoint(state, tmp_path)
        latest = tmp_path / 'checkpoint-latest.pt'
        assert latest.is_symlink()
        assert latest.resolve() == (tmp_path / 'checkpoints' / 'checkpoint-5000.pt').resolve()

    def test_update_latest_repoints_symlink(self, tmp_path: Path):
        state_a = {'step': 1000, 'model': {}}
        state_b = {'step': 2000, 'model': {}}
        save_checkpoint(state_a, tmp_path)
        save_checkpoint(state_b, tmp_path)
        latest = tmp_path / 'checkpoint-latest.pt'
        assert latest.resolve() == (tmp_path / 'checkpoints' / 'checkpoint-2000.pt').resolve()

    def test_update_latest_false_skips_symlink(self, tmp_path: Path):
        state = {'step': 5000, 'model': {}}
        # First checkpoint — creates the symlink
        save_checkpoint(state, tmp_path, update_latest=True)
        # Second — don't update
        state2 = {'step': 6000, 'model': {}}
        save_checkpoint(state2, tmp_path, update_latest=False)
        latest = tmp_path / 'checkpoint-latest.pt'
        # Still points at step 5000
        assert latest.resolve() == (tmp_path / 'checkpoints' / 'checkpoint-5000.pt').resolve()

    def test_roundtrip_loadable(self, tmp_path: Path):
        original = {'step': 5000, 'model': {'a': torch.tensor([1.0, 2.0])}}
        ckpt_path = save_checkpoint(original, tmp_path)
        loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        assert loaded['step'] == original['step']
        assert torch.equal(loaded['model']['a'], original['model']['a'])

    def test_failed_checkpoint_write_preserves_latest(self, tmp_path: Path, monkeypatch):
        save_checkpoint({'step': 100, 'model': {}}, tmp_path)
        latest = tmp_path / 'checkpoint-latest.pt'
        old_target = latest.resolve()

        def fail_save(*args, **kwargs):
            raise OSError('disk full')

        monkeypatch.setattr(torch, 'save', fail_save)
        with pytest.raises(OSError, match='disk full'):
            save_checkpoint({'step': 200, 'model': {}}, tmp_path)

        assert latest.resolve() == old_target
        assert not (tmp_path / 'checkpoints' / 'checkpoint-200.pt').exists()
        assert not list((tmp_path / 'checkpoints').glob('.*.tmp'))

    def test_failed_latest_swap_preserves_old_pointer(self, tmp_path: Path, monkeypatch):
        import pumit.train_utils as train_utils

        save_checkpoint({'step': 100, 'model': {}}, tmp_path)
        latest = tmp_path / 'checkpoint-latest.pt'
        old_target = latest.resolve()
        real_replace = train_utils.os.replace

        def fail_latest_swap(src, dst):
            if Path(dst) == latest:
                raise OSError('latest swap failed')
            return real_replace(src, dst)

        monkeypatch.setattr(train_utils.os, 'replace', fail_latest_swap)
        with pytest.raises(OSError, match='latest swap failed'):
            save_checkpoint({'step': 200, 'model': {}}, tmp_path)

        assert latest.resolve() == old_target
        assert (tmp_path / 'checkpoints' / 'checkpoint-200.pt').exists()
        assert not list(tmp_path.glob('.checkpoint-latest.*.tmp'))


# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:
    def test_returns_none_when_no_run(self, tmp_path: Path):
        base = tmp_path / 'outputs'
        assert find_latest_checkpoint(base) is None

    def test_returns_absolute_path_when_present(self, tmp_path: Path):
        base = tmp_path / 'outputs'
        # Simulate a prior run: create the dir structure
        run_dir = base / 'my-run'
        ckpt_dir = run_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / 'checkpoint-5000.pt'
        torch.save({'step': 5000}, ckpt_path)

        # Symlinks matching codec layout
        latest_run = base / 'latest-run'
        latest_run.unlink(missing_ok=True)
        latest_run.symlink_to(run_dir.name)

        ckpt_latest = run_dir / 'checkpoint-latest.pt'
        ckpt_latest.unlink(missing_ok=True)
        ckpt_latest.symlink_to(Path('checkpoints') / 'checkpoint-5000.pt')

        result = find_latest_checkpoint(base)
        assert result is not None
        assert result.is_absolute()
        assert result == ckpt_path.resolve()

    def test_path_survives_symlink_repoint(self, tmp_path: Path):
        """After latest-run is repointed, a previously-resolved path stays valid."""
        base = tmp_path / 'outputs'
        # First run
        run_a = base / 'run-A'
        (run_a / 'checkpoints').mkdir(parents=True)
        torch.save({'step': 1000}, run_a / 'checkpoints' / 'checkpoint-1000.pt')
        (base / 'latest-run').unlink(missing_ok=True)
        (base / 'latest-run').symlink_to('run-A')
        (run_a / 'checkpoint-latest.pt').symlink_to(Path('checkpoints') / 'checkpoint-1000.pt')

        result = find_latest_checkpoint(base)
        resolved = str(result)

        # Repoint latest-run to a new run (simulating wandb.init)
        (base / 'latest-run').unlink()
        (base / 'latest-run').symlink_to('run-B')

        # The resolved path still exists because it's absolute
        assert Path(resolved).exists()
