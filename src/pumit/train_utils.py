"""Shared training utilities: logging, config dump, checkpoint save, auto-resume.

Config-agnostic — helpers take primitives / Path, not config objects.
Extracted from scripts/codec/train.py and scripts/ssl/train.py to stop
per-script duplication of run-dir artifact management.
"""
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import yaml


# ---------------------------------------------------------------------------
# _TeeStream — write to both a file and the original stream
# ---------------------------------------------------------------------------

class _TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def setup_logging(save_dir: Path) -> None:
    """Tee stdout/stderr to save_dir/train.log (append mode), preserving console.

    Call once on rank 0 after the run directory is finalized.
    """
    log_file = open(save_dir / 'train.log', 'a')
    sys.stdout = _TeeStream(log_file, sys.__stdout__)
    sys.stderr = _TeeStream(log_file, sys.__stderr__)


def dump_config(cfg_dict: dict, save_dir: Path) -> None:
    """Write save_dir/config.yaml from a plain dict of YAML-serializable values.

    The caller is responsible for ensuring cfg_dict contains only
    YAML-serializable values.  Non-serializable objects (file handles,
    callbacks, lambdas) will raise cryptic errors at yaml.dump time.
    """
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def setup_run_dir(base_save_dir: Path, run_name: str) -> Path:
    """Create base_save_dir/run_name and repoint base_save_dir/latest-run -> run_name.

    The run_name is expected to come from Path(wandb.run.dir).parent.name
    in the caller.  This is wandb's internal directory layout (not a
    documented API) — if wandb changes this convention in a future
    version, the run directory label may become misleading.  The coupling
    lives in the caller, not in this helper.
    """
    run_dir = base_save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_run = base_save_dir / 'latest-run'
    latest_run.unlink(missing_ok=True)
    latest_run.symlink_to(run_dir.name)  # relative symlink — portable
    return run_dir


def save_checkpoint(
    state: dict,
    save_dir: Path,
    *,
    update_latest: bool = True,
) -> Path:
    """Save state dict to save_dir/checkpoints/checkpoint-{state['step']}.pt.

    Requires state['step'] (int) — the filename is derived from it so the
    saved step value and the filename stay in agreement.  When
    update_latest is True (the default), refreshes the
    save_dir/checkpoint-latest.pt symlink to point at the new checkpoint.
    """
    ckpt_dir = save_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f'checkpoint-{state["step"]}.pt'
    ckpt_path = ckpt_dir / ckpt_name
    tmp_path = ckpt_dir / f'.{ckpt_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, ckpt_path)
        if update_latest:
            latest = save_dir / 'checkpoint-latest.pt'
            tmp_latest = save_dir / f'.checkpoint-latest.{os.getpid()}.{uuid.uuid4().hex}.tmp'
            try:
                tmp_latest.symlink_to(Path('checkpoints') / ckpt_name)
                os.replace(tmp_latest, latest)
            finally:
                tmp_latest.unlink(missing_ok=True)
    finally:
        tmp_path.unlink(missing_ok=True)
    return ckpt_path


def _snapshot_to_cpu(obj):
    """Deep-copy a state structure with tensors cloned to CPU.

    torch.save on live training state races the next optimizer step (GPU tensors
    mutate in place); the snapshot decouples serialization from training.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict):
        return {k: _snapshot_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_snapshot_to_cpu(v) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj


class BackgroundCheckpointer:
    """Two-tier checkpointing with the serialization off the training thread.

    Permanent saves are kept forever; rotating saves keep only the newest
    ``keep`` files. The tiers are decoupled: a step on both cadences is saved
    ONCE, as permanent. Rotation only ever deletes filenames recorded in its
    own history, so a permanent checkpoint can never be rotated away even if
    the cadences change across resumes.

    ``save()`` clones the state to CPU synchronously (the only part that must
    not race the next optimizer step), then hands the snapshot to a single
    writer thread that does torch.save + latest-symlink + rotation. File I/O
    releases the GIL, so the writer coexists with the training loop. If the
    previous write is still in flight, ``save()`` blocks until it finishes
    (write ~10s vs interval ~minutes: overlap only under pathological I/O).

    Crash-safety: the latest symlink is repointed only after the write
    completes, so a mid-write crash resumes from the previous good checkpoint.

    When ``permanent_every`` is provided, existing files are classified from
    their step numbers to restore the newest ``keep`` rotating entries after a
    restart. Recovery is read-only; older untracked files are left untouched.
    """

    def __init__(
        self,
        save_dir: Path,
        *,
        keep: int = 2,
        permanent_every: int | None = None,
    ):
        self.save_dir = save_dir
        self.keep = keep
        self._rotating: list[Path] = []  # oldest-first, including recovered saves
        self._permanent: set[Path] = set()
        if permanent_every is not None:
            if permanent_every <= 0:
                raise ValueError('permanent_every must be positive')
            checkpoint_dir = save_dir / 'checkpoints'
            existing = []
            for path in checkpoint_dir.glob('checkpoint-*.pt'):
                step_text = path.name.removeprefix('checkpoint-').removesuffix('.pt')
                if not step_text.isdigit():
                    raise ValueError(f'invalid checkpoint filename: {path.name}')
                existing.append((int(step_text), path))
            recovered_rotating = []
            for step, path in sorted(existing):
                if step % permanent_every == 0:
                    self._permanent.add(path)
                else:
                    recovered_rotating.append(path)
            if keep > 0:
                self._rotating = recovered_rotating[-keep:]
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._writer, daemon=True,
                                        name='ckpt-writer')
        self._thread.start()

    def save(self, state: dict, *, rotating: bool) -> None:
        """Snapshot state to CPU and enqueue the write. Blocks only on the
        snapshot (and on a still-running previous write)."""
        self.raise_if_failed()
        snapshot = _snapshot_to_cpu(state)
        self._queue.put((snapshot, rotating))  # maxsize=1: waits out an in-flight write

    def raise_if_failed(self) -> None:
        """Raise an asynchronous checkpoint error on the training thread."""
        if self._error is not None:
            raise RuntimeError('checkpoint writer failed') from self._error

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            snapshot, rotating = item
            try:
                if self._error is not None:
                    continue
                path = save_checkpoint(snapshot, self.save_dir)
                # Invariant: _rotating never contains a permanent path, enforced at
                # insertion (and by removal on promotion), so rotation may delete
                # its victims unconditionally.
                if not rotating:
                    self._permanent.add(path)
                    self._rotating = [p for p in self._rotating if p != path]
                elif path not in self._permanent:
                    self._rotating = [p for p in self._rotating if p != path]
                    self._rotating.append(path)
                    while len(self._rotating) > self.keep:
                        self._rotating.pop(0).unlink(missing_ok=True)
            except Exception as e:
                self._error = e
            finally:
                self._queue.task_done()

    def close(self) -> None:
        """Flush pending writes and stop the writer thread."""
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError('checkpoint writer failed') from self._error


def find_latest_checkpoint(base_save_dir: Path) -> Path | None:
    """Return the absolute resolved path to the latest checkpoint, or None.

    Resolves through base_save_dir/latest-run/checkpoint-latest.pt.
    The absolute resolution is required because wandb.init repoints the
    latest-run symlink to a fresh directory; a path that still traverses
    the symlink would dangle after wandb.init runs.
    """
    candidate = base_save_dir / 'latest-run' / 'checkpoint-latest.pt'
    if candidate.exists():
        return candidate.resolve()
    return None


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize DDP if available, return (rank, world_size, device).

    Guards on RANK in env so bare `python` (no torchrun) works single-GPU.
    Sets cuda:{rank % device_count} as the local device.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif 'RANK' in __import__('os').environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return rank, world_size, device
