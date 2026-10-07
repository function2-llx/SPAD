"""Shared nnU-Net checkpoint lifecycle extensions."""

import copy
import math
import os
import threading
from numbers import Real
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# Rotating checkpoints are written off the training thread; the final one stays synchronous so it is
# on disk before nnU-Net removes checkpoint_latest.pth.
_BACKGROUND_CHECKPOINTS = frozenset({'checkpoint_latest.pth', 'checkpoint_best.pth'})


def _global_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _snapshot(value):
    """Copy a checkpoint payload off live training state so a writer thread sees one consistent epoch."""
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _maximum_logged_ema(ema_history: list | np.ndarray) -> Real | None:
    """Return the best logged EMA while enforcing nnU-Net's checkpoint schema."""
    if not isinstance(ema_history, (list, np.ndarray)):
        raise TypeError(
            "nnU-Net logging['ema_fg_dice'] must be a list or numpy.ndarray, "
            f'got {type(ema_history).__name__}'
        )
    values = np.asarray(ema_history)
    if values.ndim != 1:
        raise ValueError(
            "nnU-Net logging['ema_fg_dice'] must be one-dimensional, "
            f'got shape {values.shape}'
        )
    if values.size == 0:
        return None
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(
            "nnU-Net logging['ema_fg_dice'] must contain numeric values, "
            f'got dtype {values.dtype}'
        )
    return values.max().item()


def _merge_best_ema(best_ema: Real | None, ema_history: list | np.ndarray) -> Real | None:
    logged_best = _maximum_logged_ema(ema_history)
    if logged_best is None:
        return best_ema
    if best_ema is None:
        return logged_best
    if not isinstance(best_ema, Real):
        raise TypeError(
            "nnU-Net checkpoint['_best_ema'] must be numeric or None, "
            f'got {type(best_ema).__name__}'
        )
    return max(best_ema, logged_best)


class RetainPeriodicCheckpointsMixin:
    """Keep sparse permanent checkpoints alongside one frequently rotating latest file.

    The latest interval defaults to five epochs and can be set through
    ``PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY``. Retained checkpoints remain on
    the native 50-epoch cadence.

    Serializing a multi-GB checkpoint stalls the training loop for seconds, so the rotating
    checkpoints are snapshotted to host memory and written by a background thread. At most one write
    is in flight; the next save and any failure both surface on the training thread.
    """

    latest_checkpoint_every = 5
    retained_checkpoint_every = 50
    latest_checkpoint_every_env = 'PUMIT_NNUNET_LATEST_CHECKPOINT_EVERY'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        configured_latest_every = os.environ.get(
            self.latest_checkpoint_every_env,
            str(self.latest_checkpoint_every),
        )
        try:
            self.latest_checkpoint_every = int(configured_latest_every)
        except ValueError as error:
            raise ValueError(
                f'{self.latest_checkpoint_every_env} must be a positive integer, '
                f'got {configured_latest_every!r}'
            ) from error
        if self.latest_checkpoint_every <= 0:
            raise ValueError(
                f'{self.latest_checkpoint_every_env} must be a positive integer, '
                f'got {configured_latest_every!r}'
            )
        if self.retained_checkpoint_every <= 0:
            raise ValueError('retained_checkpoint_every must be positive')

        # nnU-Net owns the epoch-end hook and calls save_checkpoint on this cadence.
        # The save method below filters that cadence into the two independent tiers.
        self.save_every = math.gcd(
            self.latest_checkpoint_every,
            self.retained_checkpoint_every,
        )
        self._checkpoint_writer: threading.Thread | None = None
        self._checkpoint_error: BaseException | None = None

    def save_checkpoint(self, filename: str) -> None:
        checkpoint_path = Path(filename)
        is_latest = checkpoint_path.name == 'checkpoint_latest.pth'
        completed_epochs = self.current_epoch + 1
        latest_due = is_latest and completed_epochs % self.latest_checkpoint_every == 0
        retained_due = is_latest and completed_epochs % self.retained_checkpoint_every == 0

        if is_latest and not (latest_due or retained_due):
            return
        # A pending write owns the staging path and would be caught by the torch.save patch below.
        self._join_checkpoint_writer()
        if _global_rank() != 0 or self.disable_checkpointing:
            super().save_checkpoint(filename)
            return

        retained_path = (
            Path(self.output_folder) / f'checkpoint_epoch_{completed_epochs:04d}.pth'
            if retained_due
            else None
        )
        payload = self._capture_checkpoint(filename)
        if is_latest:
            # nnU-Net saves latest before updating its live best value, although logging already
            # contains the current epoch. Keep the captured pair internally consistent.
            if not isinstance(payload, dict):
                raise TypeError(f'nnU-Net checkpoint payload must be a dict, got {type(payload).__name__}')
            payload['_best_ema'] = _merge_best_ema(
                payload['_best_ema'],
                payload['logging']['ema_fg_dice'],
            )
        if checkpoint_path.name not in _BACKGROUND_CHECKPOINTS:
            self._write_checkpoint(payload, checkpoint_path, retained_path)
            return
        self._checkpoint_writer = threading.Thread(
            target=self._write_checkpoint_guarded,
            args=(payload, checkpoint_path, retained_path),
        )
        self._checkpoint_writer.start()

    def _capture_checkpoint(self, filename: str) -> object:
        """Build the payload through nnU-Net's own construction, intercepting the write to snapshot it."""
        captured = {}
        original_save = torch.save
        torch.save = lambda obj, *args, **kwargs: captured.setdefault('payload', obj)
        try:
            super().save_checkpoint(filename)
        finally:
            torch.save = original_save
        if 'payload' not in captured:
            raise RuntimeError(f'nnU-Net built no checkpoint payload for {filename}')
        payload = captured['payload']
        augment_payload = getattr(self, '_augment_checkpoint_payload', None)
        if augment_payload is not None:
            payload = augment_payload(payload)
        return _snapshot(payload)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Restore a checkpoint and reconcile stale best EMA state with its logger history."""
        super().load_checkpoint(filename_or_checkpoint)
        self._best_ema = _merge_best_ema(
            self._best_ema,
            self.logger.get_value('ema_fg_dice', step=None),
        )

    def _write_checkpoint(
        self,
        payload: object,
        checkpoint_path: Path,
        retained_path: Path | None,
    ) -> None:
        staging_path = checkpoint_path.with_name(f'.{checkpoint_path.name}.tmp')
        staging_path.unlink(missing_ok=True)
        try:
            torch.save(payload, staging_path)
            if retained_path is not None:
                self._retain_checkpoint(staging_path, retained_path)
            os.replace(staging_path, checkpoint_path)
        finally:
            staging_path.unlink(missing_ok=True)

    def _write_checkpoint_guarded(
        self,
        payload: object,
        checkpoint_path: Path,
        retained_path: Path | None,
    ) -> None:
        try:
            self._write_checkpoint(payload, checkpoint_path, retained_path)
        except BaseException as error:  # re-raised by the next _join_checkpoint_writer
            self._checkpoint_error = error

    def _join_checkpoint_writer(self) -> None:
        """Wait for the in-flight write and surface its failure on the training thread."""
        writer, self._checkpoint_writer = self._checkpoint_writer, None
        if writer is not None:
            writer.join()
        error, self._checkpoint_error = self._checkpoint_error, None
        if error is not None:
            raise error

    @staticmethod
    def _retain_checkpoint(checkpoint_path: Path, retained_path: Path) -> None:
        # Hard link rather than copy: the source is about to be renamed to checkpoint_latest.pth, and
        # the retained name keeps the inode alive once a later save replaces that name.
        staging_path = retained_path.with_name(f'.{retained_path.name}.tmp')
        staging_path.unlink(missing_ok=True)
        try:
            os.link(checkpoint_path, staging_path)
            os.replace(staging_path, retained_path)
        finally:
            staging_path.unlink(missing_ok=True)
