"""Persist the torch.compile cache of nnU-Net trainers across process restarts."""

import contextlib
import hashlib
import os
import platform
from pathlib import Path

import torch
import torch.distributed as dist

from pumit.compile_cache import (
    distributed_archive_compile_cache,
    extract_compile_cache,
)

COMPILE_MODE_ENV = 'PUMIT_NNUNET_COMPILE_MODE'
COMPILE_DYNAMIC_ENV = 'PUMIT_NNUNET_COMPILE_DYNAMIC'
CACHE_ROOT_ENV = 'PUMIT_NNUNET_COMPILE_CACHE_ROOT'

_COMPILE_MODES = ('default', 'reduce-overhead', 'max-autotune')
_DYNAMIC_VALUES = {'false': False, 'true': True, 'auto': None}


def _global_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _local_rank() -> int:
    if not dist.is_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', dist.get_rank()))


class CompileCacheMixin:
    """Reuse one Inductor/Triton cache across restarts of a compiled nnU-Net trainer.

    The live cache is node-local tmpfs; the persistent copy is a ``tar.zst`` beside the nnU-Net results.
    Restoring it avoids recompilation when training resumes on another machine.
    One archive is written after the first epoch of a cold process, when the plan's training shapes have been compiled.

    ``compile_cache_role`` separates archives whose graphs differ, so a validation process cannot overwrite the training archive.
    Sliding-window inference records a CUDA graph per patch shape, so ``reduce-overhead`` is a per-role choice.

    Attributes:
        compile_cache_role: Archive discriminator, ``train`` for the training loop and ``infer`` for standalone prediction.
    """

    compile_cache_role = 'train'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._compile_cache_dir: Path | None = None
        self._compile_cache_archive: Path | None = None
        self._compile_cache_needs_archive = False
        self._compile_cache_first_epoch: int | None = None
        self.compile_mode = self._resolve_compile_mode()
        self.compile_dynamic = self._resolve_compile_dynamic()

    @staticmethod
    def _resolve_compile_mode() -> str:
        mode = os.environ.get(COMPILE_MODE_ENV, 'default')
        if mode not in _COMPILE_MODES:
            raise ValueError(
                f'{COMPILE_MODE_ENV} must be one of {_COMPILE_MODES}, got {mode!r}'
            )
        return mode

    @staticmethod
    def _resolve_compile_dynamic() -> bool | None:
        configured = os.environ.get(COMPILE_DYNAMIC_ENV, 'false').lower()
        if configured not in _DYNAMIC_VALUES:
            raise ValueError(
                f'{COMPILE_DYNAMIC_ENV} must be one of '
                f'{tuple(_DYNAMIC_VALUES)}, got {configured!r}'
            )
        return _DYNAMIC_VALUES[configured]

    def _compile_cache_paths(self) -> tuple[Path, Path]:
        """Return the persistent archive and its node-local live directory.

        The archive is keyed by role, compile mode, dynamic setting, and machine architecture because each combination
        produces a different graph or non-portable kernels. ``output_folder_base`` carries the plan identity and, unlike
        ``output_folder``, is stable when a caller retargets per-checkpoint validation output.

        The live directory additionally carries the fold: folds of one arm share the archive but must not share a live
        directory, because concurrent extraction and archiving of a shared directory race.
        """
        dynamic = {False: 'static', True: 'dynamic', None: 'auto'}[self.compile_dynamic]
        name = (
            f'torchinductor-cache-{self.compile_cache_role}-'
            f'{self.compile_mode}-{dynamic}-{platform.machine()}.tar.zst'
        )
        archive = Path(self.output_folder_base) / name
        archive_key = hashlib.sha256(
            f'{archive.resolve()}:fold_{self.fold}'.encode()
        ).hexdigest()[:16]
        cache_root = Path(
            os.environ.get(CACHE_ROOT_ENV, '/dev/shm/pumit_nnunet_compile_cache')
        )
        return archive, cache_root / archive_key

    @contextlib.contextmanager
    def _compile_defaults(self):
        """Apply the configured mode and dynamic setting to nnU-Net's compile of the network.

        The installed nnU-Net calls ``torch.compile(self.network)`` inline with no override seam, and its wrap order
        (compile, then DDP) is the order PyTorch documents for DDP. Supplying the defaults here keeps that order instead
        of rebuilding the wrappers afterwards.

        ``initialize`` also reaches ``_build_loss``, which compiles the Dice loss. That call keeps nnU-Net's own
        defaults: the requested mode is a statement about the network, and ``reduce-overhead`` in particular would put
        the loss under CUDA graphs, which is a separate decision with its own shape and DDP constraints.
        """
        original = torch.compile
        mode = None if self.compile_mode == 'default' else self.compile_mode
        # self.network is None until initialize() builds it, so the network cannot be identified by identity here. It is
        # the first module nnU-Net compiles, and _build_loss runs afterwards.
        pending_network = True

        def compile_with_defaults(model=None, **kwargs):
            nonlocal pending_network
            if pending_network:
                pending_network = False
                kwargs.setdefault('mode', mode)
                kwargs.setdefault('dynamic', self.compile_dynamic)
            return original(model, **kwargs)

        torch.compile = compile_with_defaults
        try:
            yield
        finally:
            torch.compile = original

    def initialize(self) -> None:
        if self.was_initialized or not self._do_i_compile():
            super().initialize()
            return

        # Point Inductor at the live cache before the first compile reads its environment variables. Compilation itself is
        # lazy, so this only has to precede the first forward pass.
        archive, cache_dir = self._compile_cache_paths()
        # is_cold otherwise comes from a node-local /dev/shm probe, and every rank must agree on it: it decides who
        # reaches the archive barriers, so a split decision would hang the ranks that do.
        global_rank = _global_rank()
        archive_status = [archive.is_file() if global_rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(archive_status, src=0)
        archive_exists = archive_status[0]
        assert archive_exists is not None
        live_dir, _ = extract_compile_cache(
            archive,
            cache_dir,
            rank=_local_rank(),
            archive_exists=archive_exists,
        )
        self._compile_cache_archive = archive
        self._compile_cache_dir = live_dir
        # Keyed on the archive alone, not on whether the live dir happened to be populated: a run that crashed before
        # archiving leaves a populated /dev/shm dir whose kernels exist in no archive, and those still need writing.
        self._compile_cache_needs_archive = not archive_exists
        self.print_to_log_file(
            f'Compile cache {"missing" if not archive_exists else "reused"}: {live_dir} '
            f'(archive {archive}, mode {self.compile_mode}, '
            f'dynamic {self.compile_dynamic})'
        )
        with self._compile_defaults():
            super().initialize()

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch
        super().on_epoch_end()
        if self._compile_cache_first_epoch is None:
            self._compile_cache_first_epoch = completed_epoch
        if self._should_archive_compile_cache(completed_epoch):
            self.archive_compile_cache_now()

    def archive_compile_cache_now(self, *, force: bool = False) -> None:
        """Write the live cache to its archive, once per process.

        The training loop calls this after its first epoch. Callers that never enter the loop, such as standalone
        prediction, call it after their compiled work finishes. ``force`` refreshes an existing archive atomically.
        """
        if self._compile_cache_dir is None or (
            not force and not self._compile_cache_needs_archive
        ):
            return

        assert self._compile_cache_archive is not None
        # tar fails on files that change mid-read, so every rank must stop emitting kernels around the write.
        if dist.is_initialized():
            dist.barrier()
        distributed_archive_compile_cache(
            self._compile_cache_archive,
            self._compile_cache_dir,
            rank=_global_rank(),
            best_effort=True,
        )
        # best_effort swallows a losing race with a concurrent lazy compile. Broadcast the outcome so every rank either
        # retries the same future barrier or marks the cache complete.
        archive_status = [
            self._compile_cache_archive.is_file()
            if _global_rank() == 0
            else None
        ]
        if dist.is_initialized():
            dist.broadcast_object_list(archive_status, src=0)
        archive_exists = archive_status[0]
        assert archive_exists is not None
        if not archive_exists:
            self.print_to_log_file(
                f'Compile-cache archive not written, will retry: '
                f'{self._compile_cache_archive}'
            )
            return
        self._compile_cache_needs_archive = False
        self.print_to_log_file(
            f'Archived compile cache to {self._compile_cache_archive}'
        )

    def _should_archive_compile_cache(self, completed_epoch: int) -> bool:
        """Archive after the first epoch this process completes, until one write succeeds.

        One epoch runs both the training and per-epoch validation graphs at the plan's fixed shapes, so it covers every
        graph the loop reaches. An existing archive already holds those graphs, and a changed mode gets its own archive,
        so only a missing archive needs writing.
        """
        return (
            self._compile_cache_dir is not None
            and self._compile_cache_needs_archive
            and completed_epoch >= self._compile_cache_first_epoch
        )
