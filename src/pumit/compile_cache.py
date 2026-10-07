"""Utility for managing torch.compile cache with distributed precompilation.

Public API:
  - extract_compile_cache: extract archive into a fast dir (tmpfs) and point
    inductor/triton at it; returns (cache_dir, is_cold)
  - archive_compile_cache: tar.zst the live cache dir back to the archive
  - distributed_archive_compile_cache: archive on global rank 0 and propagate
    writer failures to every rank
  - merge_compile_cache_archive: merge a node-local cache into one persistent
    archive from its single writer
  - BackgroundCompileCachePublisher: publish per-node prewarm parts and let
    global rank 0 build the canonical union in the background
  - BackgroundCompileCacheArchiver: archive on a single background worker
  - compile_cache: context manager composing the two (extract on enter,
    archive on exit, rank 0 only)
  - parallel_precompile: distributed warm-up across all DDP ranks
  - parallel_precompile_phase: one component phase inside a shared cache

The live cache_dir is node-local tmpfs shared by all LOCAL ranks. Multi-node
callers must not concurrently update one canonical archive. Each node leader
publishes an immutable part, and global rank 0 is the only canonical writer.
"""

import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist


def _is_rank_zero() -> bool:
    """Return True if non-distributed or rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _tensor_shapes(value) -> list[tuple[int, ...]]:
    if isinstance(value, torch.Tensor):
        return [tuple(value.shape)]
    if isinstance(value, (tuple, list)):
        return [shape for item in value for shape in _tensor_shapes(item)]
    if isinstance(value, dict):
        return [shape for item in value.values() for shape in _tensor_shapes(item)]
    return []


def extract_compile_cache(
    archive: str | Path,
    cache_dir: str | Path | None,
    *,
    rank: int = 0,
    synchronize: bool = True,
    overlay: bool = False,
    archive_exists: bool | None = None,
) -> tuple[Path, bool]:
    """Extract the persistent archive into a fast live dir and point
    TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR at it.

    Only rank 0 extracts; all ranks set the env vars. By default they
    synchronize after extraction. ``archive_exists`` lets a distributed
    caller provide one globally agreed cold/warm decision.
    If ``cache_dir`` is None, a temp directory under /dev/shm is created —
    under torchrun pass an explicit shared path instead (each rank would
    otherwise mkdtemp its own dir).

    Returns (cache_dir, is_cold) where is_cold means no archive existed and
    the live dir started empty.
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="torchinductor_", dir="/dev/shm"))
    else:
        cache_dir = Path(cache_dir)
    archive = Path(archive)
    if archive_exists is None:
        archive_exists = archive.exists()
    local_error: Exception | None = None

    populated = cache_dir.exists() and any(cache_dir.iterdir())
    is_cold = not populated and not archive_exists
    try:
        if populated and not overlay:
            if rank == 0:
                print(f"[compile_cache] {cache_dir} already populated, reusing")
        elif rank == 0:
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            if overlay:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if archive_exists:
                    t0 = time.time()
                    _extract_archive(archive, cache_dir)
                    print(
                        f'[compile_cache] Refreshed {cache_dir} from {archive}'
                        f' in {time.time() - t0:.1f}s'
                    )
            elif archive_exists:
                staging = Path(tempfile.mkdtemp(
                    dir=cache_dir.parent,
                    prefix=f'{cache_dir.name}.extract-',
                ))
                try:
                    t0 = time.time()
                    _extract_archive(archive, staging)
                    if cache_dir.exists():
                        cache_dir.rmdir()
                    staging.replace(cache_dir)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
                print(f"[compile_cache] Extracted {archive} -> {cache_dir} in {time.time() - t0:.1f}s")
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"[compile_cache] No archive at {archive}, starting cold")
    except Exception as error:
        local_error = error

    if synchronize and dist.is_initialized():
        local_status = (
            None
            if local_error is None
            else (
                f'global rank {dist.get_rank()}: '
                f'{type(local_error).__name__}: {local_error}'
            )
        )
        statuses: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, local_status)
        failures = [status for status in statuses if status is not None]
        if failures:
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                'compile-cache extraction failed on another rank: '
                + '; '.join(failures)
            )
    elif local_error is not None:
        raise local_error

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    return cache_dir, is_cold


def _extract_archive(
    archive: Path,
    destination: Path,
) -> None:
    subprocess.run(
        [
            'tar', 'xf', str(archive),
            '-I', 'zstd',
            '-C', str(destination),
        ],
        check=True,
    )


def _validate_compile_cache_archive(archive: Path) -> None:
    subprocess.run(
        ['tar', 'tf', str(archive), '-I', 'zstd'],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _write_compile_cache_archive(
    archive: Path,
    source_dir: Path,
    *,
    best_effort: bool,
    zstd_threads: int,
    validate: bool = False,
) -> bool:
    archive.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=archive.parent, prefix=archive.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    t0 = time.time()
    try:
        subprocess.run(
            [
                "tar", "cf", str(tmp), "-I", f"zstd -1 -T{zstd_threads}",
                "-C", str(source_dir), ".",
            ],
            check=True,
        )
        if validate:
            _validate_compile_cache_archive(tmp)
    except BaseException as error:
        tmp.unlink(missing_ok=True)
        if best_effort and isinstance(error, subprocess.CalledProcessError):
            print(
                f'[compile_cache] best-effort archive failed '
                f'(command exit {error.returncode}); keeping previous {archive}'
            )
            return False
        raise
    tmp.replace(archive)
    size_mb = archive.stat().st_size / 1024 / 1024
    print(
        f"[compile_cache] Archived {source_dir} -> {archive} ({size_mb:.0f} MB)"
        f" in {time.time() - t0:.1f}s"
    )
    return True


def archive_compile_cache(
    archive: str | Path,
    cache_dir: str | Path,
    *,
    rank: int = 0,
    best_effort: bool = False,
    zstd_threads: int = 0,
) -> None:
    """tar.zst the live cache dir back to the persistent archive (rank 0).

    Atomic: writes a tmp file next to the archive, then renames. Call AFTER
    all ranks finish compiling (barrier upstream) — tar errors on files that
    change mid-read. Set ``best_effort=True`` for mid-training snapshots where
    a concurrent lazy compile may race the tar: the failure is logged and the
    existing archive is left intact instead of raising. ``zstd_threads=0``
    lets zstd use every available CPU.
    """
    if zstd_threads < 0:
        raise ValueError('zstd_threads must be non-negative')
    cache_dir = Path(cache_dir)
    if rank != 0 or not cache_dir.exists():
        return
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _write_compile_cache_archive(
        archive,
        cache_dir,
        best_effort=best_effort,
        zstd_threads=zstd_threads,
    )


def distributed_archive_compile_cache(
    archive: str | Path,
    cache_dir: str | Path,
    *,
    rank: int = 0,
    best_effort: bool = False,
    zstd_threads: int = 0,
) -> None:
    """Archive on global rank 0 and propagate a writer failure to every rank.

    Every process in an initialized process group must call this function.
    """
    local_error: Exception | None = None
    try:
        archive_compile_cache(
            archive,
            cache_dir,
            rank=rank,
            best_effort=best_effort,
            zstd_threads=zstd_threads,
        )
    except Exception as error:
        local_error = error

    if dist.is_initialized():
        local_status = (
            None
            if local_error is None
            else (
                f'global rank {dist.get_rank()}: '
                f'{type(local_error).__name__}: {local_error}'
            )
        )
        statuses: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, local_status)
        failures = [status for status in statuses if status is not None]
        if failures:
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                'compile-cache archive failed on another rank: '
                + '; '.join(failures)
            )
    elif local_error is not None:
        raise local_error


def _merge_compile_cache_parts(
    archive: Path,
    part_archives: list[Path],
    *,
    staging_parent: Path,
    best_effort: bool,
    zstd_threads: int,
) -> bool:
    """Merge immutable parts in order; canonical and earlier parts win conflicts."""
    if not part_archives:
        raise ValueError('part_archives must be non-empty')
    try:
        with tempfile.TemporaryDirectory(
            dir=staging_parent,
            prefix='compile-cache-union-',
        ) as staging_name:
            staging = Path(staging_name)
            if archive.exists():
                _extract_archive(archive, staging)
            for index, part in enumerate(part_archives):
                incoming = staging / f'.incoming-{index}'
                incoming.mkdir()
                _extract_archive(part, incoming)
                subprocess.run(
                    ['cp', '-a', '--update=none', f'{incoming}/.', str(staging)],
                    check=True,
                )
                shutil.rmtree(incoming)
            return _write_compile_cache_archive(
                archive,
                staging,
                best_effort=best_effort,
                zstd_threads=zstd_threads,
                validate=True,
            )
    except subprocess.CalledProcessError as e:
        if not best_effort:
            raise
        print(
            f'[compile_cache] best-effort union merge failed '
            f'(command exit {e.returncode}); keeping previous {archive}'
        )
        return False


def merge_compile_cache_archive(
    archive: str | Path,
    cache_dir: str | Path,
    *,
    rank: int = 0,
    best_effort: bool = False,
    zstd_threads: int = 0,
) -> bool:
    """Merge one node-local cache into a canonical archive from its single writer.

    This function provides no inter-process locking. The caller must ensure
    that only one process updates ``archive``. Existing canonical entries win
    path conflicts.
    """
    if zstd_threads < 0:
        raise ValueError('zstd_threads must be non-negative')
    cache_dir = Path(cache_dir)
    if rank != 0 or not cache_dir.exists():
        return False
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=archive.parent,
        prefix=f'{archive.name}.part-',
    ) as part_dir_name:
        part = Path(part_dir_name) / 'cache.tar.zst'
        if not _write_compile_cache_archive(
            part,
            cache_dir,
            best_effort=best_effort,
            zstd_threads=zstd_threads,
            validate=False,
        ):
            return False
        return _merge_compile_cache_parts(
            archive,
            [part],
            staging_parent=cache_dir.parent,
            best_effort=best_effort,
            zstd_threads=zstd_threads,
        )


def _part_error_path(part: Path) -> Path:
    return Path(f'{part}.error')


def _part_ready_path(part: Path) -> Path:
    return Path(f'{part}.ready')


def _write_empty_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + '.',
        suffix='.tmp',
    )
    os.close(fd)
    Path(tmp_name).replace(path)


def _write_part_error(part: Path, error: BaseException) -> None:
    error_path = _part_error_path(part)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=error_path.parent,
        prefix=error_path.name + '.',
        suffix='.tmp',
    )
    try:
        with os.fdopen(fd, 'w') as file:
            file.write(f'{type(error).__name__}: {error}\n')
        Path(tmp_name).replace(error_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _publish_compile_cache_part(
    part: Path,
    cache_dir: Path,
    *,
    zstd_threads: int,
) -> bool:
    try:
        published = _write_compile_cache_archive(
            part,
            cache_dir,
            best_effort=False,
            zstd_threads=zstd_threads,
            validate=False,
        )
        _write_empty_marker(_part_ready_path(part))
        return published
    except (OSError, subprocess.SubprocessError) as error:
        try:
            _write_part_error(part, error)
        except BaseException as marker_error:
            print(
                f'[compile_cache] failed to publish error marker for {part}: '
                f'{type(marker_error).__name__}: {marker_error}'
            )
        print(
            f'[compile_cache] node part publication failed for {part}: '
            f'{type(error).__name__}: {error}; keeping previous canonical archive'
        )
        return False


def _wait_for_compile_cache_parts(
    parts: list[Path],
    *,
    timeout: float,
    poll_interval: float,
) -> None:
    deadline = time.monotonic() + timeout
    pending = set(parts)
    validation_errors: dict[Path, subprocess.CalledProcessError] = {}
    while pending:
        for part in tuple(pending):
            error_path = _part_error_path(part)
            if error_path.exists():
                raise RuntimeError(
                    f'compile-cache part failed: {part}: '
                    f'{error_path.read_text().strip()}'
                )
            if not _part_ready_path(part).exists() or not part.exists():
                continue
            try:
                _validate_compile_cache_archive(part)
            except subprocess.CalledProcessError as error:
                validation_errors[part] = error
                continue
            pending.remove(part)
            validation_errors.pop(part, None)
        if not pending:
            return
        if time.monotonic() >= deadline:
            details = ', '.join(
                f'{part} ({validation_errors[part]})'
                if part in validation_errors
                else str(part)
                for part in sorted(pending)
            )
            raise TimeoutError(f'timed out waiting for compile-cache parts: {details}')
        time.sleep(poll_interval)


class BackgroundCompileCachePublisher:
    """Publish a multi-node compile-cache union without blocking training.

    Every node leader writes one immutable session part. Global rank 0 waits
    for those parts and is the only process that updates the canonical
    archive. Later snapshots from global rank 0 are queued on the same worker,
    so canonical updates cannot overlap. One training job must own the
    canonical archive path.
    """

    def __init__(
        self,
        archive: str | Path,
        cache_dir: str | Path,
        *,
        session: str,
        node_leader_ranks: list[int],
        rank: int,
        zstd_threads: int,
        part_wait_timeout: float = 600,
    ) -> None:
        if zstd_threads <= 0:
            raise ValueError('background zstd_threads must be positive')
        if part_wait_timeout <= 0:
            raise ValueError('part_wait_timeout must be positive')
        if not session or Path(session).name != session:
            raise ValueError(f'invalid compile-cache session: {session!r}')
        leaders = sorted(set(node_leader_ranks))
        if not leaders or leaders[0] != 0:
            raise ValueError('node_leader_ranks must include global rank 0')
        if rank not in leaders:
            raise ValueError(f'rank {rank} is not a node leader')

        self.archive = Path(archive)
        self.cache_dir = Path(cache_dir)
        self.rank = rank
        self.zstd_threads = zstd_threads
        self.part_wait_timeout = part_wait_timeout
        self.parts_dir = (
            self.archive.parent
            / f'{self.archive.name}.parts'
            / session
        )
        self.parts = [
            self.parts_dir / f'rank-{leader:05d}.tar.zst'
            for leader in leaders
        ]
        self.local_part = (
            self.parts_dir / f'rank-{rank:05d}.tar.zst'
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='compile-cache-publisher',
        )
        self._futures: deque[Future[object]] = deque()
        self._started = False
        self._runtime_pending = False

    def start(self) -> None:
        """Queue this node's part and, on global rank 0, the canonical union."""
        if self._started:
            raise RuntimeError('compile-cache publisher already started')
        self._started = True
        self._futures.append(self._executor.submit(self._publish_initial))

    def _publish_initial(self) -> None:
        published = _publish_compile_cache_part(
            self.local_part,
            self.cache_dir,
            zstd_threads=self.zstd_threads,
        )
        if self.rank != 0 or not published:
            return
        try:
            _wait_for_compile_cache_parts(
                self.parts,
                timeout=self.part_wait_timeout,
                poll_interval=0.25,
            )
            merged = _merge_compile_cache_parts(
                self.archive,
                self.parts,
                staging_parent=self.cache_dir.parent,
                best_effort=True,
                zstd_threads=self.zstd_threads,
            )
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TimeoutError,
        ) as error:
            print(
                f'[compile_cache] initial union publication failed: '
                f'{type(error).__name__}: {error}; '
                f'keeping previous {self.archive}'
            )
            return
        if merged:
            try:
                shutil.rmtree(self.parts_dir)
            except OSError as error:
                print(
                    f'[compile_cache] could not remove merged session parts '
                    f'at {self.parts_dir}: {error}'
                )

    def _publish_runtime(self) -> None:
        try:
            merge_compile_cache_archive(
                self.archive,
                self.cache_dir,
                best_effort=True,
                zstd_threads=self.zstd_threads,
            )
        except (OSError, subprocess.SubprocessError) as error:
            print(
                f'[compile_cache] runtime snapshot failed: '
                f'{type(error).__name__}: {error}; '
                f'keeping previous {self.archive}'
            )

    def submit(self) -> bool:
        """Queue a best-effort global-rank-0 incremental snapshot."""
        if self.rank != 0:
            raise RuntimeError('only global rank 0 may update the canonical archive')
        if not self._started:
            raise RuntimeError('compile-cache publisher has not started')
        self.raise_if_failed()
        if self._futures:
            self._runtime_pending = True
            print(
                '[compile_cache] snapshot already running; '
                'coalescing this archive request'
            )
            return False
        self._futures.append(self._executor.submit(self._publish_runtime))
        return True

    def raise_if_failed(self) -> None:
        """Raise completed background failures without waiting."""
        while self._futures and self._futures[0].done():
            self._futures.popleft().result()
        if not self._futures and self._runtime_pending:
            self._runtime_pending = False
            self._futures.append(self._executor.submit(self._publish_runtime))

    def close(self) -> None:
        """Wait for queued publication work during process teardown."""
        while self._futures:
            self._futures.popleft().result()
        if self._runtime_pending:
            self._runtime_pending = False
            self._futures.append(self._executor.submit(self._publish_runtime))
        self._executor.shutdown(wait=True)
        while self._futures:
            self._futures.popleft().result()


class BackgroundCompileCacheArchiver:
    """Archive a live compile cache without blocking the training thread.

    Exactly one archive may be in flight. Worker failures are re-raised by
    ``raise_if_failed()``, the next ``submit()``, or ``close()``.

    Args:
        archive: Atomic tar.zst destination.
        cache_dir: Live Inductor/Triton cache directory.
        zstd_threads: Number of zstd compression threads.
    """

    def __init__(
        self,
        archive: str | Path,
        cache_dir: str | Path,
        *,
        zstd_threads: int,
    ) -> None:
        if zstd_threads <= 0:
            raise ValueError('background zstd_threads must be positive')
        self.archive = Path(archive)
        self.cache_dir = Path(cache_dir)
        self.zstd_threads = zstd_threads
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='compile-cache-archiver',
        )
        self._future: Future[None] | None = None

    def submit(self) -> None:
        """Start an archive and return immediately."""
        self.raise_if_failed()
        if self._future is not None:
            raise RuntimeError('compile-cache archive is still running')
        self._future = self._executor.submit(
            archive_compile_cache,
            self.archive,
            self.cache_dir,
            best_effort=True,
            zstd_threads=self.zstd_threads,
        )

    def raise_if_failed(self) -> None:
        """Raise a completed worker failure on the training thread."""
        if self._future is None or not self._future.done():
            return
        self._future.result()
        self._future = None

    def close(self) -> None:
        """Wait for the current archive and stop the worker."""
        self._executor.shutdown(wait=True)
        if self._future is not None:
            self._future.result()
            self._future = None


@contextmanager
def compile_cache(
    archive: str | Path | None,
    cache_dir: str | Path | None,
    *,
    rank: int = 0,
) -> Generator[bool, None, None]:
    """Context manager: extract cache on enter, archive on exit.

    Pass None for ``archive`` to skip cache management entirely.
    Yields ``is_cold`` (True if cache_dir was empty AND no archive existed).
    """
    if archive is None:
        yield True
        return
    live_dir, is_cold = extract_compile_cache(archive, cache_dir, rank=rank)
    try:
        yield is_cold
    finally:
        archive_compile_cache(archive, live_dir, rank=rank)


def parallel_precompile(
    model: torch.nn.Module,
    inputs: list[tuple[tuple, dict]],
    loss_fn: Callable | None = None,
    *,
    cache_archive: str | Path | None,
    cache_dir: str | Path | None = None,
    amp_dtype: torch.dtype | None = None,
    warmup_inputs: list[tuple[tuple, dict]] | None = None,
) -> None:
    """Distributed precompilation: all ranks compile configs via shared counter.

    All ranks must call this function together (it acts as a collective).
    Each rank claims global input configs from an atomic counter. Ranks may
    then warm only the static configs they execute during training.

    Requires dist.init_process_group already called. Model must be on device.
    If training uses DDP, caller should compile + DDP-wrap before calling.

    Args:
        model: model on device (compiled, optionally DDP-wrapped)
        inputs: list of (args, kwargs) for all configs (CPU tensors)
        loss_fn: if provided, triggers backward pass for each config
        cache_archive: path to persistent tar.zst archive, or None to skip
        cache_dir: fast local cache dir (explicit > env var > auto /dev/shm)
        amp_dtype: autocast dtype for forward/backward
        warmup_inputs: Optional rank-local subset to warm after the shared
            compile phase. Defaults to all inputs.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank = dist.get_rank()

    with compile_cache(cache_archive, cache_dir, rank=rank) as is_cold:
        parallel_precompile_phase(
            model,
            inputs,
            loss_fn,
            cold_start=is_cold,
            counter_key='precompile_next',
            amp_dtype=amp_dtype,
            warmup_inputs=warmup_inputs,
            no_sync_model=model if isinstance(model, DDP) else None,
        )


def parallel_precompile_phase(
    model: torch.nn.Module,
    inputs: list[tuple[tuple, dict]],
    loss_fn: Callable | None = None,
    *,
    cold_start: bool,
    counter_key: str,
    amp_dtype: torch.dtype | None = None,
    warmup_inputs: list[tuple[tuple, dict]] | None = None,
    no_sync_model: torch.nn.Module | None = None,
) -> None:
    """Precompile one component inside a caller-managed shared cache.

    This is the multi-component counterpart to :func:`parallel_precompile`.
    Every rank must enter each phase in the same order with a distinct
    ``counter_key``.

    Args:
        model: Compiled component to execute.
        inputs: Global static input configurations.
        loss_fn: Optional loss that triggers backward compilation.
        cold_start: Whether the enclosing cache was initially empty.
        counter_key: Distributed-store counter unique to this phase.
        amp_dtype: Optional autocast dtype.
        warmup_inputs: Rank-local configurations, or all inputs by default.
        no_sync_model: Optional enclosing DDP model whose gradient
            synchronization should be disabled during component prewarming.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not counter_key:
        raise ValueError('counter_key must be non-empty')
    if no_sync_model is not None and not isinstance(no_sync_model, DDP):
        raise TypeError('no_sync_model must be DistributedDataParallel')

    device = next(model.parameters()).device
    rank = dist.get_rank()
    n = len(inputs)
    if warmup_inputs is None:
        warmup_inputs = inputs
    n_warmup = len(warmup_inputs)

    sync_context = (
        no_sync_model.no_sync() if no_sync_model is not None else nullcontext()
    )
    with sync_context:
        if cold_start:
            store = dist.distributed_c10d._get_default_store()
            if rank == 0:
                print(
                    f'[{_ts()}] [precompile:{counter_key}] Cold start: '
                    f'{n} configs across {dist.get_world_size()} ranks...'
                )
            _distributed_compile(
                model,
                inputs,
                loss_fn,
                amp_dtype,
                device,
                store,
                counter_key,
                n,
                rank,
            )
            dist.barrier()
            if rank == 0:
                print(f'[{_ts()}] [precompile:{counter_key}] Distributed compile done.')

        if rank == 0:
            print(f'[{_ts()}] [precompile:{counter_key}] Rank-local warmup...')
        _warmup(
            model,
            warmup_inputs,
            loss_fn,
            amp_dtype,
            device,
            n_warmup,
            rank,
        )
        # Cache files are shared across local ranks. Do not let rank 0 archive
        # while another rank is still writing warmup artifacts.
        dist.barrier()
        if rank == 0:
            print(f'[{_ts()}] [precompile:{counter_key}] Warmup done.')

    model.zero_grad(set_to_none=True)


def _distributed_compile(
    model: torch.nn.Module,
    inputs: list[tuple[tuple, dict]],
    loss_fn: Callable | None,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    store,
    key: str,
    n: int,
    rank: int,
) -> None:
    """Each rank claims configs from shared counter and compiles them."""
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    compiled_count = 0

    while True:
        idx = store.add(key, 1) - 1
        if idx >= n:
            break
        args, kwargs = inputs[idx]
        moved_args = _move_to_device(args, device)
        moved_kwargs = _move_to_device(kwargs, device)
        t0 = time.time()
        with autocast_ctx:
            output = model(*moved_args, **moved_kwargs)
            if loss_fn is not None:
                loss = loss_fn(output)
                loss.backward()
                del loss
        model.zero_grad(set_to_none=True)
        del output, moved_args, moved_kwargs
        torch.cuda.empty_cache()
        compiled_count += 1
        print(f"[{_ts()}] [precompile] Rank {rank} compiled [{idx + 1}/{n}] ({time.time() - t0:.1f}s)", flush=True)

    print(f"[{_ts()}] [precompile] Rank {rank} compiled {compiled_count} configs total.", flush=True)


def _warmup(
    model: torch.nn.Module,
    inputs: list[tuple[tuple, dict]],
    loss_fn: Callable | None,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    n: int,
    rank: int,
) -> None:
    """All ranks run all configs sequentially (disk cache hits)."""
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    should_print = (rank == 0)

    for i, (args, kwargs) in enumerate(inputs):
        moved_args = _move_to_device(args, device)
        moved_kwargs = _move_to_device(kwargs, device)
        if should_print:
            shapes = ", ".join(str(shape) for shape in _tensor_shapes(moved_args))
            print(f"[{_ts()}] [warmup] [{i + 1}/{n}] [{shapes}] ...", end="", flush=True)
        t0 = time.time()
        with autocast_ctx:
            output = model(*moved_args, **moved_kwargs)
            if loss_fn is not None:
                loss = loss_fn(output)
                loss.backward()
                del loss
        model.zero_grad(set_to_none=True)
        del output, moved_args, moved_kwargs
        torch.cuda.empty_cache()
        if should_print:
            print(f" done ({time.time() - t0:.1f}s)")
