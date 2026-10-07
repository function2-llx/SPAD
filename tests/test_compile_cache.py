"""Tests for pumit.compile_cache utilities."""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import pumit.compile_cache as compile_cache_module
from pumit.compile_cache import (
    BackgroundCompileCacheArchiver,
    BackgroundCompileCachePublisher,
    archive_compile_cache,
    compile_cache,
    distributed_archive_compile_cache,
    extract_compile_cache,
    merge_compile_cache_archive,
)


def test_nested_compile_inputs_move_tensors_and_preserve_metadata():
    tensor = torch.ones(2, 3)
    value = ((tensor, 2), {"skips": [tensor]})

    moved = compile_cache_module._move_to_device(value, torch.device("cpu"))

    assert moved[0][0] is tensor
    assert moved[0][1] == 2
    assert moved[1]["skips"][0] is tensor
    assert compile_cache_module._tensor_shapes(moved) == [(2, 3), (2, 3)]


def test_compile_cache_cold_start():
    """When no archive and no cache_dir content, yields is_cold=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "test.tar.zst"
        shm = Path(tmpdir) / "shm"

        with patch("torch.distributed.is_initialized", return_value=False):
            with compile_cache(archive, shm, rank=0) as is_cold:
                assert is_cold is True
                assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(shm)
                assert os.environ["TRITON_CACHE_DIR"] == str(shm / "triton")


def test_compile_cache_warm_shm():
    """When cache_dir already populated, yields is_cold=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "test.tar.zst"
        shm = Path(tmpdir) / "shm"
        shm.mkdir()
        (shm / "somefile").write_text("data")

        with patch("torch.distributed.is_initialized", return_value=False):
            with compile_cache(archive, shm, rank=0) as is_cold:
                assert is_cold is False


def test_compile_cache_archives_on_exit():
    """On exit, rank 0 creates archive from cache_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "test.tar.zst"
        shm = Path(tmpdir) / "shm"
        shm.mkdir()
        (shm / "cached_file.py").write_text("# compiled kernel")

        with patch("torch.distributed.is_initialized", return_value=False):
            with compile_cache(archive, shm, rank=0):
                pass

        assert archive.exists()
        assert archive.stat().st_size > 0


def test_compile_cache_extracts_archive():
    """When archive exists but cache_dir is empty, extracts archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "test.tar.zst"
        shm_create = Path(tmpdir) / "shm_create"
        shm_extract = Path(tmpdir) / "shm_extract"

        shm_create.mkdir()
        (shm_create / "kernel.py").write_text("# kernel code")
        with patch("torch.distributed.is_initialized", return_value=False):
            with compile_cache(archive, shm_create, rank=0):
                pass

        assert archive.exists()

        with patch("torch.distributed.is_initialized", return_value=False):
            with compile_cache(archive, shm_extract, rank=0) as is_cold:
                assert is_cold is False
                assert (shm_extract / "kernel.py").exists()


def test_compile_cache_none_archive():
    """When archive is None, yields is_cold=True (no-op)."""
    with compile_cache(None, None, rank=0) as is_cold:
        assert is_cold is True


def test_archive_compile_cache_controls_zstd_threads(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "kernel.py").write_text("# compiled kernel")
    archive = tmp_path / "cache.tar.zst"

    with patch("pumit.compile_cache.subprocess.run") as run:
        archive_compile_cache(archive, cache_dir, zstd_threads=8)

    command = run.call_args.args[0]
    assert "zstd -1 -T8" in command


def test_distributed_archive_propagates_remote_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(compile_cache_module.dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(compile_cache_module.dist, 'get_world_size', lambda: 2)

    def gather(statuses, local_status):
        assert local_status is None
        statuses[:] = ['global rank 0: OSError: archive failed', None]

    monkeypatch.setattr(compile_cache_module.dist, 'all_gather_object', gather)

    with pytest.raises(RuntimeError, match='global rank 0.*archive failed'):
        distributed_archive_compile_cache(
            tmp_path / 'cache.tar.zst',
            tmp_path / 'cache',
            rank=1,
        )


def test_distributed_archive_preserves_local_failure(tmp_path, monkeypatch):
    error = OSError('archive failed')
    monkeypatch.setattr(
        compile_cache_module,
        'archive_compile_cache',
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(compile_cache_module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(compile_cache_module.dist, 'get_world_size', lambda: 2)

    def gather(statuses, local_status):
        assert local_status == 'global rank 0: OSError: archive failed'
        statuses[:] = [local_status, None]

    monkeypatch.setattr(compile_cache_module.dist, 'all_gather_object', gather)

    with pytest.raises(OSError, match='archive failed') as raised:
        distributed_archive_compile_cache(
            tmp_path / 'cache.tar.zst',
            tmp_path / 'cache',
            rank=0,
        )

    assert raised.value is error


def test_compile_cache_decompression_omits_zstd_threads(tmp_path: Path):
    archive = tmp_path / 'cache.tar.zst'
    destination = tmp_path / 'cache'

    with patch('pumit.compile_cache.subprocess.run') as run:
        compile_cache_module._extract_archive(archive, destination)
        compile_cache_module._validate_compile_cache_archive(archive)

    assert run.call_args_list[0].args[0] == [
        'tar', 'xf', str(archive), '-I', 'zstd', '-C', str(destination),
    ]
    assert run.call_args_list[1].args[0] == [
        'tar', 'tf', str(archive), '-I', 'zstd',
    ]


def test_extract_compile_cache_can_skip_distributed_barrier(tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    (cache_dir / 'entry').touch()
    barriers = []
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(
        compile_cache_module.dist,
        'barrier',
        lambda: barriers.append(True),
    )

    extract_compile_cache(
        tmp_path / 'cache.tar.zst',
        cache_dir,
        synchronize=False,
    )

    assert not barriers


def test_extract_compile_cache_overlay_refreshes_populated_cache(tmp_path):
    archive_source = tmp_path / 'archive-source'
    archive_source.mkdir()
    (archive_source / 'remote-kernel.py').write_text('remote')
    archive = tmp_path / 'cache.tar.zst'
    archive_compile_cache(archive, archive_source)

    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    (cache_dir / 'local-kernel.py').write_text('local')
    extract_compile_cache(archive, cache_dir, overlay=True)

    assert (cache_dir / 'local-kernel.py').read_text() == 'local'
    assert (cache_dir / 'remote-kernel.py').read_text() == 'remote'


def test_extract_compile_cache_honors_global_cold_decision(tmp_path, monkeypatch):
    archive_source = tmp_path / 'archive-source'
    archive_source.mkdir()
    (archive_source / 'remote-kernel.py').write_text('remote')
    archive = tmp_path / 'cache.tar.zst'
    archive_compile_cache(archive, archive_source)
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: False)

    _, is_cold = extract_compile_cache(
        archive,
        cache_dir,
        archive_exists=False,
    )

    assert is_cold
    assert not (cache_dir / 'remote-kernel.py').exists()


def test_extract_compile_cache_uses_staging_on_failure(tmp_path, monkeypatch):
    archive = tmp_path / 'cache.tar.zst'
    archive.write_bytes(b'not an archive')
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: False)

    with pytest.raises(compile_cache_module.subprocess.CalledProcessError):
        extract_compile_cache(archive, cache_dir)

    assert not cache_dir.exists()


def test_extract_compile_cache_propagates_remote_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    (cache_dir / 'entry').touch()
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(compile_cache_module.dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(compile_cache_module.dist, 'get_world_size', lambda: 2)

    def gather(statuses, local_status):
        assert local_status is None
        statuses[:] = ['global rank 0: OSError: extract failed', None]

    monkeypatch.setattr(
        compile_cache_module.dist,
        'all_gather_object',
        gather,
    )

    with pytest.raises(RuntimeError, match='global rank 0.*extract failed'):
        extract_compile_cache(tmp_path / 'cache.tar.zst', cache_dir, rank=1)


def test_extract_compile_cache_preserves_local_failure(tmp_path, monkeypatch):
    archive = tmp_path / 'cache.tar.zst'
    archive.touch()
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr(compile_cache_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(compile_cache_module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(compile_cache_module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(
        compile_cache_module,
        '_extract_archive',
        lambda *args: (_ for _ in ()).throw(OSError('extract failed')),
    )

    def gather(statuses, local_status):
        statuses[:] = [local_status, None]

    monkeypatch.setattr(
        compile_cache_module.dist,
        'all_gather_object',
        gather,
    )

    with pytest.raises(OSError, match='extract failed'):
        extract_compile_cache(archive, cache_dir)


def test_merge_compile_cache_archive_preserves_union_and_existing_conflicts(tmp_path):
    seed = tmp_path / 'seed'
    (seed / 'aa').mkdir(parents=True)
    (seed / 'aa' / 'seed-kernel.py').write_text('seed')
    (seed / 'shared.best_config').write_text('canonical')
    archive = tmp_path / 'cache.tar.zst'
    archive_compile_cache(archive, seed)

    node_cache = tmp_path / 'node-cache'
    (node_cache / 'aa').mkdir(parents=True)
    (node_cache / 'aa' / 'node-kernel.py').write_text('node')
    (node_cache / 'shared.best_config').write_text('node')
    merge_compile_cache_archive(archive, node_cache)

    extracted = tmp_path / 'extracted'
    extract_compile_cache(archive, extracted)
    assert (extracted / 'aa' / 'seed-kernel.py').read_text() == 'seed'
    assert (extracted / 'aa' / 'node-kernel.py').read_text() == 'node'
    assert (extracted / 'shared.best_config').read_text() == 'canonical'


def test_merge_compile_cache_parts_preserves_rank_ordered_union(tmp_path):
    archive = tmp_path / 'cache.tar.zst'
    parts = []
    for node in range(2):
        cache_dir = tmp_path / f'node-{node}'
        (cache_dir / 'ab').mkdir(parents=True)
        (cache_dir / 'ab' / f'kernel-{node}.py').write_text(str(node))
        (cache_dir / 'shared.best_config').write_text(str(node))
        part = tmp_path / f'part-{node}.tar.zst'
        archive_compile_cache(part, cache_dir)
        parts.append(part)

    compile_cache_module._merge_compile_cache_parts(
        archive,
        parts,
        staging_parent=tmp_path,
        best_effort=False,
        zstd_threads=1,
    )

    extracted = tmp_path / 'extracted'
    extract_compile_cache(archive, extracted)
    assert (extracted / 'ab' / 'kernel-0.py').read_text() == '0'
    assert (extracted / 'ab' / 'kernel-1.py').read_text() == '1'
    assert (extracted / 'shared.best_config').read_text() == '0'


def test_merge_compile_cache_parts_keeps_canonical_on_validation_failure(
    tmp_path,
    monkeypatch,
):
    canonical_cache = tmp_path / 'canonical-cache'
    canonical_cache.mkdir()
    (canonical_cache / 'canonical').write_text('old')
    archive = tmp_path / 'cache.tar.zst'
    archive_compile_cache(archive, canonical_cache)
    original = archive.read_bytes()

    part_cache = tmp_path / 'part-cache'
    part_cache.mkdir()
    (part_cache / 'new').write_text('new')
    part = tmp_path / 'part.tar.zst'
    archive_compile_cache(part, part_cache)

    def fail_validation(path, **kwargs):
        raise compile_cache_module.subprocess.CalledProcessError(1, ['zstd'])

    monkeypatch.setattr(
        compile_cache_module,
        '_validate_compile_cache_archive',
        fail_validation,
    )
    with pytest.raises(compile_cache_module.subprocess.CalledProcessError):
        compile_cache_module._merge_compile_cache_parts(
            archive,
            [part],
            staging_parent=tmp_path,
            best_effort=False,
            zstd_threads=1,
        )

    assert archive.read_bytes() == original


def test_background_publisher_runs_initial_union_off_thread(
    tmp_path,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def publish(*args, **kwargs):
        calls.append(('publish', args, kwargs))
        started.set()
        assert release.wait(timeout=2)
        return True

    monkeypatch.setattr(
        compile_cache_module,
        '_publish_compile_cache_part',
        publish,
    )
    monkeypatch.setattr(
        compile_cache_module,
        '_wait_for_compile_cache_parts',
        lambda *args, **kwargs: calls.append(('wait', args, kwargs)),
    )
    monkeypatch.setattr(
        compile_cache_module,
        '_merge_compile_cache_parts',
        lambda *args, **kwargs: calls.append(('merge', args, kwargs)),
    )
    publisher = BackgroundCompileCachePublisher(
        tmp_path / 'cache.tar.zst',
        tmp_path / 'cache',
        session='session',
        node_leader_ranks=[0],
        rank=0,
        zstd_threads=1,
    )

    publisher.start()
    assert started.wait(timeout=2)
    assert not publisher._futures[0].done()
    release.set()
    publisher.close()

    assert [call[0] for call in calls] == ['publish', 'wait', 'merge']


def test_nonzero_node_publisher_never_updates_canonical(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        compile_cache_module,
        '_publish_compile_cache_part',
        lambda *args, **kwargs: calls.append(('publish', args, kwargs)),
    )
    monkeypatch.setattr(
        compile_cache_module,
        '_wait_for_compile_cache_parts',
        lambda *args, **kwargs: calls.append(('wait', args, kwargs)),
    )
    monkeypatch.setattr(
        compile_cache_module,
        '_merge_compile_cache_parts',
        lambda *args, **kwargs: calls.append(('merge', args, kwargs)),
    )
    publisher = BackgroundCompileCachePublisher(
        tmp_path / 'cache.tar.zst',
        tmp_path / 'cache',
        session='session',
        node_leader_ranks=[0, 4],
        rank=4,
        zstd_threads=1,
    )

    publisher.start()
    publisher.close()

    assert [call[0] for call in calls] == ['publish']


def test_background_publishers_build_multi_node_union(tmp_path):
    cache_0 = tmp_path / 'cache-0'
    cache_0.mkdir()
    (cache_0 / 'rank-0').write_text('zero')
    (cache_0 / 'shared.best_config').write_text('zero')
    cache_4 = tmp_path / 'cache-4'
    cache_4.mkdir()
    (cache_4 / 'rank-4').write_text('four')
    (cache_4 / 'shared.best_config').write_text('four')
    archive = tmp_path / 'cache.tar.zst'

    publisher_4 = BackgroundCompileCachePublisher(
        archive,
        cache_4,
        session='session',
        node_leader_ranks=[0, 4],
        rank=4,
        zstd_threads=1,
    )
    publisher_0 = BackgroundCompileCachePublisher(
        archive,
        cache_0,
        session='session',
        node_leader_ranks=[0, 4],
        rank=0,
        zstd_threads=1,
    )
    publisher_4.start()
    publisher_0.start()
    publisher_4.close()
    publisher_0.close()

    extracted = tmp_path / 'extracted'
    extract_compile_cache(archive, extracted)
    assert (extracted / 'rank-0').read_text() == 'zero'
    assert (extracted / 'rank-4').read_text() == 'four'
    assert (extracted / 'shared.best_config').read_text() == 'zero'


def test_background_publisher_treats_initial_archive_failure_as_best_effort(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        compile_cache_module,
        '_publish_compile_cache_part',
        lambda *args, **kwargs: False,
    )
    publisher = BackgroundCompileCachePublisher(
        tmp_path / 'cache.tar.zst',
        tmp_path / 'cache',
        session='session',
        node_leader_ranks=[0],
        rank=0,
        zstd_threads=1,
    )

    publisher.start()
    publisher.close()

    assert not (tmp_path / 'cache.tar.zst').exists()


def test_background_publisher_coalesces_runtime_snapshot_during_union(
    tmp_path,
    monkeypatch,
):
    initial_started = threading.Event()
    release_initial = threading.Event()
    calls = []

    def publish_initial(*args, **kwargs):
        calls.append('initial')
        initial_started.set()
        assert release_initial.wait(timeout=2)

    monkeypatch.setattr(
        BackgroundCompileCachePublisher,
        '_publish_initial',
        publish_initial,
    )
    monkeypatch.setattr(
        compile_cache_module,
        'merge_compile_cache_archive',
        lambda *args, **kwargs: calls.append('runtime'),
    )
    publisher = BackgroundCompileCachePublisher(
        tmp_path / 'cache.tar.zst',
        tmp_path / 'cache',
        session='session',
        node_leader_ranks=[0],
        rank=0,
        zstd_threads=1,
    )

    publisher.start()
    assert initial_started.wait(timeout=2)
    assert not publisher.submit()
    assert calls == ['initial']
    release_initial.set()
    publisher.close()

    assert calls == ['initial', 'runtime']


def test_background_publisher_treats_runtime_archive_failure_as_best_effort(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        BackgroundCompileCachePublisher,
        '_publish_initial',
        lambda self: None,
    )
    monkeypatch.setattr(
        compile_cache_module,
        'merge_compile_cache_archive',
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError('write failed')),
    )
    publisher = BackgroundCompileCachePublisher(
        tmp_path / 'cache.tar.zst',
        tmp_path / 'cache',
        session='session',
        node_leader_ranks=[0],
        rank=0,
        zstd_threads=1,
    )
    publisher.start()
    publisher._futures[0].result(timeout=2)
    publisher.raise_if_failed()

    assert publisher.submit()
    publisher.close()


def test_wait_for_compile_cache_parts_surfaces_remote_error(tmp_path):
    part = tmp_path / 'rank-00004.tar.zst'
    compile_cache_module._part_error_path(part).write_text(
        'CalledProcessError: tar failed\n'
    )

    with pytest.raises(RuntimeError, match='tar failed'):
        compile_cache_module._wait_for_compile_cache_parts(
            [part],
            timeout=1,
            poll_interval=0,
        )


def test_background_archiver_returns_immediately_and_rejects_overlap(
    tmp_path: Path,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    options = {}

    def archive(*args, **kwargs):
        options.update(kwargs)
        started.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(compile_cache_module, "archive_compile_cache", archive)
    archiver = BackgroundCompileCacheArchiver(
        tmp_path / "cache.tar.zst",
        tmp_path / "cache",
        zstd_threads=2,
    )

    archiver.submit()
    assert started.wait(timeout=2)
    assert options['best_effort'] is True
    with pytest.raises(RuntimeError, match="still running"):
        archiver.submit()
    release.set()
    archiver.close()


def test_background_archiver_surfaces_worker_failure(
    tmp_path: Path,
    monkeypatch,
):
    finished = threading.Event()

    def archive(*args, **kwargs):
        finished.set()
        raise OSError("archive failed")

    monkeypatch.setattr(compile_cache_module, "archive_compile_cache", archive)
    archiver = BackgroundCompileCacheArchiver(
        tmp_path / "cache.tar.zst",
        tmp_path / "cache",
        zstd_threads=2,
    )

    archiver.submit()
    assert finished.wait(timeout=2)
    with pytest.raises(OSError, match="archive failed"):
        archiver.raise_if_failed()
    with pytest.raises(OSError, match="archive failed"):
        archiver.close()


def test_parallel_precompile_phase_uses_its_own_counter_and_local_inputs(
    monkeypatch,
):
    model = torch.nn.Linear(2, 1)
    global_inputs = [((torch.ones(1, 2),), {}), ((torch.zeros(1, 2),), {})]
    local_inputs = global_inputs[1:]
    events = []
    store = object()

    monkeypatch.setattr(compile_cache_module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(compile_cache_module.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(
        compile_cache_module.dist.distributed_c10d,
        '_get_default_store',
        lambda: store,
    )
    monkeypatch.setattr(
        compile_cache_module.dist,
        'barrier',
        lambda: events.append(('barrier',)),
    )
    monkeypatch.setattr(
        compile_cache_module,
        '_distributed_compile',
        lambda *args: events.append(('compile', args)),
    )
    monkeypatch.setattr(
        compile_cache_module,
        '_warmup',
        lambda *args: events.append(('warmup', args)),
    )

    compile_cache_module.parallel_precompile_phase(
        model,
        global_inputs,
        cold_start=True,
        counter_key='decoder_next',
        warmup_inputs=local_inputs,
    )

    compile_args = events[0][1]
    assert compile_args[0] is model
    assert compile_args[1] is global_inputs
    assert compile_args[5] is store
    assert compile_args[6] == 'decoder_next'
    assert events[1] == ('barrier',)
    warmup_args = events[2][1]
    assert warmup_args[0] is model
    assert warmup_args[1] is local_inputs
    assert warmup_args[5] == 1
    assert events[3] == ('barrier',)
