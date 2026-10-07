"""Shared GPU encoding, materialization, and canonical latent artifact lifecycle."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import struct
import time

import msgpack
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
import yaml

from pumit.codec.config import MAX_DA
from pumit.compile_cache import archive_compile_cache, extract_compile_cache
from pumit.ucpt.affine import _canonical_inplane_grid_affine, stream_crop_size
from pumit.ucpt.latent_codec import (
    build_encoder,
    get_batch_size,
    get_latent_targets,
)
from pumit.ucpt.transforms import build_ucpt_pipeline

from .build import (
    _is_source_labeled,
    _msgpack_equal,
    _upgrade_source_unlabeled,
    distributed_context,
    partition_shard_ids,
)
from .manifest import STATS_CHUNK_ROWS, load_manifest, require_absent, sha256_file, write_json
from .metadata import (
    SAMPLE_METADATA_CONTRACT,
    canonicalize_sample_metadata,
    metadata_changes_replay,
)
from .migration import (
    spatial_migration_decision,
)

_COMPILE_CACHE_ARCHIVE_ZSTD_THREADS = 8
_LATENT_REPLAY_WORKERS = 4
# 2 outstanding items per fork worker hide parent-side refill latency; the GPUs otherwise
# drain the 4-deep prepared window during replay or feed hiccups.
_LATENT_REPLAY_PREFETCH_FACTOR = 2
CANONICAL_INPLANE_LATENT_CONTRACT = 'canonical-inplane-latent-v1'
REUSED_LATENT_CONTRACT = 'reused-latent-v1'
_CANONICAL_INPLANE_STAGING_NAME = '.canonical-inplane-latents.staging'
# Preserve resumability of parts written by the previous implementation.
_PARTS_DIR_NAME = '.migration-latent-parts'


@dataclass(frozen=True)
class LatentEncodeWork:
    """One independently encoded subset of a latent shard."""

    shard_id: int
    part_id: int
    selected_indices: tuple[int, ...]
    encoded_rows: int
    voxel_work: int


@dataclass(frozen=True)
class LatentEncodeResult:
    """One encoded work item with stage-level worker timings.

    GPU workers return the encoded rows instead of writing them; the parent writes parts on a
    thread pool so the GPU never idles behind filesystem writes. ``write_seconds`` is therefore
    zero in worker-produced results and measured parent-side during part writing.
    """

    work: LatentEncodeWork
    prepare_seconds: float
    replay_seconds: float
    transfer_seconds: float
    encode_seconds: float
    write_seconds: float
    total_seconds: float
    device_id: int | None = None
    encoded: torch.Tensor | None = None


@dataclass(frozen=True)
class _LatentEncodeBatchResult:
    """One prepared batch result with scheduler wait timings."""

    result: LatentEncodeResult
    device_id: int
    gpu_wall_seconds: float
    first_prepared_wait_seconds: float
    prepared_queue_wait_seconds: float


@dataclass(frozen=True)
class _PreparedLatentEncode:
    work: LatentEncodeWork
    sample_rows: tuple[int, ...]
    images: torch.Tensor
    da_enc: int | None
    prepare_seconds: float
    replay_seconds: float


_LATENT_WORKER_ENCODER = None
_LATENT_WORKER_DEVICE: torch.device | None = None
_LATENT_WORKER_DEVICE_ID: int | None = None
_LATENT_REPLAY_SHARD_CACHE_SIZE = 8
_LATENT_WORKER_ACTIVE_SHARD: int | None = None


def _latent_shape(path: Path) -> tuple[int, int]:
    with safe_open(str(path), framework="pt") as file:
        shape = file.get_slice("latents").get_shape()
    if len(shape) != 2 or shape[1] != 32:
        raise ValueError(f"{path}: expected latent shape [N, 32], got {shape}")
    return int(shape[0]), int(shape[1])


def _validate_materialized_latent(path: Path, expected_rows: int) -> None:
    with safe_open(str(path), framework='pt') as file:
        keys = list(file.keys())
        if keys != ['latents']:
            raise ValueError(f'{path}: expected only a latents tensor, got {keys}')
        latent_slice = file.get_slice('latents')
        shape = latent_slice.get_shape()
        dtype = latent_slice.get_dtype()
    expected_shape = [expected_rows, 32]
    if shape != expected_shape:
        raise ValueError(f'{path}: expected latent shape {expected_shape}, got {shape}')
    if dtype != 'F16':
        raise ValueError(f'{path}: expected F16 latents, got {dtype}')


def _validate_codec_provenance(
    source_latent_dir: Path,
    codec_model: str,
    codec_checkpoint: Path,
) -> None:
    receipt_path = source_latent_dir.parent / 'latent-materialized.json'
    if not receipt_path.exists():
        raise FileNotFoundError(f'missing source latent provenance: {receipt_path}')
    receipt = json.loads(receipt_path.read_text())
    source_model = receipt.get('codec_model')
    if source_model != codec_model:
        raise ValueError(
            f'codec model differs from source latents: source={source_model!r}, current={codec_model!r}'
        )
    source_checkpoint = Path(receipt['codec_checkpoint']).resolve()
    if source_checkpoint != codec_checkpoint:
        raise ValueError(
            f'codec checkpoint differs from source latents: source={source_checkpoint}, '
            f'current={codec_checkpoint}'
        )


def _prepare_materialization_rows(stream_dir: Path, rows: list[dict]) -> list[dict]:
    latent_dir = stream_dir / 'latents'
    pending = []
    completed = 0
    for row in rows:
        output_path = latent_dir / f"shard_{row['shard_id']:05d}.safetensors"
        tmp_path = output_path.with_suffix('.safetensors.tmp')
        if output_path.exists():
            if tmp_path.exists():
                raise FileExistsError(
                    f'found both completed and temporary latent outputs: {output_path}, {tmp_path}'
                )
            _validate_materialized_latent(output_path, row['logical_latent_rows'])
            completed += 1
            continue
        if tmp_path.exists():
            tmp_path.unlink()
        pending.append(row)
    print(
        json.dumps(
            {
                'completed_materialized_shards': completed,
                'pending_materialized_shards': len(pending),
            },
            indent=2,
        )
    )
    return pending


def _select_manifest_rows(
    rows: list[dict],
    *,
    shard_offset: int,
    shards: int | None,
) -> tuple[list[dict], bool]:
    if shard_offset < 0:
        raise ValueError(f'shard_offset must be non-negative, got {shard_offset}')
    if shards is not None and shards < 1:
        raise ValueError(f'shards must be positive, got {shards}')
    shard_stop = len(rows) if shards is None else shard_offset + shards
    if shard_offset >= len(rows) or shard_stop > len(rows):
        raise ValueError(
            f'requested shard range [{shard_offset}, {shard_stop}) is outside '
            f'the manifest range [0, {len(rows)})'
        )
    requested_rows = rows[shard_offset:shard_stop]
    expected_ids = list(range(shard_offset, shard_stop))
    if [row['shard_id'] for row in requested_rows] != expected_ids:
        raise ValueError('manifest shard IDs are not contiguous')
    return requested_rows, shard_offset == 0 and shards is None


def _write_or_validate_json(path: Path, value: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != value:
            raise ValueError(f'existing receipt differs from current result: {path}')
        return
    write_json(path, value)


def _json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_canonical_inplane_source(
    stream_dir: Path,
    source_latent_dir: Path,
    rows: list[dict],
    *,
    codec_model: str,
    codec_checkpoint: Path,
) -> dict:
    """Validate the exact old-latent lineage and row alignment for this one-off rewrite."""
    meta = yaml.safe_load((stream_dir / 'meta.yaml').read_text())
    if meta.get('stream_complete') is not True:
        raise RuntimeError(f'stream is not finalized: {stream_dir}')
    if meta.get('algorithm') != 'ucpt-stream-v4':
        raise ValueError(f'{stream_dir}: canonical in-plane latents require ucpt-stream-v4')
    if meta.get('sample_metadata_contract') != SAMPLE_METADATA_CONTRACT:
        raise ValueError(f'{stream_dir}: unsupported sample metadata contract')
    manifest_path = stream_dir / 'manifest.jsonl'
    manifest_sha256 = sha256_file(manifest_path)
    if meta.get('manifest_sha256') != manifest_sha256:
        raise ValueError(f'{stream_dir}: manifest hash differs from meta.yaml')

    source_stream_raw = meta.get('source_stream')
    if not isinstance(source_stream_raw, str) or not source_stream_raw:
        raise ValueError(f'{stream_dir}: meta.yaml lacks source_stream')
    source_stream = Path(source_stream_raw).resolve()
    source_meta = yaml.safe_load((source_stream / 'meta.yaml').read_text())
    if source_meta.get('stream_complete') is not True or not (source_stream / 'READY.json').exists():
        raise RuntimeError(f'source stream is not verified and ready: {source_stream}')
    source_manifest_path = source_stream / 'manifest.jsonl'
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if source_meta.get('manifest_sha256') != source_manifest_sha256:
        raise ValueError(f'{source_stream}: manifest hash differs from meta.yaml')
    source_rows = load_manifest(source_stream)
    if [row['shard_id'] for row in source_rows] != [row['shard_id'] for row in rows]:
        raise ValueError('source and target manifest shard IDs differ')

    for row, source_row in zip(rows, source_rows, strict=True):
        shard_id = row['shard_id']
        expected_zero = (
            'generated_unlabeled_samples',
            'generated_unlabeled_patches',
            'dropped_old_unlabeled_samples',
            'dropped_old_unlabeled_patches',
        )
        if any(row[key] != 0 for key in expected_zero):
            raise ValueError(f'shard {shard_id}: target does not exclusively reuse old unlabeled samples')
        if (
            row['used_old_unlabeled_samples'] != row['new_unlabeled_samples']
            or row['used_old_unlabeled_patches'] != row['logical_latent_rows']
        ):
            raise ValueError(f'shard {shard_id}: target unlabeled reuse is incomplete')
        if (
            row['new_unlabeled_samples'] != source_row['new_unlabeled_samples']
            or row['logical_latent_rows'] != source_row['logical_latent_rows']
        ):
            raise ValueError(f'shard {shard_id}: source and target unlabeled layout differs')

    links_path = source_stream / 'latent-links.json'
    links = json.loads(links_path.read_text())
    if set(links) != {'source_latent_dir', 'hardlinked_shards', 'materialize_shards'}:
        raise ValueError(f'{links_path}: unexpected latent-link receipt schema')
    shard_ids = [row['shard_id'] for row in rows]
    if links['hardlinked_shards'] != shard_ids or links['materialize_shards'] != []:
        raise ValueError(f'{links_path}: source stream does not exclusively hardlink all latent shards')
    declared_latent_dir = Path(links['source_latent_dir']).resolve()
    if declared_latent_dir != source_latent_dir:
        raise ValueError(
            f'{links_path}: declared source latent dir {declared_latent_dir} '
            f'differs from requested {source_latent_dir}'
        )

    _validate_codec_provenance(source_latent_dir, codec_model, codec_checkpoint)
    source_receipt_path = source_latent_dir.parent / 'latent-materialized.json'
    for row in rows:
        path = source_latent_dir / f"shard_{row['shard_id']:05d}.safetensors"
        _validate_materialized_latent(path, row['logical_latent_rows'])

    return {
        'stream_manifest_sha256': manifest_sha256,
        'source_stream': str(source_stream),
        'source_stream_fingerprint': source_meta['fingerprint'],
        'source_manifest_sha256': source_manifest_sha256,
        'source_latent_receipt_sha256': sha256_file(source_receipt_path),
    }


def _compile_cache_live_dir(cache_archive: Path) -> Path:
    archive_key = hashlib.sha256(str(cache_archive.resolve()).encode()).hexdigest()[:16]
    root = Path(
        os.environ.get(
            'UCPT_LATENT_COMPILE_CACHE_ROOT',
            '/dev/shm/ucpt_latent_compile_cache',
        )
    )
    return root / archive_key


def _extract_latent_compile_cache(cache_archive: Path) -> Path:
    cache_dir = _compile_cache_live_dir(cache_archive)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir, _ = extract_compile_cache(
        cache_archive,
        cache_dir,
        synchronize=False,
    )
    return cache_dir


class _PerShardCompileCacheArchiver:
    """Queue one background cache archive after every rank-0 shard."""

    def __init__(self, archive: Path, cache_dir: Path) -> None:
        self.archive = archive
        self.cache_dir = cache_dir
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='compile-cache-archiver',
        )
        self._futures: list[Future[None]] = []

    def submit(self) -> None:
        self._futures.append(
            self._executor.submit(
                archive_compile_cache,
                self.archive,
                self.cache_dir,
                best_effort=True,
                zstd_threads=_COMPILE_CACHE_ARCHIVE_ZSTD_THREADS,
            )
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for future in self._futures:
            future.result()


def _source_latent_row_spans(row: dict, source_rows: int) -> list[list[int]]:
    spans = row.get('source_latent_row_spans')
    if not isinstance(spans, list):
        raise ValueError(f"shard {row['shard_id']}: missing source latent row spans")
    previous_stop = -1
    retained_rows = 0
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
        ):
            raise ValueError(f"shard {row['shard_id']}: invalid source latent row span")
        start, stop = span
        if not 0 <= start < stop <= row['old_unlabeled_patches'] or start <= previous_stop:
            raise ValueError(f"shard {row['shard_id']}: source latent row spans must be merged and ordered")
        retained_rows += stop - start
        previous_stop = stop
    if source_rows < row['old_unlabeled_patches']:
        raise ValueError(f"shard {row['shard_id']}: source latent row contract failed")
    if retained_rows != row['logical_latent_rows']:
        raise ValueError(f"shard {row['shard_id']}: retained latent row count differs from manifest")
    if row['generated_unlabeled_samples'] != 0 or row['generated_unlabeled_patches'] != 0:
        raise ValueError('input migration cannot generate new unlabeled latents')
    if (
        row['used_old_unlabeled_patches'] != retained_rows
        or row['dropped_old_unlabeled_patches'] != row['old_unlabeled_patches'] - retained_rows
    ):
        raise ValueError(f"shard {row['shard_id']}: retained/dropped latent row counts differ")
    return spans


def _validate_reused_latent(source: Path, destination: Path, row: dict) -> bool:
    """Check retained rows against their source; return whether this must be a hardlink."""
    source_rows, _ = _latent_shape(source)
    _validate_materialized_latent(source, source_rows)
    spans = _source_latent_row_spans(row, source_rows)
    _validate_materialized_latent(destination, row['logical_latent_rows'])
    full_source = spans == [[0, source_rows]]
    if full_source:
        if not source.samefile(destination):
            raise ValueError(f'{destination}: complete reused latent must be a source hardlink')
        return True
    with safe_open(str(source), framework='pt') as source_file, safe_open(
        str(destination), framework='pt',
    ) as destination_file:
        source_slice = source_file.get_slice('latents')
        destination_slice = destination_file.get_slice('latents')
        offset = 0
        for span_start, span_stop in spans:
            for start in range(span_start, span_stop, STATS_CHUNK_ROWS):
                stop = min(start + STATS_CHUNK_ROWS, span_stop)
                if not torch.equal(
                    source_slice[start:stop], destination_slice[offset:offset + stop - start],
                ):
                    raise ValueError(f'{destination}: reused latent values differ at source row {start}')
                offset += stop - start
    return False


def _reuse_latent_shard(source_latent_dir: Path, latent_dir: Path, row: dict) -> bool:
    source = source_latent_dir / f"shard_{row['shard_id']:05d}.safetensors"
    destination = latent_dir / source.name
    source_rows, _ = _latent_shape(source)
    _validate_materialized_latent(source, source_rows)
    spans = _source_latent_row_spans(row, source_rows)
    tmp_path = destination.with_suffix('.safetensors.tmp')
    if destination.exists():
        if tmp_path.exists():
            raise FileExistsError(f'found both completed and temporary latent outputs: {destination}, {tmp_path}')
        return _validate_reused_latent(source, destination, row)
    tmp_path.unlink(missing_ok=True)
    if spans == [[0, source_rows]]:
        os.link(source, destination)
        return _validate_reused_latent(source, destination, row)

    # Standard safetensors header followed by contiguous F16 rows; keep memory bounded by one chunk.
    header = json.dumps(
        {
            'latents': {
                'dtype': 'F16',
                'shape': [row['logical_latent_rows'], 32],
                'data_offsets': [0, row['logical_latent_rows'] * 64],
            },
        },
        separators=(',', ':'),
    ).encode()
    header += b' ' * (-len(header) % 8)
    with tmp_path.open('wb') as output, safe_open(str(source), framework='pt') as source_file:
        output.write(struct.pack('<Q', len(header)))
        output.write(header)
        source_slice = source_file.get_slice('latents')
        for span_start, span_stop in spans:
            for start in range(span_start, span_stop, STATS_CHUNK_ROWS):
                stop = min(start + STATS_CHUNK_ROWS, span_stop)
                output.write(source_slice[start:stop].numpy().tobytes())
    _validate_reused_latent(source, tmp_path, row)
    tmp_path.rename(destination)
    return False


def _validate_reused_latent_operation(meta: dict) -> None:
    from .filtering import UNLABELED_FILTER_CONTRACT

    unlabeled_filter = meta.get('unlabeled_filter')
    if unlabeled_filter is None:
        raise ValueError('reused latents require unlabeled filtering metadata')
    if unlabeled_filter.get('contract') != UNLABELED_FILTER_CONTRACT:
        raise ValueError('unsupported unlabeled filter contract')


def _link_reused_latents(
    args,
    stream_dir: Path,
    source_latent_dir: Path,
    rows: list[dict],
    meta: dict,
) -> None:
    _validate_reused_latent_operation(meta)
    source_meta = yaml.safe_load((Path(meta['source_stream']) / 'meta.yaml').read_text())
    source_normalization = source_meta.get('normalization')
    if source_normalization is None:
        source_normalization = source_meta.get('composition', {}).get('normalization')
    normalization = meta.get('normalization')
    if (
        not isinstance(normalization, dict)
        or normalization.get('contract') != 'fixed-source-v1'
        or normalization != source_normalization
    ):
        raise ValueError('latent reuse must inherit the source normalization unchanged')
    source_stats = source_latent_dir / 'stats.safetensors'
    if sha256_file(source_stats) != normalization['source_stats_sha256']:
        raise ValueError('source latent stats hash differs from normalization provenance')
    source_receipt_path = source_latent_dir.parent / 'latent-materialized.json'
    source_receipt = json.loads(source_receipt_path.read_text())
    for key in ('codec_model', 'codec_checkpoint'):
        if not isinstance(source_receipt.get(key), str) or not source_receipt[key]:
            raise ValueError(f'{source_receipt_path}: missing {key} identity')

    latent_dir = stream_dir / 'latents'
    latent_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=getattr(args, 'workers', 4)) as pool:
        linked = list(pool.map(lambda row: _reuse_latent_shard(source_latent_dir, latent_dir, row), rows))
    linked_ids = [row['shard_id'] for row, is_linked in zip(rows, linked, strict=True) if is_linked]
    copied_ids = [row['shard_id'] for row, is_linked in zip(rows, linked, strict=True) if not is_linked]
    stats_path = latent_dir / 'stats.safetensors'
    if stats_path.exists():
        if sha256_file(stats_path) != normalization['source_stats_sha256']:
            raise ValueError('existing latent stats differ from source normalization')
    else:
        stats_tmp = stats_path.with_suffix('.safetensors.tmp')
        shutil.copyfile(source_stats, stats_tmp)
        if sha256_file(stats_tmp) != normalization['source_stats_sha256']:
            raise ValueError('copied latent stats differ from source normalization')
        stats_tmp.rename(stats_path)
    _write_or_validate_json(
        stream_dir / 'latent-links.json',
        {'source_latent_dir': str(source_latent_dir), 'hardlinked_shards': linked_ids, 'materialize_shards': copied_ids},
    )
    receipt = {
        'latent_contract': REUSED_LATENT_CONTRACT,
        'codec_model': source_receipt['codec_model'],
        'codec_checkpoint': source_receipt['codec_checkpoint'],
        'source_latent_dir': str(source_latent_dir),
        'source_latent_receipt_sha256': sha256_file(source_receipt_path),
        'logical_latent_rows': sum(row['logical_latent_rows'] for row in rows),
        'materialized_shards': copied_ids,
    }
    _write_or_validate_json(stream_dir / 'latent-materialized.json', receipt)
    print(json.dumps({'hardlinked_shards': len(linked_ids), 'materialized_shards': len(copied_ids)}, indent=2))


def cmd_link_latents(args) -> None:
    source_latent_dir = args.source_latent_dir.resolve()
    stream_dir = args.stream.resolve()
    latent_dir = stream_dir / "latents"
    rows = load_manifest(stream_dir)
    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get('input_migration') is not None or meta.get('unlabeled_filter') is not None:
        _link_reused_latents(args, stream_dir, source_latent_dir, rows, meta)
        return
    if (
        any(row['generated_unlabeled_samples'] == 0 for row in rows)
        and source_latent_dir.stat().st_dev != stream_dir.stat().st_dev
    ):
        raise ValueError('hardlinked latent prefixes require source and output on the same filesystem')
    require_absent(latent_dir)
    latent_dir.mkdir()
    linked = []
    materialize = []
    for row in rows:
        shard_id = row["shard_id"]
        name = f"shard_{shard_id:05d}.safetensors"
        source = source_latent_dir / name
        source_rows, _ = _latent_shape(source)
        if source_rows < row['old_unlabeled_patches']:
            raise ValueError(
                f"{source}: stored rows {source_rows} < source logical rows {row['old_unlabeled_patches']}"
            )
        destination = latent_dir / name
        require_absent(destination)
        if row["generated_unlabeled_samples"] == 0:
            os.link(source, destination)
            source_stat = source.stat()
            destination_stat = destination.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                raise ValueError(f"{destination}: hardlink inode differs from source")
            linked.append(shard_id)
        else:
            materialize.append(shard_id)
    receipt = {
        "source_latent_dir": str(source_latent_dir),
        "hardlinked_shards": linked,
        "materialize_shards": materialize,
    }
    write_json(stream_dir / 'latent-links.json', receipt)
    print(
        json.dumps(
            {
                "hardlinked_shards": len(linked),
                "materialize_shards": materialize,
            },
            indent=2,
        )
    )


def _build_pipeline(stream_dir: Path):
    meta = yaml.safe_load((stream_dir / "meta.yaml").read_text())
    stream_config = meta["config"]
    max_depth_per_da = {
        int(key): value for key, value in stream_config["max_depth_per_da"].items()
    }
    return build_ucpt_pipeline(
        size_xy_choices=stream_config["size_xy_choices"],
        size_xy_choices_2d=stream_config["size_xy_choices_2d"],
        max_depth_per_da=max_depth_per_da,
        max_da=MAX_DA,
    )


def _load_unlabeled_samples(shard_path: Path) -> list[dict]:
    with shard_path.open('rb') as file:
        shard = msgpack.unpack(file, raw=False)
    unlabeled = []
    for batch in shard['batches']:
        for sample in batch['samples']:
            if 'label_classes' in sample:
                raise ValueError(f'{shard_path}: stream embeds a full label contract')
            if not isinstance(sample.get('labeled'), bool):
                raise ValueError(f'{shard_path}: sample lacks a boolean labeled flag')
            if bool(sample.get('classes')) != sample['labeled']:
                raise ValueError(f'{shard_path}: invalid labeled sample schema')
            if not sample['labeled']:
                unlabeled.append(sample)
    return unlabeled


def _prepare_encode_work_from_samples(
    work: LatentEncodeWork,
    samples: list[dict],
    pipeline,
    replay_pool: ThreadPoolExecutor,
    *,
    prepare_start: float | None = None,
) -> _PreparedLatentEncode:
    start = time.monotonic() if prepare_start is None else prepare_start
    da_enc = samples[0]['da_enc']
    prepare_done = time.monotonic()

    def replay(sample: dict) -> torch.Tensor:
        if sample['da_enc'] != da_enc:
            raise ValueError(
                f'shard {work.shard_id} part {work.part_id}: mixed da_enc values'
            )
        return pipeline.replay(
            {'img': sample['img']},
            sample['params'],
        )['img']

    replay_futures = [replay_pool.submit(replay, sample) for sample in samples]
    try:
        images = [future.result() for future in replay_futures]
    except BaseException:
        for future in replay_futures:
            future.cancel()
        wait(replay_futures)
        raise
    replay_done = time.monotonic()
    return _PreparedLatentEncode(
        work=work,
        sample_rows=tuple(sample['n_patches'] for sample in samples),
        images=torch.stack(images),
        da_enc=da_enc,
        prepare_seconds=prepare_done - start,
        replay_seconds=replay_done - prepare_done,
    )


class _LatentReplayDataset(IterableDataset):
    """Replay queued work in forked CPU workers without touching CUDA."""

    def __init__(
        self,
        stream_dir: Path,
        work_queue,
        *,
        replay_threads: int,
    ) -> None:
        self.stream_dir = stream_dir
        self.work_queue = work_queue
        self.replay_threads = replay_threads

    def __iter__(self):
        pipeline = None
        shards: OrderedDict[int, list[dict]] = OrderedDict()
        with ThreadPoolExecutor(max_workers=self.replay_threads) as replay_pool:
            while True:
                work = self.work_queue.get()
                if work is None:
                    return
                prepare_start = time.monotonic()
                if not isinstance(work, LatentEncodeWork):
                    raise TypeError(f'unexpected latent replay work: {type(work).__name__}')
                if pipeline is None:
                    pipeline = _build_pipeline(self.stream_dir)
                samples = shards.pop(work.shard_id, None)
                if samples is None:
                    samples = _load_unlabeled_samples(
                        self.stream_dir / f'shard_{work.shard_id:05d}.msgpack'
                    )
                shards[work.shard_id] = samples
                while len(shards) > _LATENT_REPLAY_SHARD_CACHE_SIZE:
                    shards.popitem(last=False)
                selected = [samples[index] for index in work.selected_indices]
                yield _prepare_encode_work_from_samples(
                    work,
                    selected,
                    pipeline,
                    replay_pool,
                    prepare_start=prepare_start,
                )


class _LatentPreparedLoader:
    """Own one persistent fork DataLoader and its bounded global replay queue."""

    def __init__(self, stream_dir: Path, *, replay_threads: int) -> None:
        context = get_context('fork')
        self._work_queue = context.Queue()
        self._loader = DataLoader(
            _LatentReplayDataset(
                stream_dir,
                self._work_queue,
                replay_threads=replay_threads,
            ),
            batch_size=None,
            num_workers=_LATENT_REPLAY_WORKERS,
            prefetch_factor=_LATENT_REPLAY_PREFETCH_FACTOR,
            persistent_workers=False,
            in_order=False,
            pin_memory=False,
            multiprocessing_context=context,
        )
        self._iterator = None
        self._closed = False

    @property
    def capacity(self) -> int:
        return _LATENT_REPLAY_WORKERS * _LATENT_REPLAY_PREFETCH_FACTOR

    def start(self) -> None:
        if self._closed:
            raise RuntimeError('latent prepared loader is closed')
        if self._iterator is not None:
            raise RuntimeError('latent prepared loader is already started')
        # DataLoader forks all replay workers while the parent is still CPU-only.
        self._iterator = iter(self._loader)

    def submit(self, work: LatentEncodeWork) -> None:
        if self._iterator is None or self._closed:
            raise RuntimeError('latent prepared loader is not running')
        self._work_queue.put(work)

    def get(self) -> _PreparedLatentEncode:
        if self._iterator is None or self._closed:
            raise RuntimeError('latent prepared loader is not running')
        prepared = next(self._iterator)
        if not isinstance(prepared, _PreparedLatentEncode):
            raise TypeError(
                f'unexpected prepared latent result: {type(prepared).__name__}'
            )
        return prepared

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        iterator = self._iterator
        if iterator is None:
            self._work_queue.close()
            self._work_queue.join_thread()
            return
        for _ in range(_LATENT_REPLAY_WORKERS):
            self._work_queue.put(None)
        try:
            while True:
                next(iterator)
        except StopIteration:
            pass
        finally:
            shutdown = getattr(iterator, '_shutdown_workers', None)
            if shutdown is not None:
                shutdown()
            self._work_queue.close()
            self._work_queue.join_thread()


def _requires_canonical_inplane_latent(sample: dict) -> bool:
    """Return whether replay now takes the canonical in-plane grid branch."""
    try:
        affine = np.asarray(sample['params'][0]['affine'], dtype=np.float64)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError('unlabeled sample lacks a valid frozen affine') from error
    if affine.size != 16 or not np.isfinite(affine).all():
        raise ValueError('unlabeled sample affine must contain 16 finite values')
    matrix = affine.reshape(4, 4)[:3, :3]
    return _canonical_inplane_grid_affine(matrix) is not None


def _canonical_inplane_selection(shard_path: Path, row: dict) -> tuple[list[dict], list[int]]:
    """Select target samples whose source metadata canonicalization changes replay."""
    unlabeled = _load_unlabeled_samples(shard_path)
    logical_rows = sum(sample['n_patches'] for sample in unlabeled)
    if logical_rows != row['logical_latent_rows']:
        raise ValueError(
            f'{shard_path}: stream samples contain {logical_rows} latent rows, '
            f"manifest declares {row['logical_latent_rows']}"
        )
    stream_dir = shard_path.parent
    meta = yaml.safe_load((stream_dir / 'meta.yaml').read_text())
    source_stream_raw = meta.get('source_stream')
    if not isinstance(source_stream_raw, str) or not source_stream_raw:
        raise ValueError(f'{stream_dir}: meta.yaml lacks source_stream')
    source_path = Path(source_stream_raw) / shard_path.name
    with source_path.open('rb') as file:
        source_shard = msgpack.unpack(file, raw=False)
    source_unlabeled = [
        sample
        for batch in source_shard['batches']
        for sample in batch['samples']
        if not _is_source_labeled(sample)
    ]
    if len(source_unlabeled) != len(unlabeled):
        raise ValueError(f'{shard_path}: source and target unlabeled counts differ')

    selected = []
    expected = []
    for index, sample in enumerate(source_unlabeled):
        migrated, stats = canonicalize_sample_metadata(_upgrade_source_unlabeled(sample))
        expected.append(migrated)
        if metadata_changes_replay(stats):
            selected.append(index)
    if not _msgpack_equal(unlabeled, expected):
        raise ValueError(f'{shard_path}: target unlabeled metadata differs from canonical source')
    return unlabeled, selected


def _migration_selection(shard_path: Path, row: dict) -> tuple[list[dict], list[int]]:
    unlabeled = _load_unlabeled_samples(shard_path)
    used_old = row['used_old_unlabeled_samples']
    if not 0 <= used_old <= len(unlabeled):
        raise ValueError(
            f'{shard_path}: used_old_unlabeled_samples={used_old} outside [0, {len(unlabeled)}]'
        )
    selected = [
        index
        for index, sample in enumerate(unlabeled[:used_old])
        if spatial_migration_decision(sample).required
    ]
    selected.extend(range(used_old, len(unlabeled)))
    affected = [index for index in selected if index < used_old]
    affected_patches = sum(unlabeled[index]['n_patches'] for index in affected)
    if len(affected) != row['migration_affected_unlabeled_samples']:
        raise ValueError(
            f"{shard_path}: affected sample count {len(affected)} != "
            f"{row['migration_affected_unlabeled_samples']}"
        )
    if affected_patches != row['migration_affected_unlabeled_patches']:
        raise ValueError(
            f'{shard_path}: affected patch count {affected_patches} != '
            f"{row['migration_affected_unlabeled_patches']}"
        )
    return unlabeled, selected


def _sample_voxels(sample: dict) -> int:
    _, size_y, size_x = stream_crop_size(sample['params'][0])
    return int(sample['depth']) * size_y * size_x


def _build_encode_work(
    unlabeled: list[dict],
    selected_indices: list[int],
    *,
    shard_id: int,
    memory_budget_gb: float,
) -> list[LatentEncodeWork]:
    """Build shape-uniform GPU batches independently of the worker count."""
    buckets: dict[tuple[int | None, int, int], list[int]] = {}
    batch_sizes: dict[tuple[int | None, int, int], int] = {}
    work: list[LatentEncodeWork] = []

    def emit(indices: list[int]) -> None:
        part_id = len(work)
        work.append(
            LatentEncodeWork(
                shard_id=shard_id,
                part_id=part_id,
                selected_indices=tuple(indices),
                encoded_rows=sum(unlabeled[index]['n_patches'] for index in indices),
                voxel_work=sum(_sample_voxels(unlabeled[index]) for index in indices),
            )
        )

    for index in selected_indices:
        sample = unlabeled[index]
        _, size_y, _ = stream_crop_size(sample['params'][0])
        key = (sample['da_enc'], sample['depth'], size_y)
        bucket = buckets.setdefault(key, [])
        if key not in batch_sizes:
            batch_sizes[key] = get_batch_size(
                sample['da_enc'],
                sample['depth'],
                size_y,
                memory_budget_gb,
            )
        bucket.append(index)
        if len(bucket) >= batch_sizes[key]:
            emit(bucket)
            buckets[key] = []
    for bucket in buckets.values():
        if bucket:
            emit(bucket)
    return work


def _latent_part_path(stream_dir: Path, work: LatentEncodeWork) -> Path:
    return (
        stream_dir
        / _PARTS_DIR_NAME
        / f'shard_{work.shard_id:05d}'
        / f'part_{work.part_id:05d}.safetensors'
    )


def _validate_latent_part(
    path: Path,
    work: LatentEncodeWork,
    *,
    plan_sha256: str | None = None,
) -> None:
    with safe_open(str(path), framework='pt') as file:
        keys = list(file.keys())
        if keys != ['latents', 'sample_indices']:
            raise ValueError(f'{path}: unexpected tensors {keys}')
        latent_shape = file.get_slice('latents').get_shape()
        latent_dtype = file.get_slice('latents').get_dtype()
        indices = file.get_tensor('sample_indices')
        metadata = file.metadata() or {}
    if latent_shape != [work.encoded_rows, 32] or latent_dtype != 'F16':
        raise ValueError(
            f'{path}: expected F16 latents [{work.encoded_rows}, 32], '
            f'got {latent_dtype} {latent_shape}'
        )
    expected_indices = torch.tensor(work.selected_indices, dtype=torch.int64)
    if not torch.equal(indices, expected_indices):
        raise ValueError(f'{path}: sample indices differ from the current work plan')
    expected_metadata = {} if plan_sha256 is None else {'plan_sha256': plan_sha256}
    if metadata != expected_metadata:
        raise ValueError(f'{path}: part metadata differs from the current plan')


def _write_latent_part(
    stream_dir: Path,
    work: LatentEncodeWork,
    encoded: torch.Tensor,
    *,
    plan_sha256: str | None = None,
) -> None:
    if encoded.shape != (work.encoded_rows, 32):
        raise ValueError(
            f'shard {work.shard_id} part {work.part_id}: encoded shape '
            f'{tuple(encoded.shape)} != {(work.encoded_rows, 32)}'
        )
    output_path = _latent_part_path(stream_dir, work)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix('.safetensors.tmp')
    if output_path.exists():
        _validate_latent_part(output_path, work, plan_sha256=plan_sha256)
        return
    tmp_path.unlink(missing_ok=True)
    save_file(
        {
            'latents': encoded,
            'sample_indices': torch.tensor(work.selected_indices, dtype=torch.int64),
        },
        str(tmp_path),
        metadata={} if plan_sha256 is None else {'plan_sha256': plan_sha256},
    )
    _validate_latent_part(tmp_path, work, plan_sha256=plan_sha256)
    tmp_path.rename(output_path)


def _materialize_latent_parts(
    source_path: Path | None,
    unlabeled: list[dict],
    source_prefix_samples: int,
    work: list[LatentEncodeWork],
    stream_dir: Path,
    output_path: Path,
    *,
    plan_sha256: str | None = None,
) -> None:
    """Scatter encoded part files into a complete, distinct latent artifact."""
    require_absent(output_path)
    tmp_path = output_path.with_suffix('.safetensors.tmp')
    require_absent(tmp_path)
    offsets = [0]
    for sample in unlabeled:
        offsets.append(offsets[-1] + sample['n_patches'])
    if not 0 <= source_prefix_samples <= len(unlabeled):
        raise ValueError(
            f'{output_path}: source prefix sample count {source_prefix_samples} '
            f'outside [0, {len(unlabeled)}]'
        )
    source_prefix_rows = offsets[source_prefix_samples]

    selected_indices = [
        index
        for item in work
        for index in item.selected_indices
    ]
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError(f'{output_path}: latent parts contain duplicate sample indices')
    covered_indices = set(range(source_prefix_samples)) | set(selected_indices)
    expected_indices = set(range(len(unlabeled)))
    if covered_indices != expected_indices:
        raise ValueError(
            f'{output_path}: source prefix and encoded parts do not cover every sample'
        )

    output = torch.empty(offsets[-1], 32, dtype=torch.float16)
    if source_path is None:
        if source_prefix_samples != 0:
            raise ValueError(f'{output_path}: source prefix requires a source latent artifact')
    else:
        with safe_open(str(source_path), framework='pt') as file:
            source_slice = file.get_slice('latents')
            source_shape = source_slice.get_shape()
            if source_shape[1] != 32:
                raise ValueError(f'{source_path}: expected latent dim 32, got {source_shape[1]}')
            if source_prefix_rows > source_shape[0]:
                raise ValueError(
                    f'{source_path}: requested prefix {source_prefix_rows} > '
                    f'stored rows {source_shape[0]}'
                )
            for start in range(0, source_prefix_rows, STATS_CHUNK_ROWS):
                stop = min(start + STATS_CHUNK_ROWS, source_prefix_rows)
                output[start:stop].copy_(source_slice[start:stop])

    for item in work:
        part_path = _latent_part_path(stream_dir, item)
        _validate_latent_part(part_path, item, plan_sha256=plan_sha256)
        encoded = load_file(part_path)['latents']
        encoded_offset = 0
        for index in item.selected_indices:
            rows = unlabeled[index]['n_patches']
            destination = offsets[index]
            output[destination:destination + rows].copy_(
                encoded[encoded_offset:encoded_offset + rows]
            )
            encoded_offset += rows
        if encoded_offset != item.encoded_rows:
            raise ValueError(
                f'{part_path}: consumed {encoded_offset} rows, expected {item.encoded_rows}'
            )

    save_file({'latents': output}, str(tmp_path))
    with safe_open(str(tmp_path), framework='pt') as file:
        output_slice = file.get_slice('latents')
        for item in work:
            encoded = load_file(_latent_part_path(stream_dir, item))['latents']
            encoded_offset = 0
            for index in item.selected_indices:
                rows = unlabeled[index]['n_patches']
                destination = offsets[index]
                if not torch.equal(
                    output_slice[destination:destination + rows],
                    encoded[encoded_offset:encoded_offset + rows],
                ):
                    raise ValueError(
                        f'{output_path}: selected latent rows differ after materialization '
                        f'at sample {index}'
                    )
                encoded_offset += rows
    tmp_path.rename(output_path)


def _remove_latent_parts(stream_dir: Path, work: list[LatentEncodeWork]) -> None:
    directories = set()
    for item in work:
        path = _latent_part_path(stream_dir, item)
        path.unlink(missing_ok=True)
        path.with_suffix('.safetensors.tmp').unlink(missing_ok=True)
        directories.add(path.parent)
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    root = stream_dir / _PARTS_DIR_NAME
    try:
        root.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _validate_unselected_latent_rows(
    source_path: Path,
    output_path: Path,
    unlabeled: list[dict],
    selected_indices: list[int],
) -> None:
    """Require every row outside the selected sample intervals to remain bitwise unchanged."""
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError('selected sample indices must be sorted and unique')
    offsets = [0]
    for sample in unlabeled:
        offsets.append(offsets[-1] + sample['n_patches'])
    selected_intervals = [(offsets[index], offsets[index + 1]) for index in selected_indices]

    with (
        safe_open(str(source_path), framework='pt') as source_file,
        safe_open(str(output_path), framework='pt') as output_file,
    ):
        source_slice = source_file.get_slice('latents')
        output_slice = output_file.get_slice('latents')
        interval_index = 0
        for start in range(0, offsets[-1], STATS_CHUNK_ROWS):
            stop = min(start + STATS_CHUNK_ROWS, offsets[-1])
            source = source_slice[start:stop]
            output = output_slice[start:stop]
            while (
                interval_index < len(selected_intervals)
                and selected_intervals[interval_index][1] <= start
            ):
                interval_index += 1
            cursor = interval_index
            while cursor < len(selected_intervals) and selected_intervals[cursor][0] < stop:
                selected_start, selected_stop = selected_intervals[cursor]
                lo = max(selected_start, start) - start
                hi = min(selected_stop, stop) - start
                output[lo:hi].copy_(source[lo:hi])
                cursor += 1
            if not torch.equal(source, output):
                raise ValueError(
                    f'{output_path}: unselected latent rows differ from source in [{start}, {stop})'
                )


def _inplane_report_path(staging_dir: Path, shard_id: int) -> Path:
    return staging_dir / 'reports' / f'shard_{shard_id:05d}.json'


def _validate_inplane_shard_receipt(
    staging_dir: Path,
    row: dict,
    plan_sha256: str,
) -> dict:
    """Validate the durable completion marker without replaying shard metadata."""
    shard_id = row['shard_id']
    output_path = staging_dir / f'shard_{shard_id:05d}.safetensors'
    report_path = _inplane_report_path(staging_dir, shard_id)
    _validate_materialized_latent(output_path, row['logical_latent_rows'])
    report = json.loads(report_path.read_text())
    expected_keys = {
        'plan_sha256',
        'shard_id',
        'logical_latent_rows',
        'affected_unlabeled_samples',
        'affected_latent_rows',
        'output_size_bytes',
    }
    if set(report) != expected_keys:
        raise ValueError(f'{report_path}: unexpected shard receipt schema')
    expected = {
        'plan_sha256': plan_sha256,
        'shard_id': shard_id,
        'logical_latent_rows': row['logical_latent_rows'],
        'output_size_bytes': output_path.stat().st_size,
    }
    for key, expected_value in expected.items():
        if report[key] != expected_value:
            raise ValueError(f'{report_path}: shard receipt differs from its plan or output')
    for key in ('affected_unlabeled_samples', 'affected_latent_rows'):
        value = report[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{report_path}: {key} must be a non-negative integer')
    if report['affected_latent_rows'] > row['logical_latent_rows']:
        raise ValueError(f'{report_path}: affected latent rows exceed the logical shard size')
    return report


def _validate_inplane_shard(
    stream_dir: Path,
    staging_dir: Path,
    row: dict,
    plan_sha256: str,
) -> dict:
    """Validate a newly committed shard against its deterministic sample selection."""
    report = _validate_inplane_shard_receipt(staging_dir, row, plan_sha256)
    shard_id = row['shard_id']
    unlabeled, selected = _canonical_inplane_selection(
        stream_dir / f'shard_{shard_id:05d}.msgpack',
        row,
    )
    expected_counts = {
        'affected_unlabeled_samples': len(selected),
        'affected_latent_rows': sum(unlabeled[index]['n_patches'] for index in selected),
    }
    for key, expected_value in expected_counts.items():
        if report[key] != expected_value:
            raise ValueError(f'{_inplane_report_path(staging_dir, shard_id)}: {key} differs')
    return report


def _prepare_inplane_rows(
    stream_dir: Path,
    staging_dir: Path,
    rows: list[dict],
    plan_sha256: str,
) -> tuple[list[dict], list[dict]]:
    completed = []
    pending = []
    for row in rows:
        shard_id = row['shard_id']
        output_path = staging_dir / f'shard_{shard_id:05d}.safetensors'
        report_path = _inplane_report_path(staging_dir, shard_id)
        tmp_path = output_path.with_suffix('.safetensors.tmp')
        if tmp_path.exists():
            pending.append(row)
            continue
        if output_path.exists() != report_path.exists():
            raise FileExistsError(
                f'shard {shard_id}: canonical in-plane output and receipt are incomplete'
            )
        if output_path.exists():
            _validate_inplane_shard_receipt(staging_dir, row, plan_sha256)
            completed.append(row)
        else:
            pending.append(row)
    return completed, pending


def _clean_inplane_temps(staging_dir: Path, rows: list[dict]) -> list[Path]:
    """Remove interrupted temporary files only for shards owned by this invocation."""
    removed = []
    for row in rows:
        shard_id = row['shard_id']
        output_tmp = staging_dir / f'shard_{shard_id:05d}.safetensors.tmp'
        if output_tmp.exists():
            output_tmp.unlink()
            removed.append(output_tmp)
        part_dir = staging_dir / _PARTS_DIR_NAME / f'shard_{shard_id:05d}'
        if part_dir.exists():
            for path in sorted(part_dir.iterdir()):
                if path.name.startswith('.tmp') or path.name.endswith('.tmp'):
                    if not path.is_file() or path.is_symlink():
                        raise ValueError(f'{part_dir}: invalid temporary artifact {path.name}')
                    path.unlink()
                    removed.append(path)
    return removed


def _write_inplane_shard_report(
    staging_dir: Path,
    row: dict,
    result: dict,
    plan_sha256: str,
) -> None:
    output_path = staging_dir / f"shard_{row['shard_id']:05d}.safetensors"
    report = {
        'plan_sha256': plan_sha256,
        'shard_id': row['shard_id'],
        'logical_latent_rows': row['logical_latent_rows'],
        'affected_unlabeled_samples': result['samples'],
        'affected_latent_rows': result['latent_rows'],
        'output_size_bytes': output_path.stat().st_size,
    }
    path = _inplane_report_path(staging_dir, row['shard_id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)


def _remove_completed_inplane_parts(
    staging_dir: Path,
    unlabeled: list[dict],
    selected_indices: list[int],
    *,
    shard_id: int,
    memory_budget_gb: float,
    plan_sha256: str,
) -> None:
    part_dir = staging_dir / _PARTS_DIR_NAME / f'shard_{shard_id:05d}'
    if not part_dir.exists():
        return
    work = _build_encode_work(
        unlabeled,
        selected_indices,
        shard_id=shard_id,
        memory_budget_gb=memory_budget_gb,
    )
    expected = {
        _latent_part_path(staging_dir, item).name: item
        for item in work
    }
    for path in part_dir.iterdir():
        item = expected.get(path.name)
        if item is None or not path.is_file() or path.is_symlink():
            raise ValueError(f'{part_dir}: unexpected recovery artifact {path.name}')
        _validate_latent_part(path, item, plan_sha256=plan_sha256)
    _remove_latent_parts(staging_dir, work)
    try:
        part_dir.rmdir()
    except FileNotFoundError:
        pass
    root = staging_dir / _PARTS_DIR_NAME
    try:
        root.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _recover_inplane_rows(
    stream_dir: Path,
    source_latent_dir: Path,
    staging_dir: Path,
    rows: list[dict],
    *,
    plan_sha256: str,
    memory_budget_gb: float,
) -> tuple[list[int], list[dict]]:
    """Recover interrupted commits and return the pending rows in one scan."""
    recovered = []
    pending = []
    for row in rows:
        shard_id = row['shard_id']
        output_path = staging_dir / f'shard_{shard_id:05d}.safetensors'
        report_path = _inplane_report_path(staging_dir, shard_id)
        tmp_path = output_path.with_suffix('.safetensors.tmp')
        if tmp_path.exists():
            raise FileExistsError(f'incomplete canonical in-plane latent output: {tmp_path}')
        if report_path.exists() and not output_path.exists():
            raise FileExistsError(
                f'shard {shard_id}: canonical in-plane receipt exists without its output'
            )
        if not output_path.exists():
            pending.append(row)
            continue

        unlabeled = None
        selected_indices = None
        if report_path.exists():
            _validate_inplane_shard_receipt(staging_dir, row, plan_sha256)
        else:
            unlabeled, selected_indices = _canonical_inplane_selection(
                stream_dir / f'shard_{shard_id:05d}.msgpack',
                row,
            )
            _validate_materialized_latent(output_path, row['logical_latent_rows'])
            _validate_unselected_latent_rows(
                source_latent_dir / output_path.name,
                output_path,
                unlabeled,
                selected_indices,
            )
            _write_inplane_shard_report(
                staging_dir,
                row,
                {
                    'samples': len(selected_indices),
                    'latent_rows': sum(
                        unlabeled[index]['n_patches'] for index in selected_indices
                    ),
                },
                plan_sha256,
            )
            _validate_inplane_shard(stream_dir, staging_dir, row, plan_sha256)
            recovered.append(shard_id)

        part_dir = staging_dir / _PARTS_DIR_NAME / f'shard_{shard_id:05d}'
        if part_dir.exists():
            if unlabeled is None or selected_indices is None:
                unlabeled, selected_indices = _canonical_inplane_selection(
                    stream_dir / f'shard_{shard_id:05d}.msgpack',
                    row,
                )
            _remove_completed_inplane_parts(
                staging_dir,
                unlabeled,
                selected_indices,
                shard_id=shard_id,
                memory_budget_gb=memory_budget_gb,
                plan_sha256=plan_sha256,
            )
    return recovered, pending


def _init_latent_gpu_worker(
    device_queue,
    gpu_workers: int,
    codec_model: str,
    codec_checkpoint: str,
    compile_mode: str,
) -> None:
    global _LATENT_WORKER_ENCODER
    global _LATENT_WORKER_DEVICE
    global _LATENT_WORKER_DEVICE_ID
    global _LATENT_WORKER_ACTIVE_SHARD

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device_id = device_queue.get()
    os.environ['LOCAL_RANK'] = str(device_id)
    os.environ['LOCAL_WORLD_SIZE'] = str(gpu_workers)
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)

    from pumit.numa import pin_to_gpu_numa

    pin_to_gpu_numa()
    _LATENT_WORKER_ENCODER = build_encoder(
        codec_model,
        codec_checkpoint,
        device,
        compile_mode=compile_mode,
    )
    _LATENT_WORKER_DEVICE = device
    _LATENT_WORKER_DEVICE_ID = device_id
    _LATENT_WORKER_ACTIVE_SHARD = None
    print(f'[ucpt] latent GPU worker {device_id} ready in pid {os.getpid()}')


def _prepare_worker_shard(shard_id: int) -> None:
    """Release allocator cache once before a persistent GPU worker starts a new shard."""
    global _LATENT_WORKER_ACTIVE_SHARD
    if _LATENT_WORKER_ACTIVE_SHARD == shard_id:
        return
    torch.cuda.empty_cache()
    _LATENT_WORKER_ACTIVE_SHARD = shard_id


def _encode_prepared_work(prepared: _PreparedLatentEncode) -> LatentEncodeResult:
    assert _LATENT_WORKER_ENCODER is not None
    assert _LATENT_WORKER_DEVICE is not None

    work = prepared.work
    _prepare_worker_shard(work.shard_id)
    transfer_start = time.monotonic()
    x = prepared.images.to(_LATENT_WORKER_DEVICE)
    transfer_done = time.monotonic()
    try:
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            latent = get_latent_targets(
                _LATENT_WORKER_ENCODER,
                x,
                prepared.da_enc,
                max_adapt=4,
            )
    except torch.OutOfMemoryError as error:
        error.add_note(
            f'latent encoding shard={work.shard_id} part={work.part_id} '
            f'input_shape={tuple(x.shape)} voxel_work={work.voxel_work}'
        )
        raise
    latent = latent.to(dtype=torch.float16, device='cpu')
    encode_done = time.monotonic()
    encoded = torch.empty(work.encoded_rows, 32, dtype=torch.float16)
    offset = 0
    for sample_ordinal, rows in enumerate(prepared.sample_rows):
        if latent.shape[1] != rows:
            raise ValueError(
                f'shard {work.shard_id} part {work.part_id}: encoder produced '
                f'{latent.shape[1]} patches for sample {sample_ordinal}, expected {rows}'
            )
        encoded[offset:offset + rows].copy_(latent[sample_ordinal])
        offset += rows
    if offset != work.encoded_rows:
        raise ValueError(
            f'shard {work.shard_id} part {work.part_id}: encoded {offset} rows, '
            f'expected {work.encoded_rows}'
        )
    end = time.monotonic()
    transfer_seconds = transfer_done - transfer_start
    return LatentEncodeResult(
        work=work,
        prepare_seconds=prepared.prepare_seconds,
        replay_seconds=prepared.replay_seconds,
        transfer_seconds=transfer_seconds,
        encode_seconds=encode_done - transfer_done,
        write_seconds=0.0,
        total_seconds=(
            prepared.prepare_seconds
            + prepared.replay_seconds
            + end - transfer_start
        ),
        device_id=_LATENT_WORKER_DEVICE_ID,
        encoded=encoded,
    )


@contextmanager
def _latent_gpu_pool(
    args,
    *,
    rank: int,
    prepared_loader: _LatentPreparedLoader,
):
    """Create one persistent worker per local GPU for sequential shard encoding."""
    if args.num_workers < 1:
        raise ValueError(f'num_workers must be positive, got {args.num_workers}')
    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError('latent encoding requires at least one visible GPU')
    gpu_workers = available_gpus
    cache_archive = (
        Path(args.cache_archive).resolve()
        if getattr(args, 'cache_archive', None) is not None
        else None
    )
    cache_dir = None
    if cache_archive is not None:
        cache_dir = _extract_latent_compile_cache(cache_archive)
    archiver = (
        _PerShardCompileCacheArchiver(cache_archive, cache_dir)
        if rank == 0 and cache_dir is not None
        else None
    )

    max_inflight = max(1, 2 * gpu_workers)
    print(
        f'[ucpt] latent encoding: {gpu_workers} shared GPU workers, '
        f'{_LATENT_REPLAY_WORKERS} fork replay workers, '
        f'{prepared_loader.capacity} prepared batches, '
        f'{max_inflight} in-flight GPU batches'
    )
    context = get_context('spawn')
    device_queue = context.Queue()
    for device_id in range(gpu_workers):
        device_queue.put(device_id)

    try:
        with ProcessPoolExecutor(
            max_workers=gpu_workers,
            mp_context=context,
            initializer=_init_latent_gpu_worker,
            initargs=(
                device_queue,
                gpu_workers,
                args.codec_model,
                str(Path(args.codec_checkpoint).resolve()),
                args.compile_mode,
            ),
        ) as gpu_pool:
            yield gpu_pool, max_inflight, archiver
    except BaseException as error:
        if archiver is not None:
            try:
                archiver.close()
            except BaseException as archive_error:
                error.add_note(
                    'latent compile-cache archiver cleanup also failed: '
                    f'{type(archive_error).__name__}: {archive_error}'
                )
        raise
    else:
        if archiver is not None:
            archiver.close()


@dataclass(frozen=True)
class _LatentShardFinalization:
    shard_id: int
    expected_rows: int
    source_path: Path | None
    unlabeled: list[dict]
    source_prefix_samples: int
    work: tuple[LatentEncodeWork, ...]
    parts_dir: Path
    output_path: Path
    plan_sha256: str | None
    cache_archiver: _PerShardCompileCacheArchiver | None


@dataclass(frozen=True)
class _EncodedLatentShard:
    result: dict
    finalization: _LatentShardFinalization


@dataclass(frozen=True)
class _LatentShardFinalizationResult:
    shard_id: int
    materialize_seconds: float
    validate_seconds: float
    cleanup_seconds: float
    total_seconds: float


def _iter_prepared_encode_results(
    pending: list[LatentEncodeWork],
    prepared_loader: _LatentPreparedLoader,
    gpu_pool: ProcessPoolExecutor,
    *,
    max_inflight: int,
):
    pending_iter = iter(pending)
    replay_outstanding: set[LatentEncodeWork] = set()
    gpu_futures: dict[Future, tuple[LatentEncodeWork, float, bool]] = {}
    first_prepared = True

    def fill_replay() -> None:
        while len(replay_outstanding) < prepared_loader.capacity:
            work = next(pending_iter, None)
            if work is None:
                return
            if work in replay_outstanding:
                raise ValueError(f'duplicate latent replay work: {work}')
            prepared_loader.submit(work)
            replay_outstanding.add(work)

    fill_replay()
    try:
        while replay_outstanding or gpu_futures:
            while replay_outstanding and len(gpu_futures) < max_inflight:
                wait_start = time.monotonic()
                prepared = prepared_loader.get()
                prepared_wait_seconds = time.monotonic() - wait_start
                work = prepared.work
                if work not in replay_outstanding:
                    raise ValueError(
                        f'prepared replay returned unexpected work for shard '
                        f'{work.shard_id} part {work.part_id}'
                    )
                replay_outstanding.remove(work)
                future = gpu_pool.submit(_encode_prepared_work, prepared)
                gpu_futures[future] = (work, prepared_wait_seconds, first_prepared)
                first_prepared = False
                fill_replay()

            if not gpu_futures:
                continue
            done, _ = wait(gpu_futures, return_when=FIRST_COMPLETED)
            for future in done:
                expected, prepared_wait_seconds, was_first = gpu_futures.pop(future)
                result = future.result()
                if result.work != expected:
                    raise ValueError(
                        f'GPU worker returned unexpected work for shard '
                        f'{result.work.shard_id} part {result.work.part_id}'
                    )
                if result.device_id is None:
                    raise ValueError('GPU worker did not report its device ID')
                yield _LatentEncodeBatchResult(
                    result=result,
                    device_id=result.device_id,
                    gpu_wall_seconds=(
                        result.transfer_seconds
                        + result.encode_seconds
                    ),
                    first_prepared_wait_seconds=(
                        prepared_wait_seconds if was_first else 0.0
                    ),
                    prepared_queue_wait_seconds=(
                        0.0 if was_first else prepared_wait_seconds
                    ),
                )
    except BaseException:
        for future in gpu_futures:
            future.cancel()
        raise


# A single writer minimizes GIL competition with the parent's prepared-batch pickling feed.
_LATENT_PART_WRITER_THREADS = 1


def _write_encoded_part(
    parts_dir: Path,
    result: LatentEncodeResult,
    plan_sha256: str | None,
) -> tuple[LatentEncodeResult, float]:
    """Persist one GPU result off the GPU workers' critical path."""
    if result.encoded is None:
        raise ValueError(
            f'shard {result.work.shard_id} part {result.work.part_id}: '
            'GPU result lacks encoded latents'
        )
    start = time.monotonic()
    _write_latent_part(parts_dir, result.work, result.encoded, plan_sha256=plan_sha256)
    return result, time.monotonic() - start


def _encode_selected_latent_shard_parts(
    row: dict,
    unlabeled: list[dict],
    selected_indices: list[int],
    *,
    stream_dir: Path,
    source_latent_dir: Path | None,
    parts_dir: Path,
    output_dir: Path,
    source_prefix_samples: int,
    gpu_pool: ProcessPoolExecutor,
    memory_budget_gb: float,
    max_inflight: int,
    cache_archiver: _PerShardCompileCacheArchiver | None,
    prepared_loader: _LatentPreparedLoader,
    plan_sha256: str | None = None,
) -> _EncodedLatentShard:
    shard_id = row['shard_id']
    work = _build_encode_work(
        unlabeled,
        selected_indices,
        shard_id=shard_id,
        memory_budget_gb=memory_budget_gb,
    )
    planned_indices = sorted(
        index
        for item in work
        for index in item.selected_indices
    )
    if planned_indices != selected_indices:
        raise ValueError(f'shard {shard_id}: GPU work does not cover the filtered selection')

    pending = []
    completed_samples = 0
    for item in work:
        part_path = _latent_part_path(parts_dir, item)
        if part_path.exists():
            _validate_latent_part(part_path, item, plan_sha256=plan_sha256)
            completed_samples += len(item.selected_indices)
        else:
            pending.append(item)
    total_samples = len(selected_indices)
    print(
        f'[ucpt] latent shard {shard_id}: {completed_samples:,}/{total_samples:,} samples, '
        f'{len(pending)}/{len(work)} GPU batches pending, '
        f'{sum(item.voxel_work for item in pending):,} input voxels'
    )

    encode_results = _iter_prepared_encode_results(
        pending,
        prepared_loader,
        gpu_pool,
        max_inflight=max_inflight,
    )

    pipeline_start = time.monotonic()
    completed = len(work) - len(pending)
    next_progress = (completed // 25 + 1) * 25
    timings = {
        'prepare': 0.0,
        'replay': 0.0,
        'transfer': 0.0,
        'encode': 0.0,
        'write': 0.0,
        'total': 0.0,
        'first_prepared_wait': 0.0,
        'prepared_queue_wait': 0.0,
    }
    device_summaries: dict[int, dict[str, float | int]] = {}
    write_futures: list[Future[tuple[LatentEncodeResult, float]]] = []
    part_writer = ThreadPoolExecutor(
        max_workers=_LATENT_PART_WRITER_THREADS,
        thread_name_prefix='latent-part-writer',
    )
    try:
        for batch_result in encode_results:
            result = batch_result.result
            write_futures.append(
                part_writer.submit(_write_encoded_part, parts_dir, result, plan_sha256)
            )
            timings['first_prepared_wait'] += batch_result.first_prepared_wait_seconds
            timings['prepared_queue_wait'] += (
                batch_result.prepared_queue_wait_seconds
            )
            device_summary = device_summaries.setdefault(
                batch_result.device_id,
                {
                    'gpu_batches': 0,
                    'samples': 0,
                    'voxel_work': 0,
                    'gpu_wall': 0.0,
                    'max_gpu_batch_wall': 0.0,
                    'first_prepared_wait': 0.0,
                    'prepared_queue_wait': 0.0,
                    'prepare': 0.0,
                    'replay': 0.0,
                    'transfer': 0.0,
                    'encode': 0.0,
                    'write': 0.0,
                },
            )
            device_summary['gpu_batches'] += 1
            device_summary['gpu_wall'] += batch_result.gpu_wall_seconds
            device_summary['max_gpu_batch_wall'] = max(
                device_summary['max_gpu_batch_wall'],
                batch_result.gpu_wall_seconds,
            )
            device_summary['first_prepared_wait'] += (
                batch_result.first_prepared_wait_seconds
            )
            device_summary['prepared_queue_wait'] += (
                batch_result.prepared_queue_wait_seconds
            )
            completed += 1
            samples = len(result.work.selected_indices)
            completed_samples += samples
            timings['prepare'] += result.prepare_seconds
            timings['replay'] += result.replay_seconds
            timings['transfer'] += result.transfer_seconds
            timings['encode'] += result.encode_seconds
            timings['total'] += result.total_seconds
            device_summary['samples'] += samples
            device_summary['voxel_work'] += result.work.voxel_work
            device_summary['prepare'] += result.prepare_seconds
            device_summary['replay'] += result.replay_seconds
            device_summary['transfer'] += result.transfer_seconds
            device_summary['encode'] += result.encode_seconds
            if completed == len(work) or completed >= next_progress:
                fraction = completed_samples / total_samples if total_samples else 1.0
                print(
                    f'[ucpt] latent shard {shard_id}: '
                    f'{completed_samples:,}/{total_samples:,} samples ({fraction:.1%}), '
                    f'{completed}/{len(work)} GPU batches'
                )
                next_progress = (completed // 25 + 1) * 25
        # Finalization scatters parts from disk, so every write must land before this returns.
        for write_future in write_futures:
            written, write_seconds = write_future.result()
            timings['write'] += write_seconds
            timings['total'] += write_seconds
            device_summary = device_summaries[written.device_id]
            device_summary['write'] += write_seconds
    except BaseException:
        for write_future in write_futures:
            write_future.cancel()
        wait(write_futures)
        raise
    finally:
        part_writer.shutdown(wait=True)
    timings['pipeline_wall'] = time.monotonic() - pipeline_start

    source_path = (
        source_latent_dir / f'shard_{shard_id:05d}.safetensors'
        if source_latent_dir is not None
        else None
    )
    output_path = output_dir / f'shard_{shard_id:05d}.safetensors'
    return _EncodedLatentShard(
        result={
            'shard_id': shard_id,
            'samples': total_samples,
            'latent_rows': sum(
                unlabeled[index]['n_patches'] for index in selected_indices
            ),
            'gpu_batches': len(work),
            'scheduler_summary': {
                'pending_gpu_batches': len(pending),
                'prepared_capacity': prepared_loader.capacity,
                'devices': device_summaries,
            },
            'timings': timings,
        },
        finalization=_LatentShardFinalization(
            shard_id=shard_id,
            expected_rows=row['logical_latent_rows'],
            source_path=source_path,
            unlabeled=unlabeled,
            source_prefix_samples=source_prefix_samples,
            work=tuple(work),
            parts_dir=parts_dir,
            output_path=output_path,
            plan_sha256=plan_sha256,
            cache_archiver=cache_archiver,
        ),
    )


def _finalize_selected_latent_shard(
    finalization: _LatentShardFinalization,
) -> _LatentShardFinalizationResult:
    start = time.monotonic()
    _materialize_latent_parts(
        finalization.source_path,
        finalization.unlabeled,
        finalization.source_prefix_samples,
        list(finalization.work),
        finalization.parts_dir,
        finalization.output_path,
        plan_sha256=finalization.plan_sha256,
    )
    materialized = time.monotonic()
    _validate_materialized_latent(
        finalization.output_path,
        finalization.expected_rows,
    )
    validated = time.monotonic()
    _remove_latent_parts(finalization.parts_dir, list(finalization.work))
    if finalization.cache_archiver is not None:
        finalization.cache_archiver.submit()
    end = time.monotonic()
    return _LatentShardFinalizationResult(
        shard_id=finalization.shard_id,
        materialize_seconds=materialized - start,
        validate_seconds=validated - materialized,
        cleanup_seconds=end - validated,
        total_seconds=end - start,
    )


def _finalization_timing(result: _LatentShardFinalizationResult) -> dict[str, float]:
    return {
        'materialize': result.materialize_seconds,
        'validate': result.validate_seconds,
        'cleanup': result.cleanup_seconds,
        'total': result.total_seconds,
    }


def _encode_selected_latent_shard(
    row: dict,
    unlabeled: list[dict],
    selected_indices: list[int],
    *,
    stream_dir: Path,
    source_latent_dir: Path | None,
    parts_dir: Path,
    output_dir: Path,
    source_prefix_samples: int,
    gpu_pool: ProcessPoolExecutor,
    memory_budget_gb: float,
    max_inflight: int,
    cache_archiver: _PerShardCompileCacheArchiver | None,
    prepared_loader: _LatentPreparedLoader,
    plan_sha256: str | None = None,
) -> dict:
    encoded = _encode_selected_latent_shard_parts(
        row,
        unlabeled,
        selected_indices,
        stream_dir=stream_dir,
        source_latent_dir=source_latent_dir,
        parts_dir=parts_dir,
        output_dir=output_dir,
        source_prefix_samples=source_prefix_samples,
        gpu_pool=gpu_pool,
        memory_budget_gb=memory_budget_gb,
        max_inflight=max_inflight,
        cache_archiver=cache_archiver,
        prepared_loader=prepared_loader,
        plan_sha256=plan_sha256,
    )
    finalization = _finalize_selected_latent_shard(encoded.finalization)
    return {
        **encoded.result,
        'finalization': _finalization_timing(finalization),
    }


class _BoundedLatentShardFinalizer:
    """Run at most one shard finalization while the next shard encodes."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='latent-shard-finalizer',
        )
        self._future: Future[_LatentShardFinalizationResult] | None = None
        self._closed = False

    def submit(
        self,
        finalization: _LatentShardFinalization,
    ) -> _LatentShardFinalizationResult | None:
        if self._closed:
            raise RuntimeError('latent shard finalizer is closed')
        completed = self.drain()
        self._future = self._executor.submit(
            _finalize_selected_latent_shard,
            finalization,
        )
        return completed

    def drain(self) -> _LatentShardFinalizationResult | None:
        future = self._future
        if future is None:
            return None
        self._future = None
        return future.result()

    def close(self) -> _LatentShardFinalizationResult | None:
        if self._closed:
            return None
        self._closed = True
        try:
            return self.drain()
        finally:
            self._executor.shutdown(wait=True)


def _inplane_receipt(plan: dict, rows: list[dict], reports: list[dict]) -> dict:
    return {
        'latent_contract': CANONICAL_INPLANE_LATENT_CONTRACT,
        'codec_model': plan['codec_model'],
        'codec_checkpoint': plan['codec_checkpoint'],
        'source_stream': plan['source_stream'],
        'source_manifest_sha256': plan['source_manifest_sha256'],
        'source_latent_dir': plan['source_latent_dir'],
        'source_latent_receipt_sha256': plan['source_latent_receipt_sha256'],
        'materialized_shards': [row['shard_id'] for row in rows],
        'logical_latent_rows': sum(row['logical_latent_rows'] for row in rows),
        'affected_unlabeled_samples': sum(
            report['affected_unlabeled_samples'] for report in reports
        ),
        'affected_latent_rows': sum(report['affected_latent_rows'] for report in reports),
    }


def _validate_published_inplane_latents(
    stream_dir: Path,
    rows: list[dict],
    expected_plan: dict,
    *,
    recover_missing_outer: bool = False,
) -> bool:
    latent_dir = stream_dir / 'latents'
    inner_receipt = latent_dir / 'latent-materialized.json'
    outer_receipt = stream_dir / 'latent-materialized.json'
    if not inner_receipt.exists():
        raise FileNotFoundError('published canonical in-plane latents lack their inner receipt')
    plan = json.loads((latent_dir / 'plan.json').read_text())
    if plan != expected_plan:
        raise ValueError(f'{latent_dir}: published plan differs from this invocation')
    temporary = sorted(str(path) for path in latent_dir.rglob('*.tmp'))
    if temporary:
        raise FileExistsError(f'published canonical in-plane latents contain {temporary[0]}')
    parts_dir = latent_dir / _PARTS_DIR_NAME
    if parts_dir.exists():
        raise FileExistsError(f'published canonical in-plane latents contain {parts_dir}')

    plan_sha256 = _json_sha256(plan)
    reports = [
        _validate_inplane_shard_receipt(latent_dir, row, plan_sha256)
        for row in rows
    ]
    expected_receipt = _inplane_receipt(plan, rows, reports)
    if json.loads(inner_receipt.read_text()) != expected_receipt:
        raise ValueError('published canonical in-plane latent receipt differs from its plan')

    recovered = False
    if outer_receipt.exists():
        if inner_receipt.read_bytes() != outer_receipt.read_bytes():
            raise ValueError('inner and outer canonical in-plane latent receipts differ')
    elif recover_missing_outer:
        os.link(inner_receipt, outer_receipt)
        recovered = True
    else:
        raise FileNotFoundError('published canonical in-plane latents lack their outer receipt')
    return recovered


def _publish_inplane_latents(
    stream_dir: Path,
    staging_dir: Path,
    rows: list[dict],
    plan: dict,
) -> None:
    plan_sha256 = _json_sha256(plan)
    _, pending = _prepare_inplane_rows(stream_dir, staging_dir, rows, plan_sha256)
    if pending:
        raise RuntimeError(f'cannot publish canonical in-plane latents: {len(pending)} shards pending')
    temporary = sorted(str(path) for path in staging_dir.rglob('*.tmp'))
    if temporary:
        raise FileExistsError(f'cannot publish with temporary files: {temporary[:3]}')
    parts_dir = staging_dir / _PARTS_DIR_NAME
    if parts_dir.exists():
        raise FileExistsError(f'cannot publish with latent parts present: {parts_dir}')

    reports = [
        _validate_inplane_shard_receipt(staging_dir, row, plan_sha256)
        for row in rows
    ]
    receipt = _inplane_receipt(plan, rows, reports)
    inner_receipt = staging_dir / 'latent-materialized.json'
    _write_or_validate_json(inner_receipt, receipt)

    final_dir = stream_dir / 'latents'
    outer_receipt = stream_dir / 'latent-materialized.json'
    require_absent(final_dir)
    require_absent(outer_receipt)
    staging_dir.rename(final_dir)
    os.link(final_dir / inner_receipt.name, outer_receipt)


def _encode_canonical_inplane_latents(
    args,
    prepared_loader: _LatentPreparedLoader,
) -> None:
    """Selectively replace latents whose replay changes under canonical in-plane metadata."""
    if int(os.environ.get('LOCAL_WORLD_SIZE', 1)) != 1:
        raise RuntimeError(
            'encode-latents must be launched once per node, without torchrun'
        )
    stream_dir = args.stream.resolve()
    source_latent_dir = args.source_latent_dir.resolve()
    codec_checkpoint = args.codec_checkpoint.resolve()
    rows = load_manifest(stream_dir)
    if not rows:
        raise ValueError(f'{stream_dir}: manifest is empty')
    requested_rows, _ = _select_manifest_rows(
        rows,
        shard_offset=args.shard_offset,
        shards=args.shards,
    )
    rank, world_size = distributed_context()
    if world_size > 1:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

    staging_dir = stream_dir / _CANONICAL_INPLANE_STAGING_NAME
    final_dir = stream_dir / 'latents'
    try:
        if rank == 0:
            provenance = _validate_canonical_inplane_source(
                stream_dir,
                source_latent_dir,
                rows,
                codec_model=args.codec_model,
                codec_checkpoint=codec_checkpoint,
            )
            plan = {
                'latent_contract': CANONICAL_INPLANE_LATENT_CONTRACT,
                'stream': str(stream_dir),
                'stream_manifest_sha256': provenance['stream_manifest_sha256'],
                'source_stream': provenance['source_stream'],
                'source_stream_fingerprint': provenance['source_stream_fingerprint'],
                'source_manifest_sha256': provenance['source_manifest_sha256'],
                'source_latent_dir': str(source_latent_dir),
                'source_latent_receipt_sha256': provenance['source_latent_receipt_sha256'],
                'codec_model': args.codec_model,
                'codec_checkpoint': str(codec_checkpoint),
                'codec_checkpoint_sha256': sha256_file(codec_checkpoint),
                'compile_mode': args.compile_mode,
                'memory_budget_gb': args.memory_budget_gb,
                'materialized_shards': [row['shard_id'] for row in rows],
            }
            if final_dir.exists():
                if staging_dir.exists():
                    raise FileExistsError('both published and staging latent directories exist')
                recovered_outer = _validate_published_inplane_latents(
                    stream_dir,
                    rows,
                    plan,
                    recover_missing_outer=True,
                )
                if recovered_outer:
                    print('[ucpt] recovered canonical in-plane outer latent receipt')
                published = True
                pending_shard_ids = []
            else:
                published = False
                if staging_dir.exists():
                    plan_path = staging_dir / 'plan.json'
                    if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
                        raise ValueError(f'{staging_dir}: existing plan differs from this invocation')
                    removed = _clean_inplane_temps(staging_dir, requested_rows)
                    if removed:
                        print(f'[ucpt] removed {len(removed)} interrupted latent temp files')
                    recovered_shards, pending_rows = _recover_inplane_rows(
                        stream_dir,
                        source_latent_dir,
                        staging_dir,
                        requested_rows,
                        plan_sha256=_json_sha256(plan),
                        memory_budget_gb=args.memory_budget_gb,
                    )
                    if recovered_shards:
                        print(
                            '[ucpt] recovered canonical in-plane shard receipts: '
                            f'{recovered_shards}'
                        )
                else:
                    staging_dir.mkdir()
                    write_json(staging_dir / 'plan.json', plan)
                    pending_rows = requested_rows
                pending_shard_ids = [row['shard_id'] for row in pending_rows]
        else:
            plan = None
            published = None
            pending_shard_ids = None

        if dist.is_initialized():
            payload = [plan, published, pending_shard_ids]
            dist.broadcast_object_list(payload, src=0)
            plan, published, pending_shard_ids = payload
        assert (
            isinstance(plan, dict)
            and isinstance(published, bool)
            and isinstance(pending_shard_ids, list)
        )
        if published:
            print('[ucpt] canonical in-plane latents are already complete')
            return

        plan_sha256 = _json_sha256(plan)
        owned_ids = set(
            partition_shard_ids(
                pending_shard_ids,
                rank,
                world_size,
            )
        )
        owned_rows = [row for row in requested_rows if row['shard_id'] in owned_ids]
        print(
            f'[ucpt] canonical in-plane latent rank {rank}/{world_size}: '
            f'{len(owned_rows)}/{len(pending_shard_ids)} pending shards assigned'
        )

        if owned_rows:
            with _latent_gpu_pool(
                args,
                rank=rank,
                prepared_loader=prepared_loader,
            ) as (gpu_pool, max_inflight, archiver):
                for row in owned_rows:
                    shard_id = row['shard_id']
                    unlabeled, selected_indices = _canonical_inplane_selection(
                        stream_dir / f'shard_{shard_id:05d}.msgpack',
                        row,
                    )
                    result = _encode_selected_latent_shard(
                        row,
                        unlabeled,
                        selected_indices,
                        stream_dir=stream_dir,
                        source_latent_dir=source_latent_dir,
                        parts_dir=staging_dir,
                        output_dir=staging_dir,
                        source_prefix_samples=len(unlabeled),
                        gpu_pool=gpu_pool,
                        memory_budget_gb=args.memory_budget_gb,
                        max_inflight=max_inflight,
                        cache_archiver=archiver,
                        prepared_loader=prepared_loader,
                        plan_sha256=plan_sha256,
                    )
                    source_path = source_latent_dir / f'shard_{shard_id:05d}.safetensors'
                    output_path = staging_dir / source_path.name
                    _validate_unselected_latent_rows(
                        source_path,
                        output_path,
                        unlabeled,
                        selected_indices,
                    )
                    _write_inplane_shard_report(staging_dir, row, result, plan_sha256)
                    _validate_inplane_shard(stream_dir, staging_dir, row, plan_sha256)
                    print(
                        f'[ucpt] canonical in-plane shard {shard_id} complete: '
                        f"{result['samples']:,} samples, {result['latent_rows']:,} rows"
                    )

        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            completed_rows, pending_rows = _prepare_inplane_rows(
                stream_dir,
                staging_dir,
                rows,
                plan_sha256,
            )
            if pending_rows:
                print(
                    f'[ucpt] canonical in-plane staging: {len(completed_rows)}/{len(rows)} '
                    'shards complete; not publishing'
                )
            else:
                _publish_inplane_latents(stream_dir, staging_dir, rows, plan)
                print(f'[ucpt] published canonical in-plane latents: {final_dir}')
        if dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
