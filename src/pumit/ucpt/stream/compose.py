"""Compose finalized stream segments without duplicating their large artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors import safe_open
import yaml

from ..mask_store import mask_shard_path
from .manifest import (
    BUILD_PLAN_NAME,
    MANIFEST_NAME,
    SUMMARY_NAME,
    finalize_stream,
    require_absent,
    sha256_bytes,
    sha256_file,
    summarize,
    write_json,
)
from .verify import FULL_LATENT_CONTRACT


COMPOSITION_CONTRACT = 'concatenated-stream-v1'
NORMALIZATION_CONTRACT = 'fixed-source-v1'
JOURNAL_PATH = Path('.components/composition-journal.jsonl')


def _same_inode(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _link(source: Path, destination: Path) -> None:
    """Create an exact hardlink without replacing another inode."""
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f'hardlink source is not a regular file: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and not destination.is_symlink() and _same_inode(source, destination):
            return
        raise FileExistsError(f'refusing to overwrite a different inode: {destination}')
    os.link(source, destination)
    if not _same_inode(source, destination):
        raise RuntimeError(f'hardlink inode check failed: {source} -> {destination}')


def _move_by_link(
    source: Path,
    destination: Path,
    *,
    expected_inode: tuple[int, int] | None = None,
) -> None:
    """Move one same-filesystem file through a checked, no-overwrite hardlink."""
    source_exists = source.exists() or source.is_symlink()
    destination_exists = destination.exists() or destination.is_symlink()
    if not source_exists:
        if destination_exists and destination.is_file() and not destination.is_symlink():
            if expected_inode is not None:
                destination_stat = destination.stat()
                actual_inode = (destination_stat.st_dev, destination_stat.st_ino)
                if actual_inode != expected_inode:
                    raise FileExistsError(
                        f'move destination has inode {actual_inode}, expected {expected_inode}: '
                        f'{destination}'
                    )
            return
        if destination_exists:
            raise FileExistsError(f'move destination is not a regular file: {destination}')
        raise FileNotFoundError(f'move source and destination are both absent: {source} -> {destination}')
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f'move source is not a regular file: {source}')
    source_stat = source.stat()
    source_inode = (source_stat.st_dev, source_stat.st_ino)
    if expected_inode is not None and source_inode != expected_inode:
        raise FileExistsError(
            f'move source has inode {source_inode}, expected {expected_inode}: {source}'
        )
    if destination_exists:
        if (
            not destination.is_file()
            or destination.is_symlink()
            or not _same_inode(source, destination)
        ):
            raise FileExistsError(f'move destination has a different inode: {destination}')
    else:
        _link(source, destination)
    if not _same_inode(source, destination):
        raise RuntimeError(f'move inode check failed: {source} -> {destination}')
    source.unlink()


def _finish_suffix_move(
    source: Path,
    destination: Path,
    *,
    expected_inode: tuple[int, int],
) -> None:
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or destination.is_symlink():
            raise FileExistsError(f'suffix move destination is not a regular file: {destination}')
        destination_stat = destination.stat()
        actual_inode = (destination_stat.st_dev, destination_stat.st_ino)
        if actual_inode != expected_inode:
            raise FileExistsError(
                f'suffix move destination has inode {actual_inode}, expected {expected_inode}: '
                f'{destination}'
            )
        if source.exists() or source.is_symlink():
            if source.is_file() and not source.is_symlink() and _same_inode(source, destination):
                source.unlink()
        return
    _move_by_link(source, destination, expected_inode=expected_inode)


def _write_yaml(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and yaml.safe_load(path.read_text()) == value:
            return
        raise FileExistsError(f'existing YAML artifact differs: {path}')
    require_absent(path)
    data = yaml.safe_dump(value, sort_keys=True).encode()
    tmp = path.parent / f'.{path.name}.{os.getpid()}.tmp'
    require_absent(tmp)
    with tmp.open('xb') as file:
        file.write(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def _write_json_matching(path: Path, value) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and json.loads(path.read_text()) == value:
            return
        raise FileExistsError(f'existing JSON artifact differs: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _append_journal(staging: Path, event: str, **details) -> None:
    path = staging / JOURNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'event': event, **details}
    data = (json.dumps(record, sort_keys=True) + '\n').encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o664)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_journal(staging: Path) -> list[dict]:
    path = staging / JOURNAL_PATH
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f'composition staging lacks a valid journal: {path}')
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{line_number}: invalid journal record') from error
    if not records or records[0].get('event') != 'suffix-renamed':
        raise ValueError(f'{path}: first journal record is not suffix-renamed')
    return records


def _load_manifest_path(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [row['shard_id'] for row in rows] != list(range(len(rows))):
        raise ValueError(f'{path}: manifest shard IDs are not contiguous from zero')
    return rows


def _load_finalized_paths(
    *,
    display_path: Path,
    meta_path: Path,
    build_path: Path,
    manifest_path: Path,
    summary_path: Path,
) -> tuple[dict, dict, list[dict]]:
    for path in (meta_path, build_path, manifest_path, summary_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'missing finalized stream artifact: {path}')
    meta = yaml.safe_load(meta_path.read_text())
    plan = yaml.safe_load(build_path.read_text())
    rows = _load_manifest_path(manifest_path)
    if not meta.get('stream_complete'):
        raise ValueError(f'stream is not complete: {display_path}')
    if meta.get('n_shards') != len(rows):
        raise ValueError(f'{display_path}: meta shard count differs from manifest')
    if meta.get('build_fingerprint') != plan.get('fingerprint'):
        raise ValueError(f'{display_path}: build fingerprint differs from meta')
    manifest_sha256 = sha256_file(manifest_path)
    if meta.get('manifest_sha256') != manifest_sha256:
        raise ValueError(f'{display_path}: manifest hash differs from meta')
    return meta, plan, rows


def _load_finalized(stream: Path, *, require_ready: bool) -> tuple[dict, dict, list[dict]]:
    meta, plan, rows = _load_finalized_paths(
        display_path=stream,
        meta_path=stream / 'meta.yaml',
        build_path=stream / BUILD_PLAN_NAME,
        manifest_path=stream / MANIFEST_NAME,
        summary_path=stream / SUMMARY_NAME,
    )
    manifest_sha256 = meta['manifest_sha256']
    if require_ready:
        ready_path = stream / 'READY.json'
        if not ready_path.is_file():
            raise FileNotFoundError(f'prefix stream is not READY: {ready_path}')
        ready = json.loads(ready_path.read_text())
        if ready.get('fingerprint') != meta.get('fingerprint'):
            raise ValueError(f'{stream}: READY fingerprint differs from meta')
        if ready.get('manifest_sha256') != manifest_sha256:
            raise ValueError(f'{stream}: READY manifest hash differs from meta')
        if ready.get('verified_shards') != len(rows):
            raise ValueError(f'{stream}: READY shard count differs from manifest')
        latent_receipt_path = stream / 'latent-materialized.json'
        if not latent_receipt_path.is_file() or latent_receipt_path.is_symlink():
            raise FileNotFoundError(f'prefix latent receipt is missing: {latent_receipt_path}')
        if ready.get('latent_receipt_sha256') != sha256_file(latent_receipt_path):
            raise ValueError(f'{stream}: READY latent receipt hash differs from artifact')
        latent_receipt = json.loads(latent_receipt_path.read_text())
        for key in ('codec_model', 'codec_checkpoint'):
            if not isinstance(latent_receipt.get(key), str) or not latent_receipt[key]:
                raise ValueError(f'{stream}: prefix latent receipt lacks a valid {key}')
        # verify derives READY's contract from the receipt: full receipts carry no
        # latent_contract key and publish as FULL_LATENT_CONTRACT.
        if latent_receipt.get('latent_contract', FULL_LATENT_CONTRACT) != ready.get('latent_contract'):
            raise ValueError(f'{stream}: READY latent contract differs from latent receipt')
    return meta, plan, rows


def _component_artifact(staging: Path, relative: str) -> Path:
    archived = staging / '.components' / 'suffix' / relative
    top_level = staging / relative
    if archived.exists() or archived.is_symlink():
        return archived
    return top_level


def _load_suffix_snapshot(staging: Path, source_suffix: Path) -> tuple[dict, dict, list[dict]]:
    return _load_finalized_paths(
        display_path=source_suffix,
        meta_path=_component_artifact(staging, 'meta.yaml'),
        build_path=_component_artifact(staging, BUILD_PLAN_NAME),
        manifest_path=_component_artifact(staging, MANIFEST_NAME),
        summary_path=_component_artifact(staging, SUMMARY_NAME),
    )


def _require_inventory(directory: Path, suffix: str, count: int) -> list[Path]:
    expected = [directory / f'shard_{shard_id:05d}{suffix}' for shard_id in range(count)]
    actual = sorted(directory.glob(f'shard_*{suffix}')) if directory.exists() else []
    if actual != expected:
        raise ValueError(f'{directory}: shard inventory is not exactly [0, {count})')
    if any(not path.is_file() or path.is_symlink() for path in actual):
        raise ValueError(f'{directory}: shard inventory contains a non-regular file')
    return actual


def _optional_latent_inventory(stream: Path, count: int) -> list[Path]:
    latent_dir = stream / 'latents'
    actual = sorted(latent_dir.glob('shard_*.safetensors')) if latent_dir.exists() else []
    if not actual:
        return []
    return _require_inventory(latent_dir, '.safetensors', count)


def _validate_compatible(prefix_meta: dict, prefix_plan: dict, suffix_meta: dict, suffix_plan: dict) -> None:
    plan_keys = (
        'algorithm',
        'packing_policy',
        'batches_per_shard',
        'budget_ms',
        'label_budget_fraction',
        'cost_model',
        'config',
    )
    for key in plan_keys:
        if prefix_plan.get(key) != suffix_plan.get(key):
            raise ValueError(f'component build plans differ at {key!r}')
    meta_keys = (
        'mask_storage',
        'sample_metadata_contract',
        'n_total_records',
        'n_labeled_records',
    )
    for key in meta_keys:
        if prefix_meta.get(key) != suffix_meta.get(key):
            raise ValueError(f'component stream metadata differs at {key!r}')


def _stats_identity(prefix: Path, prefix_meta: dict, prefix_rows: list[dict]) -> tuple[str, int]:
    stats_path = prefix / 'latents' / 'stats.safetensors'
    if not stats_path.is_file():
        raise FileNotFoundError(f'prefix latent stats are missing: {stats_path}')
    with safe_open(str(stats_path), framework='pt') as file:
        if 'count' not in file.keys():
            raise ValueError(f'prefix latent stats lack count: {stats_path}')
        count = int(file.get_tensor('count').item())
    expected_count = sum(int(row['logical_latent_rows']) for row in prefix_rows)
    if count != expected_count:
        raise ValueError(f'prefix stats count {count} differs from manifest rows {expected_count}')
    stats_sha256 = sha256_file(stats_path)
    ready = json.loads((prefix / 'READY.json').read_text())
    if ready.get('stats_sha256') != stats_sha256:
        raise ValueError(f'{prefix}: READY stats hash differs from stats.safetensors')
    if ready.get('logical_latent_rows') != count:
        raise ValueError(f'{prefix}: READY logical latent rows differ from stats count')
    latent_hashes = ready.get('latent_shard_sha256')
    if not isinstance(latent_hashes, list) or len(latent_hashes) != len(prefix_rows):
        raise ValueError(f'{prefix}: READY latent inventory differs from manifest')
    return stats_sha256, count


def _fingerprint_identity(composition: dict) -> dict:
    return {
        'contract': composition['contract'],
        'replay_seed': composition['replay_seed'],
        'segments': [
            {key: value for key, value in segment.items() if key != 'source_stream'}
            for segment in composition['segments']
        ],
        'normalization': {
            key: value
            for key, value in composition['normalization'].items()
            if key != 'source_stream'
        },
    }


def composition_fingerprint(composition: dict) -> str:
    """Return the path-independent build fingerprint for one composition."""
    data = json.dumps(
        _fingerprint_identity(composition),
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    return sha256_bytes(data)[:16]


def _segment(
    *,
    stream: Path,
    meta: dict,
    target_start: int,
    source_count: int,
    ready_sha256: str | None = None,
) -> dict:
    segment = {
        'target_shard_start': target_start,
        'target_shard_stop': target_start + source_count,
        'source_stream': str(stream),
        'source_fingerprint': meta['fingerprint'],
        'source_manifest_sha256': meta['manifest_sha256'],
        'source_shard_start': 0,
        'source_shard_stop': source_count,
        'generation_seed': meta['seed'],
    }
    if ready_sha256 is not None:
        segment['source_ready_sha256'] = ready_sha256
    return segment


def _make_composition(
    *,
    prefix: Path,
    suffix: Path,
    prefix_meta: dict,
    suffix_meta: dict,
    prefix_count: int,
    suffix_count: int,
    stats_sha256: str,
    stats_count: int,
) -> dict:
    return {
        'contract': COMPOSITION_CONTRACT,
        'replay_seed': prefix_meta['seed'],
        'segments': [
            _segment(
                stream=prefix,
                meta=prefix_meta,
                target_start=0,
                source_count=prefix_count,
                ready_sha256=sha256_file(prefix / 'READY.json'),
            ),
            _segment(
                stream=suffix,
                meta=suffix_meta,
                target_start=prefix_count,
                source_count=suffix_count,
            ),
        ],
        'normalization': {
            'contract': NORMALIZATION_CONTRACT,
            'source_stream': str(prefix),
            'source_fingerprint': prefix_meta['fingerprint'],
            'source_manifest_sha256': prefix_meta['manifest_sha256'],
            'source_stats_sha256': stats_sha256,
            'source_stats_count': stats_count,
        },
    }


_REQUIRED_SUFFIX_ARCHIVE = (BUILD_PLAN_NAME, 'meta.yaml', MANIFEST_NAME, SUMMARY_NAME, 'reports')
_OPTIONAL_SUFFIX_ARCHIVE = (
    'READY.json',
    'latent-links.json',
    'latent-materialized.json',
    'latent-stats.json',
    'latents/plan.json',
    'latents/latent-materialized.json',
    'latents/stats.safetensors',
)


def _suffix_archive_inventory(suffix: Path) -> list[str]:
    inventory = list(_REQUIRED_SUFFIX_ARCHIVE)
    inventory.extend(
        relative for relative in _OPTIONAL_SUFFIX_ARCHIVE if (suffix / relative).exists()
    )
    return inventory


def _archive_suffix(staging: Path, expected_artifacts: list[str]) -> list[str]:
    archive = staging / '.components' / 'suffix'
    if archive.is_symlink() or (archive.exists() and not archive.is_dir()):
        raise FileExistsError(f'component archive is not a directory: {archive}')
    archive.mkdir(parents=True, exist_ok=True)
    expected = set(expected_artifacts)
    if not set(_REQUIRED_SUFFIX_ARCHIVE) <= expected:
        raise ValueError('composition journal lacks the required suffix archive inventory')
    if not expected <= set((*_REQUIRED_SUFFIX_ARCHIVE, *_OPTIONAL_SUFFIX_ARCHIVE)):
        raise ValueError(f'composition journal has unsupported suffix artifacts: {sorted(expected)}')
    complete = all(
        (
            (archive / relative).is_dir() and not (archive / relative).is_symlink()
            if relative == 'reports'
            else (archive / relative).is_file() and not (archive / relative).is_symlink()
        )
        for relative in expected
    )
    if complete:
        return sorted(expected)

    archived = []
    for relative in sorted(expected - {'reports'}):
        source = staging / relative
        destination = archive / relative
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or destination.is_symlink():
                raise FileExistsError(f'archived suffix artifact is not a regular file: {destination}')
            if source.exists() or source.is_symlink():
                if not source.is_file() or source.is_symlink() or not _same_inode(source, destination):
                    raise FileExistsError(f'suffix archive has a different inode: {destination}')
                source.unlink()
            archived.append(relative)
            continue
        if not source.exists() and not source.is_symlink():
            raise FileNotFoundError(f'missing suffix artifact to archive: {source}')
        _move_by_link(source, destination)
        archived.append(relative)
    reports = staging / 'reports'
    archived_reports = archive / 'reports'
    if archived_reports.exists() or archived_reports.is_symlink():
        if not archived_reports.is_dir() or archived_reports.is_symlink():
            raise FileExistsError(f'archived suffix reports are not a directory: {archived_reports}')
        if reports.exists() or reports.is_symlink():
            raise FileExistsError(f'suffix reports exist at both source and archive: {reports}')
    else:
        if not reports.is_dir() or reports.is_symlink():
            raise FileNotFoundError(f'missing suffix reports: {reports}')
        reports.rename(archived_reports)
    archived.append('reports')
    return sorted(archived)


def _inode(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino]


def _suffix_inode_inventory(
    suffix: Path,
    *,
    shard_count: int,
    has_latents: bool,
) -> dict[str, list[list[int]]]:
    inventory = {
        'shards': [_inode(suffix / f'shard_{shard_id:05d}.msgpack') for shard_id in range(shard_count)],
        'masks': [_inode(mask_shard_path(suffix, shard_id)) for shard_id in range(shard_count)],
        'latents': [],
    }
    if has_latents:
        inventory['latents'] = [
            _inode(suffix / 'latents' / f'shard_{shard_id:05d}.safetensors')
            for shard_id in range(shard_count)
        ]
    return inventory


def _validate_inode_inventory(
    inventory: dict,
    *,
    shard_count: int,
    has_latents: bool,
) -> None:
    expected_counts = {
        'shards': shard_count,
        'masks': shard_count,
        'latents': shard_count if has_latents else 0,
    }
    for key, expected_count in expected_counts.items():
        values = inventory.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f'composition journal has an invalid suffix {key} inode inventory')
        if any(
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(item, int) for item in value)
            for value in values
        ):
            raise ValueError(f'composition journal has an invalid suffix {key} inode')


def _composite_rows(
    prefix_rows: list[dict],
    suffix_rows: list[dict],
    fingerprint: str,
) -> list[dict]:
    prefix_count = len(prefix_rows)
    rows = [
        {**row, 'build_fingerprint': fingerprint}
        for row in prefix_rows
    ]
    rows.extend(
        {
            **row,
            'shard_id': prefix_count + row['shard_id'],
            'build_fingerprint': fingerprint,
        }
        for row in suffix_rows
    )
    return rows


def _prepare_or_validate_finalized(
    staging: Path,
    *,
    rows: list[dict],
    plan: dict,
) -> tuple[dict | None, bool]:
    manifest_path = staging / MANIFEST_NAME
    summary_path = staging / SUMMARY_NAME
    meta_path = staging / 'meta.yaml'
    expected_summary = summarize(rows, plan)
    expected_summary_json = json.loads(json.dumps(expected_summary))
    if meta_path.exists() or meta_path.is_symlink():
        if not meta_path.is_file() or meta_path.is_symlink():
            raise FileExistsError(f'composite meta is not a regular file: {meta_path}')
        meta = yaml.safe_load(meta_path.read_text())
        if not meta.get('stream_complete'):
            raise ValueError(f'composite meta is incomplete: {meta_path}')
        if meta.get('build_fingerprint') != plan['fingerprint']:
            raise ValueError(f'composite meta build fingerprint differs: {meta_path}')
        if meta.get('n_shards') != len(rows):
            raise ValueError(f'composite meta shard count differs: {meta_path}')
        if meta.get('composition') != plan['composition']:
            raise ValueError(f'composite meta composition differs: {meta_path}')
        if _load_manifest_path(manifest_path) != rows:
            raise ValueError(f'composite manifest rows differ: {manifest_path}')
        manifest_sha256 = sha256_file(manifest_path)
        if meta.get('manifest_sha256') != manifest_sha256:
            raise ValueError(f'composite manifest hash differs from meta: {manifest_path}')
        expected_fingerprint = sha256_bytes(
            f"{plan['fingerprint']}:{manifest_sha256}".encode()
        )[:16]
        if meta.get('fingerprint') != expected_fingerprint:
            raise ValueError(f'composite finalized fingerprint differs: {meta_path}')
        if json.loads(summary_path.read_text()) != expected_summary_json:
            raise ValueError(f'composite summary differs: {summary_path}')
        return expected_summary, True

    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    summary_exists = summary_path.exists() or summary_path.is_symlink()
    if manifest_exists:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileExistsError(f'partial composite manifest is not a regular file: {manifest_path}')
        if _load_manifest_path(manifest_path) != rows:
            raise ValueError(f'partial composite manifest rows differ: {manifest_path}')
    if summary_exists:
        if summary_path.is_symlink() or not summary_path.is_file():
            raise FileExistsError(f'partial composite summary is not a regular file: {summary_path}')
        if json.loads(summary_path.read_text()) != expected_summary_json:
            raise ValueError(f'partial composite summary differs: {summary_path}')
    if manifest_exists:
        manifest_path.unlink()
    if summary_exists:
        summary_path.unlink()
    return None, False


def compose_stream(
    *,
    prefix_stream: str | Path,
    suffix_stream: str | Path,
    target_stream: str | Path,
    workers: int = 0,
) -> dict:
    """Compose a READY prefix and finalized suffix into an unpublished staging stream."""
    prefix = Path(prefix_stream).resolve()
    suffix = Path(suffix_stream).resolve()
    target = Path(target_stream).resolve()
    staging = target.with_name(f'{target.name}.staging')
    if prefix == suffix:
        raise ValueError('prefix and suffix streams must differ')
    if not prefix.is_dir():
        raise FileNotFoundError(f'prefix stream is not an existing directory: {prefix}')
    if not target.parent.is_dir():
        raise FileNotFoundError(f'target parent does not exist: {target.parent}')
    require_absent(target)
    fresh = (
        suffix.is_dir()
        and not suffix.is_symlink()
        and not staging.exists()
        and not staging.is_symlink()
    )
    resuming = (
        not suffix.exists()
        and not suffix.is_symlink()
        and staging.is_dir()
        and not staging.is_symlink()
    )
    if not fresh and not resuming:
        raise RuntimeError(
            'composition requires either an existing suffix with absent staging, or '
            'an absent suffix with existing staging'
        )
    active_component = suffix if fresh else staging
    devices = {prefix.stat().st_dev, active_component.stat().st_dev, target.parent.stat().st_dev}
    if len(devices) != 1:
        raise ValueError('prefix, suffix staging, and target must reside on the same filesystem')

    prefix_meta, prefix_plan, prefix_rows = _load_finalized(prefix, require_ready=True)
    prefix_count = len(prefix_rows)
    _require_inventory(prefix, '.msgpack', prefix_count)
    _require_inventory(prefix / 'masks', '.bin', prefix_count)
    prefix_latents = _optional_latent_inventory(prefix, prefix_count)
    if len(prefix_latents) != prefix_count:
        raise ValueError('READY prefix lacks a complete latent shard inventory')
    stats_sha256, stats_count = _stats_identity(prefix, prefix_meta, prefix_rows)
    prefix_stats_receipt = prefix / 'latent-stats.json'
    if not prefix_stats_receipt.is_file():
        raise FileNotFoundError(f'prefix latent stats receipt is missing: {prefix_stats_receipt}')

    if fresh:
        if (suffix / '.components').exists() or (suffix / '.components').is_symlink():
            raise FileExistsError(
                f'suffix already contains a component archive: {suffix / ".components"}'
            )
        suffix_meta, suffix_plan, suffix_rows = _load_finalized(suffix, require_ready=False)
        suffix_count = len(suffix_rows)
        if prefix_count < 1 or suffix_count < 1:
            raise ValueError('component streams must each contain at least one shard')
        _require_inventory(suffix, '.msgpack', suffix_count)
        _require_inventory(suffix / 'masks', '.bin', suffix_count)
        suffix_has_latents = bool(_optional_latent_inventory(suffix, suffix_count))
        archive_artifacts = _suffix_archive_inventory(suffix)
        suffix_inodes = _suffix_inode_inventory(
            suffix,
            shard_count=suffix_count,
            has_latents=suffix_has_latents,
        )
    else:
        journal = _load_journal(staging)
        identity = journal[0]
        expected_paths = {
            'source_suffix': str(suffix),
            'staging_stream': str(staging),
            'target_stream': str(target),
        }
        for key, expected in expected_paths.items():
            if identity.get(key) != expected:
                raise ValueError(
                    f'composition journal {key} differs: {identity.get(key)!r} != {expected!r}'
                )
        suffix_meta, suffix_plan, suffix_rows = _load_suffix_snapshot(staging, suffix)
        suffix_count = len(suffix_rows)
        suffix_has_latents = identity.get('suffix_has_latents')
        if not isinstance(suffix_has_latents, bool):
            raise ValueError('composition journal lacks suffix_has_latents')
        archive_artifacts = identity.get('suffix_archive_artifacts')
        if not isinstance(archive_artifacts, list) or not all(
            isinstance(value, str) for value in archive_artifacts
        ):
            raise ValueError('composition journal lacks a valid suffix archive inventory')
        suffix_inodes = identity.get('suffix_inodes')
        if not isinstance(suffix_inodes, dict):
            raise ValueError('composition journal lacks suffix inode identities')
        _validate_inode_inventory(
            suffix_inodes,
            shard_count=suffix_count,
            has_latents=suffix_has_latents,
        )
        if identity.get('prefix_shards') != prefix_count:
            raise ValueError('composition journal prefix shard count differs')
        if identity.get('suffix_shards') != suffix_count:
            raise ValueError('composition journal suffix shard count differs')

    _validate_compatible(prefix_meta, prefix_plan, suffix_meta, suffix_plan)
    composition = _make_composition(
        prefix=prefix,
        suffix=suffix,
        prefix_meta=prefix_meta,
        suffix_meta=suffix_meta,
        prefix_count=prefix_count,
        suffix_count=suffix_count,
        stats_sha256=stats_sha256,
        stats_count=stats_count,
    )
    fingerprint = composition_fingerprint(composition)
    plan = {
        **prefix_plan,
        'fingerprint': fingerprint,
        'seed': prefix_meta['seed'],
        'source_stream': None,
        'source_meta_sha256': None,
        'composition': composition,
    }
    if resuming:
        identity = _load_journal(staging)[0]
        if identity.get('composition') != composition:
            raise ValueError('composition journal component identity differs from current inputs')
        if identity.get('build_fingerprint') != fingerprint:
            raise ValueError('composition journal build fingerprint differs')
    else:
        suffix.rename(staging)
        _append_journal(
            staging,
            'suffix-renamed',
            source_suffix=str(suffix),
            staging_stream=str(staging),
            target_stream=str(target),
            build_fingerprint=fingerprint,
            prefix_shards=prefix_count,
            suffix_shards=suffix_count,
            suffix_has_latents=suffix_has_latents,
            suffix_archive_artifacts=archive_artifacts,
            suffix_inodes=suffix_inodes,
            composition=composition,
        )

    try:
        archived = _archive_suffix(staging, archive_artifacts)
        _append_journal(staging, 'suffix-aggregates-archived', artifacts=archived)

        for shard_id in reversed(range(suffix_count)):
            target_id = prefix_count + shard_id
            _finish_suffix_move(
                staging / f'shard_{shard_id:05d}.msgpack',
                staging / f'shard_{target_id:05d}.msgpack',
                expected_inode=tuple(suffix_inodes['shards'][shard_id]),
            )
            _finish_suffix_move(
                mask_shard_path(staging, shard_id),
                mask_shard_path(staging, target_id),
                expected_inode=tuple(suffix_inodes['masks'][shard_id]),
            )
            if suffix_has_latents:
                _finish_suffix_move(
                    staging / 'latents' / f'shard_{shard_id:05d}.safetensors',
                    staging / 'latents' / f'shard_{target_id:05d}.safetensors',
                    expected_inode=tuple(suffix_inodes['latents'][shard_id]),
                )
        _append_journal(staging, 'suffix-shards-moved', shards=suffix_count)

        for shard_id in range(prefix_count):
            _link(
                prefix / f'shard_{shard_id:05d}.msgpack',
                staging / f'shard_{shard_id:05d}.msgpack',
            )
            _link(
                mask_shard_path(prefix, shard_id),
                mask_shard_path(staging, shard_id),
            )
            _link(
                prefix / 'latents' / f'shard_{shard_id:05d}.safetensors',
                staging / 'latents' / f'shard_{shard_id:05d}.safetensors',
            )
        _append_journal(staging, 'prefix-shards-linked', shards=prefix_count)

        _write_yaml(staging / BUILD_PLAN_NAME, plan)
        rows = _composite_rows(prefix_rows, suffix_rows, fingerprint)
        for row in rows:
            _write_json_matching(
                staging / 'reports' / f"shard_{row['shard_id']:05d}.json",
                row,
            )

        _link(prefix / 'latents' / 'stats.safetensors', staging / 'latents' / 'stats.safetensors')
        stats_receipt = json.loads(prefix_stats_receipt.read_text())
        stats_receipt['normalization'] = composition['normalization']
        _write_json_matching(staging / 'latent-stats.json', stats_receipt)
        _append_journal(staging, 'composite-plan-written', reports=prefix_count + suffix_count)

        summary, finalized = _prepare_or_validate_finalized(staging, rows=rows, plan=plan)
        if not finalized:
            summary = finalize_stream(
                staging,
                expected_shards=prefix_count + suffix_count,
                workers=workers,
            )
        assert summary is not None
        _append_journal(
            staging,
            'composite-finalized',
            fingerprint=yaml.safe_load((staging / 'meta.yaml').read_text())['fingerprint'],
        )
    except Exception as error:
        try:
            _append_journal(
                staging,
                'failed',
                error_type=type(error).__name__,
                error=str(error),
            )
        except Exception:
            pass
        raise

    return {
        'status': 'staged',
        'staging_stream': str(staging),
        'target_stream': str(target),
        'prefix_shards': prefix_count,
        'suffix_shards': suffix_count,
        'summary': summary,
    }
