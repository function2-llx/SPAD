"""Single latent-encoding entrypoint for UCPT stream artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch.distributed as dist

from .build import distributed_context, partition_shard_ids
from .latent_backend import (
    _BoundedLatentShardFinalizer,
    _LatentPreparedLoader,
    _PARTS_DIR_NAME,
    _encode_canonical_inplane_latents,
    _encode_selected_latent_shard,
    _encode_selected_latent_shard_parts,
    _latent_gpu_pool,
    _load_unlabeled_samples,
    _migration_selection,
    _prepare_materialization_rows,
    _select_manifest_rows,
    _validate_codec_provenance,
    _validate_materialized_latent,
    _write_or_validate_json,
    cmd_link_latents as cmd_link_latents,
)
from .manifest import load_manifest, sha256_file, write_json
from .migration import AFFINE_REPLAY_CONTRACT, PREVIOUS_AFFINE_REPLAY_CONTRACT


def _select_samples(
    filter_name: str,
    shard_path: Path,
    row: dict,
) -> tuple[list[dict], list[int], int]:
    """Apply one per-sample latent selection filter to a shard."""
    if filter_name == 'migration':
        unlabeled, selected = _migration_selection(shard_path, row)
        return unlabeled, selected, row['used_old_unlabeled_samples']

    unlabeled = _load_unlabeled_samples(shard_path)
    logical_rows = sum(sample['n_patches'] for sample in unlabeled)
    if logical_rows != row['logical_latent_rows']:
        raise ValueError(
            f'{shard_path}: stream contains {logical_rows} latent rows, '
            f"manifest declares {row['logical_latent_rows']}"
        )
    if filter_name == 'all':
        return unlabeled, list(range(len(unlabeled))), 0
    if filter_name == 'suffix':
        source_samples = row['used_old_unlabeled_samples']
        return unlabeled, list(range(source_samples, len(unlabeled))), source_samples
    raise ValueError(f'unsupported latent selection filter {filter_name!r}')


def _validate_filter_contract(filter_name: str, rows: list[dict]) -> None:
    if filter_name != 'migration':
        return
    contracts = {row.get('migration_contract') for row in rows}
    if len(contracts) != 1 or None in contracts:
        raise ValueError('manifest has inconsistent migration contracts')
    contract = contracts.pop()
    if contract not in {PREVIOUS_AFFINE_REPLAY_CONTRACT, AFFINE_REPLAY_CONTRACT}:
        raise ValueError(f'unsupported migration contract {contract!r}')


def _candidate_rows(filter_name: str, rows: list[dict]) -> list[dict]:
    if filter_name == 'suffix':
        return [row for row in rows if row['generated_unlabeled_samples'] > 0]
    return rows


def _latent_receipt(
    filter_name: str,
    rows: list[dict],
    *,
    codec_model: str,
    codec_checkpoint: Path,
    source_latent_dir: Path | None,
) -> dict:
    if filter_name in {'all', 'suffix'}:
        materialized = _candidate_rows(filter_name, rows)
        if filter_name == 'all':
            generated_samples = sum(row['new_unlabeled_samples'] for row in rows)
            generated_patches = sum(row['logical_latent_rows'] for row in rows)
        else:
            generated_samples = sum(row['generated_unlabeled_samples'] for row in rows)
            generated_patches = sum(row['generated_unlabeled_patches'] for row in rows)
        return {
            'codec_model': codec_model,
            'codec_checkpoint': str(codec_checkpoint),
            'materialized_shards': [row['shard_id'] for row in materialized],
            'generated_unlabeled_samples': generated_samples,
            'generated_unlabeled_patches': generated_patches,
        }

    assert filter_name == 'migration' and source_latent_dir is not None
    contract = rows[0]['migration_contract']
    return {
        'codec_model': codec_model,
        'codec_checkpoint': str(codec_checkpoint),
        'source_latent_dir': str(source_latent_dir),
        'migration_contract': contract,
        'materialized_shards': [row['shard_id'] for row in rows],
        'affected_unlabeled_samples': sum(
            row['migration_affected_unlabeled_samples'] for row in rows
        ),
        'affected_unlabeled_patches': sum(
            row['migration_affected_unlabeled_patches'] for row in rows
        ),
        'generated_unlabeled_samples': sum(
            row['generated_unlabeled_samples'] for row in rows
        ),
        'generated_unlabeled_patches': sum(
            row['generated_unlabeled_patches'] for row in rows
        ),
    }


def _guard_full_latent_plan(
    stream_dir: Path,
    latent_dir: Path,
    *,
    codec_model: str,
    codec_checkpoint: Path,
    rank: int,
) -> None:
    """Pin a full-stream encode to one codec before any partial latents are reused.

    ``--filter all`` records its codec only in the end-of-run receipt, so an interrupted run leaves parts and
    completed shards with no attributable codec. Every resume must match the plan written here at encode
    start; once the receipt exists, ``_write_or_validate_json`` enforces codec identity instead.
    """
    if (stream_dir / 'latent-materialized.json').exists():
        return
    plan_path = latent_dir / 'plan.json'
    plan = {
        'filter': 'all',
        'codec_model': codec_model,
        'codec_checkpoint': str(codec_checkpoint),
        'codec_checkpoint_sha256': sha256_file(codec_checkpoint),
    }
    if rank == 0:
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError(
                    f'{plan_path}: existing full-latent plan differs from this invocation'
                )
        else:
            has_shards = any(latent_dir.glob('shard_*.safetensors'))
            has_parts = (stream_dir / _PARTS_DIR_NAME).exists()
            if has_shards or has_parts:
                raise ValueError(
                    f'{latent_dir}: partial latents exist without plan.json; '
                    'their codec cannot be attributed'
                )
            write_json(plan_path, plan)
    if dist.is_initialized():
        dist.barrier()
        if rank != 0 and (
            not plan_path.exists() or json.loads(plan_path.read_text()) != plan
        ):
            raise ValueError(
                f'{plan_path}: existing full-latent plan differs from this invocation'
            )


def _print_shard_result(result: dict, *, finalization_pending: bool = False) -> None:
    completion = (
        'GPU pipeline complete; finalization pending'
        if finalization_pending
        else 'complete'
    )
    print(
        f"[ucpt] latent shard {result['shard_id']} {completion}: "
        f"{result['samples']:,} samples, {result['gpu_batches']} GPU batches, "
        f"{result['timings']['total']:.1f} aggregate worker-s"
    )
    timing = result['timings']
    print(
        f"[ucpt] latent shard {result['shard_id']} worker timing: "
        f"prepare={timing['prepare']:.1f}s, replay={timing['replay']:.1f}s, "
        f"H2D={timing['transfer']:.1f}s, encode+D2H={timing['encode']:.1f}s, "
        f"write={timing['write']:.1f}s, first prepared wait="
        f"{timing['first_prepared_wait']:.1f}s, prepared queue wait="
        f"{timing['prepared_queue_wait']:.1f}s, pipeline wall="
        f"{timing['pipeline_wall']:.1f}s"
    )
    scheduler = result['scheduler_summary']
    print(
        f"[ucpt] latent shard {result['shard_id']} scheduler summary: "
        f"prepared capacity={scheduler['prepared_capacity']}, "
        f"pending GPU batches={scheduler['pending_gpu_batches']}"
    )
    for device_id, device in sorted(scheduler['devices'].items()):
        print(
            f"[ucpt] latent shard {result['shard_id']} GPU {device_id}: "
            f"{device['gpu_batches']} batches, "
            f"{device['samples']:,} samples, {device['voxel_work']:,} voxels, "
            f"GPU wall={device['gpu_wall']:.1f}s, "
            f"max GPU batch={device['max_gpu_batch_wall']:.1f}s, "
            f"first prepared wait={device['first_prepared_wait']:.1f}s, "
            f"prepared queue wait={device['prepared_queue_wait']:.1f}s"
        )
        print(
            f"[ucpt] latent shard {result['shard_id']} GPU {device_id} stages: "
            f"prepare={device['prepare']:.1f}s, replay={device['replay']:.1f}s, "
            f"H2D={device['transfer']:.1f}s, "
            f"encode+D2H={device['encode']:.1f}s, write={device['write']:.1f}s"
        )
    finalization = result.get('finalization')
    if finalization is not None:
        _print_finalization_timing(result['shard_id'], finalization, background=False)


def _print_finalization_timing(
    shard_id: int,
    timing: dict[str, float],
    *,
    background: bool,
) -> None:
    mode = ' background' if background else ''
    print(
        f'[ucpt] latent shard {shard_id}{mode} finalization complete: '
        f"materialize={timing['materialize']:.1f}s, "
        f"validate={timing['validate']:.1f}s, cleanup={timing['cleanup']:.1f}s, "
        f"wall={timing['total']:.1f}s"
    )


def _print_background_finalization(result) -> None:
    _print_finalization_timing(
        result.shard_id,
        {
            'materialize': result.materialize_seconds,
            'validate': result.validate_seconds,
            'cleanup': result.cleanup_seconds,
            'total': result.total_seconds,
        },
        background=True,
    )


def _async_finalize_enabled(args) -> bool:
    requested = getattr(args, 'async_finalize', None)
    if requested is None:
        return args.filter == 'all'
    return requested


def _encode_filtered_latents(
    args,
    prepared_loader: _LatentPreparedLoader,
) -> None:
    stream_dir = args.stream.resolve()
    source_latent_dir = (
        args.source_latent_dir.resolve()
        if args.source_latent_dir is not None
        else None
    )
    codec_checkpoint = args.codec_checkpoint.resolve()
    rows = load_manifest(stream_dir)
    if not rows:
        raise ValueError(f'{stream_dir}: manifest is empty')
    _validate_filter_contract(args.filter, rows)

    all_candidates = _candidate_rows(args.filter, rows)
    receipt = _latent_receipt(
        args.filter,
        rows,
        codec_model=args.codec_model,
        codec_checkpoint=codec_checkpoint,
        source_latent_dir=source_latent_dir,
    )
    if not all_candidates:
        if int(os.environ.get('RANK', 0)) == 0:
            _write_or_validate_json(stream_dir / 'latent-materialized.json', receipt)
        return

    requested_rows, _ = _select_manifest_rows(
        rows,
        shard_offset=args.shard_offset,
        shards=args.shards,
    )
    requested_ids = {row['shard_id'] for row in requested_rows}
    requested_candidates = [
        row for row in all_candidates if row['shard_id'] in requested_ids
    ]
    rank, world_size = distributed_context()
    owned_ids = set(
        partition_shard_ids(
            [row['shard_id'] for row in requested_candidates],
            rank,
            world_size,
        )
    )
    owned_rows = [row for row in requested_candidates if row['shard_id'] in owned_ids]
    print(
        json.dumps(
            {
                'rank': rank,
                'world_size': world_size,
                'owned_shards': len(owned_rows),
                'requested_shards': len(requested_candidates),
                'selection_filter': args.filter,
            },
            indent=2,
        )
    )
    if world_size > 1:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

    try:
        if source_latent_dir is not None:
            _validate_codec_provenance(
                source_latent_dir,
                args.codec_model,
                codec_checkpoint,
            )
        latent_dir = stream_dir / 'latents'
        latent_dir.mkdir(exist_ok=True)
        if args.filter == 'all':
            _guard_full_latent_plan(
                stream_dir,
                latent_dir,
                codec_model=args.codec_model,
                codec_checkpoint=codec_checkpoint,
                rank=rank,
            )
        pending_rows = _prepare_materialization_rows(stream_dir, owned_rows)

        if pending_rows:
            with _latent_gpu_pool(
                args,
                rank=rank,
                prepared_loader=prepared_loader,
            ) as (
                gpu_pool,
                max_inflight,
                archiver,
            ):
                finalizer = (
                    _BoundedLatentShardFinalizer()
                    if _async_finalize_enabled(args)
                    else None
                )
                try:
                    for row in pending_rows:
                        shard_id = row['shard_id']
                        unlabeled, selected_indices, source_prefix_samples = (
                            _select_samples(
                                args.filter,
                                stream_dir / f'shard_{shard_id:05d}.msgpack',
                                row,
                            )
                        )
                        encode_kwargs = {
                            'stream_dir': stream_dir,
                            'source_latent_dir': source_latent_dir,
                            'parts_dir': stream_dir,
                            'output_dir': latent_dir,
                            'source_prefix_samples': source_prefix_samples,
                            'gpu_pool': gpu_pool,
                            'memory_budget_gb': args.memory_budget_gb,
                            'max_inflight': max_inflight,
                            'cache_archiver': archiver,
                            'prepared_loader': prepared_loader,
                        }
                        if finalizer is None:
                            result = _encode_selected_latent_shard(
                                row,
                                unlabeled,
                                selected_indices,
                                **encode_kwargs,
                            )
                            _print_shard_result(result)
                            continue

                        encoded = _encode_selected_latent_shard_parts(
                            row,
                            unlabeled,
                            selected_indices,
                            **encode_kwargs,
                        )
                        _print_shard_result(
                            encoded.result,
                            finalization_pending=True,
                        )
                        completed_finalization = finalizer.submit(
                            encoded.finalization
                        )
                        if completed_finalization is not None:
                            _print_background_finalization(completed_finalization)
                except BaseException:
                    if finalizer is not None:
                        try:
                            finalizer.close()
                        except BaseException:
                            pass
                    raise
                else:
                    if finalizer is not None:
                        completed_finalization = finalizer.close()
                        if completed_finalization is not None:
                            _print_background_finalization(completed_finalization)

        for row in owned_rows:
            path = latent_dir / f"shard_{row['shard_id']:05d}.safetensors"
            _validate_materialized_latent(path, row['logical_latent_rows'])
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            output_paths = [
                latent_dir / f"shard_{row['shard_id']:05d}.safetensors"
                for row in all_candidates
            ]
            if all(path.exists() for path in output_paths):
                for row, path in zip(all_candidates, output_paths, strict=True):
                    _validate_materialized_latent(path, row['logical_latent_rows'])
                _write_or_validate_json(stream_dir / 'latent-materialized.json', receipt)
            else:
                completed = sum(path.exists() for path in output_paths)
                print(
                    f'[ucpt] latent encoding: {completed}/{len(all_candidates)} '
                    'shards complete; receipt not written'
                )
        if dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def cmd_encode_latents(args) -> None:
    """Encode latent shards through one filter-driven node-level orchestration."""
    if int(os.environ.get('LOCAL_WORLD_SIZE', 1)) != 1:
        raise RuntimeError('encode-latents must be launched once per node, without torchrun')
    if _async_finalize_enabled(args) and args.filter != 'all':
        raise ValueError('--async-finalize is only valid with --filter all')
    if args.filter == 'all':
        if args.source_latent_dir is not None:
            raise ValueError('--source-latent-dir is not valid with --filter all')
    elif args.source_latent_dir is None:
        raise ValueError(f'--source-latent-dir is required with --filter {args.filter}')

    if args.num_workers < 1:
        raise ValueError(f'num_workers must be positive, got {args.num_workers}')
    prepared_loader = None
    try:
        prepared_loader = _LatentPreparedLoader(
            args.stream.resolve(),
            replay_threads=args.num_workers,
        )
        # Fork the replay workers before filtered/canonical paths can initialize Gloo.
        prepared_loader.start()
        if args.filter == 'canonical-inplane':
            _encode_canonical_inplane_latents(args, prepared_loader)
        else:
            _encode_filtered_latents(args, prepared_loader)
    except BaseException as error:
        if prepared_loader is not None:
            try:
                prepared_loader.close()
            except BaseException as loader_error:
                error.add_note(
                    'latent prepared loader cleanup also failed: '
                    f'{type(loader_error).__name__}: {loader_error}'
                )
        raise
    else:
        prepared_loader.close()
