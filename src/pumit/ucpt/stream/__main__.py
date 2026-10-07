"""Command-line interface for UCPT stream construction and artifacts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import yaml

from .build import build_stream, distributed_context
from .compose import compose_stream
from .filtering import UNLABELED_FILTER_CONTRACT, filter_stream
from .latents import (
    cmd_encode_latents,
    cmd_link_latents,
)
from .manifest import BUILD_PLAN_NAME, finalize_stream, prepare_stream_extension, sha256_file
from .stats import cmd_stats
from .verify import cmd_verify


# Report progress during long filtering phases.
_FILTER_HEARTBEAT_SECONDS = 60.0


@contextmanager
def _filter_phase_progress(stream: Path, phase: str):
    """Report published artifacts while a filtering phase blocks."""
    started = monotonic()
    stopped = Event()

    def report() -> None:
        reports = sum(1 for _ in (stream / 'reports').glob('shard_*.json'))
        latents = sum(1 for _ in (stream / 'latents').glob('shard_*.safetensors'))
        print(
            f'[ucpt] filter-unlabeled phase={phase} elapsed={monotonic() - started:.0f}s '
            f'reports={reports} latents={latents}',
            flush=True,
        )

    def heartbeat() -> None:
        while not stopped.wait(_FILTER_HEARTBEAT_SECONDS):
            report()

    report()
    thread = Thread(target=heartbeat, name='filter-progress', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _add_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--source-latent-dir', type=Path, required=True)


def _cmd_build(args) -> None:
    build_stream(
        config_path=args.config,
        shards=args.shards,
        shard_offset=args.shard_offset,
        output_dir=args.output_stream,
        seed=args.seed,
        batches_per_shard=args.batches_per_shard,
        budget_ms=args.budget_ms,
        label_budget_fraction=args.label_budget_fraction,
        cost_model_path=args.cost_model,
        use_default_cost_model=args.use_default_cost_model,
        source_stream=args.source_stream,
        shard_workers=args.shard_workers,
        sample_workers=args.sample_workers,
        sample_prefetch_factor=args.sample_prefetch_factor,
        force=args.force,
    )


def _cmd_finalize(args) -> None:
    summary = finalize_stream(args.stream, args.expected_shards, args.workers)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _validate_filter_resume(args, shard_count: int) -> None:
    stream = args.output_stream.resolve()
    source = args.source_stream.resolve()
    plan = yaml.safe_load((stream / BUILD_PLAN_NAME).read_text())
    meta = yaml.safe_load((stream / 'meta.yaml').read_text())
    expected = {
        'source_stream': str(source),
        'source_meta_sha256': sha256_file(source / 'meta.yaml'),
        'unlabeled_filter': {
            'contract': UNLABELED_FILTER_CONTRACT,
            'exclude_datasets': list(args.exclude_datasets),
        },
    }
    for artifact in (plan, meta):
        for key, value in expected.items():
            if artifact.get(key) != value:
                raise ValueError(f'filter plan mismatch for {key}: {stream}')
    if (
        meta['build_fingerprint'] != plan['fingerprint']
        or meta['n_shards'] != shard_count
        or meta.get('stream_complete') is not True
    ):
        raise ValueError(f'finalized stream differs from requested filter: {stream}')
    ready_path = stream / 'READY.json'
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if (
            ready['fingerprint'] != meta['fingerprint']
            or ready['verified_shards'] != shard_count
            or ready['manifest_sha256'] != meta['manifest_sha256']
            or sha256_file(stream / 'manifest.jsonl') != meta['manifest_sha256']
        ):
            raise ValueError(f'READY differs from finalized stream: {stream}')


def _cmd_filter_unlabeled(args) -> None:
    complete = getattr(args, 'complete', False)
    stream = args.output_stream.resolve()
    if not complete:
        filter_stream(
            source_stream=args.source_stream,
            output_dir=args.output_stream,
            exclude_datasets=args.exclude_datasets,
            shards=args.shards,
            workers=args.workers,
        )
        return
    if distributed_context()[1] != 1:
        raise ValueError('filter-unlabeled --complete requires a single node (WORLD_SIZE=1)')
    with _filter_phase_progress(stream, 'prepare'):
        source_meta = yaml.safe_load((args.source_stream / 'meta.yaml').read_text())
        shard_count = source_meta['n_shards'] if args.shards is None else args.shards
        cost_model = Path(source_meta['cost_model_source'])
        if not cost_model.is_file():
            raise FileNotFoundError(f'source cost model is not a file: {cost_model}')

    if (stream / 'meta.yaml').exists():
        with _filter_phase_progress(stream, 'resume'):
            _validate_filter_resume(args, shard_count)
    else:
        with _filter_phase_progress(stream, 'filter'):
            filter_stream(
                source_stream=args.source_stream,
                output_dir=args.output_stream,
                exclude_datasets=args.exclude_datasets,
                shards=args.shards,
                workers=args.workers,
            )
        with _filter_phase_progress(stream, 'finalize'):
            finalize_stream(stream, shard_count, args.workers)

    if (stream / 'READY.json').exists():
        print(f'[ucpt] stream is already READY: {stream}', flush=True)
        return
    with _filter_phase_progress(stream, 'latent-reuse'):
        cmd_link_latents(
            argparse.Namespace(
                stream=stream,
                source_latent_dir=args.source_stream.resolve() / 'latents',
                workers=args.workers,
            ),
        )
    with _filter_phase_progress(stream, 'verify'):
        cmd_verify(
            argparse.Namespace(
                stream=stream,
                source_stream=args.source_stream,
                source_latent_dir=args.source_stream.resolve() / 'latents',
                cost_model=cost_model,
                expected_summary=None,
                workers=args.workers,
            ),
        )
    print(f'[ucpt] filter-unlabeled complete: {stream}', flush=True)


def _cmd_prepare_extension(args) -> None:
    result = prepare_stream_extension(args.stream, args.expected_shards)
    print(json.dumps(result, indent=2, sort_keys=True))


def _cmd_compose(args) -> None:
    result = compose_stream(
        prefix_stream=args.prefix_stream,
        suffix_stream=args.suffix_stream,
        target_stream=args.output_stream,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser('build')
    build.add_argument('--output-stream', type=Path, required=True)
    build.add_argument(
        '--source-stream',
        type=Path,
        help='finalized stream supplying the complete ordered unlabeled sequence',
    )
    build.add_argument('--config', type=Path, required=True)
    cost_model = build.add_mutually_exclusive_group(required=True)
    cost_model.add_argument('--cost-model', type=Path)
    cost_model.add_argument('--use-default-cost-model', action='store_true')
    build.add_argument('--budget-ms', type=float, required=True)
    build.add_argument('--label-budget-fraction', type=float, required=True)
    build.add_argument('--batches-per-shard', type=int, required=True)
    build.add_argument('--shards', type=int, required=True)
    build.add_argument('--shard-offset', type=int, default=0)
    build.add_argument('--seed', type=int, default=42)
    build.add_argument('--shard-workers', type=int, default=8)
    build.add_argument('--sample-workers', type=int)
    build.add_argument(
        '--sample-prefetch-factor',
        type=int,
        default=8,
        help=(
            'ordered labeled-sample prefetch waves per shard; the per-shard window is '
            'factor * ceil(sample_workers / active_shards) (default: 8)'
        ),
    )
    build.add_argument('--force', action='store_true')
    build.set_defaults(func=_cmd_build)

    filtering = subparsers.add_parser('filter-unlabeled')
    filtering.add_argument('--source-stream', type=Path, required=True)
    filtering.add_argument('--output-stream', type=Path, required=True)
    filtering.add_argument('--exclude-dataset', dest='exclude_datasets', action='append', required=True)
    filtering.add_argument('--shards', type=int)
    filtering.add_argument('--workers', type=int, default=4)
    filtering.add_argument(
        '--complete',
        action='store_true',
        help='finalize, reuse latents, and verify the filtered stream on one node',
    )
    filtering.set_defaults(func=_cmd_filter_unlabeled)

    finalize = subparsers.add_parser('finalize')
    finalize.add_argument('--stream', type=Path, required=True)
    finalize.add_argument('--expected-shards', type=int, required=True)
    finalize.add_argument('--workers', type=int, default=16)
    finalize.set_defaults(func=_cmd_finalize)

    extension = subparsers.add_parser('prepare-extension')
    extension.add_argument('--stream', type=Path, required=True)
    extension.add_argument('--expected-shards', type=int, required=True)
    extension.set_defaults(func=_cmd_prepare_extension)

    compose = subparsers.add_parser('compose')
    compose.add_argument('--prefix-stream', type=Path, required=True)
    compose.add_argument('--suffix-stream', type=Path, required=True)
    compose.add_argument('--output-stream', type=Path, required=True)
    compose.add_argument('--workers', type=int, default=32)
    compose.set_defaults(func=_cmd_compose)

    links = subparsers.add_parser("link-latents")
    _add_stream_args(links)
    links.add_argument('--workers', type=int, default=4)
    links.set_defaults(func=cmd_link_latents)

    latents = subparsers.add_parser('encode-latents')
    latents.add_argument('--stream', type=Path, required=True)
    latents.add_argument(
        '--filter',
        choices=['all', 'suffix', 'migration', 'canonical-inplane'],
        required=True,
        help='per-sample selection rule for latent recomputation',
    )
    latents.add_argument('--source-latent-dir', type=Path)
    latents.add_argument('--codec-checkpoint', type=Path, required=True)
    latents.add_argument('--codec-model', choices=['flux2', 'klvae'], default='flux2')
    latents.add_argument(
        '--memory-budget-gb',
        type=float,
        default=68.0,
        help='empirical per-GPU encoder batch budget (default: 68)',
    )
    latents.add_argument('--num-workers', type=int, default=8)
    latents.add_argument(
        '--async-finalize',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'overlap shard materialization with the next GPU pipeline '
            '(default: enabled for filter=all, disabled otherwise)'
        ),
    )
    latents.add_argument('--shard-offset', type=int, default=0)
    latents.add_argument(
        '--shards',
        type=int,
        help='number of consecutive shards to encode; defaults to all remaining shards',
    )
    latents.add_argument(
        '--compile-mode',
        choices=['default', 'reduce-overhead', 'max-autotune'],
        default='default',
    )
    latents.add_argument(
        '--cache-archive',
        type=Path,
        help='Persistent tar.zst of the shared Inductor/Triton cache.',
    )
    latents.set_defaults(func=cmd_encode_latents)

    stats = subparsers.add_parser("stats")
    stats.add_argument("--stream", type=Path, required=True)
    stats.add_argument("--workers", type=int, default=16)
    stats.set_defaults(func=cmd_stats)

    verify = subparsers.add_parser("verify")
    verify.add_argument('--stream', type=Path, required=True)
    verify.add_argument('--source-latent-dir', type=Path)
    verify.add_argument('--source-stream', type=Path)
    verify.add_argument("--cost-model", type=Path, required=True)
    verify.add_argument("--expected-summary", type=Path)
    verify.add_argument("--workers", type=int, default=32)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
