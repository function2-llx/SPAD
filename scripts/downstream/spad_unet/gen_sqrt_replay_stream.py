"""Generate the flat sqrt-dataset replay shared by SPAD U-Net training plans.

The resulting stream contains only dataset, case, and foreground-crop decisions. Training consumes
the ordered records without implementing or depending on the generation policy.

Example:
    pixi run -e spad-unet python \
        scripts/downstream/spad_unet/gen_sqrt_replay_stream.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pumit.spad_unet.data import (
    UniversalExperimentManifest,
    build_region_registry,
    get_preprocessed_root,
    load_universal_datasets,
    prepare_universal_nnunet_namespace,
)
from pumit.spad_unet.replay import (
    ReplayStreamReader,
    generate_sqrt_replay_stream,
    sqrt_replay_generation_fingerprint,
)


DEFAULT_SEED = 20260822
DEFAULT_MAX_GLOBAL_BATCH_SIZE = 24
DEFAULT_RECORDS_PER_SHARD = 500_000
DEFAULT_REPLAY_DIRECTORY = (
    'SPADCTUniversalV2SqrtNFG1of3ReplaySeed20260822MaxGB24'
)
DEFAULT_STATISTICS_GLOBAL_BATCH_SIZES = (8, 16, 24)
DEFAULT_SOURCE_PLANS_IDENTIFIER = (
    'nnUNetResEncUNetLPlans6StageSampleNativeZ1x1FOV192'
)
CONFIGURATION_NAME = '3d_fullres'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate a flat replay with sqrt(training count) dataset sampling '
            'and independent foreground decisions.'
        ),
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/downstream/spad_unet/spad_ct_universal_v2.json'),
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--source-plans-identifier',
        default=DEFAULT_SOURCE_PLANS_IDENTIFIER,
        help='Source preprocessing contract used to validate cases and folds.',
    )
    parser.add_argument(
        '--max-global-batch-size',
        type=int,
        default=DEFAULT_MAX_GLOBAL_BATCH_SIZE,
        help='Generate num_updates times this many records.',
    )
    parser.add_argument(
        '--statistics-global-batch-sizes',
        type=int,
        nargs='+',
        default=None,
        help='Report prefixes corresponding to num_updates times each batch size.',
    )
    parser.add_argument(
        '--foreground-probability',
        type=float,
        default=1 / 3,
    )
    parser.add_argument(
        '--records-per-shard',
        type=int,
        default=DEFAULT_RECORDS_PER_SHARD,
    )
    parser.add_argument('--replay-directory', default=DEFAULT_REPLAY_DIRECTORY)
    parser.add_argument(
        '--reuse-existing-replay',
        action='store_true',
        help='Validate and report the named replay instead of generating it.',
    )
    return parser.parse_args()


def _print_statistics(meta: dict) -> None:
    labels = [
        label
        for label, _ in sorted(
            meta['statistics_prefix_records'].items(),
            key=lambda item: (item[1], item[0]),
        )
    ]
    print('\nRealized replay statistics:')
    print('  prefix records ' + ' '.join(
        f'{label}={meta["statistics"][label]["records"]:,}'
        for label in labels
    ))
    print('  dataset train expected ' + ' '.join(labels))
    for dataset_id in meta['dataset_ids']:
        columns = [
            f'{dataset_id:>7}',
            f'{meta["training_counts"][dataset_id]:>5}',
            f'{100 * meta["dataset_sampling_probabilities"][dataset_id]:>7.3f}%',
        ]
        columns.extend(
            f'{100 * meta["statistics"][label]["dataset_fractions"][dataset_id]:>7.3f}%'
            for label in labels
        )
        print('  ' + ' '.join(columns))
    print('  foreground ' + ' '.join(
        f'{label}={100 * meta["statistics"][label]["foreground_fraction"]:.3f}%'
        for label in labels
    ))


def main() -> None:
    args = parse_args()
    if args.max_global_batch_size < 1:
        raise ValueError('max global batch size must be positive')
    statistics_batch_sizes = tuple(
        batch_size
        for batch_size in (
            DEFAULT_STATISTICS_GLOBAL_BATCH_SIZES
            if args.statistics_global_batch_sizes is None
            else args.statistics_global_batch_sizes
        )
        if batch_size <= args.max_global_batch_size
    )
    if any(
        batch_size < 1 or batch_size > args.max_global_batch_size
        for batch_size in statistics_batch_sizes
    ):
        raise ValueError(
            'statistics batch sizes must be positive and no larger than the '
            'maximum global batch size'
        )
    if len(set(statistics_batch_sizes)) != len(statistics_batch_sizes):
        raise ValueError('statistics batch sizes must be unique')

    experiment = UniversalExperimentManifest.load(args.config)
    preprocessed_root = get_preprocessed_root()
    datasets = load_universal_datasets(
        experiment,
        preprocessed_root,
        plans_name=args.source_plans_identifier,
        configuration_name=CONFIGURATION_NAME,
    )
    registry = build_region_registry(experiment, datasets)
    reference_dataset = next(iter(datasets.values()))
    output_base = prepare_universal_nnunet_namespace(
        preprocessed_root,
        reference_dataset.data_folder,
        registry.canonical_names,
        experiment.dataset_ids,
        dataset_name=experiment.universal_dataset_name,
    )
    training_identifiers = {
        dataset_id: dataset.training_identifiers
        for dataset_id, dataset in datasets.items()
    }
    dataset_ids = experiment.dataset_ids
    total_records = experiment.num_updates * args.max_global_batch_size
    statistics_prefixes = {
        f'gb{batch_size}': experiment.num_updates * batch_size
        for batch_size in statistics_batch_sizes
    }
    output_dir = output_base / args.replay_directory
    generation_fingerprint = sqrt_replay_generation_fingerprint(
        seed=args.seed,
        total_records=total_records,
        dataset_ids=dataset_ids,
        training_identifiers=training_identifiers,
        foreground_probability=args.foreground_probability,
    )

    print('Preparing flat sqrt-dataset replay:')
    print(f'  fold: {experiment.fold}')
    print(f'  seed: {args.seed}')
    print(f'  num_updates: {experiment.num_updates:,}')
    print(f'  max_global_batch_size: {args.max_global_batch_size}')
    print(f'  total_records: {total_records:,}')
    print(f'  foreground_probability: {args.foreground_probability:.12g}')
    print(f'  records_per_shard: {args.records_per_shard:,}')
    print(f'  generation_fingerprint: {generation_fingerprint}')
    print(f'  output_dir: {output_dir}')

    started = time.perf_counter()
    if not args.reuse_existing_replay:
        generate_sqrt_replay_stream(
            seed=args.seed,
            total_records=total_records,
            dataset_ids=dataset_ids,
            training_identifiers=training_identifiers,
            output_dir=output_dir,
            foreground_probability=args.foreground_probability,
            records_per_shard=args.records_per_shard,
            statistics_prefixes=statistics_prefixes,
        )
    reader = ReplayStreamReader(
        output_dir,
        expected_fingerprint=generation_fingerprint,
        valid_identifiers=training_identifiers,
    )
    reader.validate()
    if reader.total_records != total_records:
        raise ValueError(
            f'replay contains {reader.total_records} records, expected {total_records}'
        )
    existing_prefixes = reader.meta.get('statistics_prefix_records')
    requested_prefixes = {'full': total_records, **statistics_prefixes}
    if not isinstance(existing_prefixes, dict) or any(
        existing_prefixes.get(label) != count
        for label, count in requested_prefixes.items()
    ):
        raise ValueError(
            'replay statistics do not contain the requested record prefixes'
        )

    elapsed = time.perf_counter() - started
    print(f'  stream_fingerprint: {reader.stream_fingerprint}')
    _print_statistics(reader.meta)
    print(f'\nDone in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
