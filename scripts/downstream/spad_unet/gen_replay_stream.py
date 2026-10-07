"""Generate the shared grouped replay stream for Universal training.

The stream is a plan-agnostic artifact: records are (dataset, case, forced-foreground) triples,
so every plan family binds to it by path at preparation time. Corpus-grid plan preparation lives
in prepare_corpus_grid_universal.py; SPAD universal plans use prepare_spad_universal.py.

Example:
    pixi run -e spad-unet python scripts/downstream/spad_unet/gen_replay_stream.py \
        --stream-global-batch-size-budget 32 --replay-directory <name>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from pumit.spad_unet.experiments.corpus_grid import (
    CORPUS_GRID_CONFIGURATION_NAME,
    CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
    CORPUS_GRID_SUPPORTED_SOURCE_PLANS_IDENTIFIERS,
)
from pumit.spad_unet.data import (
    UniversalExperimentManifest,
    build_region_registry,
    get_preprocessed_root,
    load_universal_datasets,
    prepare_universal_nnunet_namespace,
)
from pumit.spad_unet.replay import (
    COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    ReplayStreamReader,
    generate_replay_stream,
    replay_fingerprint,
    replay_group_count_for_budget,
)


SAMPLES_PER_DATASET = 1
DEFAULT_SEED = 20260808


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate the grouped replay stream for Universal training.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/downstream/spad_unet/spad_ct_universal_v2.json'),
    )
    parser.add_argument(
        '--source-plans-identifier',
        default=CORPUS_GRID_SOURCE_PLANS_IDENTIFIER,
        choices=CORPUS_GRID_SUPPORTED_SOURCE_PLANS_IDENTIFIERS,
        help='Source plans used only to resolve the frozen training identifiers.',
    )
    parser.add_argument(
        '--stream-global-batch-size-budget',
        type=int,
        required=True,
        help='Size the stream for this many samples per optimizer step.',
    )
    parser.add_argument(
        '--num-updates',
        type=int,
        default=None,
        help='Override the optimizer-update budget stored in the experiment manifest.',
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--replay-directory', required=True)
    parser.add_argument('--steps-per-shard', type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_experiment = json.loads(args.config.read_text())
    if args.num_updates is not None:
        raw_experiment['num_updates'] = args.num_updates
    experiment = UniversalExperimentManifest.from_dict(raw_experiment)
    datasets = load_universal_datasets(
        experiment,
        get_preprocessed_root(),
        plans_name=args.source_plans_identifier,
        configuration_name=CORPUS_GRID_CONFIGURATION_NAME,
    )
    dataset_ids = tuple(datasets)
    training_identifiers = {
        d: ds.training_identifiers for d, ds in datasets.items()
    }

    records_per_group = len(dataset_ids) * SAMPLES_PER_DATASET
    foreground_samples_per_batch = len(dataset_ids) // 2
    replay_total_groups = replay_group_count_for_budget(
        experiment.num_updates,
        args.stream_global_batch_size_budget,
        records_per_group,
        group_multiple=2,
    )
    reference_dataset = next(iter(datasets.values()))
    registry = build_region_registry(experiment, datasets)
    reference_base = prepare_universal_nnunet_namespace(
        get_preprocessed_root(),
        reference_dataset.data_folder,
        registry.canonical_names,
        experiment.dataset_ids,
        dataset_name=experiment.universal_dataset_name,
    )
    output_dir = reference_base / args.replay_directory

    print('Generating grouped replay stream:')
    print(f'  replay seed: {args.seed}')
    print(f'  optimizer_steps: {experiment.num_updates}')
    print(
        f'  stream_global_batch_size_budget: '
        f'{args.stream_global_batch_size_budget}'
    )
    print(f'  replay_total_groups: {replay_total_groups}')
    print(f'  replay_total_records: {replay_total_groups * records_per_group}')
    print(f'  datasets: {len(dataset_ids)}')
    print(f'  foreground_samples_per_batch: {foreground_samples_per_batch}')
    print(f'  n_shards: {math.ceil(replay_total_groups / args.steps_per_shard)}')
    print(f'  output_dir: {output_dir}')

    t0 = time.perf_counter()
    generate_replay_stream(
        seed=args.seed,
        total_steps=replay_total_groups,
        dataset_ids=dataset_ids,
        training_identifiers=training_identifiers,
        foreground_samples_per_batch=foreground_samples_per_batch,
        output_dir=output_dir,
        steps_per_shard=args.steps_per_shard,
        samples_per_dataset=SAMPLES_PER_DATASET,
        sampling_rule=COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
    )
    ReplayStreamReader(
        output_dir,
        expected_fingerprint=replay_fingerprint(
            seed=args.seed,
            total_steps=replay_total_groups,
            dataset_ids=dataset_ids,
            training_identifiers=training_identifiers,
            foreground_samples_per_batch=foreground_samples_per_batch,
            samples_per_dataset=SAMPLES_PER_DATASET,
            sampling_rule=COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
        ),
        valid_identifiers=training_identifiers,
    ).validate()
    print(f'Done in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
