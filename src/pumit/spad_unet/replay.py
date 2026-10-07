"""Pre-generated flat replay stream for Universal SPAD U-Net systems.

Replay groups define dataset permutation and complementary foreground decisions. Optimizer batches
independently slice the flat record sequence, and DDP partitions each optimizer batch across ranks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import torch
import torch.distributed as dist
import yaml
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from pumit.spad_unet.geometry import compute_continuous_da
from pumit.spad_unet.data import UniversalDataset
from pumit.spad_unet.universal import CanonicalRegionRegistry


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_SHARD_KEYS = frozenset({'records', 'steps', 'logical_batch_size'})
COMPLEMENTARY_FOREGROUND_REPLAY_RULE = (
    'one_sample_per_dataset_with_complementary_half_foreground_over_step_pairs'
)
SQRT_DATASET_REPLAY_RULE = (
    'sqrt_fold_training_count_dataset_sampling_with_independent_foreground'
)
SQRT_DATASET_REPLAY_FORMAT_VERSION = 6


@dataclass(frozen=True)
class ReplayRecord:
    dataset_id: str
    case_id: str
    force_foreground: bool


@dataclass(frozen=True)
class UniversalSample:
    """One augmented sample on its source dataset's native target grid."""

    dataset_id: str
    data: torch.Tensor
    target: torch.Tensor
    region_indices: torch.Tensor
    continuous_da: float | None = None

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
    ) -> 'UniversalSample':
        return UniversalSample(
            self.dataset_id,
            self.data.to(device, non_blocking=non_blocking),
            self.target.to(device, non_blocking=non_blocking),
            self.region_indices.to(device, non_blocking=non_blocking),
            self.continuous_da,
        )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def replay_fingerprint(
    seed: int,
    total_steps: int,
    dataset_ids: tuple[str, ...],
    training_identifiers: dict[str, tuple[str, ...]],
    foreground_samples_per_batch: int,
    samples_per_dataset: int = 1,
    sampling_rule: str = COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
) -> str:
    """SHA-256 prefix (32 hex chars) of the generation contract.

    Covers the sampling contract, not the emission order within a group: a stream generated before
    per-group permutation carries the same fingerprint. Give a permuted stream a fresh seed.
    """
    payload = {
        'seed': seed,
        'total_steps': total_steps,
        'dataset_ids': list(dataset_ids),
        'sampling_rule': sampling_rule,
        'foreground_samples_per_batch': foreground_samples_per_batch,
        'training_identifiers': {
            k: sorted(v) for k, v in sorted(training_identifiers.items())
        },
        'samples_per_dataset': samples_per_dataset,
    }
    canonical = repr(payload).encode()
    return hashlib.sha256(canonical).hexdigest()[:32]


def verify_legacy_replay_generation(
    reader: 'ReplayStreamReader',
    training_identifiers: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    """Verify a grouped replay's declared generation contract and return it.

    Recomputes the fingerprint from the stream's own meta plus the caller's
    training identifiers, so a stream generated against a different suite or
    split fails here instead of binding silently. Returns the generation
    parameters a plan must pin: seed, replay_total_groups,
    samples_per_dataset, foreground_samples_per_batch, and sampling_rule.
    """
    if reader.has_exact_stream_fingerprint:
        raise ValueError(
            'exact-content streams bind via stream_fingerprint, not the '
            'grouped generation contract'
        )
    seed = reader.meta.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f'replay meta must declare an integer seed, got {seed!r}'
        )
    expected_fingerprint = replay_fingerprint(
        seed=seed,
        total_steps=reader.total_steps,
        dataset_ids=reader.dataset_ids,
        training_identifiers=training_identifiers,
        foreground_samples_per_batch=reader.foreground_samples_per_batch,
        samples_per_dataset=reader.samples_per_dataset,
        sampling_rule=reader.sampling_rule,
    )
    if reader.meta.get('fingerprint') != expected_fingerprint:
        raise ValueError(
            f'replay meta fingerprint {reader.meta.get("fingerprint")!r} does '
            f'not match the generation contract it declares for this '
            f'experiment suite'
        )
    return {
        'seed': seed,
        'replay_total_groups': reader.total_steps,
        'samples_per_dataset': reader.samples_per_dataset,
        'foreground_samples_per_batch': reader.foreground_samples_per_batch,
        'sampling_rule': reader.sampling_rule,
    }


def open_grouped_replay_reader(
    stream_dir: Path,
    generation: Mapping[str, object],
    training_identifiers: dict[str, tuple[str, ...]],
) -> 'ReplayStreamReader':
    """Open a grouped stream and verify it against a plan's pinned contract.

    `generation` carries the fields a plan records for a grouped replay:
    seed, replay_total_groups, samples_per_dataset,
    foreground_samples_per_batch, and sampling_rule.
    """
    fingerprint = replay_fingerprint(
        seed=generation['seed'],
        total_steps=generation['replay_total_groups'],
        dataset_ids=tuple(training_identifiers),
        training_identifiers=training_identifiers,
        foreground_samples_per_batch=generation[
            'foreground_samples_per_batch'
        ],
        samples_per_dataset=generation['samples_per_dataset'],
        sampling_rule=generation['sampling_rule'],
    )
    return ReplayStreamReader(
        stream_dir,
        expected_fingerprint=fingerprint,
        valid_identifiers=training_identifiers,
    )


def _validate_complementary_contract(
    dataset_ids: tuple[str, ...],
    samples_per_dataset: int,
    foreground_samples_per_batch: int,
    total_steps: int,
) -> None:
    if samples_per_dataset != 1:
        raise ValueError(
            'complementary-foreground replay requires samples_per_dataset=1'
        )
    if len(dataset_ids) % 2:
        raise ValueError(
            'complementary-foreground replay requires an even dataset count'
        )
    if foreground_samples_per_batch != len(dataset_ids) // 2:
        raise ValueError(
            'complementary-foreground replay requires half of the datasets '
            'to force foreground in each batch'
        )
    if total_steps % 2:
        raise ValueError(
            'complementary-foreground replay requires an even total_steps'
        )


def sqrt_replay_generation_fingerprint(
    seed: int,
    total_records: int,
    dataset_ids: tuple[str, ...],
    training_identifiers: dict[str, tuple[str, ...]],
    foreground_probability: float,
) -> str:
    """Fingerprint the complete v6 generation contract, excluding shard layout."""
    _validate_sqrt_replay_contract(
        seed=seed,
        total_records=total_records,
        dataset_ids=dataset_ids,
        training_identifiers=training_identifiers,
        foreground_probability=foreground_probability,
        records_per_shard=1,
    )
    payload = {
        'format_version': SQRT_DATASET_REPLAY_FORMAT_VERSION,
        'seed': seed,
        'total_records': total_records,
        'dataset_ids': list(dataset_ids),
        'sampling_rule': SQRT_DATASET_REPLAY_RULE,
        'dataset_sampling': 'categorical_sqrt_fold_training_count',
        'case_sampling': 'uniform_within_selected_dataset',
        'foreground_sampling': 'independent_bernoulli_per_record',
        'foreground_probability': float(foreground_probability),
        'training_identifiers': {
            dataset_id: sorted(training_identifiers[dataset_id])
            for dataset_id in sorted(training_identifiers)
        },
        'rng_streams': {
            'dataset': 'numpy.default_rng(seed)',
            'foreground': 'numpy.default_rng(seed+1)',
            'case_per_dataset': 'numpy.default_rng(seed+2+dataset_index)',
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:32]


def _new_ordered_content_hasher():
    hasher = hashlib.sha256()
    hasher.update(b'pumit.spad_unet.replay.ordered_records.v1\0')
    return hasher


def _update_ordered_content_hash(
    hasher,
    records: list[dict],
) -> None:
    """Hash ordered record values independently of MessagePack shard boundaries."""
    for record in records:
        hasher.update(msgpack.packb(
            (record['d'], record['c'], record['f']),
            use_bin_type=True,
        ))


def replay_group_count_for_budget(
    optimizer_steps: int,
    global_batch_size_budget: int,
    logical_group_size: int,
    *,
    group_multiple: int = 1,
) -> int:
    """Return enough emission groups for a flat replay record budget.

    Emission groups define dataset permutation and foreground decisions. Optimizer batches are
    independent contiguous slices of the resulting flat record stream and may cross group bounds.
    """
    values = {
        'optimizer_steps': optimizer_steps,
        'global_batch_size_budget': global_batch_size_budget,
        'logical_group_size': logical_group_size,
        'group_multiple': group_multiple,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer, got {value!r}')
    required_records = optimizer_steps * global_batch_size_budget
    groups = math.ceil(required_records / logical_group_size)
    return math.ceil(groups / group_multiple) * group_multiple


def _shard_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------


def _validate_sqrt_replay_contract(
    *,
    seed: int,
    total_records: int,
    dataset_ids: tuple[str, ...],
    training_identifiers: dict[str, tuple[str, ...]],
    foreground_probability: float,
    records_per_shard: int,
) -> None:
    integer_values = {
        'seed': seed,
        'total_records': total_records,
        'records_per_shard': records_per_shard,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer, got {value!r}')
    if total_records < 1:
        raise ValueError('total_records must be positive')
    if records_per_shard < 1:
        raise ValueError('records_per_shard must be positive')
    if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError('dataset_ids must be non-empty and unique')
    if set(training_identifiers) != set(dataset_ids):
        raise ValueError('training_identifiers must define every dataset exactly once')
    empty_datasets = [
        dataset_id
        for dataset_id in dataset_ids
        if not training_identifiers[dataset_id]
    ]
    if empty_datasets:
        raise ValueError(
            f'every dataset must contain training cases; empty={empty_datasets}'
        )
    duplicate_cases = [
        dataset_id
        for dataset_id in dataset_ids
        if len(set(training_identifiers[dataset_id]))
        != len(training_identifiers[dataset_id])
    ]
    if duplicate_cases:
        raise ValueError(
            f'training identifiers must be unique within each dataset; '
            f'duplicates in {duplicate_cases}'
        )
    if (
        isinstance(foreground_probability, bool)
        or not isinstance(foreground_probability, (int, float))
        or not math.isfinite(float(foreground_probability))
        or not 0 <= foreground_probability <= 1
    ):
        raise ValueError(
            f'foreground_probability must be within [0, 1], got '
            f'{foreground_probability!r}'
        )


def _sqrt_dataset_probabilities(
    dataset_ids: tuple[str, ...],
    training_identifiers: dict[str, tuple[str, ...]],
) -> np.ndarray:
    weights = np.sqrt(np.asarray([
        len(training_identifiers[dataset_id]) for dataset_id in dataset_ids
    ], dtype=np.float64))
    return weights / weights.sum()


def _replay_statistics_snapshot(
    *,
    records: int,
    dataset_ids: tuple[str, ...],
    expected_probabilities: np.ndarray,
    dataset_counts: np.ndarray,
    foreground_counts: np.ndarray,
) -> dict:
    dataset_count_dict = {
        dataset_id: int(dataset_counts[index])
        for index, dataset_id in enumerate(dataset_ids)
    }
    foreground_count_dict = {
        dataset_id: int(foreground_counts[index])
        for index, dataset_id in enumerate(dataset_ids)
    }
    return {
        'records': records,
        'dataset_counts': dataset_count_dict,
        'dataset_fractions': {
            dataset_id: dataset_count_dict[dataset_id] / records
            for dataset_id in dataset_ids
        },
        'expected_dataset_fractions': {
            dataset_id: float(expected_probabilities[index])
            for index, dataset_id in enumerate(dataset_ids)
        },
        'foreground_count': int(foreground_counts.sum()),
        'foreground_fraction': float(foreground_counts.sum() / records),
        'dataset_foreground_counts': foreground_count_dict,
        'dataset_foreground_fractions': {
            dataset_id: (
                foreground_count_dict[dataset_id] / dataset_count_dict[dataset_id]
                if dataset_count_dict[dataset_id]
                else None
            )
            for dataset_id in dataset_ids
        },
    }


def generate_sqrt_replay_stream(
    *,
    seed: int,
    total_records: int,
    dataset_ids: tuple[str, ...],
    training_identifiers: dict[str, tuple[str, ...]],
    output_dir: Path,
    foreground_probability: float = 1 / 3,
    records_per_shard: int = 500_000,
    statistics_prefixes: Mapping[str, int] | None = None,
) -> dict:
    """Generate a v6 flat replay whose decisions are independent per record.

    A record first draws a dataset with probability proportional to the square root of its fold
    training count, then draws a case uniformly within that dataset. Its foreground decision is an
    independent Bernoulli draw. The serialized record contract remains the common ``(d, c, f)``
    schema consumed by training.

    Args:
        seed: Seed shared by the explicitly separated dataset, case, and foreground RNG streams.
        total_records: Number of records in the flat replay.
        dataset_ids: Ordered dataset identifiers.
        training_identifiers: Fold-training case identifiers by dataset.
        output_dir: Directory receiving MessagePack shards and ``meta.yaml``.
        foreground_probability: Per-record forced-foreground probability.
        records_per_shard: Maximum records in each shard.
        statistics_prefixes: Named prefix lengths for realized sampling statistics. Full-stream
            statistics are always recorded under ``full``.

    Returns:
        The metadata written to ``meta.yaml``.
    """
    _validate_sqrt_replay_contract(
        seed=seed,
        total_records=total_records,
        dataset_ids=dataset_ids,
        training_identifiers=training_identifiers,
        foreground_probability=foreground_probability,
        records_per_shard=records_per_shard,
    )
    prefix_records = {'full': total_records}
    if statistics_prefixes is not None:
        for label, count in statistics_prefixes.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f'statistics prefix labels must be non-empty strings, got {label!r}'
                )
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f'statistics prefix {label!r} must be an integer, got {count!r}'
                )
            if not 1 <= count <= total_records:
                raise ValueError(
                    f'statistics prefix {label!r}={count} is outside '
                    f'[1, {total_records}]'
                )
            prefix_records[label] = count

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = list(output_dir.glob('shard_*.msgpack'))
    if existing_shards:
        raise FileExistsError(
            f'output directory {output_dir} already contains '
            f'{len(existing_shards)} shard(s); remove them or use a different directory'
        )

    probabilities = _sqrt_dataset_probabilities(
        dataset_ids,
        training_identifiers,
    )
    generation_fingerprint = sqrt_replay_generation_fingerprint(
        seed=seed,
        total_records=total_records,
        dataset_ids=dataset_ids,
        training_identifiers=training_identifiers,
        foreground_probability=foreground_probability,
    )
    dataset_rng = np.random.default_rng(seed)
    foreground_rng = np.random.default_rng(seed + 1)
    case_rngs = tuple(
        np.random.default_rng(seed + 2 + dataset_index)
        for dataset_index in range(len(dataset_ids))
    )
    cases_by_dataset = tuple(
        np.asarray(sorted(training_identifiers[dataset_id]), dtype=object)
        for dataset_id in dataset_ids
    )

    boundary_labels: dict[int, list[str]] = {}
    for label, count in prefix_records.items():
        boundary_labels.setdefault(count, []).append(label)
    boundaries = sorted(boundary_labels)
    next_boundary_index = 0
    dataset_counts = np.zeros(len(dataset_ids), dtype=np.int64)
    foreground_counts = np.zeros(len(dataset_ids), dtype=np.int64)
    statistics: dict[str, dict] = {}
    content_hasher = _new_ordered_content_hasher()
    shard_hashes = {}
    n_shards = math.ceil(total_records / records_per_shard)
    emitted_records = 0

    for shard_index in range(n_shards):
        shard_records = min(records_per_shard, total_records - emitted_records)
        dataset_indices = dataset_rng.choice(
            len(dataset_ids),
            size=shard_records,
            p=probabilities,
        )
        foreground = foreground_rng.random(shard_records) < foreground_probability
        case_ids = np.empty(shard_records, dtype=object)
        for dataset_index, cases in enumerate(cases_by_dataset):
            positions = np.flatnonzero(dataset_indices == dataset_index)
            case_indices = case_rngs[dataset_index].integers(
                len(cases),
                size=len(positions),
            )
            case_ids[positions] = cases[case_indices]

        records = [
            {
                'd': dataset_ids[int(dataset_index)],
                'c': str(case_id),
                'f': bool(force_foreground),
            }
            for dataset_index, case_id, force_foreground in zip(
                dataset_indices,
                case_ids,
                foreground,
                strict=True,
            )
        ]
        _update_ordered_content_hash(content_hasher, records)

        local_start = 0
        while local_start < shard_records:
            next_boundary = boundaries[next_boundary_index]
            local_end = min(
                shard_records,
                local_start + next_boundary - emitted_records,
            )
            segment_dataset_indices = dataset_indices[local_start:local_end]
            segment_foreground = foreground[local_start:local_end]
            dataset_counts += np.bincount(
                segment_dataset_indices,
                minlength=len(dataset_ids),
            )
            foreground_counts += np.bincount(
                segment_dataset_indices[segment_foreground],
                minlength=len(dataset_ids),
            )
            consumed = local_end - local_start
            emitted_records += consumed
            local_start = local_end
            if emitted_records != next_boundary:
                continue
            for label in boundary_labels[next_boundary]:
                statistics[label] = _replay_statistics_snapshot(
                    records=emitted_records,
                    dataset_ids=dataset_ids,
                    expected_probabilities=probabilities,
                    dataset_counts=dataset_counts,
                    foreground_counts=foreground_counts,
                )
            next_boundary_index += 1

        shard_path = output_dir / f'shard_{shard_index:05d}.msgpack'
        tmp_path = shard_path.with_suffix('.msgpack.tmp')
        with open(tmp_path, 'wb') as f:
            msgpack.pack({
                'records': records,
                'steps': shard_records,
                'logical_batch_size': 1,
            }, f)
        tmp_path.rename(shard_path)
        shard_hashes[shard_path.name] = _shard_sha256(shard_path)

    stream_fingerprint = content_hasher.hexdigest()
    meta = {
        'format_version': SQRT_DATASET_REPLAY_FORMAT_VERSION,
        # Keep the common field for callers that only need to identify a replay generation.
        'fingerprint': generation_fingerprint,
        'generation_fingerprint': generation_fingerprint,
        'stream_fingerprint': stream_fingerprint,
        'seed': seed,
        # V6 is a flat record stream represented as one-record logical groups for compatibility
        # with the existing shard schema and reader arithmetic.
        'total_steps': total_records,
        'total_records': total_records,
        'logical_batch_size': 1,
        'steps_per_shard': records_per_shard,
        'records_per_shard': records_per_shard,
        'n_shards': n_shards,
        'dataset_ids': list(dataset_ids),
        'sampling_rule': SQRT_DATASET_REPLAY_RULE,
        'dataset_sampling': 'categorical_sqrt_fold_training_count',
        'case_sampling': 'uniform_within_selected_dataset',
        'foreground_sampling': 'independent_bernoulli_per_record',
        'foreground_probability': float(foreground_probability),
        'training_counts': {
            dataset_id: len(training_identifiers[dataset_id])
            for dataset_id in dataset_ids
        },
        'dataset_sampling_probabilities': {
            dataset_id: float(probabilities[index])
            for index, dataset_id in enumerate(dataset_ids)
        },
        'statistics_prefix_records': prefix_records,
        'statistics': statistics,
        'shard_hashes': shard_hashes,
    }
    (output_dir / 'meta.yaml').write_text(yaml.dump(meta, sort_keys=False))
    return meta


def generate_replay_stream(
    seed: int,
    total_steps: int,
    dataset_ids: tuple[str, ...],
    training_identifiers: dict[str, tuple[str, ...]],
    foreground_samples_per_batch: int,
    output_dir: Path,
    steps_per_shard: int = 50_000,
    samples_per_dataset: int = 1,
    sampling_rule: str = COMPLEMENTARY_FOREGROUND_REPLAY_RULE,
) -> None:
    """Generate the full replay stream as MessagePack shards + meta.yaml.

    Args:
        seed: Global RNG seed for reproducibility.
        total_steps: Number of replay emission groups. Each group contains the configured number
            of samples from every dataset; optimizer batches independently slice the flat stream.
        dataset_ids: Ordered dataset identifiers.
        training_identifiers: Per-dataset sorted training case IDs.
        foreground_samples_per_batch: Number of samples that force a foreground crop in every
            replay emission group. The name is retained for stream-format compatibility.
        output_dir: Directory to write shards and metadata into. Must not
            contain existing shard files.
        steps_per_shard: Steps per shard file.
        samples_per_dataset: Number of samples drawn from each dataset in every emission group.
        sampling_rule: Replay sampling contract.
    """
    if samples_per_dataset < 1:
        raise ValueError('samples_per_dataset must be positive')
    if sampling_rule == COMPLEMENTARY_FOREGROUND_REPLAY_RULE:
        _validate_complementary_contract(
            dataset_ids,
            samples_per_dataset,
            foreground_samples_per_batch,
            total_steps,
        )
    else:
        raise ValueError(f'unsupported sampling rule {sampling_rule!r}')

    logical_batch_size = len(dataset_ids) * samples_per_dataset
    if not 0 <= foreground_samples_per_batch <= logical_batch_size:
        raise ValueError(
            f'foreground_samples_per_batch={foreground_samples_per_batch} is '
            f'outside replay group size {logical_batch_size}'
        )
    if set(training_identifiers) != set(dataset_ids):
        raise ValueError('training_identifiers must define every dataset exactly once')
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_shards = list(output_dir.glob('shard_*.msgpack'))
    if existing_shards:
        raise FileExistsError(
            f'output directory {output_dir} already contains {len(existing_shards)} shard(s); '
            'remove them or use a different directory'
        )

    n_shards = math.ceil(total_steps / steps_per_shard)
    fingerprint = replay_fingerprint(
        seed,
        total_steps,
        dataset_ids,
        training_identifiers,
        foreground_samples_per_batch,
        samples_per_dataset,
        sampling_rule,
    )

    case_rng = np.random.default_rng(seed)
    foreground_rng = np.random.default_rng(seed + 1)
    order_rng = np.random.default_rng(seed + 2)

    shard_hashes = {}
    step = 0
    complementary_foreground_indices: set[int] | None = None
    for shard_idx in range(n_shards):
        shard_steps = min(steps_per_shard, total_steps - step)
        records = []
        for _ in range(shard_steps):
            if step % 2 == 0:
                complementary_foreground_indices = set(
                    foreground_rng.choice(
                        logical_batch_size,
                        size=foreground_samples_per_batch,
                        replace=False,
                    ).tolist()
                )
                foreground_indices = complementary_foreground_indices
            else:
                if complementary_foreground_indices is None:
                    raise RuntimeError(
                        'missing first batch of complementary step pair'
                    )
                foreground_indices = (
                    set(range(logical_batch_size))
                    - complementary_foreground_indices
                )
                complementary_foreground_indices = None
            # Permuted per group so a batch smaller than the dataset count still sees varied
            # dataset combinations. Foreground is keyed to the dataset index, not the emission
            # slot, so the complementary halves stay complementary per dataset.
            for dataset_index in order_rng.permutation(len(dataset_ids)):
                did = dataset_ids[dataset_index]
                cases = training_identifiers[did]
                case_id = cases[case_rng.integers(len(cases))]
                records.append({
                    'd': did,
                    'c': case_id,
                    'f': int(dataset_index) in foreground_indices,
                })
            step += 1

        expected_records = shard_steps * logical_batch_size
        if len(records) != expected_records:
            raise RuntimeError(
                f'shard {shard_idx}: generated {len(records)} records, '
                f'expected {expected_records}'
            )

        shard_path = output_dir / f'shard_{shard_idx:05d}.msgpack'
        tmp_path = shard_path.with_suffix('.msgpack.tmp')
        with open(tmp_path, 'wb') as f:
            msgpack.pack({
                'records': records,
                'steps': shard_steps,
                'logical_batch_size': logical_batch_size,
            }, f)
        tmp_path.rename(shard_path)
        shard_hashes[f'shard_{shard_idx:05d}.msgpack'] = _shard_sha256(shard_path)

    # Write metadata only after all shards are complete
    meta = {
        'format_version': 5,
        'fingerprint': fingerprint,
        'seed': seed,
        'total_steps': total_steps,
        'logical_batch_size': logical_batch_size,
        'steps_per_shard': steps_per_shard,
        'n_shards': n_shards,
        'dataset_ids': list(dataset_ids),
        'sampling_rule': sampling_rule,
        'samples_per_dataset': samples_per_dataset,
        'foreground_samples_per_batch': foreground_samples_per_batch,
        'training_counts': {
            did: len(training_identifiers[did]) for did in dataset_ids
        },
        'shard_hashes': shard_hashes,
    }
    meta_path = output_dir / 'meta.yaml'
    meta_path.write_text(yaml.dump(meta, sort_keys=False))


# ---------------------------------------------------------------------------
# Stream reader
# ---------------------------------------------------------------------------


class ReplayStreamReader:
    """Sequential reader over a pre-generated replay stream.

    Validates fingerprint and shard existence on construction.
    Call `validate()` for eager full-stream preflight before training.
    """

    def __init__(
        self,
        stream_dir: Path,
        expected_fingerprint: str | None = None,
        valid_identifiers: dict[str, tuple[str, ...]] | None = None,
        expected_stream_fingerprint: str | None = None,
    ):
        """
        Args:
            stream_dir: Directory containing meta.yaml and shard files.
            expected_fingerprint: If provided, reject streams with a different fingerprint.
            valid_identifiers: If provided, validate that every record in every shard
                names a case from this set (checked during validate() or lazy load).
            expected_stream_fingerprint: If provided, reject streams whose exact ordered-content
                fingerprint differs. Legacy streams do not provide this fingerprint.
        """
        self.stream_dir = stream_dir
        meta_path = stream_dir / 'meta.yaml'
        if not meta_path.exists():
            raise FileNotFoundError(f'meta.yaml not found in {stream_dir}')
        self.meta = yaml.safe_load(meta_path.read_text())
        self.format_version = self.meta.get('format_version')
        self._is_flat_sqrt_stream = (
            self.format_version == SQRT_DATASET_REPLAY_FORMAT_VERSION
        )

        if expected_fingerprint is not None:
            if self.meta['fingerprint'] != expected_fingerprint:
                raise ValueError(
                    f'fingerprint mismatch: stream has {self.meta["fingerprint"]!r}, '
                    f'expected {expected_fingerprint!r}'
                )
        if expected_stream_fingerprint is not None:
            stream_fingerprint = self.meta.get('stream_fingerprint')
            if stream_fingerprint is None:
                raise ValueError(
                    'replay metadata does not provide stream_fingerprint; '
                    'legacy streams only support expected_fingerprint'
                )
            if stream_fingerprint != expected_stream_fingerprint:
                raise ValueError(
                    f'stream fingerprint mismatch: stream has '
                    f'{stream_fingerprint!r}, expected '
                    f'{expected_stream_fingerprint!r}'
                )

        self.total_steps: int = self.meta['total_steps']
        self.logical_batch_size: int = self.meta['logical_batch_size']
        self.steps_per_shard: int = self.meta['steps_per_shard']
        self.n_shards: int = self.meta['n_shards']
        self.dataset_ids = tuple(self.meta['dataset_ids'])
        self.sampling_rule: str = self.meta['sampling_rule']
        self.samples_per_dataset: int = self.meta.get('samples_per_dataset', 1)
        self.foreground_samples_per_batch: int | None = (
            self.meta.get('foreground_samples_per_batch')
            if self._is_flat_sqrt_stream
            else self.meta['foreground_samples_per_batch']
        )
        if self._is_flat_sqrt_stream:
            if self.sampling_rule != SQRT_DATASET_REPLAY_RULE:
                raise ValueError(
                    f'v6 replay requires sampling_rule={SQRT_DATASET_REPLAY_RULE!r}'
                )
            if self.logical_batch_size != 1:
                raise ValueError('v6 replay requires logical_batch_size=1')
            if self.total_steps != self.meta.get('total_records'):
                raise ValueError(
                    'v6 replay total_steps must equal total_records'
                )
            if self.meta.get('records_per_shard') != self.steps_per_shard:
                raise ValueError(
                    'v6 replay records_per_shard must equal steps_per_shard'
                )
            if self.foreground_samples_per_batch is not None:
                raise ValueError(
                    'v6 replay must not declare foreground_samples_per_batch'
                )
            foreground_probability = self.meta.get('foreground_probability')
            if (
                isinstance(foreground_probability, bool)
                or not isinstance(foreground_probability, (int, float))
                or not math.isfinite(float(foreground_probability))
                or not 0 <= foreground_probability <= 1
            ):
                raise ValueError(
                    'v6 replay foreground_probability must be within [0, 1]'
                )
            probabilities = self.meta.get('dataset_sampling_probabilities')
            if not isinstance(probabilities, dict) or set(probabilities) != set(
                self.dataset_ids
            ):
                raise ValueError(
                    'v6 replay dataset_sampling_probabilities must define every '
                    'dataset exactly once'
                )
            probability_values = tuple(probabilities.values())
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                for value in probability_values
            ) or not math.isclose(sum(probability_values), 1.0):
                raise ValueError(
                    'v6 replay dataset sampling probabilities must be finite, '
                    'non-negative, and sum to one'
                )
            if self.meta.get('generation_fingerprint') != self.meta.get(
                'fingerprint'
            ):
                raise ValueError(
                    'v6 replay fingerprint must equal generation_fingerprint'
                )
            stream_fingerprint = self.meta.get('stream_fingerprint')
            if (
                not isinstance(stream_fingerprint, str)
                or len(stream_fingerprint) != 64
            ):
                raise ValueError(
                    'v6 replay stream_fingerprint must be a full SHA-256 digest'
                )
            statistics_prefixes = self.meta.get('statistics_prefix_records')
            if (
                not isinstance(statistics_prefixes, dict)
                or statistics_prefixes.get('full') != self.total_steps
                or any(
                    not isinstance(label, str)
                    or not label
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or not 1 <= count <= self.total_steps
                    for label, count in statistics_prefixes.items()
                )
            ):
                raise ValueError(
                    'v6 replay statistics_prefix_records must contain valid '
                    'named prefixes including the full stream'
                )
            statistics = self.meta.get('statistics')
            if (
                not isinstance(statistics, dict)
                or set(statistics) != set(statistics_prefixes)
            ):
                raise ValueError(
                    'v6 replay statistics must define every named prefix '
                    'exactly once'
                )
        elif self.sampling_rule == COMPLEMENTARY_FOREGROUND_REPLAY_RULE:
            _validate_complementary_contract(
                self.dataset_ids,
                self.samples_per_dataset,
                self.foreground_samples_per_batch,
                self.total_steps,
            )
        else:
            raise ValueError(f'unsupported sampling rule {self.sampling_rule!r}')
        expected_logical_batch_size = (
            1
            if self._is_flat_sqrt_stream
            else len(self.dataset_ids) * self.samples_per_dataset
        )
        if self.logical_batch_size != expected_logical_batch_size:
            if self._is_flat_sqrt_stream:
                raise ValueError(
                    f'metadata logical_batch_size={self.logical_batch_size} does '
                    'not match the v6 flat-record contract'
                )
            raise ValueError(
                f'metadata logical_batch_size={self.logical_batch_size} does not '
                f'match {len(self.dataset_ids)} datasets × '
                f'{self.samples_per_dataset} samples per dataset'
            )
        if (
            not self._is_flat_sqrt_stream
            and not 0
            <= self.foreground_samples_per_batch
            <= self.logical_batch_size
        ):
            raise ValueError('foreground_samples_per_batch is outside replay group size')
        self._valid_cases: dict[str, set[str]] | None = (
            {d: set(ids) for d, ids in valid_identifiers.items()}
            if valid_identifiers is not None else None
        )

        # Validate metadata internal consistency
        expected_n_shards = math.ceil(self.total_steps / self.steps_per_shard)
        if self.n_shards != expected_n_shards:
            raise ValueError(
                f'metadata inconsistency: n_shards={self.n_shards} but '
                f'ceil({self.total_steps}/{self.steps_per_shard})={expected_n_shards}'
            )

        # Validate exactly the expected shard files exist (no more, no fewer)
        expected_shard_names = {
            f'shard_{i:05d}.msgpack' for i in range(self.n_shards)
        }
        actual_shard_names = {
            p.name for p in stream_dir.glob('shard_*.msgpack')
        }
        missing = expected_shard_names - actual_shard_names
        if missing:
            raise FileNotFoundError(
                f'missing shards: {sorted(missing)} (stream is incomplete)'
            )
        extra = actual_shard_names - expected_shard_names
        if extra:
            raise ValueError(
                f'unexpected extra shards: {sorted(extra)} '
                f'(expected {self.n_shards} shards)'
            )

        self._cached_shard_idx: int | None = None
        self._cached_records: list[dict] | None = None

    def _expected_shard_steps(self, shard_idx: int) -> int:
        """Steps that shard_idx must contain according to metadata."""
        return min(self.steps_per_shard, self.total_steps - shard_idx * self.steps_per_shard)

    def _load_and_validate_shard(self, shard_idx: int) -> list[dict]:
        """Load a shard and validate its structural integrity."""
        shard_path = self.stream_dir / f'shard_{shard_idx:05d}.msgpack'
        with open(shard_path, 'rb') as f:
            shard_data = msgpack.unpack(f, raw=False)

        if not isinstance(shard_data, dict):
            raise ValueError(f'shard {shard_idx}: expected dict, got {type(shard_data).__name__}')

        # Validate top-level keys match format contract
        if set(shard_data.keys()) != _SHARD_KEYS:
            raise ValueError(
                f'shard {shard_idx}: expected keys {sorted(_SHARD_KEYS)}, '
                f'got {sorted(shard_data.keys())}'
            )

        shard_steps = shard_data['steps']
        shard_batch_size = shard_data['logical_batch_size']

        # Validate steps match what metadata dictates for this shard
        expected_steps = self._expected_shard_steps(shard_idx)
        if shard_steps != expected_steps:
            raise ValueError(
                f'shard {shard_idx}: declares {shard_steps} steps, '
                f'expected {expected_steps} from metadata'
            )

        # Validate logical_batch_size matches metadata
        if shard_batch_size != self.logical_batch_size:
            raise ValueError(
                f'shard {shard_idx}: logical_batch_size={shard_batch_size}, '
                f'expected {self.logical_batch_size} from metadata'
            )

        records = shard_data['records']
        expected_records = expected_steps * self.logical_batch_size
        if len(records) != expected_records:
            raise ValueError(
                f'shard {shard_idx}: has {len(records)} records, '
                f'expected {expected_records} ({expected_steps} steps * {self.logical_batch_size} batch)'
            )

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'shard {shard_idx} record {i}: not a dict')
            if set(record.keys()) != {'d', 'c', 'f'}:
                raise ValueError(
                    f'shard {shard_idx} record {i}: unexpected keys {set(record.keys())}'
                )
            if not isinstance(record['f'], bool):
                raise ValueError(
                    f'shard {shard_idx} record {i}: foreground flag must be boolean'
                )
            if self._is_flat_sqrt_stream and record['d'] not in self.dataset_ids:
                raise ValueError(
                    f'shard {shard_idx} record {i}: unknown dataset '
                    f'{record["d"]!r}'
                )
            if self._valid_cases is not None:
                did = record['d']
                cid = record['c']
                if did not in self._valid_cases:
                    raise ValueError(
                        f'shard {shard_idx} record {i}: unknown dataset {did!r}'
                    )
                if cid not in self._valid_cases[did]:
                    raise ValueError(
                        f'shard {shard_idx} record {i}: case {cid!r} not in '
                        f'dataset {did!r} training set'
                    )

        if self._is_flat_sqrt_stream:
            return records

        first_step = shard_idx * self.steps_per_shard
        for local_step in range(expected_steps):
            start = local_step * self.logical_batch_size
            step_records = records[start:start + self.logical_batch_size]
            # Order is permuted per group, so require the multiset rather than the sequence.
            actual_dataset_ids = sorted(record['d'] for record in step_records)
            expected_dataset_ids = sorted(
                dataset_id
                for dataset_id in self.dataset_ids
                for _ in range(self.samples_per_dataset)
            )
            if actual_dataset_ids != expected_dataset_ids:
                raise ValueError(
                    f'shard {shard_idx} step {first_step + local_step}: '
                    f'dataset schedule {actual_dataset_ids} does not match '
                    f'expected {expected_dataset_ids}'
                )
            foreground_count = sum(record['f'] for record in step_records)
            if foreground_count != self.foreground_samples_per_batch:
                raise ValueError(
                    f'shard {shard_idx} step {first_step + local_step}: '
                    f'foreground count {foreground_count} does not match '
                    f'expected {self.foreground_samples_per_batch}'
                )

        return records

    def validate(self) -> None:
        """Eagerly validate all shards (structure, record count, schema, membership, hashes).

        Call before training to guarantee fail-fast for any stream corruption.
        """
        shard_hashes = self.meta.get('shard_hashes')
        pending_foreground_datasets: frozenset[str] | None = None
        all_datasets = frozenset(self.dataset_ids)
        content_hasher = (
            _new_ordered_content_hasher()
            if self._is_flat_sqrt_stream
            else None
        )
        if self._is_flat_sqrt_stream:
            prefix_records = self.meta['statistics_prefix_records']
            boundary_labels: dict[int, list[str]] = {}
            for label, count in prefix_records.items():
                boundary_labels.setdefault(count, []).append(label)
            boundaries = sorted(boundary_labels)
            next_boundary_index = 0
            processed_records = 0
            dataset_index = {
                dataset_id: index
                for index, dataset_id in enumerate(self.dataset_ids)
            }
            dataset_counts = np.zeros(len(self.dataset_ids), dtype=np.int64)
            foreground_counts = np.zeros(len(self.dataset_ids), dtype=np.int64)
            expected_probabilities = np.asarray([
                self.meta['dataset_sampling_probabilities'][dataset_id]
                for dataset_id in self.dataset_ids
            ])
            actual_statistics: dict[str, dict] = {}
        else:
            boundaries = []
            actual_statistics = {}
        for shard_idx in range(self.n_shards):
            shard_name = f'shard_{shard_idx:05d}.msgpack'
            shard_path = self.stream_dir / shard_name

            # Verify content hash if metadata provides it
            if shard_hashes is not None and shard_name in shard_hashes:
                actual_hash = _shard_sha256(shard_path)
                expected_hash = shard_hashes[shard_name]
                if actual_hash != expected_hash:
                    raise ValueError(
                        f'{shard_name}: SHA-256 mismatch '
                        f'(expected {expected_hash[:16]}..., got {actual_hash[:16]}...)'
                    )

            # Full structural validation
            records = self._load_and_validate_shard(shard_idx)
            if content_hasher is not None:
                _update_ordered_content_hash(content_hasher, records)
                record_dataset_indices = np.fromiter(
                    (dataset_index[record['d']] for record in records),
                    dtype=np.int64,
                    count=len(records),
                )
                record_foreground = np.fromiter(
                    (record['f'] for record in records),
                    dtype=np.bool_,
                    count=len(records),
                )
                local_start = 0
                while local_start < len(records):
                    next_boundary = boundaries[next_boundary_index]
                    local_end = min(
                        len(records),
                        local_start + next_boundary - processed_records,
                    )
                    segment_indices = record_dataset_indices[
                        local_start:local_end
                    ]
                    segment_foreground = record_foreground[
                        local_start:local_end
                    ]
                    dataset_counts += np.bincount(
                        segment_indices,
                        minlength=len(self.dataset_ids),
                    )
                    foreground_counts += np.bincount(
                        segment_indices[segment_foreground],
                        minlength=len(self.dataset_ids),
                    )
                    consumed = local_end - local_start
                    processed_records += consumed
                    local_start = local_end
                    if processed_records != next_boundary:
                        continue
                    for label in boundary_labels[next_boundary]:
                        actual_statistics[label] = _replay_statistics_snapshot(
                            records=processed_records,
                            dataset_ids=self.dataset_ids,
                            expected_probabilities=expected_probabilities,
                            dataset_counts=dataset_counts,
                            foreground_counts=foreground_counts,
                        )
                    next_boundary_index += 1
            if self.sampling_rule != COMPLEMENTARY_FOREGROUND_REPLAY_RULE:
                continue
            shard_steps = self._expected_shard_steps(shard_idx)
            first_step = shard_idx * self.steps_per_shard
            for local_step in range(shard_steps):
                start = local_step * self.logical_batch_size
                step_records = records[start:start + self.logical_batch_size]
                foreground_datasets = frozenset(
                    record['d'] for record in step_records if record['f']
                )
                global_step = first_step + local_step
                if global_step % 2 == 0:
                    pending_foreground_datasets = foreground_datasets
                    continue
                if pending_foreground_datasets is None:
                    raise ValueError(
                        f'step {global_step}: missing first batch of complementary pair'
                    )
                expected = all_datasets - pending_foreground_datasets
                if foreground_datasets != expected:
                    raise ValueError(
                        f'steps {global_step - 1} and {global_step}: foreground '
                        'dataset sets are not complementary'
                    )
                pending_foreground_datasets = None
        if pending_foreground_datasets is not None:
            raise ValueError('replay ends with an incomplete complementary step pair')
        if content_hasher is not None:
            actual_fingerprint = content_hasher.hexdigest()
            expected_fingerprint = self.meta['stream_fingerprint']
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    f'ordered stream SHA-256 mismatch '
                    f'(expected {expected_fingerprint[:16]}..., '
                    f'got {actual_fingerprint[:16]}...)'
                )
            if actual_statistics != self.meta['statistics']:
                raise ValueError(
                    'v6 replay statistics do not match the ordered records'
                )

    @property
    def total_records(self) -> int:
        """Flat record count over the whole stream."""
        return self.total_steps * self.logical_batch_size

    def records_at(self, start: int, count: int) -> list[ReplayRecord]:
        """Return `count` consecutive records from the flat stream, starting at `start`.

        The stream is a flat sequence of groups; a training batch is a slice of it, so batch size is
        independent of the dataset count. Slices may span a group boundary.
        """
        if count < 1:
            raise ValueError('count must be positive')
        total_records = self.total_records
        if not 0 <= start <= total_records - count:
            raise IndexError(
                f'records [{start}, {start + count}) out of range [0, {total_records})'
            )
        records = []
        index = start
        while index < start + count:
            group = index // self.logical_batch_size
            shard_idx = group // self.steps_per_shard
            if self._cached_shard_idx != shard_idx:
                self._cached_records = self._load_and_validate_shard(shard_idx)
                self._cached_shard_idx = shard_idx
            local_group = group % self.steps_per_shard
            offset = index % self.logical_batch_size
            take = min(
                self.logical_batch_size - offset,
                start + count - index,
            )
            base = local_group * self.logical_batch_size + offset
            records.extend(
                ReplayRecord(
                    dataset_id=r['d'],
                    case_id=r['c'],
                    force_foreground=r['f'],
                )
                for r in self._cached_records[base:base + take]
            )
            index += take
        return records

    @property
    def fingerprint(self) -> str:
        return self.meta['fingerprint']

    @property
    def stream_fingerprint(self) -> str | None:
        """Exact ordered-content fingerprint for v6 streams."""
        return self.meta.get('stream_fingerprint')

    @property
    def has_exact_stream_fingerprint(self) -> bool:
        """Whether validation recomputes the stream's ordered-content identity."""
        return self._is_flat_sqrt_stream


# ---------------------------------------------------------------------------
# nnU-Net loader with prescribed case selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseLoaderConfiguration:
    """Crop, transform, and DA configuration for one case."""

    patch_size: tuple[int, int, int]
    final_patch_size: tuple[int, int, int]
    need_to_pad: tuple[int, int, int]
    transforms: object
    continuous_da: float


class PrescribedForegroundDataLoader(nnUNetDataLoader):
    """nnU-Net loader whose next singleton sample has an explicit force-fg flag."""

    def __init__(
        self,
        *args,
        case_loader_configurations: Mapping[
            str,
            CaseLoaderConfiguration,
        ] | None = None,
        default_continuous_da: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.batch_size != 1:
            raise ValueError('prescribed foreground loaders require batch_size=1')
        self._prescribed_force_fg: bool | None = None
        self._case_loader_configurations = case_loader_configurations
        self._default_continuous_da = default_continuous_da
        self._active_continuous_da = default_continuous_da
        self.get_do_oversample = self._get_prescribed_force_fg

    def _activate_case_configuration(self, case_id: str) -> None:
        configurations = getattr(self, '_case_loader_configurations', None)
        if configurations is None:
            self._active_continuous_da = getattr(
                self,
                '_default_continuous_da',
                None,
            )
            return
        try:
            configuration = configurations[case_id]
        except KeyError as error:
            raise KeyError(f'no loader geometry for case {case_id!r}') from error
        self.patch_size = np.asarray(configuration.patch_size, dtype=int)
        self.final_patch_size = configuration.final_patch_size
        self.need_to_pad = np.asarray(configuration.need_to_pad, dtype=int)
        self.transforms = configuration.transforms
        self._active_continuous_da = configuration.continuous_da

    def get_indices(self) -> list[str]:
        selected = list(super().get_indices())
        if len(selected) != 1:
            raise ValueError(
                f'prescribed foreground loader expected one case, got {selected}'
            )
        self._activate_case_configuration(selected[0])
        return selected

    def generate_train_batch(self) -> dict:
        batch = super().generate_train_batch()
        if self._active_continuous_da is None:
            raise RuntimeError('loader did not resolve a continuous DA value')
        batch['continuous_da'] = self._active_continuous_da
        return batch

    def set_next_force_fg(self, force_fg: bool) -> None:
        if self._prescribed_force_fg is not None:
            raise RuntimeError('the previous prescribed foreground flag was not consumed')
        self._prescribed_force_fg = force_fg

    def _get_prescribed_force_fg(self, sample_idx: int) -> bool:
        if sample_idx != 0:
            raise ValueError(f'prescribed singleton loader received sample index {sample_idx}')
        if self._prescribed_force_fg is None:
            raise RuntimeError(
                'PrescribedForegroundDataLoader.set_next_force_fg() must be called '
                'before iteration'
            )
        force_fg = self._prescribed_force_fg
        self._prescribed_force_fg = None
        return force_fg


class ReplayDataLoader(PrescribedForegroundDataLoader):
    """nnUNetDataLoader that produces a prescribed case instead of random selection.

    Call `set_next_case(case_id)` and `set_next_force_fg(force_fg)` before each
    `next()`. Cropping and augmentation then proceed normally on that case.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prescribed_case: str | None = None

    def set_next_case(self, case_id: str) -> None:
        """Prescribe which case the next generate_train_batch() will load."""
        if case_id not in self.indices:
            raise ValueError(
                f'case_id {case_id!r} not in loader identifiers '
                f'(has {len(self.indices)} cases)'
            )
        self._prescribed_case = case_id

    def get_indices(self) -> list[str]:
        """Override: return the prescribed case instead of random selection."""
        if self._prescribed_case is None:
            raise RuntimeError(
                'ReplayDataLoader.set_next_case() must be called before iteration'
            )
        case = self._prescribed_case
        self._prescribed_case = None
        self._activate_case_configuration(case)
        return [case]


def native_augmentation_parameters(
    patch_size: tuple[int, ...],
) -> tuple[tuple[float, float], bool, np.ndarray, tuple[int, ...]]:
    """Evaluate nnU-Net's native augmentation planning for one patch size."""
    adapter = object.__new__(nnUNetTrainer)
    adapter.configuration_manager = SimpleNamespace(patch_size=tuple(patch_size))
    adapter.print_to_log_file = lambda *args, **kwargs: None
    return nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(
        adapter
    )


def build_training_loader(
    dataset: UniversalDataset,
    configuration_name: str,
    initial_patch_size: tuple[int, ...] | np.ndarray,
    rotation_for_da: tuple[float, float],
    mirror_axes: tuple[int, ...],
    do_dummy_2d_data_aug: bool,
    foreground_oversample_probability: float,
) -> ReplayDataLoader:
    """Build one singleton native nnU-Net training loader for replay."""
    plans_manager = PlansManager(dataset.plans)
    configuration = plans_manager.get_configuration(configuration_name)
    label_manager = plans_manager.get_label_manager(dataset.dataset_json)
    dataset_class = infer_dataset_class(str(dataset.data_folder))
    training_dataset = dataset_class(
        str(dataset.data_folder),
        identifiers=list(dataset.training_identifiers),
    )
    def build_patch_configuration(
        final_patch_size: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], object]:
        (
            case_rotation,
            case_dummy_2d,
            case_initial_patch,
            case_mirror_axes,
        ) = native_augmentation_parameters(final_patch_size)
        if case_mirror_axes != mirror_axes:
            raise ValueError(
                f'case patch {final_patch_size} uses mirror axes '
                f'{case_mirror_axes}, expected {mirror_axes}'
            )
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size=np.asarray(final_patch_size),
            rotation_for_DA=case_rotation,
            deep_supervision_scales=None,
            mirror_axes=case_mirror_axes,
            do_dummy_2d_data_aug=case_dummy_2d,
            use_mask_for_norm=configuration.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=label_manager.foreground_labels,
            regions=list(dataset.region_values),
            ignore_label=None,
        )
        return (
            tuple(int(value) for value in case_initial_patch),
            transforms,
        )

    case_loader_configurations = None
    if dataset.case_geometries is None:
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size=np.asarray(dataset.patch_size),
            rotation_for_DA=rotation_for_da,
            deep_supervision_scales=None,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=configuration.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=label_manager.foreground_labels,
            regions=list(dataset.region_values),
            ignore_label=None,
        )
        final_patch_size = dataset.patch_size
    else:
        patch_configurations = {}
        case_loader_configurations = {}
        for case_id in dataset.training_identifiers:
            geometry = dataset.geometry_for_case(case_id)
            if geometry.patch_size not in patch_configurations:
                patch_configurations[geometry.patch_size] = (
                    build_patch_configuration(geometry.patch_size)
                )
            case_initial_patch, case_transforms = patch_configurations[
                geometry.patch_size
            ]
            case_loader_configurations[case_id] = CaseLoaderConfiguration(
                patch_size=case_initial_patch,
                final_patch_size=geometry.patch_size,
                need_to_pad=tuple(
                    initial - final
                    for initial, final in zip(
                        case_initial_patch,
                        geometry.patch_size,
                        strict=True,
                    )
                ),
                transforms=case_transforms,
                continuous_da=geometry.continuous_da,
            )
        first_configuration = case_loader_configurations[
            dataset.training_identifiers[0]
        ]
        initial_patch_size = first_configuration.patch_size
        final_patch_size = first_configuration.final_patch_size
        transforms = first_configuration.transforms
    return ReplayDataLoader(
        training_dataset,
        batch_size=1,
        patch_size=initial_patch_size,
        final_patch_size=final_patch_size,
        label_manager=label_manager,
        oversample_foreground_percent=foreground_oversample_probability,
        probabilistic_oversampling=False,
        transforms=transforms,
        case_loader_configurations=case_loader_configurations,
        default_continuous_da=compute_continuous_da(dataset.spacing),
    )


def build_validation_loader(
    dataset: UniversalDataset,
    configuration_name: str,
    foreground_oversample_probability: float,
) -> PrescribedForegroundDataLoader:
    """Build one singleton native nnU-Net validation loader."""
    plans_manager = PlansManager(dataset.plans)
    label_manager = plans_manager.get_label_manager(dataset.dataset_json)
    dataset_class = infer_dataset_class(str(dataset.data_folder))
    validation_dataset = dataset_class(
        str(dataset.data_folder),
        identifiers=list(dataset.validation_identifiers),
    )
    transforms = nnUNetTrainer.get_validation_transforms(
        deep_supervision_scales=None,
        is_cascaded=False,
        foreground_labels=label_manager.foreground_labels,
        regions=list(dataset.region_values),
        ignore_label=None,
    )
    case_loader_configurations = None
    initial_patch_size = dataset.patch_size
    if dataset.case_geometries is not None:
        case_loader_configurations = {}
        for case_id in dataset.validation_identifiers:
            geometry = dataset.geometry_for_case(case_id)
            case_loader_configurations[case_id] = CaseLoaderConfiguration(
                patch_size=geometry.patch_size,
                final_patch_size=geometry.patch_size,
                need_to_pad=(0, 0, 0),
                transforms=transforms,
                continuous_da=geometry.continuous_da,
            )
        initial_patch_size = case_loader_configurations[
            dataset.validation_identifiers[0]
        ].patch_size
    return PrescribedForegroundDataLoader(
        validation_dataset,
        batch_size=1,
        patch_size=initial_patch_size,
        final_patch_size=initial_patch_size,
        label_manager=label_manager,
        oversample_foreground_percent=foreground_oversample_probability,
        probabilistic_oversampling=False,
        transforms=transforms,
        case_loader_configurations=case_loader_configurations,
        default_continuous_da=compute_continuous_da(dataset.spacing),
    )


def load_universal_samples(
    dataset_ids: tuple[str, ...],
    loaders: Mapping[str, PrescribedForegroundDataLoader],
    region_indices: Mapping[str, torch.Tensor],
    force_foreground: tuple[bool, ...],
    case_ids: tuple[str, ...] | None = None,
) -> tuple[UniversalSample, ...]:
    """Load singleton native samples selected by a replay or validation schedule."""
    if len(force_foreground) != len(dataset_ids):
        raise ValueError('force_foreground must align one-to-one with dataset_ids')
    if case_ids is not None and len(case_ids) != len(dataset_ids):
        raise ValueError('case_ids must align one-to-one with dataset_ids')

    samples = []
    for sample_index, dataset_id in enumerate(dataset_ids):
        loader = loaders[dataset_id]
        if case_ids is not None:
            if not isinstance(loader, ReplayDataLoader):
                raise TypeError('prescribed case IDs require ReplayDataLoader instances')
            loader.set_next_case(case_ids[sample_index])
        loader.set_next_force_fg(force_foreground[sample_index])
        raw_sample = next(loader)
        data = raw_sample['data']
        target = raw_sample['target']
        raw_continuous_da = raw_sample.get('continuous_da')
        continuous_da = (
            None if raw_continuous_da is None else float(raw_continuous_da)
        )
        if continuous_da is not None and (
            not math.isfinite(continuous_da) or continuous_da < 0
        ):
            raise ValueError(
                f'dataset {dataset_id} produced invalid continuous DA '
                f'{raw_continuous_da!r}'
            )
        indices = region_indices[dataset_id]
        if not isinstance(data, torch.Tensor) or data.shape[0] != 1:
            raise ValueError(
                f'dataset {dataset_id} must produce singleton tensor data, '
                f'got {None if not isinstance(data, torch.Tensor) else tuple(data.shape)}'
            )
        if not isinstance(target, torch.Tensor) or target.shape[0] != 1:
            raise ValueError(
                f'dataset {dataset_id} must produce singleton tensor targets, '
                f'got {None if not isinstance(target, torch.Tensor) else tuple(target.shape)}'
            )
        if data.shape[2:] != target.shape[2:]:
            raise ValueError(
                f'dataset {dataset_id} data spatial shape {tuple(data.shape[2:])} '
                f'does not match target {tuple(target.shape[2:])}'
            )
        if target.shape[1] != len(indices):
            raise ValueError(
                f'dataset {dataset_id} target has {target.shape[1]} region channels, '
                f'but the registry maps {len(indices)}'
            )
        samples.append(
            UniversalSample(
                dataset_id,
                data,
                target,
                indices,
                continuous_da,
            )
        )
    return tuple(samples)


def _validate_rank_slice(
    global_batch_size: int,
    local_batch_size: int,
    record_offset: int,
) -> None:
    """Validate one rank's contiguous slice of the global batch."""
    if local_batch_size < 1:
        raise ValueError('local_batch_size must be positive')
    if record_offset < 0 or record_offset + local_batch_size > global_batch_size:
        raise ValueError(
            f'rank slice [{record_offset}, {record_offset + local_batch_size}) '
            f'is outside global batch size {global_batch_size}'
        )


class UniversalReplayBatchLoader:
    """Read one rank's contiguous slice of each global replay batch.

    A batch is a slice of the flat record stream, so its size is independent of the dataset count.
    """

    def __init__(
        self,
        reader: ReplayStreamReader,
        loaders: dict[str, ReplayDataLoader],
        region_indices: dict[str, torch.Tensor],
        global_batch_size: int,
        local_batch_size: int,
        record_offset: int,
        start_step: int,
        collate_fn: Callable[[tuple[UniversalSample, ...]], dict[str, object]],
    ):
        _validate_rank_slice(global_batch_size, local_batch_size, record_offset)
        self.reader = reader
        self.loaders = loaders
        self.region_indices = region_indices
        self.global_batch_size = global_batch_size
        self.local_batch_size = local_batch_size
        self.record_offset = record_offset
        self.start_step = start_step
        self.collate_fn = collate_fn
        self.step = start_step
        self.worker_stride = 1

    def configure_worker_pool(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError('num_workers must be positive')
        self.worker_stride = num_workers

    def set_thread_id(self, thread_id: int) -> None:
        if not 0 <= thread_id < self.worker_stride:
            raise ValueError(
                f'thread_id {thread_id} is outside worker pool of size {self.worker_stride}'
            )
        self.step = self.start_step + thread_id

    def __iter__(self) -> 'UniversalReplayBatchLoader':
        return self

    def __next__(self) -> dict[str, object]:
        start = self.global_batch_size * self.step + self.record_offset
        if start + self.local_batch_size > self.reader.total_records:
            raise StopIteration
        records = self.reader.records_at(start, self.local_batch_size)
        samples = load_universal_samples(
            tuple(record.dataset_id for record in records),
            self.loaders,
            self.region_indices,
            tuple(record.force_foreground for record in records),
            tuple(record.case_id for record in records),
        )
        self.step += self.worker_stride
        return self.collate_fn(samples)


class UniversalValidationBatchLoader:
    """Apply the replay dataset schedule to one rank's slice of each validation batch.

    Validation cycles the stream instead of ending with it.
    """

    def __init__(
        self,
        reader: ReplayStreamReader,
        loaders: dict[str, PrescribedForegroundDataLoader],
        region_indices: dict[str, torch.Tensor],
        global_batch_size: int,
        local_batch_size: int,
        record_offset: int,
        collate_fn: Callable[[tuple[UniversalSample, ...]], dict[str, object]],
        start_step: int = 0,
    ):
        _validate_rank_slice(global_batch_size, local_batch_size, record_offset)
        self.reader = reader
        self.loaders = loaders
        self.region_indices = region_indices
        self.global_batch_size = global_batch_size
        self.local_batch_size = local_batch_size
        self.record_offset = record_offset
        self.collate_fn = collate_fn
        self.start_step = start_step
        self.step = start_step
        self.worker_stride = 1

    def configure_worker_pool(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError('num_workers must be positive')
        self.worker_stride = num_workers

    def set_thread_id(self, thread_id: int) -> None:
        if not 0 <= thread_id < self.worker_stride:
            raise ValueError(
                f'thread_id {thread_id} is outside worker pool of size {self.worker_stride}'
            )
        self.step = self.start_step + thread_id

    def __iter__(self) -> 'UniversalValidationBatchLoader':
        return self

    def __next__(self) -> dict[str, object]:
        batches_per_stream = self.reader.total_records // self.global_batch_size
        start = (
            self.global_batch_size * (self.step % batches_per_stream)
            + self.record_offset
        )
        records = self.reader.records_at(start, self.local_batch_size)
        samples = load_universal_samples(
            tuple(record.dataset_id for record in records),
            self.loaders,
            self.region_indices,
            tuple(record.force_foreground for record in records),
        )
        self.step += self.worker_stride
        return self.collate_fn(samples)


def build_universal_replay_dataloaders(
    trainer: nnUNetTrainer,
    datasets: Mapping[str, UniversalDataset],
    registry: CanonicalRegionRegistry,
    replay_reader: ReplayStreamReader,
    collate_fn: Callable[[tuple[UniversalSample, ...]], dict[str, object]],
) -> tuple[
    UniversalReplayBatchLoader | MultiThreadedAugmenter,
    UniversalValidationBatchLoader | MultiThreadedAugmenter,
    tuple[int, ...],
]:
    """Compose native singleton loaders into ordered Universal batch streams."""
    global_batch_size = trainer.configuration_manager.batch_size
    local_batch_size = trainer.batch_size
    world_size = dist.get_world_size() if trainer.is_ddp else 1
    rank = dist.get_rank() if trainer.is_ddp else 0
    if local_batch_size * world_size != global_batch_size:
        raise ValueError(
            f'planned global batch size {global_batch_size} does not split into '
            f'{world_size} ranks of {local_batch_size}'
        )
    if replay_reader.dataset_ids != tuple(datasets):
        raise ValueError('global replay dataset order does not match loaded datasets')
    record_offset = rank * local_batch_size

    augmentation_parameters = {
        dataset_id: native_augmentation_parameters(dataset.patch_size)
        for dataset_id, dataset in datasets.items()
    }
    mirror_axes_set = {
        parameters[3] for parameters in augmentation_parameters.values()
    }
    if len(mirror_axes_set) != 1:
        raise ValueError(
            f'Universal datasets disagree on mirroring axes: {mirror_axes_set}'
        )
    mirror_axes = mirror_axes_set.pop()
    training_loaders = {
        dataset_id: build_training_loader(
            dataset,
            trainer.configuration_name,
            initial_patch_size=augmentation_parameters[dataset_id][2],
            rotation_for_da=augmentation_parameters[dataset_id][0],
            mirror_axes=augmentation_parameters[dataset_id][3],
            do_dummy_2d_data_aug=augmentation_parameters[dataset_id][1],
            foreground_oversample_probability=trainer.oversample_foreground_percent,
        )
        for dataset_id, dataset in datasets.items()
    }
    validation_loaders = {
        dataset_id: build_validation_loader(
            dataset,
            trainer.configuration_name,
            trainer.oversample_foreground_percent,
        )
        for dataset_id, dataset in datasets.items()
    }
    region_indices = {
        dataset_id: registry.indices(dataset_id)
        for dataset_id in datasets
    }
    train_loader = UniversalReplayBatchLoader(
        replay_reader,
        training_loaders,
        region_indices,
        global_batch_size,
        local_batch_size,
        record_offset,
        trainer.current_epoch * trainer.num_iterations_per_epoch,
        collate_fn,
    )
    val_loader = UniversalValidationBatchLoader(
        replay_reader,
        validation_loaders,
        region_indices,
        global_batch_size,
        local_batch_size,
        record_offset,
        collate_fn,
        trainer.current_epoch * trainer.num_val_iterations_per_epoch,
    )
    num_workers = get_allowed_n_proc_DA()
    if num_workers == 0:
        return train_loader, val_loader, mirror_axes

    val_workers = max(1, num_workers // 2)
    train_loader.configure_worker_pool(num_workers)
    val_loader.configure_worker_pool(val_workers)
    return (
        MultiThreadedAugmenter(
            data_loader=train_loader,
            transform=None,
            num_processes=num_workers,
            num_cached_per_queue=1,
            seeds=None,
            pin_memory=trainer.device.type == 'cuda',
            wait_time=0.002,
        ),
        MultiThreadedAugmenter(
            data_loader=val_loader,
            transform=None,
            num_processes=val_workers,
            num_cached_per_queue=1,
            seeds=None,
            pin_memory=trainer.device.type == 'cuda',
            wait_time=0.002,
        ),
        mirror_axes,
    )
