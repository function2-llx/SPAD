# src/pumit/ucpt/packing.py
"""UCPT cost model, two-stream greedy packer, and shard writer."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import msgpack
import orjson


_LABELED_BUCKET_DA = {'2d': None, 'da0': 0, 'da1': 1, 'da2': 2, 'da3': 3, 'da4': 4}
PACKING_POLICY = 'nearest'
_DEFAULT_FIT_RESULT = {
    'unlab': {'coef': [2.797063, 5.829850e-03, 1.056373e-06]},
    '2d': {'coef': [21.407581, 5.213848e-03, -2.186000e-06, 1.681195e-03, 9.770944e-05, 0.768647]},
    'da0': {'coef': [26.284688, -8.305255e-03, 6.525700e-06, -0.202866, 1.270988e-03, -1.672456]},
    'da1': {'coef': [24.027551, -7.324572e-03, 6.240124e-06, -0.193678, 1.278437e-03, -0.947923]},
    'da2': {'coef': [26.234922, -7.157777e-03, 5.910336e-06, -0.261840, 1.046854e-03, -1.432355]},
    'da3': {'coef': [23.516546, -9.773450e-06, 2.967442e-06, -0.300152, 7.435991e-04, -1.418871]},
    'da4': {'coef': [20.099350, 2.727218e-03, 2.408791e-06, -0.386782, 5.469417e-04, -1.285504]},
}


def _read_coef(fit_result: dict, bucket: str, expected_size: int) -> tuple[float, ...]:
    entry = fit_result[bucket]
    coef = entry['coef']
    if not isinstance(coef, list) or len(coef) != expected_size:
        raise ValueError(f'{bucket} coefficient must be a list of length {expected_size}, got {coef!r}')
    if any(isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) for value in coef):
        raise ValueError(f'{bucket} coefficient contains a non-finite number: {coef!r}')
    return tuple(float(value) for value in coef)


@dataclass(frozen=True)
class CostModel:
    """Per-sample runtime model loaded from a benchmark fit artifact.

    Attributes:
        unlabeled_coef: ``[constant, linear, quadratic]`` for the SSL path.
        labeled_coef: Per-da ``[s0, s1, s2, d0, d1, g]`` coefficients for the segmentation path.
    """

    unlabeled_coef: tuple[float, float, float]
    labeled_coef: dict[int | None, tuple[float, float, float, float, float, float]]

    @classmethod
    def from_fit_result(cls, fit_result: dict) -> CostModel:
        """Build a cost model from the JSON object emitted by ``bench_cost_model.py fit``.

        Args:
            fit_result: Mapping from benchmark bucket to an object containing ``coef``.

        Returns:
            Validated cost model.
        """
        expected_buckets = {'unlab', *_LABELED_BUCKET_DA}
        if not isinstance(fit_result, dict) or set(fit_result) != expected_buckets:
            raise ValueError(f'cost-model buckets must be {sorted(expected_buckets)}, got {sorted(fit_result)}')
        unlabeled_coef = _read_coef(fit_result, 'unlab', 3)
        labeled_coef = {
            da: _read_coef(fit_result, bucket, 6)
            for bucket, da in _LABELED_BUCKET_DA.items()
        }
        return cls(unlabeled_coef, labeled_coef)

    @classmethod
    def from_file(cls, path: str | Path) -> CostModel:
        """Load the benchmark fit JSON at ``path``."""
        return cls.from_fit_result(orjson.loads(Path(path).read_bytes()))

    @classmethod
    def default(cls) -> CostModel:
        """Return the embedded B300 cost model for explicit CLI opt-in."""
        return cls.from_fit_result(_DEFAULT_FIT_RESULT)

    def as_fit_result(self) -> dict:
        """Return the coefficient subset of the benchmark artifact for metadata and fingerprinting."""
        result = {'unlab': {'coef': list(self.unlabeled_coef)}}
        result.update({
            bucket: {'coef': list(self.labeled_coef[da])}
            for bucket, da in _LABELED_BUCKET_DA.items()
        })
        return result

    def cost_unlabeled(self, n_patches: int) -> float:
        """Estimate the SSL-path cost of an unlabeled sample in milliseconds."""
        c, a, b = self.unlabeled_coef
        return c + a * n_patches + b * n_patches ** 2

    def cost_labeled(self, n_patches: int, k_classes: int, da: int | None) -> float:
        """Estimate the standalone segmentation-path cost of a labeled sample in milliseconds."""
        s0, s1, s2, d0, d1, g = self.labeled_coef[da]
        return (s0 + s1 * n_patches + s2 * n_patches ** 2 + (d0 + d1 * n_patches) * k_classes
                + g * (k_classes > 1))

    def sample_cost(self, sample: dict) -> float:
        """Estimate one sample's contribution to step runtime in milliseconds."""
        n = sample['n_patches']
        if sample['labeled']:
            return self.cost_labeled(n, sample['seg_cost_queries'], sample['da_enc'])
        return self.cost_unlabeled(n)


def pack_samples(
    *,
    labeled_iter,
    unlabeled_iter,
    budget_ms: float,
    label_budget_fraction: float,
    cost_model: CostModel,
):
    """Pack labeled and unlabeled streams by rounding each phase to its nearest target.

    Every emitted batch contains both sample kinds. Non-fitting samples carry over without reordering, and packing stops
    when either stream cannot supply its required floor.

    Args:
        labeled_iter: Iterator of labeled sample dictionaries.
        unlabeled_iter: Iterator of unlabeled sample dictionaries.
        budget_ms: Target total runtime budget per batch.
        label_budget_fraction: Fraction of ``budget_ms`` reserved for labeled samples.
        cost_model: Benchmark-fitted per-sample runtime model.

    Yields:
        Batch dictionaries containing ``step_idx`` and labeled-first ``samples``.
    """
    lab_budget = label_budget_fraction * budget_ms
    labeled = _NearestPhase(labeled_iter, cost_model)
    unlabeled = _NearestPhase(unlabeled_iter, cost_model)
    step_idx = 0
    while True:
        try:
            labeled_samples, lab_cost = labeled.take(lab_budget)
            unlabeled_samples, _ = unlabeled.take(budget_ms - lab_cost)
        except StopIteration:
            return
        yield {
            "step_idx": step_idx,
            "samples": labeled_samples + unlabeled_samples,
        }
        step_idx += 1


class _NearestPhase:
    """Consume one ordered sample stream into nearest-rounded non-empty groups."""

    def __init__(self, samples: Iterator[dict], cost_model: CostModel) -> None:
        self.samples = iter(samples)
        self.cost_model = cost_model
        self.pending: dict | None = None

    def take(self, target_ms: float) -> tuple[list[dict], float]:
        """Take one non-empty group, admitting a crossing sample only when it is closer to the target."""
        if self.pending is None:
            first = next(self.samples)
        else:
            first, self.pending = self.pending, None
        group = [first]
        cost = self.cost_model.sample_cost(first)
        while True:
            try:
                sample = next(self.samples)
            except StopIteration:
                break
            sample_cost = self.cost_model.sample_cost(sample)
            candidate_cost = cost + sample_cost
            if candidate_cost <= target_ms:
                group.append(sample)
                cost = candidate_cost
                continue
            if candidate_cost - target_ms < target_ms - cost:
                group.append(sample)
                cost = candidate_cost
            else:
                self.pending = sample
            break
        return group, cost


def _partition_exact_samples(
    samples: Sequence[dict],
    *,
    groups: int,
    cost_model: CostModel,
) -> list[list[dict]]:
    """Partition an ordered finite stream into exactly ``groups`` nearest-rounded non-empty groups."""
    if groups < 1:
        raise ValueError(f'groups must be positive, got {groups}')
    if len(samples) < groups:
        raise ValueError(f'cannot partition {len(samples)} samples into {groups} non-empty groups')

    costs = [cost_model.sample_cost(sample) for sample in samples]
    remaining_cost = sum(costs)
    cursor = 0
    result = []
    for group_index in range(groups):
        remaining_groups = groups - group_index
        target_cost = remaining_cost / remaining_groups
        max_stop = len(samples) - (remaining_groups - 1)
        start = cursor
        group_cost = costs[cursor]
        cursor += 1
        while cursor < max_stop:
            candidate_cost = group_cost + costs[cursor]
            if candidate_cost <= target_cost:
                group_cost = candidate_cost
                cursor += 1
                continue
            if candidate_cost - target_cost < target_cost - group_cost:
                group_cost = candidate_cost
                cursor += 1
            break
        result.append(list(samples[start:cursor]))
        remaining_cost -= group_cost
    if cursor != len(samples):
        raise RuntimeError(f'exact partition consumed {cursor}/{len(samples)} samples')
    return result


def repack_samples(
    *,
    labeled_iter: Iterator[dict],
    unlabeled_samples: Sequence[dict],
    batches: int,
    budget_ms: float,
    label_budget_fraction: float,
    cost_model: CostModel,
):
    """Repack a fixed unlabeled sequence while filling the larger labeled budget.

    The complete unlabeled sequence is partitioned into exactly ``batches`` non-empty groups without reordering. Labeled
    samples use nearest rounding against their fixed budget and may extend beyond the reused source prefix.
    """
    unlabeled_groups = _partition_exact_samples(
        unlabeled_samples,
        groups=batches,
        cost_model=cost_model,
    )
    labeled = _NearestPhase(labeled_iter, cost_model)
    labeled_budget = label_budget_fraction * budget_ms
    for step_idx, unlabeled_group in enumerate(unlabeled_groups):
        try:
            labeled_group, _ = labeled.take(labeled_budget)
        except StopIteration as error:
            raise ValueError(f'labeled stream exhausted before batch {step_idx}') from error
        yield {
            'step_idx': step_idx,
            'samples': labeled_group + unlabeled_group,
        }


def write_shard(batches: list[dict], path) -> None:
    """Write one shard atomically without replacing an existing shard.

    Args:
        batches: Batch dictionaries to serialize.
        path: Destination shard path.
    """
    path = Path(path)
    total = sum(s["n_patches"] for b in batches for s in b["samples"])
    tmp = path.parent / f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp'
    with tmp.open('xb') as file:
        msgpack.pack({'batches': batches, 'total_patches': total}, file)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def write_batch_shards(batch_iter, output_dir, batches_per_shard: int) -> int:
    """Write batches to sequential msgpack shards.

    Args:
        batch_iter: Iterable of batch dictionaries.
        output_dir: Destination directory.
        batches_per_shard: Maximum batches per shard.

    Returns:
        Number of shards written, including a final partial shard.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_count = 0
    buf: list[dict] = []
    for b in batch_iter:
        buf.append(b)
        if len(buf) >= batches_per_shard:
            write_shard(buf, output_dir / f"shard_{shard_count:05d}.msgpack")
            shard_count += 1
            buf = []
    if buf:
        write_shard(buf, output_dir / f"shard_{shard_count:05d}.msgpack")
        shard_count += 1
    return shard_count
