"""Selective migration for the requested-load affine replay fix."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from pumit.ucpt.sample_stream import _load_generation_globals, generate_sample
from pumit.ucpt.seg.class_sampling import flatten_label_contract
from pumit.ucpt.seg.label_contract import normalize_label_contract
from pumit.ucpt.affine import (
    _affine_load_extent,
    _column_norm_spacing,
    _uses_interpolate,
    stream_crop_size,
)


PREVIOUS_AFFINE_REPLAY_CONTRACT = 'requested-load-padding-v2'
AFFINE_REPLAY_CONTRACT = 'requested-load-padding-v3'
LEGACY_INTERPOLATE_OFF_DIAGONAL_THRESHOLD = 1e-6
REPAIR_RNG_NAMESPACE = 2
REPLACEMENT_RNG_NAMESPACE = 3
MIGRATION_META_KEY = '_affine_migration'


@dataclass(frozen=True)
class SpatialMigrationDecision:
    """Difference between the pre-fix and fixed replay plans."""

    required: bool
    reasons: tuple[str, ...]
    old_padding: tuple[tuple[int, int], ...]
    new_padding: tuple[tuple[int, int], ...]
    spacing_scale: tuple[float, float, float]


def _padding_pairs(
    extent: np.ndarray,
    actual: np.ndarray,
    signs: np.ndarray,
    *,
    swap_on_flip: bool,
) -> tuple[tuple[int, int], ...]:
    pairs = []
    for axis, deficit in enumerate(np.maximum(extent - actual, 0)):
        low = int(deficit) // 2
        high = int(deficit) - low
        if swap_on_flip and signs[axis] < 0:
            low, high = high, low
        pairs.append((low, high))
    return tuple(pairs)


def _spatial_params(sample_or_spatial: dict) -> dict:
    if 'affine' in sample_or_spatial:
        return sample_or_spatial
    if 'spatial' in sample_or_spatial:
        return sample_or_spatial['spatial']
    params = sample_or_spatial.get('params')
    if not isinstance(params, list) or not params or not isinstance(params[0], dict):
        raise ValueError('sample lacks frozen spatial parameters')
    return params[0]


def _legacy_uses_interpolate(affine_3x3: np.ndarray) -> bool:
    off_diag = affine_3x3 - np.diag(np.diag(affine_3x3))
    return bool(np.abs(off_diag).max() < LEGACY_INTERPOLATE_OFF_DIAGONAL_THRESHOLD)


def spatial_migration_decision(sample_or_spatial: dict) -> SpatialMigrationDecision:
    """Compare pre-fix and fixed affine replay plans using frozen metadata."""
    spatial = _spatial_params(sample_or_spatial)
    reference = np.asarray(stream_crop_size(spatial), dtype=np.int64)
    start = np.asarray(spatial['load_slice_start'], dtype=np.int64)
    stop = np.asarray(spatial['load_slice_stop'], dtype=np.int64)
    if reference.shape != (3,) or start.shape != (3,) or stop.shape != (3,):
        raise ValueError('crop size and load slices must contain three axes')
    if np.any(reference <= 0) or np.any(stop <= start):
        raise ValueError(
            f'invalid frozen crop geometry: reference={reference.tolist()} '
            f'start={start.tolist()} stop={stop.tolist()}'
        )

    affine = np.asarray(spatial['affine'], dtype=np.float64)
    if affine.size != 16 or not np.isfinite(affine).all():
        raise ValueError('affine must contain 16 finite values')
    matrix = affine.reshape(4, 4)[:3, :3]
    actual = stop - start
    new_extent = _affine_load_extent(matrix, reference)
    unit_scale = (1.0, 1.0, 1.0)
    legacy_interpolate = _legacy_uses_interpolate(matrix)

    if not legacy_interpolate:
        old_padding = ((0, 0),) * 3
        new_padding = _padding_pairs(
            new_extent,
            actual,
            np.ones(3),
            swap_on_flip=False,
        )
        required = old_padding != new_padding
        reasons = ('grid_padding',) if required else ()
        return SpatialMigrationDecision(
            required,
            reasons,
            old_padding,
            new_padding,
            unit_scale,
        )

    diagonal = np.diag(matrix)
    old_extent = np.ceil(np.abs(diagonal) * reference).astype(np.int64)
    signs = np.sign(diagonal)
    old_padding = _padding_pairs(
        old_extent,
        actual,
        signs,
        swap_on_flip=False,
    )
    if not _uses_interpolate(matrix):
        new_padding = _padding_pairs(
            new_extent,
            actual,
            np.ones(3),
            swap_on_flip=False,
        )
        return SpatialMigrationDecision(
            True,
            ('near_diagonal_grid',),
            old_padding,
            new_padding,
            unit_scale,
        )

    new_padding = _padding_pairs(
        new_extent,
        actual,
        signs,
        swap_on_flip=True,
    )
    required = old_padding != new_padding
    reasons = ('diagonal_flip_pad_order',) if required else ()
    old_effective = np.maximum(actual, old_extent)
    new_effective = np.maximum(actual, new_extent)
    spacing_scale = tuple(
        float(new / old)
        for new, old in zip(new_effective, old_effective, strict=True)
    )
    if not required and spacing_scale != unit_scale:
        raise AssertionError('unaffected replay cannot change spacing')
    return SpatialMigrationDecision(
        required,
        reasons,
        old_padding,
        new_padding,
        spacing_scale,
    )


def requires_affine_v3_upgrade(sample: dict) -> bool:
    """Return whether a finalized v2 sample changes under the v3 replay contract."""
    return 'near_diagonal_grid' in spatial_migration_decision(sample).reasons


def _near_diagonal_spacing(sample: dict) -> list[float]:
    spatial = _spatial_params(sample)
    reference = np.asarray(stream_crop_size(spatial), dtype=np.float64)
    matrix = np.asarray(spatial['affine'], dtype=np.float64).reshape(4, 4)[:3, :3]
    diagonal = np.diag(matrix)
    old_extent = np.ceil(np.abs(diagonal) * reference)
    old_realized_scale = old_extent / reference
    if np.any(old_realized_scale <= 0):
        raise ValueError(f'invalid legacy realized scale: {old_realized_scale.tolist()}')
    old_spacing = np.asarray(sample['spacing_label'], dtype=np.float64)
    source_spacing = old_spacing / old_realized_scale
    return _column_norm_spacing(matrix, source_spacing)


def apply_spacing_correction(sample: dict, decision: SpatialMigrationDecision) -> dict:
    """Return a sample whose spacing target matches the fixed diagonal extent."""
    if 'near_diagonal_grid' in decision.reasons:
        return {
            **sample,
            'spacing_label': _near_diagonal_spacing(sample),
        }
    if decision.spacing_scale == (1.0, 1.0, 1.0):
        return sample
    spacing = sample.get('spacing_label')
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise ValueError('sample spacing_label must contain three axes')
    corrected = {
        **sample,
        'spacing_label': [
            float(value) * scale
            for value, scale in zip(spacing, decision.spacing_scale, strict=True)
        ],
    }
    return corrected


def migration_rng(
    seed: int,
    shard_id: int,
    namespace: int,
    ordinal: int,
) -> np.random.Generator:
    """Address one migration draw independently of execution order."""
    if min(seed, shard_id, namespace, ordinal) < 0:
        raise ValueError('migration RNG coordinates must be non-negative')
    sequence = np.random.SeedSequence(
        seed,
        spawn_key=(shard_id, namespace, ordinal),
    )
    return np.random.default_rng(sequence)


@cache
def _record_lookup(config_path: str) -> dict[tuple[str, str], dict]:
    _, pools, _ = _load_generation_globals(config_path)
    records = pools[1][0]
    lookup = {}
    for record in records:
        identity = record['dataset'], record['key']
        if identity in lookup:
            raise ValueError(f'duplicate labeled record identity: {identity!r}')
        lookup[identity] = record
    return lookup


def _record_and_contract(sample: dict, config_path: str) -> tuple[dict, dict]:
    identity = sample['dataset'], sample['key']
    try:
        record = _record_lookup(config_path)[identity]
    except KeyError:
        raise KeyError(f'migrated labeled sample is absent from the current pool: {identity!r}') from None
    if record['img'] != sample['img']:
        raise ValueError(
            f'migrated labeled image changed for {identity!r}: '
            f'source={sample["img"]!r} current={record["img"]!r}'
        )
    contract, _ = normalize_label_contract(record)
    return record, contract


def validate_reused_classes(
    sample: dict,
    label_contract: dict,
    *,
    positive_queries: int,
    negative_queries: int,
    min_positive_voxels: int,
) -> None:
    """Check that frozen selected queries remain valid under the current contract."""
    positive, explicit_negative = flatten_label_contract(label_contract)
    positive_set = set(positive)
    explicit_negative_set = set(explicit_negative)
    selected_positive = 0
    selected_negative = 0
    for item in sample['classes']:
        pair = item['source'], item['name']
        if pair not in positive_set and pair not in explicit_negative_set:
            raise ValueError(f'selected class is absent from the current label contract: {pair!r}')
        if item['is_positive']:
            if pair not in positive_set or item['target_voxels'] < min_positive_voxels:
                raise ValueError(f'invalid frozen positive class: {item!r}')
            selected_positive += 1
        else:
            if item['target_voxels'] != 0:
                raise ValueError(f'invalid frozen negative class: {item!r}')
            selected_negative += 1
    if selected_positive > positive_queries or selected_negative > negative_queries:
        raise ValueError(
            f'frozen query limits changed: positive={selected_positive}/{positive_queries} '
            f'negative={selected_negative}/{negative_queries}'
        )


def prepare_reused_labeled(
    sample: dict,
    *,
    ordinal: int,
    config_path: str,
    decision: SpatialMigrationDecision,
) -> dict:
    """Validate and tag an unaffected labeled sample without resampling masks."""
    _, _, class_sampler = _load_generation_globals(config_path)
    record, contract = _record_and_contract(sample, config_path)
    validate_reused_classes(
        sample,
        contract,
        positive_queries=class_sampler.positive_queries,
        negative_queries=class_sampler.negative_queries,
        min_positive_voxels=class_sampler.min_positive_voxels,
    )
    work = class_sampler.estimate_work(record, contract, sample['params'])
    migrated = apply_spacing_correction(sample, decision)
    return {
        **migrated,
        MIGRATION_META_KEY: {
            'result': 'reused',
            'source_ordinal': ordinal,
            'reasons': list(decision.reasons),
            **work,
        },
    }


def repair_labeled_sample(
    sample: dict,
    ordinal: int,
    *,
    seed: int,
    shard_id: int,
    config_path: str,
) -> dict:
    """Repair one affected labeled sample or generate its deterministic replacement."""
    pipeline, pools, class_sampler = _load_generation_globals(config_path)
    record, contract = _record_and_contract(sample, config_path)
    decision = spatial_migration_decision(sample)
    if not decision.required:
        raise ValueError('repair requested for an unaffected labeled sample')
    rng = migration_rng(seed, shard_id, REPAIR_RNG_NAMESPACE, ordinal)
    focus = None
    if sample['params'][0]['foreground_forced']:
        focus = next(
            (
                (item['source'], item['name'])
                for item in sample['classes']
                if item['is_positive']
            ),
            None,
        )
    classes, work = class_sampler.sample_with_stats(
        record,
        contract,
        sample['params'],
        rng,
        focus=focus,
    )
    if classes:
        migrated = apply_spacing_correction(sample, decision)
        return {
            **migrated,
            'classes': classes,
            'seg_cost_queries': len(classes),
            MIGRATION_META_KEY: {
                'result': 'repaired',
                'source_ordinal': ordinal,
                'reasons': list(decision.reasons),
                **work,
            },
        }

    replacement_rng = migration_rng(seed, shard_id, REPLACEMENT_RNG_NAMESPACE, ordinal)
    attempts = 0
    while True:
        replacement = generate_sample(
            pipeline,
            pools,
            class_sampler,
            replacement_rng,
            True,
        )
        attempts += 1
        if replacement is not None:
            return {
                **replacement,
                MIGRATION_META_KEY: {
                    'result': 'replacement',
                    'source_ordinal': ordinal,
                    'reasons': list(decision.reasons),
                    'repair_attempts': attempts,
                    **work,
                },
            }


def prepare_reused_unlabeled(sample: dict, source_index: int) -> dict:
    """Correct and tag one reused unlabeled sample."""
    decision = spatial_migration_decision(sample)
    migrated = apply_spacing_correction(sample, decision)
    return {
        **migrated,
        'labeled': False,
        MIGRATION_META_KEY: {
            'result': 'unlabeled',
            'source_index': source_index,
            'required': decision.required,
            'reasons': list(decision.reasons),
        },
    }
