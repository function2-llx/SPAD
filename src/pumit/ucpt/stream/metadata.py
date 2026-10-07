"""Canonical frozen-sample metadata for production UCPT streams."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pumit.ucpt.affine import (
    _affine_load_extent,
    _canonical_inplane_grid_affine,
    stream_crop_size,
)


SAMPLE_METADATA_CONTRACT = 'canonical-inplane-affine-crop-size-v1'


def metadata_changes_replay(stats: Counter[str]) -> bool:
    """Return whether canonicalization changes pixels produced by replay."""
    return bool(stats['canonicalized_affine_samples'] or stats['trimmed_load_samples'])


def canonicalize_sample_metadata(sample: dict) -> tuple[dict, Counter[str]]:
    """Return one sample with canonical in-plane geometry and the current crop-size key."""
    params = sample.get('params')
    if not isinstance(params, list) or not params or not isinstance(params[0], dict):
        raise ValueError('sample lacks frozen spatial parameters')
    spatial = dict(params[0])
    stats: Counter[str] = Counter()

    if 'patch_size' in spatial:
        if 'crop_size' in spatial:
            raise ValueError('spatial metadata contains both patch_size and crop_size')
        spatial['crop_size'] = spatial.pop('patch_size')
        stats['renamed_patch_size_samples'] += 1
    elif 'crop_size' not in spatial:
        raise ValueError('spatial metadata lacks crop_size')
    crop_size = np.asarray(stream_crop_size(spatial), dtype=np.int64)
    if crop_size.shape != (3,) or np.any(crop_size <= 0):
        raise ValueError(f'invalid crop_size: {crop_size.tolist()}')

    affine = np.asarray(spatial.get('affine'), dtype=np.float64)
    if affine.size != 16 or not np.isfinite(affine).all():
        raise ValueError('affine must contain 16 finite values')
    affine = affine.reshape(4, 4)
    matrix = affine[:3, :3]
    canonical = _canonical_inplane_grid_affine(matrix)
    if canonical is not None:
        if not np.array_equal(matrix, canonical):
            affine = affine.copy()
            affine[:3, :3] = canonical
            spatial['affine'] = affine.ravel().tolist()
            stats['canonicalized_affine_samples'] += 1

        start = np.asarray(spatial.get('load_slice_start'), dtype=np.int64)
        stop = np.asarray(spatial.get('load_slice_stop'), dtype=np.int64)
        if start.shape != (3,) or stop.shape != (3,) or np.any(stop <= start):
            raise ValueError(
                f'invalid load slices: start={start.tolist()} stop={stop.tolist()}'
            )
        actual = stop - start
        canonical_extent = _affine_load_extent(canonical, crop_size)
        corrected_actual = np.minimum(actual, canonical_extent)
        excess = actual - corrected_actual
        if np.any(excess):
            spatial['load_slice_stop'] = (start + corrected_actual).tolist()
            stats['trimmed_load_samples'] += 1
            stats['trimmed_load_axes'] += int(np.count_nonzero(excess))
            stats['trimmed_load_extent'] += int(excess.sum())

    if not stats:
        return sample, stats
    migrated_params = list(params)
    migrated_params[0] = spatial
    return {**sample, 'params': migrated_params}, stats
