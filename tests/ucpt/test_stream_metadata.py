import math

import numpy as np

from pumit.ucpt.stream.metadata import canonicalize_sample_metadata


def _sample(cross_axis: float) -> dict:
    theta = math.radians(31)
    matrix = np.eye(4)
    matrix[0, 0] = -1
    matrix[1:3, 1:3] = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]
    matrix[0, 1] = cross_axis
    matrix[2, 0] = -cross_axis
    return {
        'labeled': False,
        'params': [{
            'patch_size': [32, 64, 64],
            'affine': matrix.ravel().tolist(),
            'load_slice_start': [7, 11, 13],
            'load_slice_stop': [40, 99, 101],
        }],
    }


def test_metadata_migration_canonicalizes_affine_extent_and_crop_key():
    migrated, stats = canonicalize_sample_metadata(_sample(1e-16))
    spatial = migrated['params'][0]
    matrix = np.asarray(spatial['affine']).reshape(4, 4)

    assert 'patch_size' not in spatial
    assert spatial['crop_size'] == [32, 64, 64]
    assert matrix[0, 1] == matrix[2, 0] == 0
    assert spatial['load_slice_start'] == [7, 11, 13]
    assert spatial['load_slice_stop'][0] == 39
    assert stats['renamed_patch_size_samples'] == 1
    assert stats['canonicalized_affine_samples'] == 1
    assert stats['trimmed_load_samples'] == 1

    repeated, repeated_stats = canonicalize_sample_metadata(migrated)
    assert repeated is migrated
    assert not repeated_stats


def test_metadata_migration_preserves_genuine_oblique_affine_and_extent():
    sample = _sample(1e-12)
    original_affine = sample['params'][0]['affine']
    migrated, stats = canonicalize_sample_metadata(sample)

    assert migrated['params'][0]['affine'] == original_affine
    assert migrated['params'][0]['load_slice_stop'] == [40, 99, 101]
    assert stats == {'renamed_patch_size_samples': 1}


def test_metadata_migration_does_not_trim_exact_diagonal_sample():
    sample = _sample(0.0)
    affine = np.eye(4)
    sample['params'][0]['affine'] = affine.ravel().tolist()
    sample['params'][0]['load_slice_stop'] = [39, 75, 77]

    migrated, stats = canonicalize_sample_metadata(sample)

    assert migrated['params'][0]['load_slice_stop'] == [39, 75, 77]
    assert stats == {'renamed_patch_size_samples': 1}
