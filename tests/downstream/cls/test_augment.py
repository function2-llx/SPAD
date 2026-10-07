"""Train-time flip augmentation: independent p=0.5 per spatial axis.

Flips are exact on the cubic 64^3 MedMNIST3D volumes (no interpolation, no resampling loss).
Applied to raw volumes before the per-backbone transform_batch, so every backbone sees the same
augmentation regardless of its own normalization/resize.

organmnist3d is excluded: its label set has three explicit laterality pairs (kidney-right/left,
femur-right/left, lung-right/left), so a mirrored right-kidney would be trained as 'right'.
The other five label by appearance (malignancy, hyperplasia, aneurysm, synapse type, fracture
type), which is mirror-invariant.
"""
import numpy as np
import pytest
import torch

from pumit.downstream.cls.augment import FLIP_EXCLUDED_FLAGS, random_flip_3d, flips_enabled_for


def test_flip_is_exact_and_shape_preserving():
    rng = torch.Generator().manual_seed(0)
    x = np.arange(2 * 4 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4, 4)
    out = random_flip_3d(x, generator=rng)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # every output voxel came from the input unchanged (a flip is a permutation of voxels)
    for i in range(2):
        assert sorted(out[i].ravel().tolist()) == sorted(x[i].ravel().tolist())


def test_each_axis_flips_independently_at_p_half():
    """Over many draws each of the 8 orientations appears, at roughly equal frequency."""
    rng = torch.Generator().manual_seed(0)
    x = np.arange(8, dtype=np.uint8).reshape(1, 2, 2, 2)
    seen = {}
    for _ in range(400):
        out = random_flip_3d(x, generator=rng)
        seen[out.tobytes()] = seen.get(out.tobytes(), 0) + 1
    assert len(seen) == 8, f'expected all 8 flip combinations, saw {len(seen)}'
    counts = sorted(seen.values())
    assert counts[0] > 400 / 8 * 0.4, f'a combination was far too rare: {counts}'


def test_samples_in_a_batch_flip_independently():
    """Per-sample randomness: a batch must not receive one shared flip."""
    rng = torch.Generator().manual_seed(3)
    x = np.tile(np.arange(64, dtype=np.uint8).reshape(1, 4, 4, 4), (64, 1, 1, 1))
    out = random_flip_3d(x, generator=rng)
    distinct = {out[i].tobytes() for i in range(64)}
    assert len(distinct) > 1, 'all samples got the identical flip'


def test_is_deterministic_for_a_seed():
    x = np.arange(2 * 4 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4, 4)
    a = random_flip_3d(x, generator=torch.Generator().manual_seed(7))
    b = random_flip_3d(x, generator=torch.Generator().manual_seed(7))
    c = random_flip_3d(x, generator=torch.Generator().manual_seed(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_organ_is_excluded_by_laterality():
    """organmnist3d labels kidney/femur/lung by side, so mirroring contradicts the label."""
    assert 'organmnist3d' in FLIP_EXCLUDED_FLAGS
    assert not flips_enabled_for('organmnist3d')
    for flag in ('nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d',
                 'vesselmnist3d', 'synapsemnist3d'):
        assert flips_enabled_for(flag), f'{flag} labels by appearance and is mirror-invariant'
