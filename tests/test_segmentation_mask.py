import numpy as np
import pytest

from pumit.segmentation_mask import pack_binary_mask, unpack_binary_mask, unpack_binary_mask_crop


def test_row_aligned_mask_round_trip():
    rng = np.random.default_rng(0)
    mask = rng.random((3, 5, 19)) > 0.7

    packed = pack_binary_mask(mask)

    assert packed.shape == (3, 5, 3)
    assert np.array_equal(unpack_binary_mask(packed, mask.shape), mask)


def test_crop_unpack_matches_full_unpack_then_crop():
    rng = np.random.default_rng(1)
    mask = rng.random((7, 11, 29)) > 0.7
    packed = pack_binary_mask(mask)
    crop_slices = (slice(2, 6), slice(3, 10), slice(5, 24))

    crop = unpack_binary_mask_crop(packed, mask.shape, crop_slices)
    full = unpack_binary_mask(packed, mask.shape)

    assert crop.shape == (4, 7, 19)
    assert np.array_equal(crop, full[crop_slices])


def test_flat_packed_mask_is_rejected():
    mask = np.zeros((2, 3, 9), dtype=bool)
    legacy = np.packbits(mask.ravel())

    with pytest.raises(ValueError, match='regenerate segmentation artifacts'):
        unpack_binary_mask(legacy, mask.shape)
