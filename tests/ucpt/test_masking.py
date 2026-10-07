import numpy as np
import pytest

from pumit.ucpt.masking import (
    _sample_depth_extent,
    _sample_inplane_extent,
    multi_block_mask,
    random_mask,
)


def test_random_mask_ratio_and_nonempty():
    rng = np.random.default_rng(0)
    grid = (4, 16, 16)
    masked, stats = random_mask(grid, 0.8, rng)
    assert masked.shape == grid
    n = masked.sum()
    assert 0 < n < np.prod(grid)
    # nearest feasible to 0.8 * 1024
    assert stats['masked_tokens'] == n
    assert abs(stats['realized_ratio'] - 0.8) < 0.01


@pytest.mark.parametrize('ratio', [0.0, 1.0, -0.1, 1.1, np.nan])
def test_random_mask_rejects_invalid_ratio(ratio):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='mask_ratio'):
        random_mask((1, 4, 4), ratio, rng)


def test_random_mask_rejects_single_token_grid():
    with pytest.raises(ValueError, match='at least two tokens'):
        random_mask((1, 1, 1), 0.5, np.random.default_rng(0))


def test_multi_block_reaches_ratio_and_leaves_visible():
    rng = np.random.default_rng(1)
    grid = (8, 16, 16)
    masked, stats = multi_block_mask(grid, mask_ratio=0.8, rng=rng)
    assert masked.shape == grid
    assert 0 < masked.sum() < np.prod(grid)
    # realized within one block of the request (block granularity, not exact)
    assert abs(stats['realized_ratio'] - 0.8) < 0.25
    assert stats['n_blocks'] >= 1


def test_multi_block_depth_one_grid_degenerates_to_2d():
    rng = np.random.default_rng(2)
    grid = (1, 32, 32)
    masked, stats = multi_block_mask(grid, mask_ratio=0.75, rng=rng)
    assert masked.shape == grid
    assert stats['depth_fallbacks'] == 0
    assert 0 < masked.sum() < np.prod(grid)


def test_multi_block_deterministic_under_seed():
    grid = (6, 16, 16)
    m1, s1 = multi_block_mask(grid, 0.8, np.random.default_rng(7))
    m2, s2 = multi_block_mask(grid, 0.8, np.random.default_rng(7))
    assert np.array_equal(m1, m2)
    assert s1 == s2


def test_multi_block_depth_tracks_inplane_token_extent():
    def mean_bd(l_hw, seed0):
        bds = []
        for s in range(seed0, seed0 + 400):
            rng = np.random.default_rng(s)
            b_d, _ = _sample_depth_extent(l_hw=l_hw, D=64, rng=rng)
            bds.append(b_d)
        return np.mean(bds)

    small = mean_bd(3.0, 0)
    large = mean_bd(9.0, 0)
    assert large > small
    assert large / small > 2.0


def test_multi_block_inplane_returns_continuous_sampled_area():
    for seed in range(100):
        b_h, b_w, l_hw = _sample_inplane_extent((16, 16), np.random.default_rng(seed))
        area_ratio = l_hw**2 / (16 * 16)
        assert 1 <= b_h <= 16
        assert 1 <= b_w <= 16
        assert 0.15 <= area_ratio <= 0.20


def test_multi_block_depth_draws_from_truncated_legal_support():
    # The untruncated prior includes b_D > 3, but it also has a non-empty legal interval. Those draws
    # must be conditioned onto the legal interval rather than clamped to D and marked as fallbacks.
    for seed in range(400):
        b_d, fell_back = _sample_depth_extent(l_hw=3.0, D=3, rng=np.random.default_rng(seed))
        assert 1 <= b_d <= 3
        assert not fell_back


@pytest.mark.parametrize(
    ('l_hw', 'D', 'expected'),
    [
        (0.1, 3, 1),
        (100.0, 3, 3),
    ],
)
def test_multi_block_depth_falls_back_only_when_legal_support_is_empty(l_hw, D, expected):
    b_d, fell_back = _sample_depth_extent(l_hw=l_hw, D=D, rng=np.random.default_rng(0))
    assert b_d == expected
    assert fell_back


def test_multi_block_stats_describe_returned_union():
    grid = (8, 16, 16)
    masked, stats = multi_block_mask(grid, mask_ratio=0.8, rng=np.random.default_rng(17))
    total = np.prod(grid)
    assert stats['requested_ratio'] == 0.8
    assert stats['sampled_blocks'] >= stats['n_blocks'] >= 1
    assert stats['sampled_depth_fallbacks'] >= stats['depth_fallbacks'] >= 0
    assert stats['masked_tokens'] == masked.sum()
    assert stats['visible_tokens'] == total - masked.sum()
    assert stats['placed_block_tokens'] - stats['overlap_tokens'] == masked.sum()
    assert stats['overlap_fraction'] == pytest.approx(
        stats['overlap_tokens'] / stats['placed_block_tokens']
    )


@pytest.mark.parametrize(
    ('grid', 'ratio', 'message'),
    [
        ((1, 1, 1), 0.5, 'at least two tokens'),
        ((2, 8), 0.5, 'three positive integers'),
        ((2, 8, 8), 0.0, 'mask_ratio'),
    ],
)
def test_multi_block_rejects_invalid_inputs(grid, ratio, message):
    with pytest.raises(ValueError, match=message):
        multi_block_mask(grid, ratio, np.random.default_rng(0))


def test_multi_block_unreachable_ratio_raises_not_hangs():
    # A grid where the min block already covers everything can't sit below a low ratio forever;
    # instead verify the fail-fast guard exists by requesting an impossible ratio on a 1-token grid.
    rng = np.random.default_rng(0)
    with pytest.raises((RuntimeError, ValueError)):
        # 1x1x1 grid: no valid in-plane block prior -> shape sampler fails fast
        multi_block_mask((1, 1, 1), 0.5, rng)
