"""Multi-block masking for UCPT self-supervised views.

Generates a spatially structured target mask over a ViT patch-token grid as a Boolean union of I-JEPA-style blocks.
The in-plane shape follows the I-JEPA area/aspect prior, and the depth extent is sampled relative to the geometric-mean in-plane extent.

The sampler is pure: it takes the token grid, a requested mask ratio, and a seeded numpy Generator, and returns the masked-token Boolean grid.
"""
from __future__ import annotations

import numpy as np

# I-JEPA target-block prior.
_AREA_RANGE = (0.15, 0.20)          # fraction of the in-plane token grid
_ASPECT_RANGE = (0.75, 1.5)         # r_HW = L_H / L_W
_DEPTH_GAMMA = 1.5                  # r_D log-uniform on [1/gamma, gamma]
_MAX_SHAPE_RESAMPLES = 1000         # fail-fast guard on in-plane shape rejection
_MAX_BLOCKS = 4096                  # fail-fast guard on the union-until-crossing loop


def _validate_grid_and_ratio(grid: tuple[int, int, int], mask_ratio: float) -> tuple[int, int, int]:
    if len(grid) != 3 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in grid):
        raise ValueError(f'grid must contain three positive integers, got {grid!r}')
    if np.prod(grid) < 2:
        raise ValueError(f'grid must contain at least two tokens, got {grid!r}')
    if not np.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
        raise ValueError(f'mask_ratio must satisfy 0 < mask_ratio < 1, got {mask_ratio!r}')
    return int(grid[0]), int(grid[1]), int(grid[2])


def _sample_inplane_extent(
    grid_hw: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    """Sample one block's ``(b_H, b_W)`` extent from the I-JEPA area/aspect prior in token space.

    Invalid shapes are resampled rather than clipped. The continuous characteristic length
    ``L_HW = sqrt(area)`` is returned for depth sampling before token rounding.
    """
    H, W = grid_hw
    for _ in range(_MAX_SHAPE_RESAMPLES):
        a = rng.uniform(*_AREA_RANGE)
        r_hw = rng.uniform(*_ASPECT_RANGE)
        area = a * H * W
        l_h = np.sqrt(area * r_hw)
        l_w = np.sqrt(area / r_hw)
        b_h = round(l_h)
        b_w = round(l_w)
        if 1 <= b_h <= H and 1 <= b_w <= W:
            return b_h, b_w, float(np.sqrt(area))
    raise RuntimeError(
        f'in-plane block sampling failed to find a valid shape in {_MAX_SHAPE_RESAMPLES} tries '
        f'(grid_hw={grid_hw}); the grid may be too small for the area prior'
    )


def _sample_depth_extent(
    l_hw: float,
    D: int,
    rng: np.random.Generator,
) -> tuple[int, bool]:
    """Sample one block's depth extent from the token-space depth-aspect prior.

    The characteristic in-plane length is the geometric mean ``L_HW = sqrt(A_HW)``. A multiplicatively
    symmetric depth aspect sets ``b_D = round(r_D L_HW)``.

    A depth-one grid degenerates directly to the 2D sampler. If the base ``r_D`` support contains no legal
    extent, the nearest feasible boundary extent is returned and marked as a fallback.
    """
    if D == 1:
        return 1, False
    if not np.isfinite(l_hw) or l_hw <= 0:
        raise ValueError(f'l_hw must be finite and positive, got {l_hw!r}')

    scale = l_hw
    base_low, base_high = 1 / _DEPTH_GAMMA, _DEPTH_GAMMA
    # round(r_D * scale) is legal for values strictly between 0.5 and D + 0.5. Intersect this
    # interval with the base support before drawing so boundary extents do not acquire clamp mass.
    valid_low = np.nextafter(0.5 / scale, np.inf)
    valid_high = np.nextafter((D + 0.5) / scale, -np.inf)
    low = max(base_low, valid_low)
    high = min(base_high, valid_high)
    if low < high:
        r_d = np.exp(rng.uniform(np.log(low), np.log(high)))
        b_d = round(r_d * scale)
        if not 1 <= b_d <= D:
            raise RuntimeError(
                f'truncated depth draw produced invalid extent b_D={b_d} '
                f'(D={D}, L_HW={l_hw}, support=({low}, {high}))'
            )
        return b_d, False

    # The complete base support lies outside the feasible interval, so the nearest boundary extent is
    # the only meaningful fallback.
    if base_high * scale <= 0.5:
        return 1, True
    if base_low * scale >= D + 0.5:
        return D, True
    raise RuntimeError(
        f'failed to resolve depth support '
        f'(D={D}, L_HW={l_hw}, base_support=({base_low}, {base_high}))'
    )


def _place_block(
    grid: tuple[int, int, int],
    extent: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[slice, slice, slice]:
    """Uniformly sample an in-bounds origin for a block of the given extent; return its slice per axis."""
    slices = []
    for n_i, b_i in zip(grid, extent):
        o_i = int(rng.integers(0, n_i - b_i + 1))
        slices.append(slice(o_i, o_i + b_i))
    return slices[0], slices[1], slices[2]


def multi_block_mask(
    grid: tuple[int, int, int],
    mask_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Build a multi-block target mask over a ``(D, H, W)`` token grid.

    Independently sampled blocks are merged into a Boolean union until coverage first crosses ``mask_ratio``;
    the returned union is whichever of the pre-crossing / post-crossing unions is closer to the requested
    ratio (post-crossing wins an exact tie, favoring the harder view). The block count is an outcome, never
    fixed. The crossing block is kept or dropped whole; no independent tokens are added to hit the ratio,
    so the realized union ratio may fall slightly outside the requested value.

    Args:
        grid: Token grid ``(D, H, W)``.
        mask_ratio: Requested fraction of tokens to mask.
        rng: Seeded generator; all randomness is drawn from it for replay determinism.

    Returns:
        ``(masked, stats)`` where ``masked`` is a ``(D, H, W)`` bool array (True = masked target) and ``stats``
        holds audit fields (block count, realized ratio, overshoot side, depth-fallback count, overlap).
    """
    D, H, W = _validate_grid_and_ratio(grid, mask_ratio)

    total = D * H * W
    masked = np.zeros(grid, dtype=bool)
    sampled_blocks = 0
    sampled_depth_fallbacks = 0
    sampled_block_tokens = 0
    sampled_overlap_tokens = 0
    covered_before = 0
    prev_masked = masked.copy()
    last_fallback = False
    last_block_tokens = 0
    last_overlap_tokens = 0

    for _ in range(_MAX_BLOCKS):
        prev_masked = masked.copy()
        covered_before = int(masked.sum())
        b_h, b_w, l_hw = _sample_inplane_extent((H, W), rng)
        b_d, fell_back = _sample_depth_extent(l_hw, D, rng)
        sd, sh, sw = _place_block(grid, (b_d, b_h, b_w), rng)
        masked[sd, sh, sw] = True
        covered_after = int(masked.sum())
        block_tokens = b_d * b_h * b_w
        overlap_tokens = block_tokens - (covered_after - covered_before)
        sampled_blocks += 1
        sampled_depth_fallbacks += int(fell_back)
        sampled_block_tokens += block_tokens
        sampled_overlap_tokens += overlap_tokens
        last_fallback = fell_back
        last_block_tokens = block_tokens
        last_overlap_tokens = overlap_tokens
        if covered_after / total >= mask_ratio:
            break
    else:
        raise RuntimeError(
            f'multi-block union did not reach mask_ratio={mask_ratio} within {_MAX_BLOCKS} blocks '
            f'(grid={grid}); mask_ratio may be unreachable for this grid'
        )

    # Keep whichever complete union is closer to the requested ratio; post-crossing wins an exact tie.
    r_before = covered_before / total
    r_after = int(masked.sum()) / total
    before_valid = covered_before > 0
    after_valid = int(masked.sum()) < total
    if not before_valid and not after_valid:
        raise RuntimeError(f'block prior cannot leave both visible and masked tokens for grid={grid!r}')
    choose_before = before_valid and (
        not after_valid or abs(r_before - mask_ratio) < abs(r_after - mask_ratio)
    )
    if choose_before:
        masked = prev_masked
        kept_blocks = sampled_blocks - 1
        depth_fallbacks = sampled_depth_fallbacks - int(last_fallback)
        placed_block_tokens = sampled_block_tokens - last_block_tokens
        overlap_tokens = sampled_overlap_tokens - last_overlap_tokens
        realized = r_before
        overshoot_side = 'before'
    else:
        kept_blocks = sampled_blocks
        depth_fallbacks = sampled_depth_fallbacks
        placed_block_tokens = sampled_block_tokens
        overlap_tokens = sampled_overlap_tokens
        realized = r_after
        overshoot_side = 'after'

    stats = {
        'requested_ratio': float(mask_ratio),
        'n_blocks': kept_blocks,
        'sampled_blocks': sampled_blocks,
        'realized_ratio': realized,
        'overshoot_side': overshoot_side,
        'depth_fallbacks': depth_fallbacks,
        'sampled_depth_fallbacks': sampled_depth_fallbacks,
        'placed_block_tokens': placed_block_tokens,
        'overlap_tokens': overlap_tokens,
        'overlap_fraction': overlap_tokens / placed_block_tokens if placed_block_tokens else 0.0,
        'masked_tokens': int(masked.sum()),
        'visible_tokens': total - int(masked.sum()),
    }
    return masked, stats


def random_mask(
    grid: tuple[int, int, int],
    mask_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Uniform random token masking over the (D, H, W) grid (the complementary baseline view).

    The realized integer masked count is the nearest feasible value to ``mask_ratio * N_p``, clamped so both
    the visible and masked sets stay non-empty.
    """
    D, H, W = _validate_grid_and_ratio(grid, mask_ratio)
    total = D * H * W
    n_masked = round(mask_ratio * total)
    n_masked = int(np.clip(n_masked, 1, total - 1))
    flat_idx = rng.permutation(total)[:n_masked]
    masked = np.zeros(total, dtype=bool)
    masked[flat_idx] = True
    masked = masked.reshape(grid)
    stats = {
        'requested_ratio': float(mask_ratio),
        'realized_ratio': n_masked / total,
        'masked_tokens': n_masked,
        'visible_tokens': total - n_masked,
    }
    return masked, stats
