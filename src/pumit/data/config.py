"""Shared data configuration: depth tiers and spatial constraints."""

from dataclasses import dataclass

MAX_INPLANE_RATIO = 1.1


@dataclass
class DepthTierConfig:
    tiers: tuple[int, ...]
    batch_sizes: tuple[int, ...]

    def __post_init__(self):
        if len(self.tiers) != len(self.batch_sizes):
            raise ValueError(
                f"tiers and batch_sizes must have same length, "
                f"got {len(self.tiers)} and {len(self.batch_sizes)}"
            )


def validate_depth_tiers(depth_tiers: dict[int | None, DepthTierConfig], *, max_da: int) -> None:
    if None not in depth_tiers:
        raise ValueError("depth_tiers must contain key None for 2D samples")
    if max_da not in depth_tiers:
        raise ValueError(f"depth_tiers must contain catch-all key {max_da} for DA >= {max_da}")
    for da_key in range(max_da):
        if da_key not in depth_tiers:
            raise ValueError(f"depth_tiers missing key {da_key}")

    for da_key, tier_cfg in depth_tiers.items():
        if list(tier_cfg.tiers) != sorted(tier_cfg.tiers):
            raise ValueError(
                f"tiers at DA={da_key} must be sorted ascending, got {tier_cfg.tiers}"
            )
        if da_key is None:
            continue
        for d in tier_cfg.tiers:
            da_min = max(da_key - 1, 0)
            da_max = min(da_key + 1, max_da)
            for da in range(da_min, da_max + 1):
                nds = 3 - min(da, 3)
                divisor = 1 << nds
                if d % divisor != 0:
                    raise ValueError(
                        f"Tier depth {d} at DA={da_key} is not divisible by {divisor} "
                        f"(required for cross-DA with da={da}, nds={nds})"
                    )
