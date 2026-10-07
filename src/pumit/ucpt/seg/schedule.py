"""SPAD depth schedule for the segmentation neck and pixel decoder.

Each consumer resolves the schedule from the raw sample DA. Depth follows in-plane upsampling while features are
isotropic, capped by ``min(2, 4 - da)`` stages.
"""

from dataclasses import dataclass

NUM_NECK_UPSAMPLE_STAGES = 2


@dataclass(frozen=True)
class StageDA:
    """Resolved depth behavior for one upsampling stage.

    Attributes:
        da: Local SPAD depth-adaptation level.
        upsample_depth: Whether the stage doubles depth.
    """

    da: int | None
    upsample_depth: bool


def neck_da_schedule(da: int | None, num_stages: int = NUM_NECK_UPSAMPLE_STAGES) -> list[StageDA]:
    """Resolve the neck's coarse-to-fine depth schedule.

    Args:
        da: Input depth-adaptation level, or ``None`` for 2D.
        num_stages: Number of neck upsampling stages.

    Returns:
        Per-stage adaptation levels and depth-upsampling decisions.
    """
    if da is None:
        return [StageDA(da=None, upsample_depth=False) for _ in range(num_stages)]
    nds = max(0, min(num_stages, 4 - da))  # depth-upsample budget
    out: list[StageDA] = []
    cur = da
    for _ in range(num_stages):
        if nds > 0:
            out.append(StageDA(da=cur, upsample_depth=True))
            nds -= 1
        else:
            # In-plane-only upsampling raises local anisotropy by one octave.
            # Values above four are safe because SPAD convolutions collapse all da >= 2.
            cur = cur + 1
            out.append(StageDA(da=cur, upsample_depth=False))
    return out
