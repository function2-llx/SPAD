from typing import TypedDict

from pumit.types import tuple3_t

__all__ = [
    'TransInfo',
]

class TransInfo(TypedDict):
    da_enc: int | None
    da_dec: int | None
    t: float                # frac(log2(ratio)), in [0, 1)
    scale: tuple3_t[float]
    patch_size: tuple3_t[int]
