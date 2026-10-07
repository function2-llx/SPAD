import numpy as np
import torch
from torch.nn import functional as nnf

from pumit.types import tuple3_t

__all__ = [
    'resample',
]

def resample(
    x: torch.Tensor,
    shape: tuple[int, ...],
    *,
    downsample_mode: str = 'area',
    upsample_mode: str | None = None,
    scale: bool = False,
):
    """
    Perform spatial resampling to a target shape by sequentially applying downsampling and upsampling.
    Downsampling uses area mode by default, while upsampling uses bicubic for 2D and trilinear for 3D.

    Args:
        scale: whether to scale the values based on size, this can be useful for preserving order of magnitude through resampling
        upsample_mode: interpolation mode for upsampling, default: (2D, bicubic), (3D, trilinear)
    """
    scale_ratio = np.prod(x.shape[2:]) / np.prod(shape) if scale else 1.
    # without `.tolist()`, PyTorch will complain it is not int
    downsample_shape = tuple(np.minimum(x.shape[2:], shape).tolist())
    if downsample_shape != x.shape[2:]:
        x = nnf.interpolate(x, downsample_shape, mode=downsample_mode)
    if shape != x.shape[2:]:
        if upsample_mode is None:
            upsample_mode = 'trilinear' if x.ndim == 5 else 'bicubic'
        x = nnf.interpolate(x, shape, mode=upsample_mode)
    if scale:
        x *= scale_ratio
    return x
