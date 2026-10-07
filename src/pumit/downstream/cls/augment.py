"""Train-time augmentation and test-time flip views for MedMNIST3D classification finetuning.

Only random flips: the volumes are cubic 64^3, so an axis flip is an exact voxel permutation --
no interpolation, no resampling loss, and no change to the input shape (which CUDA graphs under
compile reduce-overhead require). Each spatial axis flips independently with p=0.5, giving 8
equally likely orientations.

Flips are applied to the raw uint8 volumes before the per-backbone `transform_batch`, so every
backbone sees the identical augmentation regardless of its own normalization and resize. The same
8 orientations serve as the test-time augmentation views (`FLIP_VIEWS`), and both are gated on
`flips_enabled_for`.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

# organmnist3d labels three organs by side (kidney-right/left, femur-right/left, lung-right/left),
# i.e. 6 of its 11 classes. These are single-organ crops, so a mirrored right kidney is
# indistinguishable from a left one while keeping the 'right' label -- label-contradicting noise.
# The other five datasets label by appearance (malignancy, hyperplasia, aneurysm, synapse type,
# fracture type), all of which are mirror-invariant.
FLIP_EXCLUDED_FLAGS = frozenset({'organmnist3d'})

# The 8 flip orientations as axis tuples into an (N, D, H, W) batch; () is the identity view.
FLIP_VIEWS: tuple[tuple[int, ...], ...] = tuple(
    tuple(axis for axis, on in zip((1, 2, 3), bits) if on)
    for bits in itertools.product((False, True), repeat=3)
)


def flips_enabled_for(flag: str) -> bool:
    """Whether random flips preserve label semantics for this dataset."""
    return flag not in FLIP_EXCLUDED_FLAGS


def flip_view(images: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """One TTA view of a raw (N, D, H, W) batch; `axes` is a member of FLIP_VIEWS."""
    if not axes:
        return images
    return np.ascontiguousarray(np.flip(images, axis=axes))



def random_flip_3d(images: np.ndarray, *, generator: torch.Generator) -> np.ndarray:
    """Flip each spatial axis of each sample independently with p=0.5.

    Args:
        images: raw batch, shape (N, D, H, W).
        generator: CPU generator, so augmentation is reproducible from the run seed and
            independent of the sampler's own stream.

    Returns:
        A new array of the same shape and dtype.
    """
    if images.ndim != 4:
        raise ValueError(f'expected (N, D, H, W) volumes, got shape {images.shape}')
    decisions = torch.rand(images.shape[0], 3, generator=generator) < 0.5
    out = np.empty_like(images)
    for i, flags in enumerate(decisions.tolist()):
        axes = tuple(1 + axis for axis, do_flip in enumerate(flags) if do_flip)
        out[i] = np.flip(images[i], axis=tuple(a - 1 for a in axes)) if axes else images[i]
    return out
