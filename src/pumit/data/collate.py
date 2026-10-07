"""Collation for DA-bucketed batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DABatch:
    img: torch.Tensor
    not_rgb: list[bool]
    da_enc: int | None
    da_dec: int | None
    t: float                # frac(log2(ratio))
    spacing: tuple
    paths: list[str]
    original_spacing: torch.Tensor | None = None  # (B, 3), physical voxel spacing in mm


def da_collate_fn(batch: list[tuple]) -> DABatch:
    """Collate DA-bucketed batch: all samples must have the same shape (stack)."""
    items, trans_infos = zip(*batch)
    imgs, not_rgbs, da_encs, spacings, paths, original_spacings = zip(*items)
    ti = trans_infos[0]  # uniform within bucket
    da_enc = ti['da_enc']
    da_dec = ti['da_dec']
    t = ti['t']
    da_enc_val = da_encs[0]
    assert all(im.shape == imgs[0].shape for im in imgs), \
        f"Shape mismatch in batch: {[im.shape for im in imgs]}"
    img = torch.stack(imgs)
    if da_enc_val is None:
        spacing_z = float('inf')
    else:
        spacing_z = 2.0 ** da_enc_val
    return DABatch(
        img=img,
        not_rgb=list(not_rgbs),
        da_enc=da_enc,
        da_dec=da_dec,
        t=t,
        spacing=(spacing_z, 1.0, 1.0),
        paths=list(paths),
        original_spacing=torch.stack(original_spacings),
    )
