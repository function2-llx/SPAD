"""UCPT batch contract shared by the data pipeline and model.

Coordinates, masks, and packing indices are precomputed on CPU. Python metadata stays outside compiled graphs through
``field(metadata={'device': 'cpu'})``.

``patches`` stores every sample once. SSL tensors use unlabeled-local indexing, while student, teacher, and segmentation
gather indices map back to the full patch array.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

_SEG_PATCH_SIZE = 16


def seg_output_grid(da: int | None, patch_grid: tuple[int, int, int]) -> tuple[int, int, int]:
    """Compute the segmentation decoder's output grid.

    In-plane dimensions are upsampled by four. Depth follows the closed form of ``neck_da_schedule`` and never exceeds
    native input depth.

    Args:
        da: Sample SPAD depth-adaptation level, or ``None`` for 2D.
        patch_grid: ViT patch grid ``(D_p, H_p, W_p)``.

    Returns:
        Decoder output grid ``(D, H, W)`` before logits are interpolated to the full-resolution target.
    """
    return seg_supervision_grid(da, patch_grid, stride=4)


def seg_supervision_grid(
    da: int | None,
    patch_grid: tuple[int, int, int],
    *,
    stride: int,
) -> tuple[int, int, int]:
    """Compute a SPAD-aware segmentation supervision grid.

    Args:
        da: Sample SPAD depth-adaptation level, or ``None`` for 2D.
        patch_grid: ViT patch grid ``(D_p, H_p, W_p)``.
        stride: In-plane supervision stride in input voxels. Must be 1, 2, or 4.

    Returns:
        Supervision grid ``(D, H, W)`` covering the input crop's full FOV.
    """
    if stride not in (1, 2, 4):
        raise ValueError(f'segmentation supervision stride must be 1, 2, or 4, got {stride}')
    assert da is None or da >= 0, f'da must be None or >= 0, got {da!r}'
    D_p, H_p, W_p = patch_grid
    n_inplane_up = (_SEG_PATCH_SIZE // stride).bit_length() - 1
    scale_xy = 1 << n_inplane_up
    if da is None:
        n_depth_up = 0
    else:
        n_depth_up = max(0, min(n_inplane_up, 4 - da))
    D_out = D_p * (2 ** n_depth_up)
    return (D_out, H_p * scale_xy, W_p * scale_xy)


def _move_field(val: Any, device, non_blocking: bool) -> Any:
    """Move a field value to device. Handles Tensor, list[Tensor], and .to()-able."""
    if isinstance(val, torch.Tensor):
        return val.to(device, non_blocking=non_blocking)
    if isinstance(val, list):
        return [t.to(device, non_blocking=non_blocking) if isinstance(t, torch.Tensor) else t for t in val]
    if hasattr(val, 'to'):
        return val.to(device)
    return val


def _pin_field(val: Any) -> Any:
    """Pin tensors in a field."""
    if isinstance(val, torch.Tensor):
        return val.pin_memory()
    if isinstance(val, list):
        return [t.pin_memory() if isinstance(t, torch.Tensor) else t for t in val]
    return val


@dataclass(slots=True)
class UCPTBatch:
    # --- SSL half (verbatim SSLBatch fields) ---
    patches: Tensor
    latents: Tensor
    student_attn_bias: BlockDiagonalMask
    teacher_attn_bias: BlockDiagonalMask
    # One decoder packing shared by BOTH the recon and patch-distill decoders: each unlabeled sample
    # contributes V masked-view blocks (one per view), each of full grid length n_p, in (sample, view) order.
    view_decoder_attn_bias: BlockDiagonalMask
    student_coords: Tensor
    teacher_coords: Tensor
    view_decoder_coords: Tensor
    student_patch_mask: Tensor
    student_patch_gather_idx: Tensor
    teacher_patch_mask: Tensor
    # Decoder-space indices shared by both decoders. Masked positions align with view_target_gather_idx.
    view_visible_idx: Tensor
    view_masked_idx: Tensor
    # Maps each view's masked tokens (in decoder (sample, view)-major order) to unlabeled-local patch index.
    # Serves as the gather for BOTH the recon target (latents[...]) and the distill target
    # (teacher_patch_feats[...]); a token masked in multiple views is a target once per view.
    view_target_gather_idx: Tensor
    total_student_len: int = field(metadata={'device': 'cpu'})
    total_teacher_len: int = field(metadata={'device': 'cpu'})
    num_blocks: int = field(metadata={'device': 'cpu'})
    n_views: int = field(metadata={'device': 'cpu'}, default=2)

    # --- Seg extension ---
    # SSL-half field: gather index for unlabeled-only teacher forward.
    # Kept in the defaulted block for dataclass ordering, but moved to GPU for indexing.
    teacher_patch_gather_idx: Tensor | None = None
    # (num_samples,) bool: which samples came from the labeled stream and must be excluded from SSL.
    sample_is_labeled: Tensor | None = None
    seg_patch_gather_idx: Tensor | None = None
    seg_patch_mask: Tensor | None = None
    seg_coords: Tensor | None = None
    seg_attn_bias: BlockDiagonalMask | None = None
    seg_sample_shapes: list[tuple[int, int, int]] = field(metadata={'device': 'cpu'}, default_factory=list)
    seg_das: list[int | None] = field(metadata={'device': 'cpu'}, default_factory=list)
    text_embeddings: list[Tensor] = field(default_factory=list)
    text_valid_masks: list[Tensor] = field(default_factory=list)
    is_positive: list[Tensor] = field(default_factory=list)
    # Large dense targets stay on pinned CPU memory until their per-sample loss is evaluated.
    target_masks: list[Tensor] = field(metadata={'device': 'lazy'}, default_factory=list)
    total_seg_len: int = field(metadata={'device': 'cpu'}, default=0)
    # Number of UNLABELED SSL patches (student+teacher operate on these).
    # SSL masks, coords, and latents use this local length; gather indices point into the full patch array.
    n_ssl_patches: int = field(metadata={'device': 'cpu'}, default=0)

    def to(self, device, non_blocking=False):
        for f in dataclasses.fields(self):
            if f.metadata.get('device') in ('cpu', 'lazy'):
                continue
            setattr(self, f.name, _move_field(getattr(self, f.name), device, non_blocking))
        return self

    def pin_memory(self):
        for f in dataclasses.fields(self):
            if f.metadata.get('device') == 'cpu':
                continue
            setattr(self, f.name, _pin_field(getattr(self, f.name)))
        return self
