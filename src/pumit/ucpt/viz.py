# src/pumit/ucpt/viz.py
"""UCPT batch-shard visualization: render a sample's augmentation + label on the fly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pumit.ucpt.mask import _apply_affine_to_mask, _load_mask


def _to_uint8(vol: np.ndarray) -> np.ndarray:
    """Scale a float volume to [0,255] uint8 via per-slice percentile clipping (display only)."""
    lo = float(np.percentile(vol, 1))
    hi = float(np.percentile(vol, 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (vol - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def render_sample(
    pipeline,
    sample: dict,
    *,
    data_root: Path | str = 'PUMIT-data/preprocess',
) -> dict:
    """Render one UCPT sample: augmented image + raw image + both mask variants + meta.

    Args:
        pipeline: A ``build_ucpt_pipeline()`` transform pipeline.
        sample: a UCPT sample dict (from generate_sample / a batch shard).
        data_root: root for mask loading (default 'PUMIT-data/preprocess').

    Returns {'image', 'raw', 'masks', 'raw_masks', 'meta'}.
    """
    data_root = Path(data_root)
    img_path = sample['img']

    # --- shape_3d recovery (UCPT samples don't carry 'shape') ---
    raw_full = np.load(img_path, mmap_mode='r')   # (C, D_raw, H_raw, W_raw)
    shape_3d = tuple(int(x) for x in raw_full.shape[1:])

    # --- augmented image ---
    with torch.inference_mode():
        aug = pipeline.replay({'img': img_path}, sample['params'])['img']  # (3, D, H, W) [-1,1]
        aug = aug.detach().cpu().numpy()
    aug_gray = aug.mean(0)                                   # (D, H, W)
    image = ((aug_gray + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    # --- raw image (full FOV, no pipeline) ---
    raw_gray = np.asarray(raw_full[0])                       # (D_raw, H_raw, W_raw)
    raw = _to_uint8(raw_gray.astype(np.float32))

    # --- masks ---
    masks: dict[str, np.ndarray] = {}
    raw_masks: dict[str, np.ndarray] = {}
    positive_classes = [
        (item['source'], item['name'])
        for item in sample.get('classes', [])
        if item['is_positive']
    ]
    for source, name in positive_classes:
        display_name = f'{source}:{name}'
        raw_m = _load_mask(data_root, sample['dataset'], sample['key'], source, name, shape_3d)
        if raw_m is None:
            raise FileNotFoundError(
                f'missing positive mask: dataset={sample["dataset"]!r} key={sample["key"]!r} '
                f'source={source!r} class={name!r}'
            )
        raw_masks[display_name] = raw_m
        aug_m = _apply_affine_to_mask(raw_m, sample['params'])  # (1, D, H, W)
        masks[display_name] = aug_m.squeeze(0).numpy().astype(bool)

    meta = dict(sample)

    return {
        'image': image,
        'raw': raw,
        'masks': masks,
        'raw_masks': raw_masks,
        'meta': meta,
    }


def phase_indices(D: int, *, grid_h: int, grid_w: int) -> list[list[int]]:
    """Phase-sweep slice indices: one slice per part per phase, clamped at the index.

    Returns a list of phases; each phase is a list of n_cells slice indices.
    n_cells = min(grid_h*grid_w, D). No slices dropped; shorter parts repeat
    their last slice at later phases. Slider ranges 0..max_part_len-1.
    """
    n_cells = min(grid_h * grid_w, D)
    parts = np.array_split(np.arange(D), n_cells)
    max_len = max(len(p) for p in parts)
    return [
        [int(parts[i][min(p, len(parts[i]) - 1)]) for i in range(n_cells)]
        for p in range(max_len)
    ]
