# src/pumit/ucpt/sample_stream.py
"""UCPT sample stream generator: SSL generation + per-sample labeled/unlabeled fork.

Pure library code (no CLI). Consumed in-memory by the downstream packing stage.
Copies — does not import — from archive pumit.ssl / pumit.seg.
"""
from __future__ import annotations

import math
from functools import cache

import numpy as np
import yaml

from pumit.codec.config import MAX_DA
from pumit.data import DATA_ROOT, build_training_data
from pumit.transforms.pipeline import TransformPipeline
from pumit.ucpt.input import InputNormalizer
from pumit.ucpt.mask_store import MASK_FRAME_KEY
from pumit.ucpt.seg.label_contract import (
    LabelContract,
    labeled_eligible,
    normalize_label_contract,
)
from pumit.ucpt.seg.data import CropClassSampler
from pumit.ucpt.transforms import build_ucpt_pipeline


# One dataset record (a row of build_training_data's train DataFrame, via
# .reset_index().to_dict('records')). Keyed fields used here: 'img' (str path),
# 'weight' (float), 'label' (bool), 'label_classes' (dict), 'key'/'dataset'/
# 'modality' (str), 'shape'/'spacing' (np.ndarray). Labeled stream samples retain only final raw-class queries.
type Record = dict

# A sampling pool: the records to draw from and the per-record sampling weight
# (normalized to sum 1).
type Pool = tuple[list[Record], np.ndarray]

# The two pools UCPT draws from: (full_pool, labeled_pool). The labeled pool is
# a subset of the full pool (records passing _labeled_eligible), with its
# weights re-normalized among themselves.
type Pools = tuple[Pool, Pool]


def _labeled_eligible(record: dict) -> bool:
    """Check whether a record can enter the labeled sampling pool.

    Legacy bare-list ``label_classes`` records remain eligible for unlabeled SSL but cannot enter the segmentation
    stream.

    Args:
        record: Dataset record to classify.

    Returns:
        Whether the record is labeled and has the expected per-source class mapping.
    """
    return labeled_eligible(record)


def build_pools(records: list[Record]) -> Pools:
    """Build full and labeled sampling pools.

    Each pool normalizes its own record weights. Records rejected by ``_labeled_eligible`` remain available to the
    full pool.

    Args:
        records: Dataset records used for sampling.

    Returns:
        ``(full_pool, labeled_pool)``, where each pool contains records and normalized float64 weights.
    """
    weights_all = np.array([r['weight'] for r in records], dtype=np.float64)
    weights_all = weights_all / weights_all.sum()

    n_labeled = sum(1 for r in records if r.get('label') == True)  # noqa: E712
    records_lab = [r for r in records if _labeled_eligible(r)]
    n_dropped = n_labeled - len(records_lab)
    if n_dropped:
        print(
            f'[ucpt] excluded {n_dropped}/{n_labeled} labeled records with '
            f'non-dict label_classes from the labeled pool (still used as '
            f'unlabeled SSL samples)'
        )
    weights_lab = np.array([r['weight'] for r in records_lab], dtype=np.float64)
    weights_lab = weights_lab / weights_lab.sum()

    return (records, weights_all), (records_lab, weights_lab)


@cache
def _load_generation_globals(config_path: str):
    """Build and cache per-worker generation state.

    Returned objects are shared across chunks in the worker and must remain read-only.

    Args:
        config_path: Generation YAML path.

    Returns:
        Augmentation pipeline, sampling pools, and crop-level class sampler.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    train_data, _, _ = build_training_data(depth_tiers=None, verbose=False, max_da=MAX_DA)
    records = train_data.reset_index().to_dict('records')
    pools = build_pools(records)

    if not raw.get('fg_coords', False):
        raise ValueError('generation-time class classification requires fg_coords with exact bboxes')
    from pumit.ucpt.fg_coords import FgCoordCache
    datasets = sorted({r['dataset'] for r in records})
    fg_cache = FgCoordCache(DATA_ROOT, datasets=datasets)

    pipeline = build_ucpt_pipeline(
        size_xy_choices=raw['size_xy_choices'],
        size_xy_choices_2d=raw['size_xy_choices_2d'],
        max_depth_per_da={int(k): v for k, v in raw['max_depth_per_da'].items()},
        max_da=MAX_DA,
        scale_xy=tuple(raw['scale_xy']) if 'scale_xy' in raw else (3 / 4, 4 / 3),
        labeled_size_xy_max=raw.get('labeled_size_xy_max'),
        labeled_size_xy_max_2d=raw.get('labeled_size_xy_max_2d'),
        fg_cache=fg_cache,
        force_fraction=raw.get('force_fraction', 0.5),
        input_normalizer=InputNormalizer(DATA_ROOT),
    )

    class_sampler = CropClassSampler(
        data_root=DATA_ROOT,
        fg_cache=fg_cache,
        positive_queries=raw['seg_positive_queries'],
        negative_queries=raw['seg_negative_queries'],
        min_positive_voxels=raw['seg_min_positive_voxels'],
        mask_batch_size=raw['seg_mask_batch_size'],
    )
    return pipeline, pools, class_sampler


def _volume_positives(label_contract: LabelContract) -> list[tuple[str, str]]:
    """All source/class pairs declared positive in one volume."""
    out: list[tuple[str, str]] = []
    for source, info in label_contract.items():
        for cls_name in info.get('positive', []):
            out.append((source, cls_name))
    return out


def generate_sample(
    pipeline: TransformPipeline,
    pools: Pools,
    class_sampler: CropClassSampler,
    rng: np.random.Generator,
    labeled: bool,
) -> Record | None:
    """Draw and emit one UCPT sample.

    Args:
        pipeline: Augmentation pipeline used to sample crop parameters.
        pools: Full and labeled pools returned by ``build_pools``.
        class_sampler: Exact post-affine raw-class classifier and final query sampler.
        rng: Sampling generator.
        labeled: Whether to draw from the labeled pool.

    Returns:
        Sample metadata, or ``None`` when augmentation rejects the record.
    """
    (records_all, weights_all), (records_lab, weights_lab) = pools

    label_contract = None
    if labeled:
        idx = int(rng.choice(len(records_lab), p=weights_lab))
        # Shallow copy: pool records are cached read-only. The flag routes the
        # loader's labeled size_xy cap (labeled_size_xy_max*).
        state = {**records_lab[idx], 'labeled_draw': True}
        label_contract, _ = normalize_label_contract(state)

        # Pick an optional center class. Negative-only records remain valid labeled draws and use random placement.
        vol_pos = _volume_positives(label_contract)
        center = None
        if vol_pos:
            center = vol_pos[int(rng.choice(len(vol_pos)))]
            state['_center_class'] = center
    else:
        idx = int(rng.choice(len(records_all), p=weights_all))
        state = records_all[idx]

    # Bake the crop. The spatial transform records whether foreground forcing actually occurred.
    params = pipeline.sample_params(state, rng)
    if params is None:
        return None
    sp = params[0]

    ln2 = math.log(2)
    rope_rescale = float(math.exp(rng.uniform(-ln2, ln2)))

    sample = {
        'img': state['img'],
        'spacing_label': [float(x) for x in sp['spacing_label']],
        'params': params,
        'da_enc': sp['da_enc'],
        'n_patches': sp['n_patches'],
        'depth': sp['crop_size'][0],
        'rope_rescale': rope_rescale,
        'labeled': labeled,
    }
    if labeled:
        assert label_contract is not None
        focus = center if sp['foreground_forced'] else None
        classes, mask_frame, _ = class_sampler.sample_materialized_with_stats(
            state,
            label_contract,
            params,
            rng,
            focus=focus,
        )
        if not classes:
            return None
        sample.update({
            'dataset': state['dataset'],
            'key': state['key'],
            'modality': state.get('modality', 'CT'),
            'classes': classes,
            'seg_cost_queries': len(classes),
            MASK_FRAME_KEY: mask_frame,
        })
    return sample
