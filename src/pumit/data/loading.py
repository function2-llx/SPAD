"""Data loading: metadata ingestion, weighting, filtering, train/val split."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from .config import DepthTierConfig, MAX_INPLANE_RATIO

DATA_ROOT = Path('PUMIT-data/preprocess')

# Downstream benchmark datasets: their images stay in training as SSL-only; segmentation supervision is withheld.
# Applied when generating streams; existing streams retain their original supervision.
SSL_ONLY_DATASETS = {'AMOS22', 'ISLES22', 'KiTS23', 'autoPET-III'}


def sqrt_weight_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Default weighting: sqrt-proportional per dataset (alpha=0.5), no 2D/3D ratio."""
    df = df.copy()
    dataset_sizes = df.groupby("dataset")["dataset"].transform("count").astype(float)
    df["weight"] = 1.0 / np.sqrt(dataset_sizes)
    return df


def compute_da(spacing: np.ndarray, depth: int | None = None, *, max_da: int) -> int | None:
    """Compute deterministic DA from raw spacing for filtering.

    Returns None for 2D samples (depth == 1).
    """
    if depth == 1:
        return None
    if np.any(np.isnan(spacing)):
        return None
    spacing_z = spacing[0]
    spacing_xy = spacing[1:].min()
    ratio = spacing_z / spacing_xy
    if ratio < 1:
        return 0
    return min(int(np.log2(ratio)), max_da)


def drop_filter(depth: int, spacing: np.ndarray, depth_tiers: dict, *, max_da: int) -> bool:
    """Return True if sample should be kept, False if dropped."""
    da = compute_da(spacing, depth, max_da=max_da)
    if da is None:
        return True  # 2D: never drop
    tier_key = min(da, max_da)
    tier_cfg = depth_tiers[tier_key]
    drop_threshold = tier_cfg.tiers[0] // 2
    return depth >= drop_threshold


def build_training_data(
    data_root: Path = DATA_ROOT,
    weight_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = sqrt_weight_fn,
    depth_tiers: dict[int, DepthTierConfig] | None = None,
    verbose: bool = True,
    *,
    max_da: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all meta.json files, apply weights, split train/val.

    Returns:
        train_data: DataFrame indexed by key, with columns: shape, spacing, weight, modality, img, ...
        val_data: DataFrame, keys listed in per-dataset val.json files.
        dataset_info: DataFrame with dataset-level stats.
    """
    rows = []
    for meta_path in sorted(data_root.glob("*/meta.json")):
        meta = orjson.loads(meta_path.read_bytes())
        for key, entry in meta.items():
            entry["img"] = str(meta_path.parent / "images" / f"{key}.npy")
            # Convert shape from (C,D,H,W) to spatial-only (D,H,W)
            entry["shape"] = entry["shape"][1:]
            entry["spacing"] = np.array(
                [x if x is not None else np.nan for x in entry["spacing"]]
                if entry["spacing"] is not None
                else [np.nan, np.nan, np.nan],
                dtype=np.float64,
            )
            rows.append(entry)
    if not rows:
        raise FileNotFoundError(f"No meta.json found under {data_root}")

    df = pd.DataFrame(rows).set_index("key")
    df["weight"] = 1.0

    # Demote downstream benchmarks to unlabeled; labeled_eligible then rejects them in every consumer.
    df.loc[df["dataset"].isin(SSL_ONLY_DATASETS), "label"] = False

    if weight_fn is not None:
        df = weight_fn(df)

    # In-plane anisotropy filter: drop samples with excessive in-plane spacing ratio
    has_valid_inplane = df["spacing"].map(lambda sp: not np.any(np.isnan(sp[1:])))
    inplane_ratios = df.loc[has_valid_inplane, "spacing"].map(lambda sp: sp[1:].max() / sp[1:].min())
    inplane_drop_mask = pd.Series(False, index=df.index)
    inplane_drop_mask.loc[has_valid_inplane] = inplane_ratios > MAX_INPLANE_RATIO
    n_inplane_dropped = inplane_drop_mask.sum()
    if n_inplane_dropped > 0:
        from collections import Counter
        drop_by_ds = Counter(df.loc[inplane_drop_mask, "dataset"])
        if verbose:
            print(
                f"in-plane filter: dropped {n_inplane_dropped}/{len(df)} samples "
                f"({n_inplane_dropped/len(df):.1%}): {dict(drop_by_ds)}"
            )
        df = df[~inplane_drop_mask]

    # Dataset-level info
    dataset_info = {}
    for dataset_name, group in df.groupby("dataset"):
        is_2d = all(s[0] == 1 for s in group["shape"])
        dataset_info[dataset_name] = {
            "dims": 2 if is_2d else 3,
            "count": len(group),
            "weights": group["weight"].sum(),
        }
    dataset_info = pd.DataFrame.from_dict(dataset_info, orient="index").sort_index()

    # Train/val split: read per-dataset val.json
    val_mask = pd.Series(False, index=df.index)
    for dataset_name in df["dataset"].unique():
        val_path = data_root / dataset_name / "val.json"
        if not val_path.exists():
            raise FileNotFoundError(
                f"No val.json for {dataset_name}. "
                f"Provide the validation sample keys in {val_path}."
            )
        dataset_val_keys = set(orjson.loads(val_path.read_bytes()))
        dataset_mask = df["dataset"] == dataset_name
        val_mask |= dataset_mask & df.index.isin(dataset_val_keys)

    val_data = df[val_mask]
    train_data = df[~val_mask]

    # Drop filtering: remove training samples too thin for their tier
    if depth_tiers is not None:
        n_before = len(train_data)
        keep_mask = train_data.apply(
            lambda row: drop_filter(row["shape"][0], row["spacing"], depth_tiers, max_da=max_da),
            axis=1,
        )
        dropped = train_data[~keep_mask]
        if len(dropped) > 0:
            from collections import Counter
            drop_by_ds = Counter(dropped["dataset"])
            if verbose:
                print(
                    f"drop_filter: dropped {len(dropped)}/{n_before} training samples "
                    f"({len(dropped)/n_before:.1%}): {dict(drop_by_ds)}"
                )
        train_data = train_data[keep_mask]

    return train_data, val_data, dataset_info
