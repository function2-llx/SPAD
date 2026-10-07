"""Factory: build_train_dataloader wires dataset + sampler + DataLoader."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch
import torch.utils.data

from .collate import da_collate_fn
from .config import DepthTierConfig
from .dataset import PUMITDataset
from .sampler import DABatchSampler
from .trans_info import TransformConf


def build_train_dataloader(
    train_data: pd.DataFrame,
    trans_conf: TransformConf,
    transform: Callable,
    num_batches: int,
    *,
    max_da: int,
    num_workers: int,
    rank: int,
    seed: int = 42,
    smooth_spad: bool,
    depth_tiers: dict[int | None, DepthTierConfig],
) -> torch.utils.data.DataLoader:
    """Build training DataLoader with DA-aware batch sampling."""
    data = train_data.to_dict('records')
    weights = torch.from_numpy(train_data['weight'].to_numpy().copy())
    dataset = PUMITDataset(data, transform)
    sampler = DABatchSampler(
        data=data,
        trans_conf=trans_conf,
        weights=weights,
        num_batches=num_batches,
        max_da=max_da,
        seed=seed,
        rank=rank,
        smooth_spad=smooth_spad,
        depth_tiers=depth_tiers,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=da_collate_fn,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
