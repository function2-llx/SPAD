"""DA-aware batch sampler with depth-tier bucketing."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler

from pumit.transforms import TransInfo

from .config import DepthTierConfig
from .trans_info import TransformConf, gen_trans_info


class DABatchSampler(Sampler):
    """DA-aware batch sampler that buckets samples by anisotropy degree.

    Bucket key is (da_enc, da_dec, depth_tier). Batch size from tier config.

    Yields list[(index, TransInfo)] batches.
    """

    def __init__(
        self,
        data: list[dict],
        trans_conf: TransformConf,
        weights: torch.Tensor,
        num_batches: int,
        *,
        max_da: int,
        seed: int = 42,
        rank: int = 0,
        smooth_spad: bool = True,
        depth_tiers: dict[int | None, DepthTierConfig],
    ):
        self.data = data
        self.trans_conf = trans_conf
        self.weights = weights.double()
        self.num_batches = num_batches
        self.max_da = max_da
        self.smooth_spad = smooth_spad
        self.depth_tiers = depth_tiers
        self.R = np.random.RandomState(seed + rank)
        self.bucket_yield_counts: dict[tuple, int] = {}

    def __iter__(self):
        buckets: dict[tuple, list[tuple[int, TransInfo]]] = {}
        self.bucket_yield_counts = {}
        remain = self.num_batches

        while remain > 0:
            idx = self.weights.multinomial(1, replacement=True).item()
            trans_info = gen_trans_info(
                self.data[idx], self.trans_conf, self.R,
                max_da=self.max_da,
                smooth_spad=self.smooth_spad, depth_tiers=self.depth_tiers,
            )
            if trans_info is None:
                continue  # dropped sample

            da_enc = trans_info['da_enc']
            da_dec = trans_info['da_dec']
            depth_tier = trans_info['patch_size'][0]

            # Look up batch size from tier config
            tier_key = None if da_enc is None else min(da_enc, self.max_da)
            tier_cfg = self.depth_tiers[tier_key]
            tier_idx = tier_cfg.tiers.index(depth_tier)
            batch_size = tier_cfg.batch_sizes[tier_idx]

            key = (da_enc, da_dec, depth_tier)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append((idx, trans_info))

            if len(buckets[key]) >= batch_size:
                yield buckets.pop(key)
                self.bucket_yield_counts[key] = self.bucket_yield_counts.get(key, 0) + 1
                remain -= 1

    def __len__(self):
        return self.num_batches
