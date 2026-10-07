"""Replay dataset: reads pre-generated stream shards, produces DABatch."""

from __future__ import annotations

from pathlib import Path

import msgpack
import torch
import yaml

from pumit.data.collate import DABatch
from pumit.transforms.intensity import ensure_rgb
from pumit.transforms.pipeline import TransformPipeline


class CodecReplayDataset(torch.utils.data.Dataset):
    """Map-style dataset that reads msgpack shards lazily and replays transforms.

    Rank-exclusive shard assignment: rank i owns shards i, i+W, i+2W, ...
    Each rank reads its shards sequentially front-to-back.
    """

    def __init__(
        self,
        stream_dir: Path | str,
        rank: int,
        world_size: int,
        pipeline: TransformPipeline,
        start_offset: int = 0,
    ):
        self.pipeline = pipeline
        self.start_offset = start_offset

        stream_dir = Path(stream_dir)
        all_shards = sorted(stream_dir.glob('shard_*.msgpack'))
        if not all_shards:
            raise FileNotFoundError(f"No shard files found in {stream_dir}")

        num_shards = len(all_shards)
        assert num_shards % world_size == 0, (
            f"num_shards ({num_shards}) must be divisible by world_size ({world_size})"
        )

        meta_path = stream_dir / 'meta.yaml'
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        self.batches_per_shard = meta['batches_per_shard']

        self.my_shards = all_shards[rank::world_size]

        # Validate meta against real shard contents before training starts (not lazily
        # in __getitem__, which would only surface inside a DataLoader worker after the
        # compile warmup). Shard-boundary indexing assumes every shard holds exactly
        # batches_per_shard; the generator enforces total_batches % batches_per_shard == 0
        # so there is no partial trailing shard.
        with open(self.my_shards[0], 'rb') as f:
            n = len(msgpack.unpack(f, raw=False)['batches'])
        assert n == self.batches_per_shard, (
            f"{self.my_shards[0]} holds {n} batches but meta.yaml says "
            f"batches_per_shard={self.batches_per_shard}. Fix meta.yaml to match "
            f"the real shard size."
        )

        self._length = len(self.my_shards) * self.batches_per_shard - start_offset
        self._cached_shard: list | None = None
        self._cached_shard_idx: int = -1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> DABatch:
        local_idx = self.start_offset + idx
        shard_local = local_idx // self.batches_per_shard
        batch_idx = local_idx % self.batches_per_shard

        if shard_local != self._cached_shard_idx:
            with open(self.my_shards[shard_local], 'rb') as f:
                shard = msgpack.unpack(f, raw=False)
            self._cached_shard = shard['batches']
            self._cached_shard_idx = shard_local

        batch_info = self._cached_shard[batch_idx]

        imgs = []
        not_rgb_flags = []
        paths = []
        for s in batch_info['samples']:
            data = {'img': s['img'], 'spacing': s['spacing']}
            data = self.pipeline.replay(data, s['params'])
            img = data['img']
            if hasattr(img, 'as_tensor'):
                img = img.as_tensor()
            img, not_rgb = ensure_rgb(img)
            imgs.append(img)
            not_rgb_flags.append(not_rgb)
            paths.append(s['img'])

        da_enc = batch_info['da_enc']
        return DABatch(
            img=torch.stack(imgs),
            not_rgb=not_rgb_flags,
            da_enc=da_enc,
            da_dec=batch_info['da_dec'],
            t=0.0,
            spacing=(float('inf') if da_enc is None else 2.0 ** da_enc, 1.0, 1.0),
            paths=paths,
        )
