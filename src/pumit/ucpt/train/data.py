"""Shared UCPT replay-dataset construction."""

from pathlib import Path

import yaml

from pumit.codec.config import MAX_DA
from pumit.ucpt.input import InputNormalizer
from pumit.ucpt.replay_dataset import UCPTReplayDataset
from pumit.ucpt.transforms import build_ucpt_pipeline

from .config import UCPTDataConfig


def build_replay_dataset(
    cfg: UCPTDataConfig,
    *,
    rank: int,
    world_size: int,
    start_offset: int,
    n_prefix: int,
    augment_threads: int,
    tcmalloc_release_every: int,
) -> UCPTReplayDataset:
    """Build the replay dataset and augmentation pipeline from stream metadata."""
    with open(Path(cfg.stream_dir) / 'meta.yaml') as file:
        stream_config = yaml.safe_load(file)['config']
    max_depth_per_da = {
        int(key): value for key, value in stream_config.get('max_depth_per_da', {}).items()
    }
    pipeline = build_ucpt_pipeline(
        size_xy_choices=stream_config['size_xy_choices'],
        size_xy_choices_2d=stream_config['size_xy_choices_2d'],
        max_depth_per_da=max_depth_per_da,
        max_da=MAX_DA,
        scale_xy=tuple(stream_config.get('scale_xy', (3 / 4, 4 / 3))),
        input_normalizer=InputNormalizer(cfg.data_root),
    )
    return UCPTReplayDataset(
        stream_dir=cfg.stream_dir,
        latent_dir=cfg.latent_dir,
        rank=rank,
        world_size=world_size,
        virtual_lanes=cfg.virtual_lanes,
        pipeline=pipeline,
        start_offset=start_offset,
        view_specs=[
            {
                'strategy': 'random',
                'ratio_2d': cfg.random_mask_ratio_2d,
                'ratio_3d': cfg.random_mask_ratio_3d,
            },
            {
                'strategy': 'block',
                'ratio_2d': cfg.block_mask_ratio_2d,
                'ratio_3d': cfg.block_mask_ratio_3d,
            },
        ],
        n_prefix=n_prefix,
        augment_threads=augment_threads,
        tcmalloc_release_every=tcmalloc_release_every,
        verify_ready_hashes=False,
        data_root=cfg.data_root,
        text_cache_path=cfg.text_cache_path,
        class_captions_dir=cfg.class_captions_dir,
    )
