"""Validate UCPT training inputs before distributed model startup."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml

from pumit.ucpt.replay_dataset import _validate_v4_ready

from .config import UCPTTrainConfig, parse_config


def _require_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'{name} is missing: {resolved}')
    return resolved


def _require_dir(path: str | Path, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f'{name} is missing: {resolved}')
    return resolved


def validate_training_preflight(
    cfg: UCPTTrainConfig,
    *,
    global_world_size: int,
) -> dict:
    """Check the finalized stream, launch geometry, and static training inputs."""
    if global_world_size < 1:
        raise ValueError(f'global_world_size must be positive, got {global_world_size}')
    if cfg.data.virtual_lanes < 1:
        raise ValueError(f'virtual_lanes must be positive, got {cfg.data.virtual_lanes}')
    if cfg.data.virtual_lanes % global_world_size:
        raise ValueError(
            f'virtual_lanes ({cfg.data.virtual_lanes}) must be divisible by '
            f'global GPU process count ({global_world_size})'
        )

    stream_dir = _require_dir(cfg.data.stream_dir, 'stream directory')
    latent_dir = (
        stream_dir / 'latents'
        if cfg.data.latent_dir is None
        else _require_dir(cfg.data.latent_dir, 'latent directory')
    )
    meta_path = _require_file(stream_dir / 'meta.yaml', 'stream metadata')
    meta = yaml.safe_load(meta_path.read_text())
    if not isinstance(meta, dict) or meta.get('stream_complete') is not True:
        raise RuntimeError(f'stream is not finalized: {stream_dir}')

    n_shards = meta.get('n_shards')
    batches_per_shard = meta.get('batches_per_shard')
    if isinstance(n_shards, bool) or not isinstance(n_shards, int) or n_shards < 1:
        raise ValueError(f'{meta_path}: invalid n_shards')
    if (
        isinstance(batches_per_shard, bool)
        or not isinstance(batches_per_shard, int)
        or batches_per_shard < 1
    ):
        raise ValueError(f'{meta_path}: invalid batches_per_shard')
    if n_shards % cfg.data.virtual_lanes:
        raise ValueError(
            f'n_shards ({n_shards}) must be divisible by virtual_lanes '
            f'({cfg.data.virtual_lanes})'
        )
    available_steps = n_shards // cfg.data.virtual_lanes * batches_per_shard
    if cfg.optim.steps > available_steps:
        raise ValueError(
            f'training requests {cfg.optim.steps} steps, but the replay stream '
            f'provides {available_steps}'
        )

    if meta.get('algorithm') == 'ucpt-stream-v4':
        _validate_v4_ready(
            stream_dir,
            latent_dir,
            meta,
            verify_content_hashes=False,
        )
    else:
        _require_file(latent_dir / 'stats.safetensors', 'latent statistics')

    _require_dir(cfg.data.data_root, 'preprocessed data root')
    _require_dir(cfg.data.class_captions_dir, 'class caption directory')
    text_cache_path = _require_file(cfg.data.text_cache_path, 'text embedding cache')
    text_cache = torch.load(
        text_cache_path,
        map_location='cpu',
        weights_only=True,
        mmap=True,
    )
    if text_cache.get('dim') != cfg.model.text_embed_dim:
        raise ValueError(
            f'text embedding cache dim {text_cache.get("dim")!r} differs from '
            f'model.text_embed_dim {cfg.model.text_embed_dim}'
        )
    _require_file(cfg.model.pretrained, 'DINOv3 checkpoint')
    if cfg.model.sam3_checkpoint:
        _require_file(cfg.model.sam3_checkpoint, 'SAM 3 checkpoint')

    save_dir = Path(cfg.run.save_dir).resolve()
    run_manifest_path = save_dir / 'run.yaml'
    if run_manifest_path.exists():
        run_manifest = yaml.safe_load(run_manifest_path.read_text())
        if not isinstance(run_manifest, dict):
            raise ValueError(f'invalid run manifest: {run_manifest_path}')
        run_config = run_manifest.get('config')
        run_data = run_config.get('data') if isinstance(run_config, dict) else None
        bound_stream = run_data.get('stream_dir') if isinstance(run_data, dict) else None
        if not isinstance(bound_stream, str):
            raise ValueError(f'{run_manifest_path}: missing bound data.stream_dir')
        if Path(bound_stream).resolve() != stream_dir:
            raise ValueError(
                f'training output {save_dir} is already bound to stream '
                f'{Path(bound_stream).resolve()}, not {stream_dir}'
            )

    return {
        'stream_dir': str(stream_dir),
        'latent_dir': str(latent_dir.resolve()),
        'algorithm': meta.get('algorithm'),
        'n_shards': n_shards,
        'batches_per_shard': batches_per_shard,
        'available_steps': available_steps,
        'requested_steps': cfg.optim.steps,
        'virtual_lanes': cfg.data.virtual_lanes,
        'global_gpu_processes': global_world_size,
        'save_dir': str(save_dir),
    }


def main() -> None:
    cfg, _, _ = parse_config()
    world_size = int(os.environ.get('UCPT_PREFLIGHT_WORLD_SIZE', '1'))
    result = validate_training_preflight(cfg, global_world_size=world_size)
    print('[ucpt] training preflight passed')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
