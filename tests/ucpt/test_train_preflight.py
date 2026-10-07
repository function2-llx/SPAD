from pathlib import Path

import pytest
import torch
import yaml

from pumit.ucpt.train.config import UCPTTrainConfig
from pumit.ucpt.train.preflight import validate_training_preflight


def _preflight_config(tmp_path: Path) -> UCPTTrainConfig:
    stream = tmp_path / 'stream'
    latents = stream / 'latents'
    latents.mkdir(parents=True)
    (latents / 'stats.safetensors').touch()
    (stream / 'meta.yaml').write_text(
        yaml.safe_dump(
            {
                'algorithm': 'ucpt-stream-v3',
                'stream_complete': True,
                'n_shards': 8,
                'batches_per_shard': 10,
            }
        )
    )
    data_root = tmp_path / 'data'
    captions = tmp_path / 'captions'
    data_root.mkdir()
    captions.mkdir()
    text_cache = tmp_path / 'text.pt'
    torch.save({'dim': 1024}, text_cache)
    pretrained = tmp_path / 'dinov3.safetensors'
    pretrained.touch()

    cfg = UCPTTrainConfig()
    cfg.data.stream_dir = str(stream)
    cfg.data.virtual_lanes = 4
    cfg.data.data_root = str(data_root)
    cfg.data.text_cache_path = str(text_cache)
    cfg.data.class_captions_dir = str(captions)
    cfg.model.pretrained = str(pretrained)
    cfg.model.text_embed_dim = 1024
    cfg.optim.steps = 20
    cfg.run.save_dir = str(tmp_path / 'output')
    return cfg


def test_training_preflight_uses_runtime_global_gpu_count(tmp_path):
    cfg = _preflight_config(tmp_path)

    result = validate_training_preflight(cfg, global_world_size=2)

    assert result['global_gpu_processes'] == 2
    assert result['available_steps'] == 20
    with pytest.raises(ValueError, match=r'global GPU process count \(3\)'):
        validate_training_preflight(cfg, global_world_size=3)


def test_training_preflight_rejects_stream_shorter_than_training(tmp_path):
    cfg = _preflight_config(tmp_path)
    cfg.optim.steps = 21

    with pytest.raises(ValueError, match='provides 20'):
        validate_training_preflight(cfg, global_world_size=2)


def test_training_preflight_rejects_output_bound_to_another_stream(tmp_path):
    cfg = _preflight_config(tmp_path)
    save_dir = Path(cfg.run.save_dir)
    save_dir.mkdir()
    cfg.run.save_dir = str(save_dir)
    (save_dir / 'run.yaml').write_text(
        yaml.safe_dump({'config': {'data': {'stream_dir': str(tmp_path / 'old-stream')}}})
    )

    with pytest.raises(ValueError, match='already bound to stream'):
        validate_training_preflight(cfg, global_world_size=2)
