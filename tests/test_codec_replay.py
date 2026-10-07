"""Test replay dataset reads shards and produces DABatch."""

import tempfile
from pathlib import Path

import msgpack
import numpy as np
import torch

from pumit.codec.replay_dataset import CodecReplayDataset
from pumit.codec.transforms import build_codec_pipeline
from pumit.codec.datamodule import TransformConf
from pumit.data.config import DepthTierConfig


def _create_test_stream(tmp_dir: Path, num_batches: int = 4, num_shards: int = 1):
    """Create a minimal stream with real .npy files for replay testing.

    Writes ``num_shards`` shards, each holding ``num_batches`` batches (matching the
    real generator's invariant that every shard is exactly batches_per_shard).
    """
    conf = TransformConf()
    depth_tiers = {None: DepthTierConfig(tiers=(1,), batch_sizes=(2,))}
    pipeline = build_codec_pipeline(conf, depth_tiers=depth_tiers, max_da=4, smooth_spad=True)

    # Create fake .npy files (size_xy=256 in TransformConf, so images must be >= 256)
    npy_dir = tmp_dir / 'data'
    npy_dir.mkdir()
    npy_paths = []
    for i in range(8):
        img = np.random.rand(3, 1, 512, 512).astype(np.float32)
        path = npy_dir / f'sample_{i}.npy'
        np.save(path, img)
        npy_paths.append(str(path))

    rng = np.random.default_rng(42)

    def _make_batch():
        samples = []
        for _ in range(2):  # batch_size=2
            idx = int(rng.integers(0, len(npy_paths)))
            state = {
                'shape': np.array([1, 512, 512]),
                'spacing': np.array([float('nan'), 1.0, 1.0]),
            }
            params = pipeline.sample_params(state, rng)
            assert params is not None
            samples.append({
                'img': npy_paths[idx],
                'spacing': [float('nan'), 1.0, 1.0],
                'params': params,
            })
        return {'da_enc': None, 'da_dec': None, 'patch_size': [1, 256, 256], 'samples': samples}

    for shard_i in range(num_shards):
        batches = [_make_batch() for _ in range(num_batches)]
        with open(tmp_dir / f'shard_{shard_i:05d}.msgpack', 'wb') as f:
            msgpack.pack({'batches': batches}, f)

    import yaml
    with open(tmp_dir / 'meta.yaml', 'w') as f:
        yaml.dump(
            {'batches_per_shard': num_batches, 'total_batches': num_batches * num_shards}, f
        )

    return tmp_dir, pipeline


def test_replay_dataset_iterates():
    with tempfile.TemporaryDirectory() as tmp:
        stream_dir, pipeline = _create_test_stream(Path(tmp), num_batches=4)
        ds = CodecReplayDataset(stream_dir, rank=0, world_size=1, pipeline=pipeline)
        batches = list(ds)
        assert len(batches) == 4
        for batch in batches:
            assert batch.img.shape[0] == 2  # batch_size
            assert batch.img.shape[1] == 3  # channels
            assert batch.da_enc is None


def test_replay_dataset_skip_offset():
    with tempfile.TemporaryDirectory() as tmp:
        stream_dir, pipeline = _create_test_stream(Path(tmp), num_batches=4)
        ds = CodecReplayDataset(stream_dir, rank=0, world_size=1, pipeline=pipeline, start_offset=2)
        batches = list(ds)
        assert len(batches) == 2


def test_replay_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        stream_dir, pipeline = _create_test_stream(Path(tmp), num_batches=2)
        ds1 = CodecReplayDataset(stream_dir, rank=0, world_size=1, pipeline=pipeline)
        ds2 = CodecReplayDataset(stream_dir, rank=0, world_size=1, pipeline=pipeline)
        for b1, b2 in zip(ds1, ds2):
            assert torch.allclose(b1.img, b2.img)


def test_replay_rank_exclusive_shards():
    # Rank-exclusive assignment: rank i owns shards i, i+W, i+2W, ...
    # With 2 shards and world_size=2, rank 0 reads shard 0, rank 1 reads shard 1.
    with tempfile.TemporaryDirectory() as tmp:
        stream_dir, pipeline = _create_test_stream(Path(tmp), num_batches=4, num_shards=2)
        ds_r0 = CodecReplayDataset(stream_dir, rank=0, world_size=2, pipeline=pipeline)
        ds_r1 = CodecReplayDataset(stream_dir, rank=1, world_size=2, pipeline=pipeline)
        r0_batches = list(ds_r0)
        r1_batches = list(ds_r1)
        assert len(r0_batches) == 4
        assert len(r1_batches) == 4
        # Different shards -> the two ranks see different data.
        assert not torch.allclose(r0_batches[0].img, r1_batches[0].img)


def test_replay_rejects_meta_shard_mismatch():
    # meta.yaml claims more batches_per_shard than the shard actually holds:
    # must fail loudly at construction, before any training/warmup.
    import yaml
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        stream_dir, pipeline = _create_test_stream(Path(tmp), num_batches=4)
        with open(stream_dir / 'meta.yaml', 'w') as f:
            yaml.dump({'batches_per_shard': 1000, 'total_batches': 1000}, f)
        with pytest.raises(AssertionError, match='batches_per_shard'):
            CodecReplayDataset(stream_dir, rank=0, world_size=1, pipeline=pipeline)
