"""Tests for codec transforms following sample_params/apply protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pumit.codec.transforms import (
    AdjustContrastTransform,
    GammaCorrectionTransform,
    NormalizeTransform,
    ScaleIntensityTransform,
    SpatialTransform,
    build_codec_pipeline,
)
from pumit.codec.datamodule import TransformConf
from pumit.data.config import DepthTierConfig


def _make_depth_tiers() -> dict[int | None, DepthTierConfig]:
    """Depth tiers suitable for testing."""
    return {
        None: DepthTierConfig(tiers=(1,), batch_sizes=(32,)),
        0: DepthTierConfig(tiers=(64, 96, 128), batch_sizes=(8, 6, 4)),
        1: DepthTierConfig(tiers=(48, 64, 96), batch_sizes=(8, 6, 4)),
        2: DepthTierConfig(tiers=(32, 48, 64), batch_sizes=(8, 6, 4)),
        3: DepthTierConfig(tiers=(24, 32, 48), batch_sizes=(8, 6, 4)),
        4: DepthTierConfig(tiers=(16, 24, 32), batch_sizes=(8, 6, 4)),
    }


@pytest.fixture
def conf() -> TransformConf:
    return TransformConf()


@pytest.fixture
def depth_tiers() -> dict[int | None, DepthTierConfig]:
    return _make_depth_tiers()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestScaleIntensityTransform:
    def test_enabled(self, rng):
        t = ScaleIntensityTransform(prob=1.0, factor_range=(-0.2, 0.2), channel_wise=True)
        state = {'num_channels': 3}
        params = t.sample_params(state, rng)
        assert params is not None
        assert params['enabled'] is True
        assert 'factors' in params
        assert len(params['factors']) == 3

        img = torch.rand(3, 16, 16, 16) * 0.5 + 0.25  # in (0.25, 0.75)
        data = {'img': img.clone()}
        result = t(data, **params)
        # Image should be modified
        assert not torch.allclose(result['img'], img)
        # Should be clamped to [0, 1]
        assert result['img'].min() >= 0.0
        assert result['img'].max() <= 1.0

    def test_disabled(self, rng):
        t = ScaleIntensityTransform(prob=0.0, factor_range=(-0.2, 0.2), channel_wise=True)
        state = {'num_channels': 3}
        params = t.sample_params(state, rng)
        assert params is not None
        assert params['enabled'] is False

        img = torch.rand(3, 16, 16, 16)
        data = {'img': img.clone()}
        result = t(data, **params)
        assert torch.allclose(result['img'], img)


class TestAdjustContrastTransform:
    def test_round_trip(self, rng):
        t = AdjustContrastTransform(prob=1.0, contrast_range=(0.75, 1.25), preserve_range=True)
        state = {'num_channels': 1}
        params = t.sample_params(state, rng)
        assert params is not None
        assert params['enabled'] is True
        assert 'factor' in params

        img = torch.rand(1, 16, 16, 16)
        data = {'img': img.clone()}
        result = t(data, **params)
        # With preserve_range=True, output should be in [0, 1]
        assert result['img'].min() >= 0.0
        assert result['img'].max() <= 1.0


class TestGammaCorrectionTransform:
    def test_params(self, rng):
        t = GammaCorrectionTransform(prob=1.0, gamma_range=(0.7, 1.5), prob_invert=0.15)
        state = {'num_channels': 1}
        params = t.sample_params(state, rng)
        assert params is not None
        assert params['enabled'] is True
        assert 'gammas' in params
        assert isinstance(params['gammas'], list)
        assert len(params['gammas']) == 1
        assert params['gammas'][0] > 0
        assert 'invert' in params
        assert isinstance(params['invert'], bool)


class TestNormalizeTransform:
    def test_deterministic(self, rng):
        t = NormalizeTransform()

        img = torch.rand(1, 16, 16, 16)
        expected = (img * 2 - 1).expand(3, -1, -1, -1)  # ensure_rgb expands to 3ch
        data = {'img': img.clone()}
        result = t(data)
        assert torch.allclose(result['img'], expected)


class TestSpatialTransform:
    def test_sample_params_2d(self, conf, depth_tiers, rng):
        t = SpatialTransform(conf=conf, depth_tiers=depth_tiers, max_da=4, smooth_spad=True)
        state = {'shape': np.array([1, 512, 512]), 'spacing': np.array([1.0, 0.5, 0.5])}
        params = t.sample_params(state, rng)
        assert params is not None
        assert params['da_enc'] is None
        assert params['da_dec'] is None
        assert params['patch_size'] == [1, 256, 256]
        assert len(params['affine']) == 16
        assert len(params['load_slice_start']) == 3
        assert len(params['load_slice_stop']) == 3

    def test_sample_params_3d(self, conf, depth_tiers, rng):
        t = SpatialTransform(conf=conf, depth_tiers=depth_tiers, max_da=4, smooth_spad=True)
        state = {'shape': np.array([100, 512, 512]), 'spacing': np.array([1.0, 0.5, 0.5])}
        params = t.sample_params(state, rng)
        assert params is not None
        assert params['da_enc'] is not None
        assert params['da_dec'] is not None
        assert len(params['affine']) == 16
        assert len(params['patch_size']) == 3
        assert len(params['load_slice_start']) == 3
        assert len(params['load_slice_stop']) == 3

    def test_drop_thin(self, conf, rng):
        """A volume with only 5 slices should be dropped if tier requires >= 64."""
        # Use tiers where smallest is 64, so 5 < 64//2 = 32 triggers drop
        depth_tiers = {
            None: DepthTierConfig(tiers=(1,), batch_sizes=(32,)),
            0: DepthTierConfig(tiers=(64, 96, 128), batch_sizes=(8, 6, 4)),
            1: DepthTierConfig(tiers=(64, 96, 128), batch_sizes=(8, 6, 4)),
            2: DepthTierConfig(tiers=(64, 96, 128), batch_sizes=(8, 6, 4)),
            3: DepthTierConfig(tiers=(64, 96, 128), batch_sizes=(8, 6, 4)),
            4: DepthTierConfig(tiers=(64, 96, 128), batch_sizes=(8, 6, 4)),
        }
        t = SpatialTransform(conf=conf, depth_tiers=depth_tiers, max_da=4, smooth_spad=True)
        # shape[0]=5, spacing ratio ~2 so da_enc likely 1.
        # With tiers[0]=64, drop threshold is 64//2=32, 5 < 32 -> drop
        state = {'shape': np.array([5, 512, 512]), 'spacing': np.array([1.0, 0.5, 0.5])}
        # Try multiple seeds: at least one should trigger drop
        dropped = False
        for seed in range(100):
            params = t.sample_params(state, np.random.default_rng(seed))
            if params is None:
                dropped = True
                break
        assert dropped, "Expected SpatialTransform to drop a thin volume (5 slices)"

    def test_apply(self, conf, depth_tiers, rng, tmp_path):
        """End-to-end: sample_params then apply on a real .npy file."""
        t = SpatialTransform(conf=conf, depth_tiers=depth_tiers, max_da=4, smooth_spad=True)
        # Create a temp .npy file (C=1, D=100, H=256, W=256)
        arr = np.random.rand(1, 100, 256, 256).astype(np.float32)
        npy_path = str(tmp_path / "test.npy")
        np.save(npy_path, arr)

        state = {'shape': np.array([100, 256, 256]), 'spacing': np.array([1.0, 1.0, 1.0])}
        params = t.sample_params(state, rng)
        assert params is not None

        data = {'img': npy_path}
        result = t(data, **params)
        assert isinstance(result['img'], torch.Tensor)
        assert result['img'].ndim == 4  # C, D, H, W
        assert 'da_enc' in result
        assert 'da_dec' in result


class TestBuildCodecPipeline:
    def test_pipeline_has_5_transforms(self, depth_tiers):
        pipeline = build_codec_pipeline(depth_tiers=depth_tiers)
        assert len(pipeline.transforms) == 5
