from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pumit.transforms.loader import PUMITLoaderV2, RandPUMITLoaderV2


@pytest.fixture
def sample_data(tmp_path):
    vol = np.random.RandomState(0).rand(1, 32, 64, 64).astype(np.float32)
    npy_path = str(tmp_path / 'test_vol.npy')
    np.save(npy_path, vol)
    return {
        'img': npy_path,
        'shape': np.array([32, 64, 64]),
        'spacing': np.array([1.0, 0.5, 0.5]),
        '_trans': {
            'da_enc': 0,
            'da_dec': 0,
            't': 0.0,
            'scale': (1.0, 1.0, 1.0),
            'patch_size': (16, 32, 32),
        },
    }


class TestPUMITLoaderV2:
    def test_identity_affine_output_shape(self, sample_data):
        loader = PUMITLoaderV2()
        affine = np.eye(4)
        load_slice = (slice(0, 16), slice(0, 32), slice(0, 32))
        result = loader(dict(sample_data), affine=affine, load_slice=load_slice)
        assert result['img'].shape[1:] == (16, 32, 32)

    def test_same_params_same_output(self, sample_data):
        loader = PUMITLoaderV2()
        affine = np.eye(4)
        affine[0, 0] = 0.5
        load_slice = (slice(0, 32), slice(0, 64), slice(0, 64))
        r1 = loader(dict(sample_data), affine=affine, load_slice=load_slice)
        r2 = loader(dict(sample_data), affine=affine, load_slice=load_slice)
        assert torch.equal(r1['img'].as_tensor(), r2['img'].as_tensor())


class TestRandPUMITLoaderV2:
    def test_get_params_replay_roundtrip(self, sample_data):
        rand_loader = RandPUMITLoaderV2(rotate_p=0.5, rotate_axis_p=0.2)
        rng = np.random.default_rng(42)
        result1 = rand_loader(dict(sample_data), rng)
        params = rand_loader.get_params()
        affine = rand_loader.get_affine()
        result2 = rand_loader.replay(dict(sample_data), params, affine)
        img1 = result1['img'].as_tensor() if hasattr(result1['img'], 'as_tensor') else result1['img']
        img2 = result2['img'].as_tensor() if hasattr(result2['img'], 'as_tensor') else result2['img']
        assert torch.allclose(img1, img2, atol=1e-5)

    def test_different_seeds_different_output(self, sample_data):
        rand_loader = RandPUMITLoaderV2(rotate_p=1.0, rotate_axis_p=1.0)
        r1 = rand_loader(dict(sample_data), np.random.default_rng(1))
        r2 = rand_loader(dict(sample_data), np.random.default_rng(2))
        img1 = r1['img'].as_tensor() if hasattr(r1['img'], 'as_tensor') else r1['img']
        img2 = r2['img'].as_tensor() if hasattr(r2['img'], 'as_tensor') else r2['img']
        assert not torch.equal(img1, img2)

    def test_params_json_serializable(self, sample_data):
        rand_loader = RandPUMITLoaderV2(rotate_p=1.0, rotate_axis_p=0.5)
        rand_loader(dict(sample_data), np.random.default_rng(42))
        params = rand_loader.get_params()
        json.dumps(params)

    def test_affine_shape(self, sample_data):
        rand_loader = RandPUMITLoaderV2(rotate_p=1.0, rotate_axis_p=0.5)
        rand_loader(dict(sample_data), np.random.default_rng(42))
        affine = rand_loader.get_affine()
        assert affine.shape == (4, 4)
        assert affine.dtype == np.float64

    def test_no_rotation_when_nan_spacing(self, sample_data):
        sample_data['spacing'] = np.array([float('nan'), 0.5, 0.5])
        rand_loader = RandPUMITLoaderV2(rotate_p=1.0, rotate_axis_p=1.0)
        rand_loader(dict(sample_data), np.random.default_rng(42))
        affine = rand_loader.get_affine()
        # With NaN spacing: no rotation, scale only + flips
        # The 3x3 block should be diagonal (possibly with sign flips)
        off_diag = affine[:3, :3].copy()
        np.fill_diagonal(off_diag, 0)
        assert np.allclose(off_diag, 0, atol=1e-10)
