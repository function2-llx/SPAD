"""Tests for the data pipeline after Lightning removal."""

import numpy as np
from pathlib import Path

import torch

from pumit.codec.datamodule import TransformConf
from pumit.data import gen_trans_info, DABatchSampler, compute_da, drop_filter
from pumit.data.trans_info import GenTransInfo
from pumit.data.config import DepthTierConfig
from pumit.codec.config import MAX_DA


def _default_depth_tiers():
    return {
        0: DepthTierConfig(tiers=(32, 64, 96), batch_sizes=(18, 9, 6)),
        1: DepthTierConfig(tiers=(16, 32, 48), batch_sizes=(30, 15, 10)),
        2: DepthTierConfig(tiers=(12, 24), batch_sizes=(40, 20)),
        3: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
        4: DepthTierConfig(tiers=(6,), batch_sizes=(80,)),
        None: DepthTierConfig(tiers=(1,), batch_sizes=(480,)),
    }


class TestGenTransInfo:
    """gen_trans_info is pure logic, no I/O needed."""

    def test_isotropic_spacing(self):
        data = {'shape': np.array([64, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA)
        assert 'da_enc' in info
        assert 'scale' in info
        assert 'patch_size' in info
        assert info['da_enc'] == 0  # isotropic -> DA=0

    def test_anisotropic_spacing(self):
        # spacing_z = 5mm, spacing_xy = 1mm -> ratio ~5 -> DA=2
        data = {'shape': np.array([20, 128, 128]), 'spacing': np.array([5.0, 1.0, 1.0])}
        conf = TransformConf()
        R = np.random.RandomState(0)
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        info = gen_trans_info(data, conf, R, max_da=MAX_DA)
        assert info['da_enc'] == 2
        assert info['patch_size'][0] <= 32

    def test_2d_data(self):
        data = {'shape': np.array([1, 256, 256]), 'spacing': np.array([999.0, 1.0, 1.0])}
        conf = TransformConf()
        R = np.random.RandomState(0)
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        info = gen_trans_info(data, conf, R, max_da=MAX_DA)
        assert info['patch_size'][0] == 1
        assert info['da_enc'] is None
        assert info['da_dec'] is None
        assert info['t'] == 0.0

    def test_2d_returns_none_da(self):
        """2D samples must return None for all DA fields."""
        data = {'shape': np.array([1, 512, 512]), 'spacing': np.array([1.0, 0.5, 0.5])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(123)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA)
        assert info['da_enc'] is None
        assert info['da_dec'] is None

    def test_high_da_clamped_to_max(self):
        """Very anisotropic 3D data should have DA clamped to MAX_DA."""
        # spacing_z=64, spacing_xy=1 -> ratio=64 -> log2=6, clamped to MAX_DA=4
        data = {'shape': np.array([10, 128, 128]), 'spacing': np.array([64.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False)
        assert info['da_enc'] == MAX_DA
        assert info['da_dec'] == MAX_DA


class TestGenTransInfoTransform:
    """GenTransInfo is a monai Randomizable Transform."""

    def test_injects_trans_key(self):
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        t = GenTransInfo(conf, max_da=MAX_DA)
        data = {'shape': np.array([64, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        result = t(data)
        assert '_trans' in result
        assert 'da_enc' in result['_trans']

    def test_does_not_mutate_input(self):
        conf = TransformConf()
        t = GenTransInfo(conf, max_da=MAX_DA)
        data = {'shape': np.array([64, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        original_keys = set(data.keys())
        t(data)
        assert set(data.keys()) == original_keys  # input not mutated


class TestGenTransInfoWithTiers:
    def test_isotropic_small_depth(self):
        """Depth 20 at DA=0 -> tier 32 (smallest >= 20)."""
        data = {'shape': np.array([20, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False, depth_tiers=_default_depth_tiers())
        assert info['patch_size'][0] == 32

    def test_isotropic_exact_tier(self):
        """Depth 64 at DA=0 -> tier 64."""
        data = {'shape': np.array([64, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False, depth_tiers=_default_depth_tiers())
        assert info['patch_size'][0] == 64

    def test_isotropic_overflow_crops_to_largest(self):
        """Depth 200 at DA=0 exceeds max tier 96 -> tier 96."""
        data = {'shape': np.array([200, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False, depth_tiers=_default_depth_tiers())
        assert info['patch_size'][0] == 96

    def test_2d_routes_to_depth_1(self):
        data = {'shape': np.array([1, 256, 256]), 'spacing': np.array([999.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False, depth_tiers=_default_depth_tiers())
        assert info['patch_size'][0] == 1
        assert info['da_enc'] is None

    def test_drop_returns_none(self):
        """Depth 5 at DA=0 with smallest tier 32 -> threshold 16 -> drop."""
        data = {'shape': np.array([5, 128, 128]), 'spacing': np.array([1.0, 1.0, 1.0])}
        conf = TransformConf()
        conf.scale_z_p = 0.0
        conf.scale_xy_p = 0.0
        R = np.random.RandomState(42)
        info = gen_trans_info(data, conf, R, max_da=MAX_DA, smooth_spad=False, depth_tiers=_default_depth_tiers())
        assert info is None


class TestComputeDa:
    def test_isotropic(self):
        assert compute_da(np.array([1.0, 1.0, 1.0]), max_da=MAX_DA) == 0

    def test_anisotropic(self):
        assert compute_da(np.array([4.0, 1.0, 1.0]), max_da=MAX_DA) == 2

    def test_negative_ratio(self):
        assert compute_da(np.array([0.5, 1.0, 1.0]), max_da=MAX_DA) == 0

    def test_2d_returns_none(self):
        assert compute_da(np.array([999.0, 1.0, 1.0]), depth=1, max_da=MAX_DA) is None

    def test_clamps_at_max_da(self):
        # ratio=64 -> log2=6, clamped to MAX_DA=4
        assert compute_da(np.array([64.0, 1.0, 1.0]), max_da=MAX_DA) == MAX_DA

    def test_da5_clamped_to_4(self):
        # ratio=32 -> log2=5, clamped to MAX_DA=4
        assert compute_da(np.array([32.0, 1.0, 1.0]), max_da=MAX_DA) == MAX_DA


class TestDropFilter:
    def test_keeps_adequate_depth(self):
        tiers = _default_depth_tiers()
        assert drop_filter(depth=20, spacing=np.array([1.0, 1.0, 1.0]), depth_tiers=tiers, max_da=MAX_DA) is True

    def test_drops_tiny_depth(self):
        tiers = _default_depth_tiers()
        assert drop_filter(depth=5, spacing=np.array([1.0, 1.0, 1.0]), depth_tiers=tiers, max_da=MAX_DA) is False

    def test_keeps_2d(self):
        tiers = _default_depth_tiers()
        assert drop_filter(depth=1, spacing=np.array([999.0, 1.0, 1.0]), depth_tiers=tiers, max_da=MAX_DA) is True


class TestDABatchSamplerTiered:
    def _make_data(self, n=100):
        data = []
        R = np.random.RandomState(0)
        for i in range(n):
            depth = R.choice([16, 32, 64, 96])
            spacing_z = R.choice([1.0, 2.0, 4.0, 8.0])
            data.append({
                'shape': np.array([depth, 128, 128]),
                'spacing': np.array([spacing_z, 1.0, 1.0]),
                'dataset': 'test',
                'modality': 'CT',
                'key': f'sample_{i}',
            })
        return data

    def test_yields_correct_number_of_batches(self):
        data = self._make_data(200)
        weights = torch.ones(len(data))
        sampler = DABatchSampler(
            data=data,
            trans_conf=TransformConf(),
            weights=weights,
            num_batches=10,
            depth_tiers=_default_depth_tiers(),
            smooth_spad=False,
            max_da=MAX_DA,
        )
        batches = list(sampler)
        assert len(batches) == 10

    def test_all_samples_in_batch_share_da_and_depth(self):
        data = self._make_data(500)
        weights = torch.ones(len(data))
        sampler = DABatchSampler(
            data=data,
            trans_conf=TransformConf(),
            weights=weights,
            num_batches=20,
            depth_tiers=_default_depth_tiers(),
            smooth_spad=False,
            max_da=MAX_DA,
        )
        for batch in sampler:
            da_encs = set()
            da_decs = set()
            depths = set()
            for idx, ti in batch:
                da_encs.add(ti['da_enc'])
                da_decs.add(ti['da_dec'])
                depths.add(ti['patch_size'][0])
            assert len(da_encs) == 1, f"Mixed da_enc: {da_encs}"
            assert len(da_decs) == 1, f"Mixed da_dec: {da_decs}"
            assert len(depths) == 1, f"Mixed depths: {depths}"

    def test_bucket_yield_counts_logged(self):
        data = self._make_data(500)
        weights = torch.ones(len(data))
        sampler = DABatchSampler(
            data=data,
            trans_conf=TransformConf(),
            weights=weights,
            num_batches=50,
            depth_tiers=_default_depth_tiers(),
            smooth_spad=False,
            max_da=MAX_DA,
        )
        list(sampler)
        assert hasattr(sampler, 'bucket_yield_counts')
        assert sum(sampler.bucket_yield_counts.values()) == 50


class TestModuleInterface:
    def test_no_lightning_import(self):
        import importlib
        mod = importlib.import_module('pumit.codec.datamodule')
        source = Path(mod.__file__).read_text()
        assert 'lightning' not in source.lower()

    def test_exports_functions(self):
        import pumit.codec.datamodule as dm
        import pumit.data as data
        assert callable(dm.build_train_transform)
        assert callable(data.build_training_data)
        assert callable(data.da_collate_fn)
