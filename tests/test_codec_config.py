"""Tests for VAE training config dataclasses."""

import pytest
from pumit.codec.config import DepthTierConfig, TrainConfig, validate_depth_tiers


class TestDepthTierConfig:
    def test_basic_construction(self):
        cfg = DepthTierConfig(tiers=(32, 64, 96), batch_sizes=(18, 9, 6))
        assert cfg.tiers == (32, 64, 96)
        assert cfg.batch_sizes == (18, 9, 6)

    def test_mismatched_lengths_caught_by_validation(self):
        with pytest.raises(ValueError, match="same length"):
            DepthTierConfig(tiers=(32, 64), batch_sizes=(18,))


class TestValidateDepthTiers:
    def test_valid_default_config(self):
        tiers = {
            0: DepthTierConfig(tiers=(32, 64, 96), batch_sizes=(18, 9, 6)),
            1: DepthTierConfig(tiers=(16, 32, 48), batch_sizes=(30, 15, 10)),
            2: DepthTierConfig(tiers=(12, 24), batch_sizes=(40, 20)),
            3: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
            4: DepthTierConfig(tiers=(6,), batch_sizes=(80,)),
            None: DepthTierConfig(tiers=(1,), batch_sizes=(480,)),
        }
        validate_depth_tiers(tiers)

    def test_missing_2d_key(self):
        tiers = {
            0: DepthTierConfig(tiers=(32,), batch_sizes=(18,)),
        }
        with pytest.raises(ValueError, match="key None"):
            validate_depth_tiers(tiers)

    def test_missing_catchall_key(self):
        tiers = {
            0: DepthTierConfig(tiers=(32,), batch_sizes=(18,)),
            None: DepthTierConfig(tiers=(1,), batch_sizes=(480,)),
        }
        with pytest.raises(ValueError, match="catch-all key 4"):
            validate_depth_tiers(tiers)

    def test_cross_da_divisibility_violation(self):
        tiers = {
            0: DepthTierConfig(tiers=(32,), batch_sizes=(18,)),
            1: DepthTierConfig(tiers=(10,), batch_sizes=(48,)),  # 10 not divisible by 8 (DA=0 divisor)
            2: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
            3: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
            4: DepthTierConfig(tiers=(6,), batch_sizes=(80,)),
            None: DepthTierConfig(tiers=(1,), batch_sizes=(480,)),
        }
        with pytest.raises(ValueError, match="divisib"):
            validate_depth_tiers(tiers)

    def test_tiers_not_sorted(self):
        tiers = {
            0: DepthTierConfig(tiers=(96, 32), batch_sizes=(6, 18)),
            1: DepthTierConfig(tiers=(16,), batch_sizes=(30,)),
            2: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
            3: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
            4: DepthTierConfig(tiers=(6,), batch_sizes=(80,)),
            None: DepthTierConfig(tiers=(1,), batch_sizes=(480,)),
        }
        with pytest.raises(ValueError, match="sorted"):
            validate_depth_tiers(tiers)


class TestTrainConfig:
    def test_required_fields(self):
        cfg = TrainConfig(
            pretrained="pretrained/flux1-vae.pt",
            depth_tiers={
                0: DepthTierConfig(tiers=(32,), batch_sizes=(18,)),
                1: DepthTierConfig(tiers=(16,), batch_sizes=(30,)),
                2: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
                3: DepthTierConfig(tiers=(12,), batch_sizes=(40,)),
                4: DepthTierConfig(tiers=(6,), batch_sizes=(80,)),
                None: DepthTierConfig(tiers=(1,), batch_sizes=(480,)),
            },
            steps=10000,
            lr=1e-4,
            warmup_steps=1000,
            save_dir="outputs/vae",
        )
        assert cfg.weight_decay == 0.01
        assert cfg.grad_ckpt is True
