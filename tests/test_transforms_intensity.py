from __future__ import annotations

import numpy as np
import pytest
import torch

from pumit.transforms.intensity import (
    AdjustContrast,
    GammaCorrection,
    GaussianNoise,
    GaussianSmooth,
    RandAdjustContrast,
    RandGammaCorrection,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandScaleIntensity,
    ScaleIntensity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_img(c: int = 2, d: int = 4, h: int = 4, w: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(c, d, h, w, generator=g)


def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Deterministic: ScaleIntensity
# ---------------------------------------------------------------------------

def test_scale_intensity_scalar():
    img = torch.ones(2, 4, 4, 4)
    out = ScaleIntensity()(img, 0.5)
    assert torch.allclose(out, img * 1.5)


def test_scale_intensity_per_channel():
    img = torch.ones(2, 4, 4, 4)
    out = ScaleIntensity()(img, [0.5, 1.0])
    assert torch.allclose(out[0], img[0] * 1.5)
    assert torch.allclose(out[1], img[1] * 2.0)


def test_scale_intensity_zero_factor():
    img = make_img()
    out = ScaleIntensity()(img, 0.0)
    assert torch.allclose(out, img)


# ---------------------------------------------------------------------------
# Deterministic: AdjustContrast
# ---------------------------------------------------------------------------

def test_adjust_contrast_factor_one_is_identity():
    """factor=1 -> out = img * 1 + mean * 0 = img (mean cancels)."""
    img = make_img()
    out = AdjustContrast()(img, 1.0, preserve_range=False)
    assert torch.allclose(out, img)


def test_adjust_contrast_factor_zero_is_mean():
    """factor=0 -> out = mean per channel."""
    img = make_img()
    c = img.shape[0]
    out = AdjustContrast()(img, 0.0, preserve_range=False)
    for ch in range(c):
        expected_mean = img[ch].mean()
        assert torch.allclose(out[ch], torch.full_like(out[ch], expected_mean), atol=1e-5)


def test_adjust_contrast_preserve_range():
    img = make_img()
    out = AdjustContrast()(img, 2.0, preserve_range=True)
    for ch in range(img.shape[0]):
        assert out[ch].min() >= img[ch].min() - 1e-5
        assert out[ch].max() <= img[ch].max() + 1e-5


def test_adjust_contrast_per_channel():
    img = make_img()
    out = AdjustContrast()(img, [1.0, 0.0], preserve_range=False)
    assert torch.allclose(out[0], img[0])
    expected_mean = img[1].mean()
    assert torch.allclose(out[1], torch.full_like(out[1], expected_mean), atol=1e-5)


# ---------------------------------------------------------------------------
# Deterministic: GammaCorrection
# ---------------------------------------------------------------------------

def test_gamma_one_is_near_identity():
    """gamma=1 -> img^1 = img (after normalize/denormalize, output equals input)."""
    img = make_img()
    out = GammaCorrection()(img, 1.0, invert=False, retain_stats=False)
    assert torch.allclose(out, img, atol=1e-5)


def test_gamma_retain_stats_preserves_mean_std():
    img = make_img()
    original_mean = img.view(img.shape[0], -1).mean(1)
    original_std = img.view(img.shape[0], -1).std(1, correction=0)
    out = GammaCorrection()(img, 2.0, invert=False, retain_stats=True)
    out_mean = out.view(out.shape[0], -1).mean(1)
    out_std = out.view(out.shape[0], -1).std(1, correction=0)
    assert torch.allclose(out_mean, original_mean, atol=1e-4)
    assert torch.allclose(out_std, original_std, atol=1e-4)


def test_gamma_invert_double_negate_is_identity():
    """invert=True applied twice should be identity."""
    img = make_img()
    out = GammaCorrection()(img, 1.0, invert=True, retain_stats=False)
    out2 = GammaCorrection()(out, 1.0, invert=True, retain_stats=False)
    assert torch.allclose(out2, img, atol=1e-5)


# ---------------------------------------------------------------------------
# Deterministic: GaussianSmooth
# ---------------------------------------------------------------------------

def test_gaussian_smooth_zero_sigma_is_identity():
    img = make_img()
    out = GaussianSmooth()(img, [0.0, 0.0, 0.0])
    assert torch.allclose(out, img)


def test_gaussian_smooth_reduces_variance():
    img = make_img(seed=99)
    out = GaussianSmooth()(img, [1.0, 1.0, 1.0])
    assert out.var() < img.var()


def test_gaussian_smooth_output_shape():
    img = make_img(c=3, d=8, h=8, w=8)
    out = GaussianSmooth()(img, [0.5, 1.0, 1.5])
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Deterministic: GaussianNoise
# ---------------------------------------------------------------------------

def test_gaussian_noise_adds_noise():
    img = make_img()
    noise = torch.randn_like(img) * 0.1
    out = GaussianNoise()(img, noise)
    assert torch.allclose(out, img + noise)


# ---------------------------------------------------------------------------
# RandScaleIntensity
# ---------------------------------------------------------------------------

def test_rand_scale_prob0_no_transform():
    img = make_img()
    t = RandScaleIntensity(prob=0.0, factor_range=(-0.5, 0.5))
    out = t(img, rng())
    assert torch.allclose(out, img)
    assert not t.get_params()['do_transform']


def test_rand_scale_prob1_transforms():
    img = make_img()
    t = RandScaleIntensity(prob=1.0, factor_range=(0.5, 0.5))
    out = t(img, rng())
    assert not torch.allclose(out, img)


def test_rand_scale_replay():
    img = make_img()
    t = RandScaleIntensity(prob=1.0, factor_range=(-0.3, 0.3))
    out = t(img, rng(7))
    params = t.get_params()
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_scale_channel_wise_replay():
    img = make_img(c=3)
    t = RandScaleIntensity(prob=1.0, factor_range=(-0.3, 0.3), channel_wise=True)
    out = t(img, rng(8))
    params = t.get_params()
    assert len(params['factors']) == 3
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_scale_params_serializable():
    img = make_img()
    t = RandScaleIntensity(prob=1.0, factor_range=(-0.3, 0.3))
    t(img, rng())
    params = t.get_params()
    import json
    json.dumps(params)  # must not raise


# ---------------------------------------------------------------------------
# RandAdjustContrast
# ---------------------------------------------------------------------------

def test_rand_adjust_contrast_prob0_no_transform():
    img = make_img()
    t = RandAdjustContrast(prob=0.0, contrast_range=(0.5, 2.0))
    out = t(img, rng())
    assert torch.allclose(out, img)


def test_rand_adjust_contrast_replay():
    img = make_img()
    t = RandAdjustContrast(prob=1.0, contrast_range=(0.5, 2.0))
    out = t(img, rng(1))
    params = t.get_params()
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_adjust_contrast_per_channel_replay():
    img = make_img(c=4)
    t = RandAdjustContrast(prob=1.0, contrast_range=(0.5, 2.0), per_channel=True)
    out = t(img, rng(2))
    params = t.get_params()
    assert len(params['factors']) == 4
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_adjust_contrast_preserve_range():
    img = make_img()
    t = RandAdjustContrast(prob=1.0, contrast_range=(3.0, 5.0), preserve_range=True)
    out = t(img, rng(3))
    for ch in range(img.shape[0]):
        assert out[ch].min() >= img[ch].min() - 1e-5
        assert out[ch].max() <= img[ch].max() + 1e-5


# ---------------------------------------------------------------------------
# RandGammaCorrection
# ---------------------------------------------------------------------------

def test_rand_gamma_prob0_no_transform():
    img = make_img()
    t = RandGammaCorrection(prob=0.0, gamma_range=(0.5, 2.0))
    out = t(img, rng())
    assert torch.allclose(out, img)


def test_rand_gamma_replay():
    img = make_img()
    t = RandGammaCorrection(prob=1.0, gamma_range=(0.5, 2.0), prob_invert=0.5)
    out = t(img, rng(10))
    params = t.get_params()
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_gamma_per_channel_replay():
    img = make_img(c=3)
    t = RandGammaCorrection(prob=1.0, gamma_range=(0.5, 2.0), per_channel=True)
    out = t(img, rng(11))
    params = t.get_params()
    assert len(params['gammas']) == 3
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_gamma_low_range_samples_both_sides():
    """With gamma_range=(0.5, 2.0), gammas should appear both below and above 1."""
    t = RandGammaCorrection(prob=1.0, gamma_range=(0.5, 2.0))
    img = make_img()
    gammas = []
    r = rng(0)
    for _ in range(200):
        t(img, r)
        gammas.append(t.get_params()['gammas'][0])
    assert any(g < 1.0 for g in gammas), 'expected some gammas < 1'
    assert any(g >= 1.0 for g in gammas), 'expected some gammas >= 1'


def test_rand_gamma_params_serializable():
    img = make_img()
    t = RandGammaCorrection(prob=1.0, gamma_range=(0.5, 2.0), prob_invert=0.5)
    t(img, rng())
    import json
    json.dumps(t.get_params())


# ---------------------------------------------------------------------------
# RandGaussianSmooth
# ---------------------------------------------------------------------------

def test_rand_gaussian_smooth_prob0_no_transform():
    img = make_img(d=8, h=8, w=8)
    t = RandGaussianSmooth(prob=0.0, sigma_range=(0.5, 1.5))
    out = t(img, rng())
    assert torch.allclose(out, img)


def test_rand_gaussian_smooth_replay():
    img = make_img(d=8, h=8, w=8)
    t = RandGaussianSmooth(prob=1.0, sigma_range=(0.5, 1.5))
    out = t(img, rng(20))
    params = t.get_params()
    out_replay = t.replay(img, params)
    assert torch.allclose(out, out_replay)


def test_rand_gaussian_smooth_isotropic():
    img = make_img(d=8, h=8, w=8)
    t = RandGaussianSmooth(prob=1.0, sigma_range=(0.5, 1.5), isotropic=True)
    t(img, rng(21))
    params = t.get_params()
    sigma = params['sigma']
    assert sigma[0] == sigma[1] == sigma[2]


def test_rand_gaussian_smooth_params_serializable():
    img = make_img(d=8, h=8, w=8)
    t = RandGaussianSmooth(prob=1.0, sigma_range=(0.5, 1.5))
    t(img, rng())
    import json
    json.dumps(t.get_params())


# ---------------------------------------------------------------------------
# RandGaussianNoise
# ---------------------------------------------------------------------------

def test_rand_gaussian_noise_prob0_no_transform():
    img = make_img()
    t = RandGaussianNoise(prob=0.0, std_range=(0.01, 0.1))
    out = t(img, rng())
    assert torch.allclose(out, img)


def test_rand_gaussian_noise_prob1_changes_img():
    img = make_img()
    t = RandGaussianNoise(prob=1.0, std_range=(0.1, 0.1))
    out = t(img, rng())
    assert not torch.allclose(out, img)


def test_rand_gaussian_noise_replay_bit_identical():
    """Replay from seed must produce bit-identical results."""
    img = make_img()
    t = RandGaussianNoise(prob=1.0, std_range=(0.05, 0.1))
    out = t(img, rng(30))
    params = t.get_params()
    out_replay = t.replay(img, params)
    assert torch.equal(out, out_replay), 'replay must be bit-identical'


def test_rand_gaussian_noise_params_no_tensor():
    """Params must not contain tensors (only seed + std)."""
    img = make_img()
    t = RandGaussianNoise(prob=1.0, std_range=(0.05, 0.1))
    t(img, rng(31))
    params = t.get_params()
    assert 'seed' in params
    assert 'std' in params
    assert isinstance(params['seed'], int)
    assert isinstance(params['std'], float)
    import json
    json.dumps(params)


def test_rand_gaussian_noise_different_seeds_differ():
    """Different rng seeds should (almost surely) produce different outputs."""
    img = make_img()
    t = RandGaussianNoise(prob=1.0, std_range=(0.1, 0.1))
    out1 = t(img, rng(0))
    params1 = t.get_params()
    out2 = t(img, rng(1))
    params2 = t.get_params()
    # Seeds themselves must differ with very high probability
    assert params1['seed'] != params2['seed']
    assert not torch.equal(out1, out2)
