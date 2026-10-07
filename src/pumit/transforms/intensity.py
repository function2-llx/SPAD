from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    'ensure_rgb',
    'rgb_to_gray',
    'ScaleIntensity',
    'RandScaleIntensity',
    'AdjustContrast',
    'RandAdjustContrast',
    'GammaCorrection',
    'RandGammaCorrection',
    'GaussianSmooth',
    'RandGaussianSmooth',
    'GaussianNoise',
    'RandGaussianNoise',
]

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)


def ensure_rgb(x: Tensor, batched: bool = False) -> tuple[Tensor, bool]:
    x = x.contiguous()
    if x.shape[batched] == 3:
        not_rgb = False
    else:
        assert x.shape[batched] == 1
        maybe_batch = 'n' if batched else ''
        x = einops.repeat(x, f'{maybe_batch} 1 ... -> c ...', c=3)
        not_rgb = True
    return x, not_rgb


def rgb_to_gray(x: Tensor, batched: bool = False) -> Tensor:
    """x need not be scaled to [0, 1] since sum(RGB_TO_GRAY_WEIGHT) ≈ 1"""
    maybe_batch = 'n' if batched else ''
    return einops.rearrange(
        einops.einsum(x, x.new_tensor(RGB_TO_GRAY_WEIGHT), f'{maybe_batch} c ..., c ... -> {maybe_batch} ...'),
        f'{maybe_batch} ... -> {maybe_batch} 1 ...',
    )


class ScaleIntensity:
    """Multiply intensity by (1 + factor), optionally per-channel."""

    def __call__(self, img: Tensor, factor: list[float] | float) -> Tensor:
        if isinstance(factor, list):
            f = img.new_tensor(factor).view(-1, *([1] * (img.ndim - 1)))
        else:
            f = factor
        return img * (1.0 + f)


class RandScaleIntensity:
    def __init__(
        self,
        prob: float,
        factor_range: tuple[float, float],
        channel_wise: bool = False,
    ) -> None:
        self.prob = prob
        self.factor_range = factor_range
        self.channel_wise = channel_wise
        self._inner = ScaleIntensity()
        self._params: dict | None = None

    def __call__(self, img: Tensor, rng: np.random.Generator) -> Tensor:
        if rng.random() >= self.prob:
            self._params = {'do_transform': False}
            return img
        n = img.shape[0] if self.channel_wise else 1
        factors = rng.uniform(self.factor_range[0], self.factor_range[1], size=n).tolist()
        self._params = {'do_transform': True, 'factors': factors}
        factor = factors if self.channel_wise else factors[0]
        return self._inner(img, factor)

    def get_params(self) -> dict:
        assert self._params is not None, 'call __call__ before get_params'
        return dict(self._params)

    def replay(self, img: Tensor, params: dict) -> Tensor:
        if not params['do_transform']:
            return img
        factors: list[float] = params['factors']
        factor = factors if self.channel_wise else factors[0]
        return self._inner(img, factor)


class AdjustContrast:
    """out = img * factor + mean * (1 - factor), optionally clamped to original range."""

    def __call__(
        self,
        img: Tensor,
        factor: list[float] | float,
        preserve_range: bool,
    ) -> Tensor:
        c = img.shape[0]
        flat = img.view(c, -1)
        if preserve_range:
            min_v = flat.amin(1, keepdim=True)
            max_v = flat.amax(1, keepdim=True)
        mean = flat.mean(1, keepdim=True)
        if isinstance(factor, list):
            f = img.new_tensor(factor).view(c, 1)
        else:
            f = factor
        out = flat * f + mean * (1.0 - f)
        if preserve_range:
            out = out.clamp(min_v, max_v)
        return out.view_as(img)


class RandAdjustContrast:
    def __init__(
        self,
        prob: float,
        contrast_range: tuple[float, float],
        preserve_range: bool = True,
        per_channel: bool = False,
    ) -> None:
        self.prob = prob
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self._inner = AdjustContrast()
        self._params: dict | None = None

    def __call__(self, img: Tensor, rng: np.random.Generator) -> Tensor:
        if rng.random() >= self.prob:
            self._params = {'do_transform': False}
            return img
        n = img.shape[0] if self.per_channel else 1
        factors = rng.uniform(
            self.contrast_range[0], self.contrast_range[1], size=n
        ).tolist()
        self._params = {'do_transform': True, 'factors': factors}
        factor = factors if self.per_channel else factors[0]
        return self._inner(img, factor, self.preserve_range)

    def get_params(self) -> dict:
        assert self._params is not None
        return dict(self._params)

    def replay(self, img: Tensor, params: dict) -> Tensor:
        if not params['do_transform']:
            return img
        factors: list[float] = params['factors']
        factor = factors if self.per_channel else factors[0]
        return self._inner(img, factor, self.preserve_range)


class GammaCorrection:
    """Apply gamma correction per channel, with optional invert and retain_stats."""

    def __call__(
        self,
        img: Tensor,
        gamma: list[float] | float,
        invert: bool,
        retain_stats: bool,
        eps: float = 1e-7,
    ) -> Tensor:
        c = img.shape[0]
        spatial_shape = img.shape[1:]
        flat = img.view(c, -1)
        if invert:
            flat = -flat
        if retain_stats:
            mean = flat.mean(1, keepdim=True)
            std = flat.std(1, keepdim=True, correction=0)
        min_v = flat.amin(1, keepdim=True)
        range_v = flat.amax(1, keepdim=True) - min_v + eps
        flat = (flat - min_v) / range_v
        if isinstance(gamma, list):
            g = flat.new_tensor(gamma).view(c, 1)
        else:
            g = gamma
        flat = flat.pow(g)
        if retain_stats:
            new_mean = flat.mean(1, keepdim=True)
            new_std = flat.std(1, keepdim=True, correction=0)
            flat = (flat - new_mean) * (std / torch.clip(new_std, min=1e-8)) + mean
        else:
            flat = flat * range_v + min_v
        if invert:
            flat = -flat
        return flat.view(c, *spatial_shape)


class RandGammaCorrection:
    def __init__(
        self,
        prob: float,
        gamma_range: tuple[float, float],
        prob_invert: float = 0.0,
        retain_stats: bool = False,
        eps: float = 1e-7,
        per_channel: bool = False,
    ) -> None:
        self.prob = prob
        self.gamma_range = gamma_range
        self.prob_invert = prob_invert
        self.retain_stats = retain_stats
        self.eps = eps
        self.per_channel = per_channel
        self._inner = GammaCorrection()
        self._params: dict | None = None

    def _sample_gamma(self, rng: np.random.Generator) -> float:
        if self.gamma_range[0] < 1 and rng.random() < 0.5:
            return float(rng.uniform(self.gamma_range[0], 1.0))
        else:
            return float(rng.uniform(max(self.gamma_range[0], 1.0), self.gamma_range[1]))

    def __call__(self, img: Tensor, rng: np.random.Generator) -> Tensor:
        if rng.random() >= self.prob:
            self._params = {'do_transform': False}
            return img
        n = img.shape[0] if self.per_channel else 1
        gammas = [self._sample_gamma(rng) for _ in range(n)]
        invert = bool(rng.random() < self.prob_invert)
        self._params = {'do_transform': True, 'gammas': gammas, 'invert': invert}
        gamma = gammas if self.per_channel else gammas[0]
        return self._inner(img, gamma, invert, self.retain_stats, self.eps)

    def get_params(self) -> dict:
        assert self._params is not None
        return dict(self._params)

    def replay(self, img: Tensor, params: dict) -> Tensor:
        if not params['do_transform']:
            return img
        gammas: list[float] = params['gammas']
        gamma = gammas if self.per_channel else gammas[0]
        return self._inner(img, gamma, params['invert'], self.retain_stats, self.eps)


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = round(3.0 * sigma)
    size = 2 * radius + 1
    x = torch.arange(size, dtype=dtype, device=device) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


class GaussianSmooth:
    """3D separable Gaussian smoothing with per-axis sigma."""

    def __call__(self, img: Tensor, sigma: list[float]) -> Tensor:
        # img: (C, D, H, W), sigma: [sigma_d, sigma_h, sigma_w]
        c = img.shape[0]
        x = img.unsqueeze(0)  # (1, C, D, H, W)
        for axis, s in enumerate(sigma):
            if s == 0.0:
                continue
            k = _gaussian_kernel_1d(s, img.device, img.dtype)
            k_len = k.shape[0]
            pad = k_len // 2
            # axis 0->D, 1->H, 2->W corresponds to dim 2,3,4 in (1,C,D,H,W)
            spatial_axis = axis + 2
            ndim = x.ndim  # 5
            # build conv weight: shape (C, 1, 1, ..., k_len, ..., 1)
            weight_shape = [c, 1] + [1] * (ndim - 2)
            weight_shape[spatial_axis] = k_len
            weight = k.view(*([1] * spatial_axis), k_len, *([1] * (ndim - spatial_axis - 1)))
            weight = weight.expand(weight_shape)
            pad_cfg = [0] * (2 * (ndim - 2))
            pad_idx = 2 * (ndim - 2 - 1 - axis)  # F.pad reverses order
            pad_cfg[pad_idx] = pad
            pad_cfg[pad_idx + 1] = pad
            x = F.pad(x, pad_cfg, mode='replicate')
            x = F.conv3d(x, weight, groups=c)
        return x.squeeze(0)


class RandGaussianSmooth:
    def __init__(
        self,
        prob: float,
        sigma_range: tuple[float, float],
        isotropic: bool = False,
    ) -> None:
        self.prob = prob
        self.sigma_range = sigma_range
        self.isotropic = isotropic
        self._inner = GaussianSmooth()
        self._params: dict | None = None

    def __call__(self, img: Tensor, rng: np.random.Generator) -> Tensor:
        if rng.random() >= self.prob:
            self._params = {'do_transform': False}
            return img
        if self.isotropic:
            s = float(rng.uniform(self.sigma_range[0], self.sigma_range[1]))
            sigma = [s, s, s]
        else:
            sigma = rng.uniform(
                self.sigma_range[0], self.sigma_range[1], size=3
            ).tolist()
        self._params = {'do_transform': True, 'sigma': sigma}
        return self._inner(img, sigma)

    def get_params(self) -> dict:
        assert self._params is not None
        return dict(self._params)

    def replay(self, img: Tensor, params: dict) -> Tensor:
        if not params['do_transform']:
            return img
        return self._inner(img, params['sigma'])


class GaussianNoise:
    """Add a pre-generated noise tensor to the image."""

    def __call__(self, img: Tensor, noise: Tensor) -> Tensor:
        return img + noise


class RandGaussianNoise:
    def __init__(
        self,
        prob: float,
        std_range: tuple[float, float],
    ) -> None:
        self.prob = prob
        self.std_range = std_range
        self._inner = GaussianNoise()
        self._params: dict | None = None

    def __call__(self, img: Tensor, rng: np.random.Generator) -> Tensor:
        if rng.random() >= self.prob:
            self._params = {'do_transform': False}
            return img
        std = float(rng.uniform(self.std_range[0], self.std_range[1]))
        # Store a seed for deterministic replay instead of storing the full tensor.
        seed = int(rng.integers(0, 2**31))
        self._params = {'do_transform': True, 'std': std, 'seed': seed}
        noise = self._make_noise(img, std, seed)
        return self._inner(img, noise)

    def _make_noise(self, img: Tensor, std: float, seed: int) -> Tensor:
        gen = torch.Generator(device=img.device)
        gen.manual_seed(seed)
        return torch.randn(img.shape, dtype=img.dtype, device=img.device, generator=gen) * std

    def get_params(self) -> dict:
        assert self._params is not None
        return dict(self._params)

    def replay(self, img: Tensor, params: dict) -> Tensor:
        if not params['do_transform']:
            return img
        noise = self._make_noise(img, params['std'], params['seed'])
        return self._inner(img, noise)
