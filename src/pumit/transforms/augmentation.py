from collections.abc import Callable, Hashable, Mapping

import einops
import numpy as np
import torch

from monai import transforms as mt
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import get_track_meta
from monai.transforms import Randomizable
from monai.utils import convert_to_tensor

from pumit.transforms.intensity import AdjustContrast, GammaCorrection, ScaleIntensity
from pumit.types import tuple2_t


class DictWrapper(mt.MapTransform):
    def __init__(self, keys: KeysCollection, trans: Callable, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.trans = trans

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        data = dict(data)
        for key in self.key_iterator(data):
            data[key] = self.trans(data[key], *args, **kwargs)
        return data


class RandDictWrapper(DictWrapper, mt.Randomizable):
    def __init__(self, keys: KeysCollection, trans: Callable, allow_missing_keys: bool = False):
        DictWrapper.__init__(self, keys, trans, allow_missing_keys)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        if isinstance(self.trans, mt.Randomizable):
            self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data=None):
        pass

    def __call__(self, data: Mapping[Hashable, ...], *args, **kwargs):
        return super().__call__(data, *args, **kwargs, randomize=True)


class ClampIntensityD(mt.MapTransform):
    def __init__(self, keys: KeysCollection, min_v: float = 0., max_v: float = 1., allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        data = dict(data)
        for k in self.key_iterator(data):
            data[k] = data[k].clamp(self.min_v, self.max_v)
        return data


class RandAdjustContrast(mt.RandomizableTransform):
    def __init__(self, prob: float, contrast_range: tuple2_t[float], preserve_intensity_range: bool = True, per_channel: bool = False):
        mt.RandomizableTransform.__init__(self, prob)
        self.contrast_range = contrast_range
        self.preserve_intensity_range = preserve_intensity_range
        self.per_channel = per_channel

    def randomize(self, num_channels: int):
        super().randomize(None)
        if not self._do_transform:
            return
        n = num_channels if self.per_channel else 1
        self.factor = self.R.uniform(*self.contrast_range, size=(n, 1))

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True):
        img_t: torch.Tensor = convert_to_tensor(img, track_meta=get_track_meta())
        num_channels = img_t.shape[0]
        if randomize:
            self.randomize(num_channels)
        if not self._do_transform:
            return img_t
        spatial_size = img_t.shape[1:]
        img_t = einops.rearrange(img_t, 'c ... -> c (...)')
        if self.preserve_intensity_range:
            min_v = img_t.amin(1, True)
            max_v = img_t.amax(1, True)
        mean = img_t.mean(1, True)
        factor = img_t.new_tensor(self.factor)
        ret = img_t * factor + mean * (1 - factor)
        if self.preserve_intensity_range:
            ret.clamp_(min_v, max_v)
        return ret.view(num_channels, *spatial_size)


class RandGammaCorrection(mt.RandomizableTransform):
    def __init__(self, prob: float, gamma_range: tuple2_t[float], prob_invert: float, retain_stats: bool, eps: float = 1e-7, per_channel: bool = False):
        mt.RandomizableTransform.__init__(self, prob)
        self.gamma_range = gamma_range
        self.prob_invert = prob_invert
        self.retain_stats = retain_stats
        self.eps = eps
        self.per_channel = per_channel

    def randomize(self, num_channels: int):
        super().randomize(None)
        if not self._do_transform:
            return
        n = num_channels if self.per_channel else 1
        self.gamma = np.empty((n, 1))
        for i in range(n):
            if self.gamma_range[0] < 1 and self.R.uniform() < 0.5:
                self.gamma[i] = self.R.uniform(self.gamma_range[0], 1)
            else:
                self.gamma[i] = self.R.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
        self.invert = self.R.uniform() < self.prob_invert

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True):
        img_t: torch.Tensor = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize(img_t.shape[0])
        if not self._do_transform:
            return img_t
        spatial_shape = img_t.shape[1:]
        img_t = einops.rearrange(img_t, 'c ... -> c (...)')
        if self.invert:
            img_t = -img_t
        if self.retain_stats:
            mean = img_t.mean(1, True)
            std = img_t.std(1, keepdim=True, correction=0)
        min_v = img_t.amin(1, True)
        range_v = img_t.amax(1, True) - min_v + self.eps
        img_t = (img_t - min_v) / range_v
        img_t = img_t.pow(img_t.new_tensor(self.gamma))
        if self.retain_stats:
            new_mean = img_t.mean(1, True)
            new_std = img_t.std(1, keepdim=True, correction=0)
            img_t = (img_t - new_mean) * (std / torch.clip(new_std, 1e-8)) + mean
        else:
            img_t = img_t * range_v + min_v
        if self.invert:
            img_t = -img_t
        return img_t.view(img_t.shape[0], *spatial_shape)


# ---------------------------------------------------------------------------
# ScaleIntensityTransform
# ---------------------------------------------------------------------------


class ScaleIntensityTransform:
    """Probabilistic per-channel intensity scaling."""

    def __init__(self, *, prob: float, factor_range: tuple[float, float], channel_wise: bool = True):
        self.prob = prob
        self.factor_range = factor_range
        self.channel_wise = channel_wise
        self._inner = ScaleIntensity()

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict:
        if rng.random() >= self.prob:
            return {'enabled': False}
        n = state.get('num_channels', 1) if self.channel_wise else 1
        factors = rng.uniform(self.factor_range[0], self.factor_range[1], size=n).tolist()
        return {'enabled': True, 'factors': factors}

    def __call__(self, data: dict, *, enabled: bool, factors: list[float] | None = None) -> dict:
        data = dict(data)
        if not enabled:
            return data
        img = data['img']
        if hasattr(img, 'as_tensor'):
            img = img.as_tensor()
        img = self._inner(img, factors)
        data['img'] = img.clamp(0, 1)
        return data


# ---------------------------------------------------------------------------
# AdjustContrastTransform
# ---------------------------------------------------------------------------


class AdjustContrastTransform:
    """Probabilistic contrast adjustment."""

    def __init__(self, *, prob: float, contrast_range: tuple[float, float], preserve_range: bool = True):
        self.prob = prob
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self._inner = AdjustContrast()

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict:
        if rng.random() >= self.prob:
            return {'enabled': False}
        factor = float(rng.uniform(self.contrast_range[0], self.contrast_range[1]))
        return {'enabled': True, 'factor': factor}

    def __call__(self, data: dict, *, enabled: bool, factor: float | None = None) -> dict:
        data = dict(data)
        if not enabled:
            return data
        img = data['img']
        if hasattr(img, 'as_tensor'):
            img = img.as_tensor()
        img = self._inner(img, factor, self.preserve_range)
        data['img'] = img.clamp(0, 1)
        return data


# ---------------------------------------------------------------------------
# GammaCorrectionTransform
# ---------------------------------------------------------------------------


class GammaCorrectionTransform:
    """Probabilistic gamma correction with optional inversion."""

    def __init__(self, *, prob: float, gamma_range: tuple[float, float], prob_invert: float = 0.0):
        self.prob = prob
        self.gamma_range = gamma_range
        self.prob_invert = prob_invert
        self._inner = GammaCorrection()

    def _sample_gamma(self, rng: np.random.Generator) -> float:
        if self.gamma_range[0] < 1 and rng.random() < 0.5:
            return float(rng.uniform(self.gamma_range[0], 1.0))
        else:
            return float(rng.uniform(max(self.gamma_range[0], 1.0), self.gamma_range[1]))

    def sample_params(self, state: dict, rng: np.random.Generator) -> dict:
        if rng.random() >= self.prob:
            return {'enabled': False}
        n = state.get('num_channels', 1)
        gammas = [self._sample_gamma(rng) for _ in range(n)]
        invert = bool(rng.random() < self.prob_invert)
        return {'enabled': True, 'gammas': gammas, 'invert': invert}

    def __call__(
        self, data: dict, *, enabled: bool, gammas: list[float] | None = None, invert: bool = False,
    ) -> dict:
        data = dict(data)
        if not enabled:
            return data
        img = data['img']
        if hasattr(img, 'as_tensor'):
            img = img.as_tensor()
        img = self._inner(img, gammas[0], invert, False)
        data['img'] = img.clamp(0, 1)
        return data


# ---------------------------------------------------------------------------
# NormalizeTransform
# ---------------------------------------------------------------------------


class NormalizeTransform:
    """Deterministic: as_tensor, normalize [0,1] -> [-1,1], ensure RGB."""

    def __call__(self, data: dict, **kwargs) -> dict:
        from pumit.transforms.intensity import ensure_rgb
        data = dict(data)
        img = data['img']
        if hasattr(img, 'as_tensor'):
            img = img.as_tensor()
        img = img * 2 - 1
        img, _ = ensure_rgb(img)
        data['img'] = img
        return data
