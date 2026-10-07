"""Generic transform pipeline with Protocol-based dispatch.

Transforms follow one of two protocols:
- Transform: any callable (data, **kwargs) -> data
- RandomizableTransform: additionally implements sample_params(state, rng) -> dict
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class Transform(Protocol):
    def __call__(self, data: dict, **kwargs) -> dict: ...


@runtime_checkable
class RandomizableTransform(Transform, Protocol):
    def sample_params(self, state: dict, rng: np.random.Generator) -> dict | None: ...


class TransformPipeline:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def sample_params(self, state: dict, rng: np.random.Generator) -> list | None:
        all_params = []
        for t in self.transforms:
            if isinstance(t, RandomizableTransform):
                p = t.sample_params(state, rng)
                if p is None:
                    return None
            else:
                p = {}
            all_params.append(p)
        return all_params

    def replay(self, data: dict, params: list) -> dict:
        assert len(params) == len(self.transforms)
        for t, p in zip(self.transforms, params):
            data = t(data, **p)
        return data
