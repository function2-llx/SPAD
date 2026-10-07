"""Prepare normalized Pass 2 inputs and route augmentation by Pass 1 cache dtype."""

from functools import cache
from pathlib import Path

import orjson
from torch import Tensor


def normalize_input(image: Tensor, scheme: str, *, batched: bool = False) -> Tensor:
    """Preserve Pass 2 intensities and expand single-channel medical inputs."""
    if scheme not in ('rgb', 'zscore'):
        raise ValueError(f'unknown UCPT input normalization scheme {scheme!r}')
    image = image.float()
    channel_axis = int(batched)
    channels = image.shape[channel_axis]
    if channels == 1 and scheme == 'zscore':
        image = image.repeat_interleave(3, dim=channel_axis)
    elif channels != 3:
        raise ValueError(f'expected three channels or single-channel zscore input, got {channels} ({scheme})')
    return image.contiguous()


class InputNormalizer:
    """Lazily retain only per-record schemes from sibling fingerprint metadata."""

    def __init__(self, data_root: Path | str):
        # Keep the logical parent: preprocess may be a symlink to a different storage root.
        self.fingerprint_root = Path(data_root).parent / 'fingerprint'

    @cache
    def _dataset_schemes(self, dataset: str) -> dict[str, str]:
        records = orjson.loads((self.fingerprint_root / dataset / 'samples.json').read_bytes())
        schemes = {}
        for key, record in records.items():
            dtype = record['image_dtype']
            cache_dtype = dtype['cache'] if isinstance(dtype, dict) else dtype
            schemes[key] = 'rgb' if cache_dtype == 'uint8' else 'zscore'
        return schemes

    def scheme(self, image_path: Path | str) -> str:
        path = Path(image_path)
        # VerSe legacy stream names link to the renamed fingerprint records.
        if path.parent.parent.name == 'VerSe' and path.is_symlink():
            path = path.resolve()
        return self.scheme_for(path.parent.parent.name, path.stem)

    def scheme_for(self, dataset: str, key: str) -> str:
        return self._dataset_schemes(dataset)[key]

    def __call__(self, data: dict, **kwargs) -> dict:
        return {**data, 'img': normalize_input(data['img'], data['input_scheme'])}
