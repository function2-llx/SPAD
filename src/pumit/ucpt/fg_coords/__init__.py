# src/pumit/ucpt/fg_coords/__init__.py
"""Foreground-coordinate sidecar: reader + crop placement helper.

The precompute entrypoint lives in __main__.py (python -m pumit.ucpt.fg_coords).
This module holds the generation-time reader (FgCoordCache) and the pure
placement helper, kept dependency-light for the generation hot path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson


FG_COORD_VERSION = 3
FG_COORD_CAP = 1024


def place_crop_start(
    voxel: np.ndarray,
    load_size: np.ndarray,
    shape: np.ndarray,
    crop_size: np.ndarray,
    output_to_source: np.ndarray,
    rng: np.random.Generator,
    *,
    jitter_fraction: float,
) -> np.ndarray:
    """Place a crop so ``voxel`` has a sampled post-affine in-plane offset.

    Jitter is sampled in the final crop coordinate system, then mapped to the source grid by ``output_to_source``.
    Consequently scale, rotation, and flip do not change its distribution in the model-visible crop. The final depth offset is
    zero; an oblique transform may still require moving the source-space depth center. Boundary clamping truncates the requested
    distribution.

    Args:
        voxel: (3,) int array, (d, h, w) in the raw voxel grid.
        load_size: (3,) int array, source region loaded for affine resampling.
        shape: (3,) int array, volume spatial shape (D, H, W).
        crop_size: (3,) int array, final model-visible crop size.
        output_to_source: (3, 3) linear map from centered final-crop coordinates to source coordinates.
        rng: numpy Generator.
        jitter_fraction: Maximum per-axis in-plane offset as a fraction of the final crop extent.

    Returns:
        (3,) int64 array, load_slice_start.
    """
    crop_size = np.asarray(crop_size)
    output_to_source = np.asarray(output_to_source, dtype=np.float64)
    assert np.all(load_size <= shape), f'load_size {load_size} exceeds shape {shape}'
    assert crop_size.shape == (3,) and np.all(crop_size > 0), f'invalid crop_size {crop_size}'
    assert output_to_source.shape == (3, 3) and np.isfinite(output_to_source).all(), (
        f'invalid output_to_source {output_to_source}'
    )
    assert 0 <= jitter_fraction < 0.5, f'invalid jitter_fraction {jitter_fraction}'

    output_offset = np.zeros(3, dtype=np.float64)
    if jitter_fraction > 0:
        radius = crop_size[1:] * jitter_fraction
        output_offset[1:] = rng.uniform(-radius, radius)
    center = np.rint(voxel - output_to_source @ output_offset).astype(np.int64)
    start = center - load_size // 2
    start = np.clip(start, 0, shape - load_size)
    return start


class FgCoordCache:
    """Reads the per-dataset fg-coord sidecar.

    Loads every listed dataset's index.json fully into one RAM dict, and
    mmaps each coords.npy (read-only; fork-safe, shared page cache when built
    pre-fork in the parent). Coordinates are int16 (d, h, w) in the raw voxel
    grid.

    ``index.json`` carries an explicit format version and coordinate cap so a
    stale sidecar cannot be consumed silently.
    """

    def __init__(self, data_root: Path | str, datasets: list[str]):
        self.data_root = Path(data_root)
        self._index: dict[
            tuple[str, str, str, str],
            tuple[int, int, np.ndarray, np.ndarray],
        ] = {}
        self._coords: dict[str, np.ndarray] = {}
        for ds in datasets:
            d = self.data_root / ds / 'fg_coords'
            index = orjson.loads((d / 'index.json').read_bytes())
            if not isinstance(index, dict) or set(index) != {'version', 'coord_cap', 'records'}:
                raise ValueError(f'{ds}: invalid fg-coord index envelope')
            if index['version'] != FG_COORD_VERSION:
                raise ValueError(
                    f'{ds}: fg-coord version {index["version"]!r}, expected {FG_COORD_VERSION}; '
                    'rebuild the sidecar'
                )
            if index['coord_cap'] != FG_COORD_CAP:
                raise ValueError(
                    f'{ds}: fg-coord cap {index["coord_cap"]!r}, expected {FG_COORD_CAP}; '
                    'rebuild the sidecar'
                )
            records = index['records']
            expected_offset = 0
            for key, source, cls, off, cnt, bbox_start, bbox_stop in records:
                pair = (ds, key, source, cls)
                if pair in self._index:
                    raise ValueError(f'{ds}: duplicate fg-coord pair {pair[1:]!r}')
                if off != expected_offset:
                    raise ValueError(
                        f'{ds}: non-contiguous fg-coord offset {off} for {pair[1:]!r}, '
                        f'expected {expected_offset}'
                    )
                if cnt <= 0 or cnt > FG_COORD_CAP:
                    raise ValueError(f'{ds}: invalid fg-coord count {cnt} for {pair[1:]!r}')
                bbox_start_arr = np.asarray(bbox_start, dtype=np.int64)
                bbox_stop_arr = np.asarray(bbox_stop, dtype=np.int64)
                if (
                    bbox_start_arr.shape != (3,)
                    or bbox_stop_arr.shape != (3,)
                    or np.any(bbox_start_arr < 0)
                    or np.any(bbox_stop_arr <= bbox_start_arr)
                ):
                    raise ValueError(
                        f'{ds}: invalid foreground bbox {bbox_start!r}:{bbox_stop!r} '
                        f'for {pair[1:]!r}'
                    )
                self._index[pair] = (off, cnt, bbox_start_arr, bbox_stop_arr)
                expected_offset += cnt
            coords = np.load(d / 'coords.npy', mmap_mode='r')
            if coords.dtype != np.int16 or coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f'{ds}: invalid fg-coord array shape/dtype {coords.shape}/{coords.dtype}')
            expected_rows = sum(record[4] for record in records)
            if len(coords) != expected_rows:
                raise ValueError(f'{ds}: index describes {expected_rows} rows but coords.npy has {len(coords)}')
            self._coords[ds] = coords

    def has_pair(self, dataset: str, key: str, source: str, cls: str) -> bool:
        return (dataset, key, source, cls) in self._index

    def _rows(self, dataset: str, key: str, source: str, cls: str) -> np.ndarray:
        off, cnt, _, _ = self._index[(dataset, key, source, cls)]
        return self._coords[dataset][off:off + cnt]

    def bbox(
        self,
        dataset: str,
        key: str,
        source: str,
        cls: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact half-open foreground bbox in raw voxel coordinates."""
        _, _, start, stop = self._index[(dataset, key, source, cls)]
        return start, stop

    def sample_voxel(
        self, dataset: str, key: str, source: str, cls: str, rng: np.random.Generator,
    ) -> np.ndarray:
        rows = self._rows(dataset, key, source, cls)
        return np.asarray(rows[rng.integers(len(rows))], dtype=np.int64)
