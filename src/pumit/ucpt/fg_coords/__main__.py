# src/pumit/ucpt/fg_coords/__main__.py
"""Precompute the fg-coord sidecar. Run: python -m pumit.ucpt.fg_coords

Scans every dataset's meta.json manifest, resolves each positive (key, source,
class) to its mask file, decodes, subsamples foreground voxels to cap, and
writes a per-dataset sidecar (coords.npy + index.json) atomically.

Parallelism is at the PAIR level, flat across all datasets, not per dataset: one
mega-dataset (AbdomenAtlas3.0 alone is ~60% of the ~612K-pair corpus) would
otherwise serialize on a single worker while the rest sit idle. Phase 1 resolves
every pair (serial, cheap: no decode); phase 2 decodes all pairs in one flat
ProcessPool; phase 3 writes each dataset's sidecar. Per dataset the write is
all-or-nothing: any pair failure aborts that dataset (no partial sidecar),
because a manifest positive with no index entry would KeyError at generation.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import zstandard as zstd

from pumit.segmentation_mask import unpack_binary_mask

from pumit.data import DATA_ROOT
from pumit.ucpt.fg_coords import FG_COORD_CAP, FG_COORD_VERSION

CAP = FG_COORD_CAP


def resolve_mask_file(label_dir: Path, raw_cls: str, files: list[Path] | None = None) -> Path:
    """Resolve a raw class name to its mask file in label_dir.

    Mirrors _load_mask: sanitize '/'->'_', accept a bare stem (== query) or a
    prefixed one (endswith '__<query>'). Asserts EXACTLY ONE match. `files` lets
    the caller pass a pre-listed directory so a label_dir with many classes is
    listed once instead of once per class.
    """
    stem = raw_cls.replace('/', '_')
    if files is None:
        files = list(label_dir.iterdir())
    matches = []
    for f in files:
        if not f.name.endswith('.npy.zst'):
            continue
        base = f.name[:-len('.npy.zst')]
        if base == stem or base.endswith(f'__{stem}'):
            matches.append(f)
    assert len(matches) == 1, f'expected 1 match for {raw_cls!r} in {label_dir}, got {matches}'
    return matches[0]


def assert_injective(resolved: dict[tuple[str, str, str], Path]) -> None:
    """No two distinct (key, source, class) queries resolve to the same file."""
    seen: dict[Path, tuple] = {}
    for query, path in resolved.items():
        if path in seen:
            raise AssertionError(
                f'non-injective mask resolution: {query} and {seen[path]} '
                f'both resolve to {path}'
            )
        seen[path] = query


def subsample_coords(mask: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Return up to `cap` foreground (d,h,w) coords as int16, subsampled uniformly."""
    flat = np.flatnonzero(mask)
    if len(flat) == 0:
        raise ValueError('positive mask is empty')
    if len(flat) > cap:
        flat = rng.choice(flat, cap, replace=False)
    coords = np.stack(np.unravel_index(flat, mask.shape), axis=1)
    return coords.astype(np.int16)


def foreground_bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact half-open bbox of a nonempty binary mask."""
    occupied = np.nonzero(mask)
    if not occupied[0].size:
        raise ValueError('positive mask is empty')
    start = np.array([axis.min() for axis in occupied], dtype=np.int64)
    stop = np.array([axis.max() + 1 for axis in occupied], dtype=np.int64)
    return start, stop


def _decode_mask(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    dctx = zstd.ZstdDecompressor()
    packed = np.load(io.BytesIO(dctx.decompress(path.read_bytes())))
    return unpack_binary_mask(packed, shape).astype(bool)


def _resolve_dataset(
    dataset: str, data_root: Path, *, force: bool = False,
) -> tuple[dict[tuple[str, str, str], Path], dict[str, tuple[int, int, int]]] | str:
    """Resolve every positive (key, source, class) of one dataset to its mask
    file (serial, no decode). Returns (resolved, shapes), or 'skip' when the
    sidecar already exists.

    `resolved` preserves manifest insertion order, which fixes the coords.npy
    row order at write time (so generation's `rng.integers(count)` is stable).
    label_classes is null (not just absent) for unlabeled images inside a mixed
    labeled/unlabeled dataset; `or {}` handles both.
    """
    out_dir = data_root / dataset / 'fg_coords'
    if not force and (out_dir / 'coords.npy').exists() and (out_dir / 'index.json').exists():
        return 'skip'

    meta = orjson.loads((data_root / dataset / 'meta.json').read_bytes())
    resolved: dict[tuple[str, str, str], Path] = {}
    shapes: dict[str, tuple[int, int, int]] = {}
    for key, entry in meta.items():
        shape = tuple(int(x) for x in entry['shape'][1:])  # strip channel
        assert max(shape) < 2 ** 15, f'{dataset}/{key} dim exceeds int16: {shape}'
        shapes[key] = shape
        for source, info in (entry.get('label_classes') or {}).items():
            positives = info.get('positive', [])
            if not positives:
                continue
            label_dir = data_root / dataset / 'labels' / key / source
            files = list(label_dir.iterdir())  # list once, match all classes against it
            for cls in positives:
                resolved[(key, source, cls)] = resolve_mask_file(label_dir, cls, files)
    assert_injective(resolved)
    return resolved, shapes


def _decode_pair(
    arg: tuple[str, str, str, str, str, tuple[int, int, int], int],
) -> tuple[
    str,
    tuple[str, str, str],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
]:
    """Decode+subsample one pair. Catches its own failure so a flat map never
    loses a whole chunk's healthy siblings; the caller aborts the failed pair's
    dataset. Returns (dataset, pair_key, coords_or_None, traceback_or_None)."""
    dataset, key, source, cls, path_str, shape, seed = arg
    try:
        mask = _decode_mask(Path(path_str), shape)
        coords = subsample_coords(mask, CAP, np.random.default_rng(seed))
        bbox_start, bbox_stop = foreground_bbox(mask)
        return dataset, (key, source, cls), coords, bbox_start, bbox_stop, None
    except Exception:
        return dataset, (key, source, cls), None, None, None, traceback.format_exc()


def _write_sidecar(out_dir: Path, arr: np.ndarray, records: list) -> None:
    """Atomic per-dataset write. np.save appends '.npy' unless the name already
    ends in '.npy', so the tmp name ends in '.npy' to keep the written path ==
    the os.replace source. The index is a flat list of
    ``[key, source, cls, offset, count, bbox_start, bbox_stop]`` records serialized with orjson. The dataset is
    implicit in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_coords = out_dir / 'coords.tmp.npy'
    tmp_index = out_dir / 'index.tmp.json'
    np.save(tmp_coords, arr)
    tmp_index.write_bytes(orjson.dumps({
        'version': FG_COORD_VERSION,
        'coord_cap': FG_COORD_CAP,
        'records': records,
    }))
    os.replace(tmp_coords, out_dir / 'coords.npy')
    os.replace(tmp_index, out_dir / 'index.json')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=DATA_ROOT)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--force', action='store_true', help='Rebuild existing sidecars for the selected datasets')
    args = ap.parse_args()

    datasets = args.datasets or sorted(
        p.parent.name for p in args.data_root.glob('*/meta.json')
    )

    # Phase 1: resolve every pair (serial, cheap).
    plan: dict[str, tuple[dict, dict]] = {}
    failures: dict[str, str] = {}
    for ds in datasets:
        try:
            r = _resolve_dataset(ds, args.data_root, force=args.force)
        except Exception:
            failures[ds] = traceback.format_exc()
            print(f'[fg_coords] FAILED (resolve) {ds}', file=sys.stderr)
            continue
        if r == 'skip':
            print(f'[fg_coords] {ds}: skip', flush=True)
        else:
            plan[ds] = r

    # Phase 2: flat pair-level decode across all datasets.
    work: list[tuple] = []
    seed = 0
    for ds, (resolved, shapes) in plan.items():
        for (key, source, cls), path in resolved.items():
            work.append((ds, key, source, cls, str(path), shapes[key], seed))
            seed += 1
    print(f'[fg_coords] decoding {len(work)} pairs across {len(plan)} datasets '
          f'on {args.workers} workers', flush=True)

    summaries_by_ds: dict[
        str,
        dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = defaultdict(dict)
    if work:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for ds, pair_key, coords, bbox_start, bbox_stop, err in ex.map(
                _decode_pair,
                work,
                chunksize=16,
            ):
                if err is not None:
                    failures.setdefault(ds, err)
                else:
                    assert coords is not None and bbox_start is not None and bbox_stop is not None
                    summaries_by_ds[ds][pair_key] = coords, bbox_start, bbox_stop

    # Phase 3: write each dataset's sidecar (all-or-nothing).
    for ds, (resolved, shapes) in plan.items():
        if ds in failures:
            print(f'[fg_coords] FAILED (decode) {ds}', file=sys.stderr)
            continue
        pairs = summaries_by_ds[ds]
        records: list = []
        blocks: list[np.ndarray] = []
        off = 0
        for pk in resolved:  # resolved preserves manifest order -> fixed row order
            coords, bbox_start, bbox_stop = pairs[pk]
            key, source, cls = pk
            records.append([
                key,
                source,
                cls,
                off,
                len(coords),
                bbox_start.tolist(),
                bbox_stop.tolist(),
            ])
            blocks.append(coords)
            off += len(coords)
        arr = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 3), np.int16)
        _write_sidecar(args.data_root / ds / 'fg_coords', arr, records)
        print(f'[fg_coords] {ds}: {len(records)} pairs', flush=True)

    if failures:
        print(f'\n{len(failures)} dataset(s) failed:', file=sys.stderr)
        for ds, tb in failures.items():
            print(f'--- {ds} ---\n{tb}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
