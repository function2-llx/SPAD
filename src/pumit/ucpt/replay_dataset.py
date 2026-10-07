"""Map-style UCPT replay dataset backed by stream and latent shards.

The full patch array stores labeled and unlabeled samples once. SSL metadata uses unlabeled-local offsets, while
student, teacher, and segmentation gather indices address the full array. Latent shards contain unlabeled samples only.

Each rank owns disjoint virtual lanes and lazily caches the corresponding stream/latent shards.
"""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
import json
import os
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import yaml
from safetensors import safe_open
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from pumit.transforms.patchify import patchify
from pumit.transforms.pipeline import TransformPipeline
from pumit.text_prompt import SegmentationPromptResolver

# Collation and segmentation helpers shared with the training pipeline.
from pumit.ucpt.batch import UCPTBatch
from pumit.ucpt.input import InputNormalizer
from pumit.ucpt.mask_store import MASK_REF_KEY, MASK_STORAGE, mask_shard_path, read_mask_frame
from pumit.ucpt.masking import multi_block_mask, random_mask
from pumit.ucpt.seg.data import build_seg_payload
from pumit.ucpt.seg.text_encoding import TextEmbeddingCache
from pumit.ucpt.stream.manifest import load_manifest, sha256_file
from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT, canonicalize_sample_metadata

# ``(shard_id, 0)`` and ``(shard_id, 1)`` address the labeled and unlabeled generation streams.
# Keep replay-time augmentation in a disjoint SeedSequence subtree.
_BATCH_RNG_DOMAIN = 2
_CONCATENATED_STREAM_CONTRACT = 'concatenated-stream-v1'
_FIXED_SOURCE_STATS_CONTRACT = 'fixed-source-v1'


def _require_sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _validate_composition_structure(meta: dict) -> tuple[dict, list[dict]]:
    composition = meta.get('composition')
    if not isinstance(composition, dict):
        raise ValueError('v4 fixed-source stats require stream composition metadata')
    if composition.get('contract') != _CONCATENATED_STREAM_CONTRACT:
        raise ValueError('v4 stream has an unsupported composition contract')
    replay_seed = composition.get('replay_seed')
    if (
        isinstance(replay_seed, bool)
        or not isinstance(replay_seed, int)
        or replay_seed != meta.get('seed')
    ):
        raise ValueError('v4 concatenated stream replay_seed differs from stream seed')
    n_shards = meta.get('n_shards')
    if isinstance(n_shards, bool) or not isinstance(n_shards, int) or n_shards < 2:
        raise ValueError('v4 concatenated stream has an invalid shard count')
    segments = composition.get('segments')
    if not isinstance(segments, list) or len(segments) != 2:
        raise ValueError('v4 concatenated stream must contain exactly two segments')

    target_cursor = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f'v4 concatenated stream segment {index} must be an object')
        ranges = {}
        for key in (
            'target_shard_start',
            'target_shard_stop',
            'source_shard_start',
            'source_shard_stop',
        ):
            value = segment.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'v4 concatenated stream segment {index} has invalid {key}')
            ranges[key] = value
        target_start = ranges['target_shard_start']
        target_stop = ranges['target_shard_stop']
        source_start = ranges['source_shard_start']
        source_stop = ranges['source_shard_stop']
        if (
            target_start != target_cursor
            or target_stop <= target_start
            or source_stop <= source_start
            or target_stop - target_start != source_stop - source_start
        ):
            raise ValueError(f'v4 concatenated stream segment {index} has an invalid mapping')
        if not isinstance(segment.get('source_stream'), str) or not segment['source_stream']:
            raise ValueError(
                f'v4 concatenated stream segment {index} has invalid source_stream'
            )
        if (
            not isinstance(segment.get('source_fingerprint'), str)
            or not segment['source_fingerprint']
        ):
            raise ValueError(
                f'v4 concatenated stream segment {index} has invalid source_fingerprint'
            )
        _require_sha256(
            segment.get('source_manifest_sha256'),
            f'v4 concatenated stream segment {index} source_manifest_sha256',
        )
        generation_seed = segment.get('generation_seed')
        if (
            isinstance(generation_seed, bool)
            or not isinstance(generation_seed, int)
            or generation_seed < 0
        ):
            raise ValueError(
                f'v4 concatenated stream segment {index} has invalid generation_seed'
            )
        target_cursor = target_stop
    if target_cursor != n_shards:
        raise ValueError('v4 concatenated stream segments do not cover every shard')
    _require_sha256(
        segments[0].get('source_ready_sha256'),
        'v4 concatenated stream prefix source_ready_sha256',
    )
    return composition, segments


def _validate_fixed_source_stats(
    meta: dict,
    ready: dict,
    *,
    stats_sha256: str,
    stats_count: int,
) -> bool:
    """Validate statistics retained from a stream's fixed normalization source."""
    composition = meta.get('composition')
    normalization = meta.get('normalization')
    stats_contract = ready.get('stats_contract')
    if composition is None and normalization is None and stats_contract is None:
        return False
    if normalization is not None:
        segments = None
    else:
        composition, segments = _validate_composition_structure(meta)
        normalization = composition.get('normalization')
    if stats_contract != _FIXED_SOURCE_STATS_CONTRACT:
        raise ValueError('v4 concatenated stream must declare fixed-source-v1 stats')

    if not isinstance(normalization, dict):
        raise ValueError('v4 concatenated stream lacks normalization provenance')
    if normalization.get('contract') != _FIXED_SOURCE_STATS_CONTRACT:
        raise ValueError('v4 composition has an invalid normalization contract')
    source_stream = normalization.get('source_stream')
    source_fingerprint = normalization.get('source_fingerprint')
    source_count = normalization.get('source_stats_count')
    if not isinstance(source_stream, str) or not source_stream:
        raise ValueError('v4 fixed-source normalization has an invalid source_stream')
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError('v4 fixed-source normalization has an invalid source_fingerprint')
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 1:
        raise ValueError('v4 fixed-source normalization has an invalid source_stats_count')
    source_manifest_sha256 = _require_sha256(
        normalization.get('source_manifest_sha256'),
        'v4 fixed-source normalization source_manifest_sha256',
    )
    source_stats_sha256 = _require_sha256(
        normalization.get('source_stats_sha256'),
        'v4 fixed-source normalization source_stats_sha256',
    )

    if segments is not None:
        prefix = segments[0]
        if (
            not isinstance(prefix.get('source_stream'), str)
            or Path(prefix['source_stream']).resolve() != Path(source_stream).resolve()
            or prefix.get('source_fingerprint') != source_fingerprint
            or prefix.get('source_manifest_sha256') != source_manifest_sha256
        ):
            raise ValueError('v4 fixed-source normalization identity differs from prefix segment')

    if stats_sha256 != source_stats_sha256:
        raise ValueError('v4 fixed-source stats hash differs from composition provenance')
    if stats_count != source_count:
        raise ValueError(
            f'v4 fixed-source stats count {stats_count} differs from source count {source_count}'
        )
    if ready.get('stats_source_fingerprint') != source_fingerprint:
        raise ValueError('v4 READY stats source fingerprint differs from composition')
    if ready.get('stats_source_count') != source_count:
        raise ValueError('v4 READY stats source count differs from composition')
    return True


def _validate_v4_ready(
    stream_dir: Path,
    latent_dir: Path,
    meta: dict,
    *,
    verify_content_hashes: bool = True,
    validation_rank: int = 0,
    validation_world_size: int = 1,
) -> None:
    """Validate the cheap training-time projection of a finalized v4 READY proof."""
    if validation_world_size < 1:
        raise ValueError(
            f'validation_world_size must be positive, got {validation_world_size}'
        )
    if not 0 <= validation_rank < validation_world_size:
        raise ValueError(
            f'validation_rank must be in [0, {validation_world_size}), '
            f'got {validation_rank}'
        )
    ready_path = stream_dir / 'READY.json'
    if not ready_path.is_file():
        raise RuntimeError(f'ucpt-stream-v4 is not READY: {ready_path}')
    try:
        ready = json.loads(ready_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f'invalid v4 READY proof: {ready_path}') from error
    if not isinstance(ready, dict):
        raise ValueError(f'invalid v4 READY proof: {ready_path}')

    manifest_path = stream_dir / 'manifest.jsonl'
    manifest_sha256 = meta.get('manifest_sha256')
    if not isinstance(manifest_sha256, str) or sha256_file(manifest_path) != manifest_sha256:
        raise ValueError('v4 manifest hash differs from meta.yaml')
    if ready.get('manifest_sha256') != manifest_sha256:
        raise ValueError('v4 READY manifest hash differs from meta.yaml')
    if ready.get('mask_storage') != MASK_STORAGE:
        raise ValueError('v4 READY has invalid mask storage')
    metadata_contract = meta.get('sample_metadata_contract')
    if metadata_contract != SAMPLE_METADATA_CONTRACT:
        raise ValueError('v4 meta.yaml has an invalid sample metadata contract')
    if ready.get('sample_metadata_contract') != metadata_contract:
        raise ValueError('v4 READY sample metadata contract differs from meta.yaml')
    rows = load_manifest(stream_dir)
    n_shards = meta.get('n_shards')
    if len(rows) != n_shards:
        raise ValueError(f'v4 manifest contains {len(rows)} shards, meta.yaml declares {n_shards!r}')
    latent_shard_sha256 = ready.get('latent_shard_sha256')
    if not isinstance(latent_shard_sha256, list) or len(latent_shard_sha256) != len(rows):
        raise ValueError('v4 READY has an invalid latent shard hash inventory')

    logical_latent_rows = 0
    latent_paths = []
    for index, row in enumerate(rows):
        shard_id = row['shard_id']
        if row.get('mask_storage') != MASK_STORAGE:
            raise ValueError(f'v4 manifest shard {shard_id} has invalid mask storage')
        if row.get('sample_metadata_contract') != metadata_contract:
            raise ValueError(
                f'v4 manifest shard {shard_id} has an invalid sample metadata contract'
            )
        expected_size = row.get('mask_storage_bytes')
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f'v4 manifest shard {shard_id} has invalid mask sidecar size')
        logical_rows = row.get('logical_latent_rows')
        if isinstance(logical_rows, bool) or not isinstance(logical_rows, int) or logical_rows < 0:
            raise ValueError(f'v4 manifest shard {shard_id} has invalid logical latent rows')
        logical_latent_rows += logical_rows
        expected_sha256 = latent_shard_sha256[index]
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in '0123456789abcdef' for character in expected_sha256)
        ):
            raise ValueError(f'v4 READY has an invalid latent shard hash for shard {shard_id}')
        if index % validation_world_size != validation_rank:
            continue

        mask_path = mask_shard_path(stream_dir, shard_id)
        try:
            actual_size = mask_path.stat().st_size
        except FileNotFoundError:
            raise FileNotFoundError(f'v4 mask sidecar is missing: {mask_path}') from None
        if actual_size != expected_size:
            raise ValueError(
                f'v4 mask sidecar size mismatch for shard {shard_id}: '
                f'expected {expected_size}, got {actual_size}'
            )
        latent_path = latent_dir / f'shard_{shard_id:05d}.safetensors'
        if not latent_path.is_file():
            raise FileNotFoundError(f'v4 latent shard is missing: {latent_path}')
        with safe_open(str(latent_path), framework='pt') as file:
            keys = list(file.keys())
            if keys != ['latents']:
                raise ValueError(f'{latent_path}: expected only a latents tensor, got {keys}')
            latent_slice = file.get_slice('latents')
            shape = latent_slice.get_shape()
            dtype = latent_slice.get_dtype()
        expected_shape = [logical_rows, 32]
        if shape != expected_shape:
            raise ValueError(f'{latent_path}: expected latent shape {expected_shape}, got {shape}')
        if dtype != 'F16':
            raise ValueError(f'{latent_path}: expected F16 latents, got {dtype}')
        latent_paths.append((row, latent_path, expected_sha256))

    if verify_content_hashes and latent_paths:
        with ThreadPoolExecutor(max_workers=min(16, len(latent_paths))) as pool:
            actual_hashes = list(
                pool.map(sha256_file, (path for _, path, _ in latent_paths))
            )
        for (row, _, expected), actual in zip(latent_paths, actual_hashes, strict=True):
            if actual != expected:
                raise ValueError(
                    f"v4 latent shard content hash differs from READY.json for shard {row['shard_id']}"
                )

    required_ready = {
        'fingerprint': meta.get('fingerprint'),
        'verified_shards': n_shards,
        'logical_latent_rows': logical_latent_rows,
    }
    for key, expected in required_ready.items():
        if ready.get(key) != expected:
            raise ValueError(
                f'v4 READY {key} differs from the finalized stream: '
                f'expected {expected!r}, got {ready.get(key)!r}'
            )

    latent_contract = ready.get('latent_contract')
    if not isinstance(latent_contract, str) or not latent_contract:
        raise ValueError('v4 READY has an invalid latent contract')
    latent_receipt_path = stream_dir / 'latent-materialized.json'
    latent_receipt_sha256 = ready.get('latent_receipt_sha256')
    if (
        not isinstance(latent_receipt_sha256, str)
        or sha256_file(latent_receipt_path) != latent_receipt_sha256
    ):
        raise ValueError('v4 latent receipt hash differs from READY.json')

    stats_path = latent_dir / 'stats.safetensors'
    stats_sha256 = ready.get('stats_sha256')
    if not isinstance(stats_sha256, str) or sha256_file(stats_path) != stats_sha256:
        raise ValueError('v4 latent stats hash differs from READY.json')
    with safe_open(str(stats_path), framework='pt') as file:
        if set(file.keys()) != {'count', 'mean', 'std'}:
            raise ValueError(f'v4 latent stats have an invalid tensor schema: {stats_path}')
        stats_count = int(file.get_tensor('count'))
        mean = file.get_tensor('mean')
        std = file.get_tensor('std')
    uses_fixed_source_stats = _validate_fixed_source_stats(
        meta,
        ready,
        stats_sha256=stats_sha256,
        stats_count=stats_count,
    )
    if not uses_fixed_source_stats and stats_count != logical_latent_rows:
        raise ValueError(
            f'v4 latent stats count {stats_count} differs from manifest rows {logical_latent_rows}'
        )
    if mean.shape != (32,) or std.shape != (32,):
        raise ValueError(f'v4 latent stats must contain 32-channel mean/std: {stats_path}')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError(f'v4 latent stats contain invalid values: {stats_path}')


@dataclass(slots=True)
class _LoadedShard:
    """One stream shard and its aligned latent storage."""

    shard_id: int
    batches: list[dict]
    latent_handle: Any
    latent_offsets: list[int]
    mask_fd: int | None


def default_view_specs() -> list[dict]:
    """Default two-view policy: one uniform-random view and one multi-block view.

    Both use 0.70-0.80 (2D) / 0.75-0.85 (3D) requested mask-ratio ranges. The four leaf ranges are
    independently overridable; the shared defaults avoid confounding strategy with difficulty.
    """
    ranges = {'ratio_2d': (0.70, 0.80), 'ratio_3d': (0.75, 0.85)}
    return [
        {'strategy': 'random', **ranges},
        {'strategy': 'block', **ranges},
    ]


def validate_view_specs(view_specs: list[dict]) -> list[dict]:
    """Validate and normalize the fixed random-then-block view policy."""
    if len(view_specs) != 2:
        raise ValueError(f'exactly two view specs are required, got {len(view_specs)}')

    normalized: list[dict] = []
    for index, (spec, expected_strategy) in enumerate(zip(view_specs, ('random', 'block'))):
        if not isinstance(spec, dict):
            raise TypeError(f'view spec {index} must be a dict, got {type(spec).__name__}')
        strategy = spec.get('strategy')
        if strategy != expected_strategy:
            raise ValueError(
                f'view spec {index} must use strategy {expected_strategy!r}, got {strategy!r}'
            )
        normalized_spec = {'strategy': strategy}
        for key in ('ratio_2d', 'ratio_3d'):
            value = spec.get(key)
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f'view spec {index} {key} must be a length-2 range, got {value!r}')
            low, high = map(float, value)
            if not np.isfinite((low, high)).all() or not 0 < low <= high < 1:
                raise ValueError(
                    f'view spec {index} {key} must satisfy 0 < low <= high < 1, '
                    f'got {(low, high)!r}'
                )
            normalized_spec[key] = (low, high)
        normalized.append(normalized_spec)
    return normalized


def batch_rng(seed: int, shard_id: int, batch_idx: int) -> np.random.Generator:
    """Derive replay-time augmentation RNGs from one batch's absolute stream address."""
    if shard_id < 0 or batch_idx < 0:
        raise ValueError(
            f'shard_id and batch_idx must be non-negative, got {(shard_id, batch_idx)!r}'
        )
    return np.random.default_rng(
        np.random.SeedSequence(
            seed,
            spawn_key=(shard_id, _BATCH_RNG_DOMAIN, batch_idx),
        )
    )


def _make_view_masks(
    grid: tuple[int, int, int],
    ratios: list[float],
    strategies: list[str],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[dict]]:
    """Build one flat visible-token boolean mask per view for a single unlabeled sample.

    Args:
        grid: token grid (D, H, W).
        ratios: requested mask ratio per view.
        strategies: 'random' or 'block' per view (same length as ratios).
        rng: per-sample seeded generator (shared across this sample's views).

    Returns:
        Visible masks and sampler statistics in view order.
    """
    if len(ratios) != len(strategies):
        raise ValueError(
            f'ratios and strategies must have equal length, got {len(ratios)} and {len(strategies)}'
        )
    vis_masks: list[np.ndarray] = []
    view_stats: list[dict] = []
    for ratio, strat in zip(ratios, strategies):
        if strat == 'random':
            masked, stats = random_mask(grid, ratio, rng)
        elif strat == 'block':
            masked, stats = multi_block_mask(grid, ratio, rng)
        else:
            raise ValueError(f'unknown mask strategy {strat!r}')
        vis_masks.append(~masked.reshape(-1))  # visible = complement of masked
        view_stats.append({'strategy': strat, **stats})
    return vis_masks, view_stats



def _load_tcmalloc_release():
    """Resolve tcmalloc's page-release function.

    Returns:
        ``MallocExtension_ReleaseFreeMemory``, or ``None`` when tcmalloc is unavailable.
    """
    try:
        lib = ctypes.CDLL('libtcmalloc.so.4')
        return lib.MallocExtension_ReleaseFreeMemory
    except (OSError, AttributeError):
        return None


def _sample_coords(shape: list[int], r: float) -> torch.Tensor:
    """Compute RoPE coordinates for one patch grid.

    Args:
        shape: Patch grid ``[D, H, W]``.
        r: Coordinate half-range.

    Returns:
        Coordinates shaped ``(D * H * W, 3)`` in ``[-r, r]``.
    """
    D, H, W = shape
    coords_d = ((2 * (torch.arange(D, dtype=torch.float32) + 0.5) / D) - 1) * r
    coords_h = ((2 * (torch.arange(H, dtype=torch.float32) + 0.5) / H) - 1) * r
    coords_w = ((2 * (torch.arange(W, dtype=torch.float32) + 0.5) / W) - 1) * r
    grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')
    return torch.stack([grid_d.flatten(), grid_h.flatten(), grid_w.flatten()], dim=1)


def _collate(
    all_patches: torch.Tensor,
    latent: torch.Tensor,
    n_patches_list: list[int],
    shapes: list[tuple[int, int, int]],
    rescales: list[float],
    seg_payloads: list[dict | None],
    mask_rng: np.random.Generator | list[np.random.Generator],
    *,
    da_list: list[int | None],
    view_specs: list[dict],
    n_prefix: int,
    labeled_flags: list[bool],
) -> UCPTBatch:
    """Build a UCPT batch from loaded per-sample tensors and metadata.

    Each unlabeled sample produces ``V = len(view_specs)`` masked student views (masked view = one visibility
    mask + its own ratio); every view is encoded once by the student and its output feeds BOTH the
    reconstruction and patch-distillation decoders plus the CLS predictor. The teacher encodes the full grid
    once per sample and its targets are reused across views. Losses stay fp32 sums over all views; the model
    divides by DDP-global element counts.

    Packing order is sample-major, view-minor: ``[s0v0, s0v1, s1v0, s1v1, ...]`` for both the student blocks
    and the shared decoder blocks, so a view's CLS, visible tokens, and masked targets stay aligned.

    SSL masks, coordinates, and latents use unlabeled-local offsets. Student, teacher, and segmentation gather
    indices use full-array offsets; keeping these spaces separate prevents silent alignment errors.

    Args:
        all_patches: Packed patches for all samples.
        latent: Z-scored latents for unlabeled patches.
        n_patches_list: Patch count per sample.
        shapes: Patch grid per sample.
        rescales: RoPE rescale per sample.
        seg_payloads: Optional segmentation payload per sample.
        mask_rng: One batch-addressed generator, or one generator per sample when several virtual-lane
            microbatches are merged.
        da_list: Per-sample depth-adaptation level (None = 2D), used to select the dimension-specific ratio.
        view_specs: One dict per view: ``{'strategy': 'random'|'block', 'ratio_2d': (lo, hi),
            'ratio_3d': (lo, hi)}``. Shared across samples; ratio drawn per sample from the dim-appropriate
            range.
        n_prefix: Prefix-token count per packed block.
        labeled_flags: Whether each sample is labeled.

    Returns:
        CPU-resident collation metadata and packed tensors ready for device transfer.
    """
    view_specs = validate_view_specs(view_specs)
    n_views = len(view_specs)
    if isinstance(mask_rng, np.random.Generator):
        sample_mask_rngs = [mask_rng] * len(labeled_flags)
    else:
        sample_mask_rngs = list(mask_rng)
        if len(sample_mask_rngs) != len(labeled_flags):
            raise ValueError(
                f'mask RNG count must match sample count, got '
                f'{len(sample_mask_rngs)} and {len(labeled_flags)}'
            )
    # Full-array offsets over ALL samples (gather indices into all_tokens).
    full_offset_table: list[int] = []
    off = 0
    for n_p in n_patches_list:
        full_offset_table.append(off)
        off += n_p

    # SSL supports an empty unlabeled subset for pure-labeled evaluation;
    # training batches always include the unlabeled floor.
    unlab_idx = [i for i, lab in enumerate(labeled_flags) if not lab]
    unlab_n_patches = [n_patches_list[i] for i in unlab_idx]
    unlab_shapes = [shapes[i] for i in unlab_idx]
    unlab_rescales = [rescales[i] for i in unlab_idx]
    n_unlab_patches = sum(unlab_n_patches)
    num_samples_ssl = len(unlab_idx)

    # Unlabeled-local offsets (into length-n_unlab_patches SSL arrays).
    local_offset_table: list[int] = []
    off = 0
    for n_p in unlab_n_patches:
        local_offset_table.append(off)
        off += n_p

    # Per-sample masked views: draw ratios then masks from one seeded generator (replay-deterministic).
    # view_vis[k] is a list of V (n_p,) bool visible masks for unlabeled sample k, in view order.
    view_strategies = [spec['strategy'] for spec in view_specs]
    view_vis: list[list[np.ndarray]] = []
    for k, i in enumerate(unlab_idx):
        rng = sample_mask_rngs[i]
        da = da_list[i]
        ratios = [
            float(rng.uniform(*(spec['ratio_2d'] if da is None else spec['ratio_3d'])))
            for spec in view_specs
        ]
        masks, _ = _make_view_masks(unlab_shapes[k], ratios, view_strategies, rng)
        view_vis.append(masks)

    cpu = torch.device('cpu')
    prefix_coords = torch.zeros(n_prefix, 3)

    # --- Student: V blocks per sample, each [prefix, view-visible tokens], sample-major/view-minor. ---
    # Each block is one view encoded once; its output feeds both decoders + the CLS predictor.
    student_seqlens: list[int] = []
    student_coords_parts: list[torch.Tensor] = [torch.zeros(0, 3)]
    student_patch_mask_parts: list[torch.Tensor] = [torch.zeros(0, dtype=torch.bool)]
    student_patch_gather_parts: list[torch.Tensor] = [torch.zeros(0, dtype=torch.long)]

    for k, i in enumerate(unlab_idx):
        full_coords = _sample_coords(unlab_shapes[k], unlab_rescales[k])
        full_off_i = full_offset_table[i]
        for vis in view_vis[k]:
            vis_local = np.nonzero(vis)[0]
            n_vis = len(vis_local)
            student_seqlens.append(n_prefix + n_vis)
            student_coords_parts.append(prefix_coords)
            student_coords_parts.append(full_coords[vis_local])
            student_patch_mask_parts.append(torch.zeros(n_prefix, dtype=torch.bool))
            student_patch_mask_parts.append(torch.ones(n_vis, dtype=torch.bool))
            student_patch_gather_parts.append(
                torch.from_numpy((vis_local + full_off_i).astype(np.int64))
            )

    student_coords = torch.cat(student_coords_parts, dim=0)
    student_patch_mask = torch.cat(student_patch_mask_parts, dim=0)
    student_patch_gather_idx = torch.cat(student_patch_gather_parts, dim=0)

    # --- Teacher: full grid with prefix, once per sample. ---
    teacher_seqlens = [n_prefix + n_p for n_p in unlab_n_patches]
    teacher_coords_parts: list[torch.Tensor] = [torch.zeros(0, 3)]
    teacher_patch_mask_parts: list[torch.Tensor] = [torch.zeros(0, dtype=torch.bool)]
    for k in range(num_samples_ssl):
        full_coords = _sample_coords(unlab_shapes[k], 1.0)
        teacher_coords_parts.append(prefix_coords)
        teacher_coords_parts.append(full_coords)
        teacher_patch_mask_parts.append(torch.zeros(n_prefix, dtype=torch.bool))
        teacher_patch_mask_parts.append(torch.ones(unlab_n_patches[k], dtype=torch.bool))
    teacher_coords = torch.cat(teacher_coords_parts, dim=0)
    teacher_patch_mask = torch.cat(teacher_patch_mask_parts, dim=0)

    # Full-array indices of unlabeled patches in sample order (teacher forward gather).
    teacher_patch_gather_idx = torch.cat([
        torch.zeros(0, dtype=torch.long),
        *(torch.arange(full_offset_table[i], full_offset_table[i] + n_patches_list[i], dtype=torch.long)
          for i in unlab_idx),
    ])

    # --- Shared decoder packing: V blocks per sample, each the FULL grid, sample-major/view-minor. ---
    # Both decoders reconstruct/predict each view's masked tokens from that view's visible encoder output.
    # Integer indices partition decoder space into visible and masked positions. view_target_gather_idx maps
    # each masked position to unlabeled-local patch index (the target for BOTH latents and teacher features).
    view_decoder_seqlens = [n_p for n_p in unlab_n_patches for _ in range(n_views)]
    view_decoder_coords_parts: list[torch.Tensor] = [torch.zeros(0, 3)]
    view_visible_idx_parts: list[torch.Tensor] = [torch.zeros(0, dtype=torch.long)]
    view_masked_idx_parts: list[torch.Tensor] = [torch.zeros(0, dtype=torch.long)]
    view_target_gather_parts: list[torch.Tensor] = [torch.zeros(0, dtype=torch.long)]
    decoder_off = 0
    for k in range(num_samples_ssl):
        full_coords = _sample_coords(unlab_shapes[k], unlab_rescales[k])
        local_off = local_offset_table[k]
        for vis in view_vis[k]:
            view_decoder_coords_parts.append(full_coords)
            visible_local = np.nonzero(vis)[0]
            masked_local = np.nonzero(~vis)[0]
            view_visible_idx_parts.append(
                torch.from_numpy((visible_local + decoder_off).astype(np.int64))
            )
            view_masked_idx_parts.append(
                torch.from_numpy((masked_local + decoder_off).astype(np.int64))
            )
            view_target_gather_parts.append(
                torch.from_numpy((masked_local + local_off).astype(np.int64))
            )
            decoder_off += len(vis)
    view_decoder_coords = torch.cat(view_decoder_coords_parts, dim=0)
    view_visible_idx = torch.cat(view_visible_idx_parts, dim=0)
    view_masked_idx = torch.cat(view_masked_idx_parts, dim=0)
    view_target_gather_idx = torch.cat(view_target_gather_parts, dim=0)

    total_student_len = sum(student_seqlens)
    total_teacher_len = sum(teacher_seqlens)

    # --- Seg half (stream-labeled samples only) ---
    if any((payload is not None) != labeled for payload, labeled in zip(seg_payloads, labeled_flags, strict=True)):
        raise ValueError('segmentation payload presence must match the stream labeled flag')
    seg_idx = [i for i, labeled in enumerate(labeled_flags) if labeled]

    if seg_idx:
        seg_sample_shapes = [seg_payloads[i]['patch_grid'] for i in seg_idx]
        seg_das = [seg_payloads[i]['da'] for i in seg_idx]
        text_embeddings = [seg_payloads[i]['text_embeddings'] for i in seg_idx]
        text_valid_masks = [seg_payloads[i]['text_valid_mask'] for i in seg_idx]
        is_positive = [seg_payloads[i]['is_positive'] for i in seg_idx]
        target_masks = [seg_payloads[i]['target_masks'] for i in seg_idx]

        seg_n_p_list = [n_patches_list[i] for i in seg_idx]
        seg_seqlens = [n_prefix + n for n in seg_n_p_list]
        total_seg_len = sum(seg_seqlens)

        # gather idx: concat of arange(full_offset_i, full_offset_i + n_p_i)
        gather_parts: list[torch.Tensor] = []
        for li, n_p_i in zip(seg_idx, seg_n_p_list):
            full_off = full_offset_table[li]
            gather_parts.append(torch.arange(full_off, full_off + n_p_i, dtype=torch.long))
        seg_patch_gather_idx = torch.cat(gather_parts)

        # mask: False at prefix, True at patches (per labeled sample)
        mask_parts: list[torch.Tensor] = []
        for n_p_i in seg_n_p_list:
            mask_parts.append(torch.zeros(n_prefix, dtype=torch.bool))
            mask_parts.append(torch.ones(n_p_i, dtype=torch.bool))
        seg_patch_mask = torch.cat(mask_parts)

        # coords: per labeled sample [prefix_coords, _sample_coords(shape, rescale)]
        # rescale is the labeled sample's STUDENT-side rope_rescale.
        coord_parts: list[torch.Tensor] = []
        for shape_i, rescale_i in zip(
            seg_sample_shapes, (rescales[i] for i in seg_idx),
        ):
            coord_parts.append(prefix_coords)
            coord_parts.append(_sample_coords(list(shape_i), rescale_i))
        seg_coords = torch.cat(coord_parts, dim=0)

        seg_attn_bias = BlockDiagonalMask.from_seqlens(seg_seqlens, device=cpu)
    else:
        seg_sample_shapes = []
        seg_das = []
        text_embeddings = []
        text_valid_masks = []
        is_positive = []
        target_masks = []
        seg_patch_gather_idx = None
        seg_patch_mask = None
        seg_coords = None
        seg_attn_bias = None
        total_seg_len = 0

    return UCPTBatch(
        patches=all_patches,
        latents=latent.to(torch.bfloat16),
        student_attn_bias=BlockDiagonalMask.from_seqlens(student_seqlens, device=cpu),
        teacher_attn_bias=BlockDiagonalMask.from_seqlens(teacher_seqlens, device=cpu),
        view_decoder_attn_bias=BlockDiagonalMask.from_seqlens(view_decoder_seqlens, device=cpu),
        student_coords=student_coords,
        teacher_coords=teacher_coords,
        view_decoder_coords=view_decoder_coords,
        student_patch_mask=student_patch_mask,
        student_patch_gather_idx=student_patch_gather_idx,
        teacher_patch_mask=teacher_patch_mask,
        teacher_patch_gather_idx=teacher_patch_gather_idx,
        sample_is_labeled=torch.tensor(labeled_flags, dtype=torch.bool),
        view_visible_idx=view_visible_idx,
        view_masked_idx=view_masked_idx,
        view_target_gather_idx=view_target_gather_idx,
        total_student_len=total_student_len,
        total_teacher_len=total_teacher_len,
        num_blocks=n_views * num_samples_ssl,
        n_views=n_views,
        n_ssl_patches=n_unlab_patches,
        seg_patch_gather_idx=seg_patch_gather_idx,
        seg_patch_mask=seg_patch_mask,
        seg_coords=seg_coords,
        seg_attn_bias=seg_attn_bias,
        seg_sample_shapes=seg_sample_shapes,
        seg_das=seg_das,
        text_embeddings=text_embeddings,
        text_valid_masks=text_valid_masks,
        is_positive=is_positive,
        target_masks=target_masks,
        total_seg_len=total_seg_len,
    )


class UCPTReplayDataset(torch.utils.data.Dataset):
    """Map-style UCPT dataset with fixed virtual lanes and precomputed latents.

    Every global step consumes one microbatch from each virtual lane. Distributed ranks own disjoint lane subsets
    and merge their lane-local microbatches before collation. ``start_offset`` advances every lane together for
    deterministic resume.
    """

    def __init__(
        self,
        stream_dir: Path | str,
        latent_dir: Path | str | None = None,
        rank: int = 0,
        world_size: int = 1,
        virtual_lanes: int = 1,
        pipeline: TransformPipeline | None = None,
        start_offset: int = 0,
        view_specs: list[dict] | None = None,
        n_prefix: int = 5,
        augment_threads: int = 4,
        tcmalloc_release_every: int = 0,
        verify_ready_hashes: bool = True,
        *,
        data_root: Path | str,
        text_cache_path: Path | str,
        class_captions_dir: Path | str,
    ):
        """
        Args:
            stream_dir: Directory containing stream shards and metadata.
            latent_dir: Optional latent-shard directory; defaults to ``stream_dir / 'latents'``.
            rank: Global distributed rank used for virtual-lane assignment.
            world_size: Global number of distributed data-parallel ranks.
            virtual_lanes: Fixed number of logical microbatch lanes consumed per global step.
            pipeline: Replayable augmentation pipeline.
            start_offset: Number of global lane steps to skip on resume.
            view_specs: One dict per masked view (see ``_collate``); defaults to a random view + a SPAD
                multi-block view, both 0.70-0.80 (2D) / 0.75-0.85 (3D).
            n_prefix: Prefix-token count per packed block.
            augment_threads: Per-worker augmentation thread count.
            tcmalloc_release_every: Page-release interval; zero disables release.
            verify_ready_hashes: Rehash latent contents against READY. Training disables this because final verification
                already produced the immutable READY proof.
            data_root: Root directory containing preprocessed masks.
            text_cache_path: Precomputed text-embedding cache path.
            class_captions_dir: Directory containing one caption document per label source.
        """
        self.pipeline = pipeline
        self.start_offset = start_offset
        self.view_specs = validate_view_specs(
            view_specs if view_specs is not None else default_view_specs()
        )
        self.n_prefix = n_prefix
        self.augment_threads = augment_threads

        # End-of-__getitem__ tcmalloc page-heap release to bound per-worker RSS
        # under variable batch sizes (0 = disabled).
        self.tcmalloc_release_every = tcmalloc_release_every
        self._getitem_count = 0
        self._tcmalloc_release = _load_tcmalloc_release() if tcmalloc_release_every > 0 else None

        stream_dir = Path(stream_dir)
        if latent_dir is None:
            latent_dir = stream_dir / 'latents'
        else:
            latent_dir = Path(latent_dir)

        # Load meta
        with open(stream_dir / 'meta.yaml') as f:
            meta = yaml.safe_load(f)
        if meta.get('stream_complete') is not True:
            raise RuntimeError(f'stream is not finalized: {stream_dir}')
        self.batches_per_shard = meta['batches_per_shard']
        self.stream_seed = meta['seed']
        self._mask_storage = meta.get('mask_storage')
        is_v4 = meta.get('algorithm') == 'ucpt-stream-v4'
        if is_v4 and self._mask_storage != MASK_STORAGE:
            raise ValueError(
                f'ucpt-stream-v4 requires mask_storage={MASK_STORAGE!r}'
            )
        if self._mask_storage not in (None, MASK_STORAGE):
            raise ValueError(
                f'unsupported positive-mask storage {self._mask_storage!r}; '
                f'expected {MASK_STORAGE!r}'
            )
        if is_v4:
            _validate_v4_ready(
                stream_dir,
                latent_dir,
                meta,
                verify_content_hashes=verify_ready_hashes,
                validation_rank=rank,
                validation_world_size=world_size,
            )
        self._sample_metadata_contract = meta.get('sample_metadata_contract') if is_v4 else None

        if virtual_lanes < 1:
            raise ValueError(f'virtual_lanes must be positive, got {virtual_lanes}')
        if world_size < 1:
            raise ValueError(f'world_size must be positive, got {world_size}')
        if not 0 <= rank < world_size:
            raise ValueError(f'rank must be in [0, {world_size}), got {rank}')
        assert virtual_lanes % world_size == 0, (
            f'virtual_lanes ({virtual_lanes}) must be divisible by world_size ({world_size})'
        )

        # Virtual lane l owns shard IDs l, l + virtual_lanes, ...
        all_shards = sorted(stream_dir.glob('shard_*.msgpack'))
        num_shards = len(all_shards)
        if not all_shards:
            raise FileNotFoundError(f'No shard files found in {stream_dir}')
        if num_shards != meta['n_shards']:
            raise RuntimeError(
                f'stream {stream_dir} is incomplete: found {num_shards} shard files, '
                f"meta.yaml declares {meta['n_shards']} (generation interrupted or "
                f'files missing)'
            )
        if num_shards % virtual_lanes != 0:
            raise ValueError(
                f'num_shards ({num_shards}) must be divisible by virtual_lanes ({virtual_lanes})'
            )
        shard_paths_by_id = {
            int(path.stem.removeprefix('shard_')): path
            for path in all_shards
        }
        expected_shard_ids = set(range(num_shards))
        if set(shard_paths_by_id) != expected_shard_ids:
            raise RuntimeError(f'stream shard IDs must be contiguous in [0, {num_shards})')

        self.virtual_lanes = virtual_lanes
        self.my_lane_ids = list(range(rank, virtual_lanes, world_size))
        self.shards_per_lane = num_shards // virtual_lanes
        self._shard_paths = [shard_paths_by_id[shard_id] for shard_id in range(num_shards)]
        self._latent_paths = [
            latent_dir / path.name.replace('.msgpack', '.safetensors')
            for path in self._shard_paths
        ]
        total_steps = self.shards_per_lane * self.batches_per_shard
        if not 0 <= start_offset <= total_steps:
            raise ValueError(
                f'start_offset must be in [0, {total_steps}], got {start_offset}'
            )
        self._length = total_steps - start_offset

        # Latent normalization stats
        stats_path = latent_dir / 'stats.safetensors'
        with safe_open(str(stats_path), framework='pt') as f:
            self._latent_mean = f.get_tensor('mean')
            self._latent_std = f.get_tensor('std')

        # All rank-local lanes advance together, so one cache entry per lane covers the current shard round.
        self._cached_shard_round = -1
        self._cached_shards: list[_LoadedShard] = []

        # Segmentation resources loaded lazily by _build_seg_payload.
        self.data_root = Path(data_root)
        self._text_cache_path = Path(text_cache_path)
        self._class_captions_dir = Path(class_captions_dir)

    @cached_property
    def _thread_pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self.augment_threads)

    @cached_property
    def _text_cache(self) -> TextEmbeddingCache:
        return TextEmbeddingCache(self._text_cache_path)

    @cached_property
    def _prompt_resolver(self) -> SegmentationPromptResolver:
        return SegmentationPromptResolver(self._class_captions_dir)

    def __len__(self) -> int:
        return self._length

    def _load_shard(self, shard_id: int) -> _LoadedShard:
        """Load one stream shard, its latent handle, and per-batch latent offsets."""
        shard_path = self._shard_paths[shard_id]
        with open(shard_path, 'rb') as f:
            shard = msgpack.unpack(f, raw=False)
        batches = shard['batches']
        if len(batches) != self.batches_per_shard:
            raise ValueError(
                f'{shard_path} contains {len(batches)} batches, expected '
                f'{self.batches_per_shard}'
            )
        for batch_idx, batch in enumerate(batches):
            if batch.get('step_idx') != batch_idx:
                raise ValueError(
                    f'{shard_path} batch {batch_idx} has '
                    f'step_idx={batch.get("step_idx")!r}'
                )
            for sample in batch['samples']:
                if not isinstance(sample.get('labeled'), bool):
                    raise ValueError(
                        f'{shard_path} sample lacks a boolean labeled flag'
                    )
                has_classes = isinstance(sample.get('classes'), list) and bool(sample['classes'])
                if has_classes != sample['labeled'] or 'label_classes' in sample:
                    raise ValueError(
                        f'{shard_path} has an invalid labeled sample schema'
                    )
                if self._sample_metadata_contract is not None:
                    canonical, changes = canonicalize_sample_metadata(sample)
                    if changes or canonical is not sample:
                        raise ValueError(f'{shard_path} contains obsolete sample metadata')

        latent_handle = safe_open(str(self._latent_paths[shard_id]), framework='pt')
        mask_fd = None
        for batch in batches:
            for sample in batch['samples']:
                has_reference = MASK_REF_KEY in sample
                if self._mask_storage is not None:
                    if sample['labeled'] and not has_reference:
                        raise ValueError(
                            f'{shard_path} labeled sample lacks {MASK_REF_KEY!r}'
                        )
                    if not sample['labeled'] and has_reference:
                        raise ValueError(
                            f'{shard_path} unlabeled sample contains {MASK_REF_KEY!r}'
                        )
                elif has_reference:
                    raise ValueError(
                        f'{shard_path} contains {MASK_REF_KEY!r} but meta.yaml does not '
                        f'declare mask_storage'
                    )
        if self._mask_storage is not None:
            mask_fd = os.open(mask_shard_path(shard_path.parent, shard_id), os.O_RDONLY)

        # Latents are stored for unlabeled samples only.
        offsets = [0]
        for batch in batches:
            batch_patches = sum(s['n_patches'] for s in batch['samples']
                                if not s['labeled'])
            offsets.append(offsets[-1] + batch_patches)
        return _LoadedShard(shard_id, batches, latent_handle, offsets, mask_fd)

    def _load_shard_round(self, shard_round: int) -> None:
        """Load the shard currently addressed by every rank-local virtual lane."""
        if shard_round == self._cached_shard_round:
            return
        loaded_shards = []
        try:
            for lane_id in self.my_lane_ids:
                loaded_shards.append(
                    self._load_shard(lane_id + self.virtual_lanes * shard_round)
                )
        except BaseException:
            for loaded in loaded_shards:
                if loaded.mask_fd is not None:
                    os.close(loaded.mask_fd)
            raise
        for loaded in self._cached_shards:
            if loaded.mask_fd is not None:
                os.close(loaded.mask_fd)
        self._cached_shards = loaded_shards
        self._cached_shard_round = shard_round

    def _build_seg_payload(
        self,
        sample: dict,
        patch_grid: tuple[int, int, int],
        da: int | None,
        *,
        text_rng: np.random.Generator | None = None,
        positive_mask_frame: bytes | None = None,
    ) -> dict:
        """Build segmentation targets for one labeled sample.

        Masks use the same crop and affine as the image and are nearest-resampled to the full image crop.

        Args:
            sample: Labeled stream sample with final raw-class queries.
            patch_grid: ViT patch grid ``(D, H, W)``.
            da: Sample SPAD depth-adaptation level.

        Returns:
            Target masks, text embeddings, concept IDs, positivity flags, patch grid, and decoder schedule.
        """
        return build_seg_payload(
            sample,
            patch_grid,
            da,
            data_root=self.data_root,
            prompt_resolver=self._prompt_resolver,
            text_cache=self._text_cache,
            text_rng=text_rng,
            positive_mask_frame=positive_mask_frame,
        )

    def __getitem__(self, idx: int) -> UCPTBatch:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        lane_step = self.start_offset + idx
        shard_round, batch_idx = divmod(lane_step, self.batches_per_shard)
        self._load_shard_round(shard_round)

        samples: list[dict] = []
        sample_mask_fds: list[int | None] = []
        sample_mask_rngs: list[np.random.Generator] = []
        text_rngs: list[np.random.Generator] = []
        latent_parts: list[torch.Tensor] = []
        for loaded in self._cached_shards:
            batch_info = loaded.batches[batch_idx]
            lane_samples = batch_info['samples']
            replay_rng = batch_rng(self.stream_seed, loaded.shard_id, batch_idx)
            mask_rng, text_root_rng = replay_rng.spawn(2)
            samples.extend(lane_samples)
            sample_mask_fds.extend([loaded.mask_fd] * len(lane_samples))
            sample_mask_rngs.extend([mask_rng] * len(lane_samples))
            text_rngs.extend(text_root_rng.spawn(len(lane_samples)))
            start = loaded.latent_offsets[batch_idx]
            end = loaded.latent_offsets[batch_idx + 1]
            latent_parts.append(loaded.latent_handle.get_slice('latents')[start:end])

        normalizer = self.pipeline.transforms[-1]
        if isinstance(normalizer, InputNormalizer):
            for sample in samples:
                normalizer.scheme(sample['img'])

        # Replay transforms per sample, writing directly into pre-allocated buffer
        total_patches = sum(s['n_patches'] for s in samples)
        all_patches = torch.empty(total_patches, 3, 16, 16, 16, dtype=torch.bfloat16)

        # Pre-compute per-sample offsets from stream metadata (n_patches is exact)
        full_offset_table = []
        off = 0
        for s in samples:
            full_offset_table.append(off)
            off += s['n_patches']

        def _process_sample(args: tuple[int, dict]) -> tuple[list[int], int, int | None, dict | None]:
            i, s = args
            data = self.pipeline.replay({'img': s['img']}, s['params'])
            img = data['img']
            da = s['da_enc']
            patches, shape = patchify(img, da=da if da is not None else 4)
            n = patches.shape[0]
            all_patches[full_offset_table[i]:full_offset_table[i] + n] = patches
            seg_payload = None
            if s['labeled']:
                positive_mask_frame = None
                if self._mask_storage is not None:
                    mask_fd = sample_mask_fds[i]
                    assert mask_fd is not None
                    positive_mask_frame = read_mask_frame(mask_fd, s[MASK_REF_KEY])
                seg_payload = self._build_seg_payload(
                    s,
                    shape,
                    da,
                    text_rng=text_rngs[i],
                    positive_mask_frame=positive_mask_frame,
                )
            return shape, n, da, seg_payload

        # Initialize large read-only segmentation resources once in this worker before augmentation threads access them.
        if any(sample['labeled'] for sample in samples):
            _ = self._text_cache, self._prompt_resolver
        results = list(self._thread_pool.map(_process_sample, enumerate(samples)))

        shapes = [r[0] for r in results]
        n_patches_list = [r[1] for r in results]
        da_list = [r[2] for r in results]
        seg_payloads = [r[3] for r in results]

        latent_raw = latent_parts[0] if len(latent_parts) == 1 else torch.cat(latent_parts)

        # Z-score
        latent = (latent_raw.float() - self._latent_mean) / self._latent_std

        # Per-sample RoPE rescale: read from stream (decoupled from mask RNG).
        rescales = [s['rope_rescale'] for s in samples]

        # Latents and SSL tensors are both ordered over the stream-unlabeled subset.
        labeled_flags = [s['labeled'] for s in samples]

        batch = _collate(
            all_patches, latent, n_patches_list, shapes, rescales,
            seg_payloads,
            sample_mask_rngs,
            da_list=da_list,
            view_specs=self.view_specs,
            n_prefix=self.n_prefix,
            labeled_flags=labeled_flags,
        )

        # Release retained pages after augmentation threads drop their transient fp32 buffers.
        if self._tcmalloc_release is not None:
            self._getitem_count += 1
            if self._getitem_count % self.tcmalloc_release_every == 0:
                self._tcmalloc_release()

        return batch
