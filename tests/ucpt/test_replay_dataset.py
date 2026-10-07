"""Tests for UCPTReplayDataset. Synthetic stream fixtures, no real dataset."""
import io
import json

import msgpack
import numpy as np
import orjson
import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from pumit.segmentation_mask import pack_binary_mask
from pumit.text_prompt import build_segmentation_prompt
from pumit.ucpt.batch import UCPTBatch
from pumit.ucpt.replay_dataset import (
    UCPTReplayDataset,
    _make_view_masks,
    batch_rng,
    default_view_specs,
    validate_view_specs,
)
from pumit.ucpt.transforms import build_ucpt_pipeline


def _write_image(tmp_path, name='vol.npy', shape=(3, 16, 64, 64)):
    vol = np.random.RandomState(0).random(shape).astype(np.float32)
    np.save(tmp_path / name, vol)
    return tmp_path / name


def _write_mask(tmp_path, dataset, key, source, cls, shape_3d, bright='center'):
    import zstandard as zstd
    d = tmp_path / dataset / 'labels' / key / source
    d.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape_3d, dtype=bool)
    if bright == 'periphery':
        # Fill the first coarse-grid cell so center-aligned nearest sampling preserves it.
        mask[:, :, :4] = True
    elif bright == 'point':
        mask[shape_3d[0] // 2, shape_3d[1] // 2, shape_3d[2] // 2] = True
    else:
        shape = np.array(shape_3d)
        lo = shape // 4
        hi = np.maximum(lo + 1, shape // 2)
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    packed = pack_binary_mask(mask)
    buf = io.BytesIO(); np.save(buf, packed)
    (d / f'masks__{cls}.npy.zst').write_bytes(zstd.ZstdCompressor().compress(buf.getvalue()))


def _write_text_cache(
    path,
    classes=('liver', 'kidney'),
    modalities=('CT',),
    source='src',
    *,
    variants=None,
):
    L = 24
    captions = variants or {class_name: [class_name] for class_name in classes}
    prompts = [
        build_segmentation_prompt(description, modality)
        for descriptions in captions.values()
        for description in descriptions
        for modality in modalities
    ]
    emb = {prompt: torch.zeros(L, 1152) for prompt in prompts}
    for k in emb:
        emb[k][:6] = torch.randn(6, 1152)
    masks = {prompt: torch.arange(L) < 6 for prompt in prompts}
    torch.save(
        {'key_type': 'prompt_text', 'embeddings': emb, 'valid_masks': masks, 'dim': 1152, 'seq_len': L},
        path,
    )
    captions_dir = path.parent / 'captions'
    captions_dir.mkdir(exist_ok=True)
    (captions_dir / f'{source}.json').write_bytes(orjson.dumps({
        'source': source,
        'classes': captions,
    }))


def _make_stream(
    tmp_path, samples_spec, n_shards=1, batches_per_shard=None, *,
    unlabeled_only_latents=False, latent_dim=32, latent_dtype=torch.bfloat16,
):
    """samples_spec: list of sample dicts (already shaped like generate_sample output).

    By default latents are written for ALL samples (legacy behavior). When
    ``unlabeled_only_latents=True`` (T2 unlabeled-only latent storage), only
    samples with ``labeled=False`` contribute latents, so total rows ==
    sum of unlabeled n_patches, matching ``_load_shard``'s unlabeled-only
    offset logic. ``latent_dim`` selects the per-row width (default 32).
    """
    stream_dir = tmp_path / 'stream'; stream_dir.mkdir()
    latent_dir = tmp_path / 'latents'; latent_dir.mkdir()
    n = len(samples_spec)
    bps = batches_per_shard or n
    assert n % n_shards == 0
    per_shard = n // n_shards
    latents_for = (
        [s for s in samples_spec if not s['labeled']]
        if unlabeled_only_latents else samples_spec
    )
    all_latents = [
        torch.randn(s['n_patches'], latent_dim, dtype=latent_dtype)
        for s in latents_for
    ]
    flat = torch.cat(all_latents) if all_latents else torch.zeros(0, latent_dim, dtype=latent_dtype)
    for sh in range(n_shards):
        shard_batches = []
        for k in range(per_shard):
            s = samples_spec[sh * per_shard + k]
            shard_batches.append({'step_idx': k, 'samples': [s]})
        with open(stream_dir / f'shard_{sh:05d}.msgpack', 'wb') as f:
            msgpack.pack({'batches': shard_batches}, f)
    # one latent shard per stream shard (same rows replicated; _load_shard slices
    # by per-batch unlabeled-only offsets, so replication is harmless).
    for sh in range(n_shards):
        save_file({'latents': flat}, latent_dir / f'shard_{sh:05d}.safetensors')
    save_file(
        {'mean': torch.zeros(latent_dim), 'std': torch.ones(latent_dim)},
        latent_dir / 'stats.safetensors',
    )
    with open(stream_dir / 'meta.yaml', 'w') as f:
        yaml.dump({
            'stream_complete': True,
            'n_shards': n_shards,
            'batches_per_shard': bps,
            'seed': 42,
        }, f)
    return stream_dir, latent_dir


def _write_v4_ready(stream_dir, latent_dir):
    """Publish the minimal finalized-v4 proof consumed by UCPTReplayDataset."""
    from pumit.ucpt.mask_store import MASK_STORAGE, mask_shard_path
    from pumit.ucpt.stream.manifest import sha256_file
    from pumit.ucpt.stream.metadata import SAMPLE_METADATA_CONTRACT

    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    rows = []
    for shard_id in range(meta['n_shards']):
        with (stream_dir / f'shard_{shard_id:05d}.msgpack').open('rb') as file:
            shard = msgpack.unpack(file, raw=False)
        logical_rows = sum(
            sample['n_patches']
            for batch in shard['batches']
            for sample in batch['samples']
            if not sample['labeled']
        )
        mask_path = mask_shard_path(stream_dir, shard_id)
        rows.append({
            'shard_id': shard_id,
            'logical_latent_rows': logical_rows,
            'mask_storage': MASK_STORAGE,
            'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
            'mask_storage_bytes': mask_path.stat().st_size,
            'mask_sha256': sha256_file(mask_path),
        })
    manifest_path = stream_dir / 'manifest.jsonl'
    manifest_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    latent_receipt_path = stream_dir / 'latent-materialized.json'
    latent_receipt_path.write_text(json.dumps({'fixture': 'v4-latents'}))

    fingerprint = 'fixture-v4-fingerprint'
    meta.update({
        'algorithm': 'ucpt-stream-v4',
        'fingerprint': fingerprint,
        'manifest_sha256': sha256_file(manifest_path),
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
    })
    meta_path.write_text(yaml.safe_dump(meta))

    logical_rows = sum(row['logical_latent_rows'] for row in rows)
    stats_path = latent_dir / 'stats.safetensors'
    stats_path.unlink()
    save_file(
        {
            'count': torch.tensor(logical_rows, dtype=torch.int64),
            'mean': torch.zeros(32),
            'std': torch.ones(32),
        },
        stats_path,
    )
    (stream_dir / 'READY.json').write_text(json.dumps({
        'fingerprint': fingerprint,
        'verified_shards': len(rows),
        'logical_latent_rows': logical_rows,
        'manifest_sha256': sha256_file(manifest_path),
        'mask_storage': MASK_STORAGE,
        'sample_metadata_contract': SAMPLE_METADATA_CONTRACT,
        'latent_contract': 'fixture-v4-latents',
        'latent_receipt_sha256': sha256_file(latent_receipt_path),
        'latent_shard_sha256': [
            sha256_file(latent_dir / f"shard_{row['shard_id']:05d}.safetensors")
            for row in rows
        ],
        'stats_sha256': sha256_file(stats_path),
    }))


def _make_ready_v4_stream(tmp_path, *, n_shards=1):
    from pumit.ucpt.mask_store import mask_shard_path, write_mask_shard

    img = _write_image(tmp_path)
    sample = _sample(img)
    stream_dir, latent_dir = _make_stream(
        tmp_path,
        [sample] * n_shards,
        n_shards=n_shards,
        batches_per_shard=1,
        unlabeled_only_latents=True,
        latent_dtype=torch.float16,
    )
    for shard_id in range(n_shards):
        with (stream_dir / f'shard_{shard_id:05d}.msgpack').open('rb') as file:
            shard = msgpack.unpack(file, raw=False)
        write_mask_shard(shard['batches'], mask_shard_path(stream_dir, shard_id))
        latent_path = latent_dir / f'shard_{shard_id:05d}.safetensors'
        latent_path.unlink()
        save_file(
            {
                'latents': torch.zeros(
                    sample['n_patches'],
                    32,
                    dtype=torch.float16,
                )
            },
            latent_path,
        )
    _write_v4_ready(stream_dir, latent_dir)
    return stream_dir, latent_dir


def _use_fixed_source_stats(stream_dir, latent_dir, *, source_count=7):
    from pumit.ucpt.stream.manifest import sha256_file

    stats_path = latent_dir / 'stats.safetensors'
    stats_path.unlink()
    save_file(
        {
            'count': torch.tensor(source_count, dtype=torch.int64),
            'mean': torch.zeros(32),
            'std': torch.ones(32),
        },
        stats_path,
    )
    source_stream = (stream_dir.parent / 'prefix-stream').resolve()
    source_fingerprint = 'fixture-prefix-fingerprint'
    source_manifest_sha256 = 'a' * 64
    stats_sha256 = sha256_file(stats_path)
    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['composition'] = {
        'contract': 'concatenated-stream-v1',
        'replay_seed': 42,
        'segments': [
            {
                'target_shard_start': 0,
                'target_shard_stop': 1,
                'source_stream': str(source_stream),
                'source_fingerprint': source_fingerprint,
                'source_manifest_sha256': source_manifest_sha256,
                'source_ready_sha256': 'b' * 64,
                'source_shard_start': 0,
                'source_shard_stop': 1,
                'generation_seed': 42,
            },
            {
                'target_shard_start': 1,
                'target_shard_stop': 2,
                'source_stream': str((stream_dir.parent / 'suffix-stream').resolve()),
                'source_fingerprint': 'fixture-suffix-fingerprint',
                'source_manifest_sha256': 'c' * 64,
                'source_shard_start': 0,
                'source_shard_stop': 1,
                'generation_seed': 43,
            },
        ],
        'normalization': {
            'contract': 'fixed-source-v1',
            'source_stream': str(source_stream),
            'source_fingerprint': source_fingerprint,
            'source_manifest_sha256': source_manifest_sha256,
            'source_stats_sha256': stats_sha256,
            'source_stats_count': source_count,
        },
    }
    meta_path.write_text(yaml.safe_dump(meta))

    ready_path = stream_dir / 'READY.json'
    ready = json.loads(ready_path.read_text())
    ready.update(
        {
            'stats_sha256': stats_sha256,
            'stats_contract': 'fixed-source-v1',
            'stats_source_fingerprint': source_fingerprint,
            'stats_source_count': source_count,
        }
    )
    ready_path.write_text(json.dumps(ready))
    return meta, ready


def _make_virtual_lane_stream(tmp_path, *, n_shards=8, batches_per_shard=2):
    """Build a stream whose latent values identify each shard/batch."""
    img = _write_image(tmp_path)
    samples = [_sample(img) for _ in range(n_shards * batches_per_shard)]
    stream_dir, latent_dir = _make_stream(
        tmp_path,
        samples,
        n_shards=n_shards,
        batches_per_shard=batches_per_shard,
        unlabeled_only_latents=True,
    )
    rows_per_batch = samples[0]['n_patches']
    for shard_id in range(n_shards):
        latent_path = latent_dir / f'shard_{shard_id:05d}.safetensors'
        latent_path.unlink()
        save_file(
            {
                'latents': torch.cat(
                    [
                        torch.full(
                            (rows_per_batch, 32),
                            10 * shard_id + batch_idx,
                            dtype=torch.bfloat16,
                        )
                        for batch_idx in range(batches_per_shard)
                    ],
                ),
            },
            latent_path,
        )
    return stream_dir, latent_dir


def _sample(img_path, *, label=False, da=0, patch_grid=(1, 4, 4), rope_rescale=1.0):
    n_p = patch_grid[0] * patch_grid[1] * patch_grid[2]
    s = {
        'img': str(img_path),
        'spacing_label': [2.0, 1.0, 1.0],
        'params': [{
            'da_enc': da, 'crop_size': [patch_grid[0]*16, patch_grid[1]*16, patch_grid[2]*16],
            'affine': np.eye(4).ravel().tolist(),
            'load_slice_start': [0, 0, 0],
            'load_slice_stop': [patch_grid[0]*16, patch_grid[1]*16, patch_grid[2]*16],
            'n_patches': n_p, 'spacing_label': [2.0, 1.0, 1.0],
        }, {'enabled': False}, {'enabled': False}, {'enabled': False}, {}],
        'da_enc': da, 'n_patches': n_p, 'depth': patch_grid[0]*16,
        'rope_rescale': rope_rescale,
        'labeled': label,
    }
    if label:
        s['dataset'] = 'ds'
        s['key'] = 'key'
        s['modality'] = 'CT'
        s['classes'] = [
            {'source': 'src', 'name': 'liver', 'is_positive': True, 'target_voxels': 1024},
            {'source': 'src', 'name': 'kidney', 'is_positive': False, 'target_voxels': 0},
        ]
        s['seg_cost_queries'] = 2
    return s


@pytest.fixture
def pipeline():
    return build_ucpt_pipeline(
        size_xy_choices=[64], size_xy_choices_2d=[64],
        max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12}, max_da=4,
    )


def test_getitem_returns_ucpt_batch(tmp_path, pipeline):
    img = _write_image(tmp_path)
    s = _sample(img)
    stream_dir, latent_dir = _make_stream(tmp_path, [s])
    ds = UCPTReplayDataset(
        stream_dir=stream_dir, latent_dir=latent_dir, rank=0, world_size=1,
        pipeline=pipeline, data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    batch = ds[0]
    assert isinstance(batch, UCPTBatch)
    assert batch.patches.shape[0] == s['n_patches']
    assert batch.n_ssl_patches == s['n_patches']  # not the default 0
    # seg fields default (no labeled samples)
    assert batch.total_seg_len == 0
    assert batch.seg_patch_gather_idx is None


def test_replay_rejects_unfinalized_stream(tmp_path, pipeline):
    img = _write_image(tmp_path)
    stream_dir, latent_dir = _make_stream(tmp_path, [_sample(img)])
    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['stream_complete'] = False
    meta_path.write_text(yaml.safe_dump(meta))

    with pytest.raises(RuntimeError, match='not finalized'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            rank=0,
            world_size=1,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_without_mask_storage(tmp_path, pipeline):
    img = _write_image(tmp_path)
    stream_dir, latent_dir = _make_stream(tmp_path, [_sample(img)])
    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['algorithm'] = 'ucpt-stream-v4'
    meta_path.write_text(yaml.safe_dump(meta))

    with pytest.raises(ValueError, match='ucpt-stream-v4 requires mask_storage'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            rank=0,
            world_size=1,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_without_ready_proof(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    (stream_dir / 'READY.json').unlink()

    with pytest.raises(RuntimeError, match='not READY'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_manifest_not_bound_to_meta(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    with (stream_dir / 'manifest.jsonl').open('a') as file:
        file.write('\n')

    with pytest.raises(ValueError, match='manifest hash differs'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_ready_not_bound_to_meta(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    ready_path = stream_dir / 'READY.json'
    ready = json.loads(ready_path.read_text())
    ready['fingerprint'] = 'different-stream'
    ready_path.write_text(json.dumps(ready))

    with pytest.raises(ValueError, match='READY fingerprint differs'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_ready_manifest_not_bound_to_meta(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    ready_path = stream_dir / 'READY.json'
    ready = json.loads(ready_path.read_text())
    ready['manifest_sha256'] = 'different-manifest'
    ready_path.write_text(json.dumps(ready))

    with pytest.raises(ValueError, match='READY manifest hash differs'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_latent_receipt_not_bound_to_ready(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    (stream_dir / 'latent-materialized.json').write_text(json.dumps({'fixture': 'different'}))

    with pytest.raises(ValueError, match='latent receipt hash differs'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_same_schema_latent_replacement(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    latent_path = latent_dir / 'shard_00000.safetensors'
    replacement = torch.full_like(load_file(latent_path)['latents'], 17)
    latent_path.unlink()
    save_file({'latents': replacement}, latent_path)

    with pytest.raises(ValueError, match='latent shard content hash differs'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_training_ready_projection_skips_rehash_after_final_verification(
    tmp_path,
    monkeypatch,
):
    from pumit.ucpt import replay_dataset

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    meta = yaml.safe_load((stream_dir / 'meta.yaml').read_text())
    sha256_file = replay_dataset.sha256_file

    def reject_latent_rehash(path):
        if path.parent == latent_dir and path.name.startswith('shard_'):
            raise AssertionError(f'unexpected latent rehash: {path}')
        return sha256_file(path)

    monkeypatch.setattr(replay_dataset, 'sha256_file', reject_latent_rehash)

    replay_dataset._validate_v4_ready(
        stream_dir,
        latent_dir,
        meta,
        verify_content_hashes=False,
    )


def test_training_ready_accepts_fixed_prefix_stats_for_composite_stream(tmp_path):
    from pumit.ucpt import replay_dataset

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path, n_shards=2)
    meta, _ = _use_fixed_source_stats(stream_dir, latent_dir, source_count=7)

    replay_dataset._validate_v4_ready(stream_dir, latent_dir, meta)


@pytest.mark.parametrize('input_migration', [False, True])
def test_training_ready_accepts_top_level_fixed_stats(tmp_path, input_migration):
    from pumit.ucpt import replay_dataset

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path, n_shards=2)
    meta, _ = _use_fixed_source_stats(stream_dir, latent_dir, source_count=7)
    meta['normalization'] = meta.pop('composition')['normalization']
    if input_migration:
        meta['input_migration'] = {'contract': 'pass2-input-v1', 'data_root': str(tmp_path)}

    replay_dataset._validate_v4_ready(stream_dir, latent_dir, meta)

    meta['normalization']['source_stats_count'] = 6
    with pytest.raises(ValueError, match='stats count .* differs from source count'):
        replay_dataset._validate_v4_ready(stream_dir, latent_dir, meta)
    meta['normalization']['source_stats_count'] = 7
    meta['normalization']['source_stats_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='stats hash differs'):
        replay_dataset._validate_v4_ready(stream_dir, latent_dir, meta)


@pytest.mark.parametrize(
    ('field', 'value', 'match'),
    [
        ('source_stats_count', 6, 'stats count .* differs from source count'),
        ('source_stats_sha256', 'b' * 64, 'stats hash differs'),
        ('source_fingerprint', 'different-prefix', 'identity differs'),
    ],
)
def test_training_ready_rejects_invalid_fixed_source_stats_provenance(
    tmp_path,
    field,
    value,
    match,
):
    from pumit.ucpt import replay_dataset

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path, n_shards=2)
    meta, _ = _use_fixed_source_stats(stream_dir, latent_dir, source_count=7)
    meta['composition']['normalization'][field] = value

    with pytest.raises(ValueError, match=match):
        replay_dataset._validate_v4_ready(stream_dir, latent_dir, meta)


def test_training_ready_rejects_invalid_source_ready_digest_without_external_io(
    tmp_path,
):
    from pumit.ucpt import replay_dataset

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path, n_shards=2)
    meta, _ = _use_fixed_source_stats(stream_dir, latent_dir, source_count=7)
    meta['composition']['segments'][0]['source_ready_sha256'] = 'not-a-sha256'

    with pytest.raises(ValueError, match='source_ready_sha256'):
        replay_dataset._validate_v4_ready(stream_dir, latent_dir, meta)


def test_replay_rejects_missing_v4_mask_sidecar_at_startup(tmp_path, pipeline):
    from pumit.ucpt.mask_store import mask_shard_path

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    mask_shard_path(stream_dir, 0).unlink()

    with pytest.raises(FileNotFoundError, match='mask sidecar is missing'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_v4_mask_sidecar_size_mismatch_at_startup(tmp_path, pipeline):
    from pumit.ucpt.mask_store import mask_shard_path

    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    with mask_shard_path(stream_dir, 0).open('ab') as file:
        file.write(b'x')

    with pytest.raises(ValueError, match='mask sidecar size mismatch'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_replay_rejects_missing_v4_latent_shard_at_startup(tmp_path, pipeline):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    (latent_dir / 'shard_00000.safetensors').unlink()

    with pytest.raises(FileNotFoundError, match='latent shard is missing'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


@pytest.mark.parametrize(
    ('tensors', 'match'),
    [
        ({'other': torch.zeros(1)}, 'expected only a latents tensor'),
        ({'latents': torch.zeros(2, 32, dtype=torch.float16)}, 'expected latent shape'),
        ({'latents': torch.zeros(16, 32, dtype=torch.bfloat16)}, 'expected F16 latents'),
    ],
)
def test_replay_rejects_invalid_v4_latent_shard_at_startup(
    tmp_path,
    pipeline,
    tensors,
    match,
):
    stream_dir, latent_dir = _make_ready_v4_stream(tmp_path)
    save_file(tensors, latent_dir / 'shard_00000.safetensors')

    with pytest.raises(ValueError, match=match):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_n_ssl_patches_set_not_default_zero(tmp_path, pipeline):
    """The dataclass default is 0; teacher_vit_forward slices all_tokens[:0] and crashes
    if the SSL block forgets to assign it."""
    img = _write_image(tmp_path)
    s = _sample(img)
    stream_dir, latent_dir = _make_stream(tmp_path, [s])
    ds = UCPTReplayDataset(
        stream_dir=stream_dir, latent_dir=latent_dir, rank=0, world_size=1,
        pipeline=pipeline, data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    assert ds[0].n_ssl_patches > 0


def test_replay_rejects_classes_on_unlabeled_sample(tmp_path, pipeline):
    img = _write_image(tmp_path)
    sample = _sample(img)
    sample['classes'] = [
        {'source': 'src', 'name': 'liver', 'is_positive': True, 'target_voxels': 1},
    ]
    stream_dir, latent_dir = _make_stream(tmp_path, [sample])
    ds = UCPTReplayDataset(
        stream_dir=stream_dir,
        latent_dir=latent_dir,
        pipeline=pipeline,
        data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    with pytest.raises(ValueError, match='invalid labeled sample schema'):
        ds[0]


def test_collate_direct_builds_ucpt_batch():
    """_collate is a pure fn: builds a UCPTBatch from synthetic per-sample tensors
    with no I/O and no `self` (the seam the cost-model bench builds against)."""
    from pumit.ucpt.replay_dataset import _collate

    n_prefix = 5
    shapes = [(1, 4, 4), (1, 4, 4)]
    n_patches_list = [16, 16]
    total = sum(n_patches_list)
    all_patches = torch.randn(total, 3, 16, 16, 16, dtype=torch.bfloat16)
    latent = torch.randn(total, 32)
    rescales = [1.0, 0.7]
    seg_payloads = [None, None]
    b = _collate(
        all_patches, latent, n_patches_list, shapes, rescales,
        seg_payloads, mask_rng=np.random.default_rng(42),
        da_list=[0, 0],
        view_specs=[
            {'strategy': 'random', 'ratio_2d': (0.70, 0.80), 'ratio_3d': (0.75, 0.85)},
            {'strategy': 'block', 'ratio_2d': (0.70, 0.80), 'ratio_3d': (0.75, 0.85)},
        ],
        n_prefix=n_prefix,
        labeled_flags=[False, False],
    )
    assert b.n_ssl_patches == total
    assert b.total_seg_len == 0
    assert b.seg_patch_gather_idx is None
    assert b.patches.shape[0] == total
    assert b.total_teacher_len == sum(n_prefix + n for n in n_patches_list)
    assert b.num_blocks == 2 * len(n_patches_list)  # V=2 views per sample


def test_batch_rng_is_addressed_by_stream_position():
    def draw(seed, shard_id, batch_idx):
        return batch_rng(seed, shard_id, batch_idx).integers(
            0,
            2**63,
            size=8,
            dtype=np.int64,
        )

    expected = draw(42, 7, 11)
    assert np.array_equal(expected, draw(42, 7, 11))
    assert not np.array_equal(expected, draw(42, 8, 11))
    assert not np.array_equal(expected, draw(42, 7, 12))
    assert not np.array_equal(expected, draw(43, 7, 11))


def test_batch_rng_splits_masking_and_text_augmentation():
    mask_rng_a, text_rng_a = batch_rng(42, 7, 11).spawn(2)
    mask_rng_b, _ = batch_rng(42, 7, 11).spawn(2)

    text_rng_a.integers(0, 100, size=100)
    assert np.array_equal(
        mask_rng_a.integers(0, 2**63, size=8, dtype=np.int64),
        mask_rng_b.integers(0, 2**63, size=8, dtype=np.int64),
    )


def test_virtual_lanes_map_shards_and_resume_by_global_step(tmp_path, pipeline):
    stream_dir, latent_dir = _make_virtual_lane_stream(tmp_path)
    ds = UCPTReplayDataset(
        stream_dir=stream_dir,
        latent_dir=latent_dir,
        rank=0,
        world_size=2,
        virtual_lanes=4,
        pipeline=pipeline,
        data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )

    assert ds.my_lane_ids == [0, 2]
    assert len(ds) == 4
    assert ds[0].latents[::16, 0].tolist() == [0.0, 20.0]
    assert ds[1].latents[::16, 0].tolist() == [1.0, 21.0]
    assert ds[2].latents[::16, 0].tolist() == [40.0, 60.0]
    assert ds[2].latents[:, 0].unique().tolist() == [40.0, 60.0]

    resumed = UCPTReplayDataset(
        stream_dir=stream_dir,
        latent_dir=latent_dir,
        rank=0,
        world_size=2,
        virtual_lanes=4,
        start_offset=2,
        pipeline=pipeline,
        data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    assert len(resumed) == 2
    assert resumed[0].latents[::16, 0].tolist() == [40.0, 60.0]


def test_virtual_lanes_require_world_size_divisor(tmp_path, pipeline):
    stream_dir, latent_dir = _make_virtual_lane_stream(tmp_path)
    with pytest.raises(AssertionError, match='virtual_lanes .* divisible by world_size'):
        UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            rank=0,
            world_size=3,
            virtual_lanes=4,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )


def test_32_virtual_lanes_scale_by_global_dp_world_size(tmp_path, pipeline):
    stream_dir, latent_dir = _make_virtual_lane_stream(
        tmp_path,
        n_shards=32,
        batches_per_shard=1,
    )
    expected_rank_zero_lanes = {
        4: list(range(0, 32, 4)),
        8: [0, 8, 16, 24],
        16: [0, 16],
        32: [0],
    }
    for world_size, expected in expected_rank_zero_lanes.items():
        ds = UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            rank=0,
            world_size=world_size,
            virtual_lanes=32,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )
        assert ds.my_lane_ids == expected
        assert len(ds.my_lane_ids) == 32 // world_size


def test_virtual_lane_replay_rng_is_world_size_invariant(tmp_path, pipeline):
    stream_dir, latent_dir = _make_virtual_lane_stream(tmp_path)
    merged = UCPTReplayDataset(
        stream_dir=stream_dir,
        latent_dir=latent_dir,
        rank=0,
        world_size=1,
        virtual_lanes=4,
        pipeline=pipeline,
        data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )[0]

    n_patches = 16
    n_views = 2

    def sample_masked_indices(batch, sample_idx):
        blocks = []
        for view_idx in range(n_views):
            start = (sample_idx * n_views + view_idx) * n_patches
            stop = start + n_patches
            selected = batch.view_masked_idx[
                (batch.view_masked_idx >= start) & (batch.view_masked_idx < stop)
            ]
            blocks.append((selected - start).tolist())
        return blocks

    for lane_id in range(4):
        split = UCPTReplayDataset(
            stream_dir=stream_dir,
            latent_dir=latent_dir,
            rank=lane_id,
            world_size=4,
            virtual_lanes=4,
            pipeline=pipeline,
            data_root=tmp_path,
            text_cache_path=tmp_path / 'text.pt',
            class_captions_dir=tmp_path / 'captions',
        )[0]
        assert sample_masked_indices(merged, lane_id) == sample_masked_indices(split, 0)


def test_2d_view_masks_use_token_grid_only():
    masks, stats = _make_view_masks(
        (1, 32, 32),
        ratios=[0.75, 0.75],
        strategies=['random', 'block'],
        rng=np.random.default_rng(0),
    )

    assert len(masks) == len(stats) == 2
    assert [stat['strategy'] for stat in stats] == ['random', 'block']
    for mask, stat in zip(masks, stats):
        assert mask.shape == (32 * 32,)
        assert mask.any() and not mask.all()
        assert stat['visible_tokens'] == mask.sum()
        assert stat['masked_tokens'] == (~mask).sum()


def test_validate_view_specs_normalizes_ranges():
    specs = default_view_specs()
    specs[0]['ratio_2d'] = [0.7, 0.8]
    normalized = validate_view_specs(specs)
    assert normalized[0]['ratio_2d'] == (0.7, 0.8)
    assert normalized[1]['strategy'] == 'block'


@pytest.mark.parametrize(
    ('specs', 'error'),
    [
        (default_view_specs()[:1], 'exactly two'),
        (list(reversed(default_view_specs())), 'must use strategy'),
        (
            [
                {'strategy': 'random', 'ratio_2d': (0.0, 0.8), 'ratio_3d': (0.75, 0.85)},
                default_view_specs()[1],
            ],
            '0 < low <= high < 1',
        ),
        (
            [
                default_view_specs()[0],
                {'strategy': 'block', 'ratio_2d': (0.8, 0.7), 'ratio_3d': (0.75, 0.85)},
            ],
            '0 < low <= high < 1',
        ),
    ],
)
def test_validate_view_specs_rejects_invalid_policy(specs, error):
    with pytest.raises(ValueError, match=error):
        validate_view_specs(specs)


# ---------------------------------------------------------------------------
# _build_seg_payload (Task 4): per-sample seg targets, direct-call style.
# The batch-level seg loop (Task 5) wires these payloads into UCPTBatch; here
# we test the payload dict directly to keep Task 4 scoped.
# ---------------------------------------------------------------------------


def _make_seg_dataset(tmp_path, pipeline, s):
    stream_dir, latent_dir = _make_stream(tmp_path, [s])
    return UCPTReplayDataset(
        stream_dir=stream_dir, latent_dir=latent_dir, rank=0, world_size=1,
        pipeline=pipeline, data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )


def test_seg_payload_target_shape_and_stacked_structure(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True, da=0, patch_grid=(1, 4, 4))
    ds = _make_seg_dataset(tmp_path, pipeline, s)
    payload = ds._build_seg_payload(s, (1, 4, 4), 0)
    tm = payload['target_masks']
    assert tm.shape == (2, 16, 64, 64)
    assert tm.dtype == torch.bool
    # exact model indexing succeeds (stacked tensor, not list)
    _ = tm[0:1].unsqueeze(1)
    assert payload['text_embeddings'].shape == (2, 24, 1152)
    assert payload['text_valid_mask'].shape == (2, 24)
    assert payload['text_valid_mask'].dtype == torch.bool
    assert payload['text_valid_mask'].any(-1).all()  # no all-pad row
    assert payload['is_positive'].shape == (2,)
    assert payload['is_positive'].tolist() == [True, False]
    assert payload['patch_grid'] == (1, 4, 4)
    assert payload['da'] == 0


def test_negative_target_is_zeros(tmp_path, pipeline):
    img = _write_image(tmp_path)
    # kidney is negative -> no mask file needed; liver positive file exists
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True)
    ds = _make_seg_dataset(tmp_path, pipeline, s)
    payload = ds._build_seg_payload(s, (1, 4, 4), 0)
    # is_positive = [True, False] -> target_masks[1] (kidney, neg) all zero
    assert not payload['target_masks'][1].any()
    # positive target non-zero where mask present
    assert payload['target_masks'][0].any()


def test_seg_payload_samples_text_variant(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(
        tmp_path / 'text.pt',
        variants={
            'liver': ['liver', 'hepatic parenchyma'],
            'kidney': ['kidney'],
        },
    )
    s = _sample(img, label=True)
    ds = _make_seg_dataset(tmp_path, pipeline, s)

    payload = ds._build_seg_payload(
        s,
        (1, 4, 4),
        0,
        text_rng=np.random.default_rng(0),
    )

    assert payload['prompts'][0] == build_segmentation_prompt('hepatic parenchyma', 'CT')


def test_negative_only_contract_builds_payload_without_mask_file(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True)
    s['classes'] = [
        {'source': 'src', 'name': 'kidney', 'is_positive': False, 'target_voxels': 0},
    ]
    s['seg_cost_queries'] = 1
    ds = _make_seg_dataset(tmp_path, pipeline, s)

    payload = ds._build_seg_payload(s, (1, 4, 4), 0)

    assert payload['prompts'] == [build_segmentation_prompt('kidney', 'CT')]
    assert payload['is_positive'].tolist() == [False]
    assert not payload['target_masks'].any()


def test_single_voxel_positive_survives_full_resolution_supervision(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64), bright='point')
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True)
    s['classes'][0]['target_voxels'] = 1
    ds = _make_seg_dataset(tmp_path, pipeline, s)

    payload = ds._build_seg_payload(s, (1, 4, 4), 0)

    assert payload['target_masks'][0].count_nonzero() == 1
    assert payload['is_positive'].tolist() == [True, False]


def test_positive_target_that_vanishes_during_replay_fails(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True, patch_grid=(1, 2, 2))
    # The liver mask is in the upper-left quarter; replay a disjoint lower-right crop.
    s['params'][0]['load_slice_start'] = [0, 32, 32]
    s['params'][0]['load_slice_stop'] = [16, 64, 64]

    ds = _make_seg_dataset(tmp_path, pipeline, s)
    with pytest.raises(RuntimeError, match='positive target vanished during replay'):
        ds._build_seg_payload(s, (1, 2, 2), 0)


def test_nonzero_generation_replay_count_difference_is_allowed(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True)
    s['classes'][0]['target_voxels'] += 1

    ds = _make_seg_dataset(tmp_path, pipeline, s)
    payload = ds._build_seg_payload(s, (1, 4, 4), 0)

    assert payload['target_masks'][0].any()


def test_missing_positive_mask_raises(tmp_path, pipeline):
    img = _write_image(tmp_path)
    # no mask file written for 'liver' (positive)
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True)
    ds = _make_seg_dataset(tmp_path, pipeline, s)
    with pytest.raises(FileNotFoundError):
        ds._build_seg_payload(s, (1, 4, 4), 0)


def test_periphery_mask_survives_resampling(tmp_path, pipeline):
    """A mask touching the first W plane survives the frozen spatial transform."""
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64), bright='periphery')
    _write_text_cache(tmp_path / 'text.pt')
    s = _sample(img, label=True)
    s['classes'][0]['target_voxels'] = 16 * 64 * 4
    ds = _make_seg_dataset(tmp_path, pipeline, s)
    payload = ds._build_seg_payload(s, (1, 4, 4), 0)
    # liver is positive (index 0); the first W cell survives
    assert payload['target_masks'][0][..., 0, :].any()


def test_v4_replays_positive_and_negative_masks_without_raw_mask_files(tmp_path, pipeline):
    from pumit.ucpt.mask_store import (
        MASK_FRAME_KEY,
        encode_positive_masks,
        mask_shard_path,
        write_mask_shard,
    )

    img = _write_image(tmp_path)
    _write_text_cache(tmp_path / 'text.pt')
    labeled = _sample(img, label=True)
    unlabeled = _sample(img, label=False)
    positive_mask = np.zeros((16, 64, 64), dtype=np.bool_)
    positive_mask[:, 8:16, 8:16] = True
    labeled['classes'][0]['target_voxels'] = int(positive_mask.sum())
    labeled[MASK_FRAME_KEY] = encode_positive_masks([positive_mask])

    stream_dir, latent_dir = _make_stream(
        tmp_path,
        [labeled, unlabeled],
        unlabeled_only_latents=True,
        latent_dtype=torch.float16,
    )
    batches = [{'step_idx': 0, 'samples': [labeled, unlabeled]}]
    write_mask_shard(batches, mask_shard_path(stream_dir, 0))
    with (stream_dir / 'shard_00000.msgpack').open('wb') as file:
        msgpack.pack({'batches': batches}, file)
    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['batches_per_shard'] = 1
    meta_path.write_text(yaml.safe_dump(meta))
    _write_v4_ready(stream_dir, latent_dir)

    ds = UCPTReplayDataset(
        stream_dir=stream_dir,
        latent_dir=latent_dir,
        rank=0,
        world_size=1,
        pipeline=pipeline,
        data_root=tmp_path / 'raw-masks-do-not-exist',
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    batch = ds[0]

    assert len(batch.target_masks) == 1
    assert torch.equal(batch.target_masks[0][0], torch.from_numpy(positive_mask))
    assert not batch.target_masks[0][1].any()


def test_v4_replays_negative_only_labeled_sample_without_raw_mask_files(tmp_path, pipeline):
    from pumit.ucpt.mask_store import MASK_FRAME_KEY, mask_shard_path, write_mask_shard

    img = _write_image(tmp_path)
    _write_text_cache(tmp_path / 'text.pt')
    labeled = _sample(img, label=True)
    labeled['classes'] = [
        {'source': 'src', 'name': 'kidney', 'is_positive': False, 'target_voxels': 0},
    ]
    labeled['seg_cost_queries'] = 1
    labeled[MASK_FRAME_KEY] = b''
    unlabeled = _sample(img, label=False)
    stream_dir, latent_dir = _make_stream(
        tmp_path,
        [labeled, unlabeled],
        batches_per_shard=1,
        unlabeled_only_latents=True,
        latent_dtype=torch.float16,
    )
    batches = [{'step_idx': 0, 'samples': [labeled, unlabeled]}]
    write_mask_shard(batches, mask_shard_path(stream_dir, 0))
    with (stream_dir / 'shard_00000.msgpack').open('wb') as file:
        msgpack.pack({'batches': batches}, file)
    meta_path = stream_dir / 'meta.yaml'
    meta = yaml.safe_load(meta_path.read_text())
    meta['batches_per_shard'] = 1
    meta_path.write_text(yaml.safe_dump(meta))
    _write_v4_ready(stream_dir, latent_dir)

    ds = UCPTReplayDataset(
        stream_dir=stream_dir,
        latent_dir=latent_dir,
        pipeline=pipeline,
        data_root=tmp_path / 'raw-masks-do-not-exist',
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    batch = ds[0]

    assert len(batch.target_masks) == 1
    assert batch.is_positive[0].tolist() == [False]
    assert not batch.target_masks[0].any()


# ---------------------------------------------------------------------------
# Batch-level seg half (Task 5): gather-from-SSL-region, no second copy.
# ---------------------------------------------------------------------------


def _make_multi_sample_batch(tmp_path, samples, pipeline):
    """Build a stream+latent fixture with one batch containing all *samples*.

    _make_stream packs one sample per batch; here we overwrite the shard so the
    full sample list is a single batch (preserving order).
    """
    stream_dir, latent_dir = _make_stream(tmp_path, samples)
    shard = [{'step_idx': 0, 'samples': samples}]
    with open(stream_dir / 'shard_00000.msgpack', 'wb') as f:
        msgpack.pack({'batches': shard}, f)
    with open(stream_dir / 'meta.yaml', 'w') as f:
        yaml.dump({'stream_complete': True, 'n_shards': 1, 'batches_per_shard': 1, 'seed': 42}, f)
    return UCPTReplayDataset(
        stream_dir=stream_dir, latent_dir=latent_dir, rank=0, world_size=1,
        pipeline=pipeline, data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )


def test_seg_gather_points_into_ssl_region(tmp_path, pipeline):
    """seg_patch_gather_idx gathers each labeled sample's patches from its
    FULL-ARRAY slot (full_offset_table[i]); total_seg_len == sum(n_prefix +
    n_p_i); mask True-count == total_seg_patches == len(gather_idx). Under the
    unlabeled-only SSL design, seg gathers from the full patches array (which
    holds ALL samples), so the gather idx may exceed n_ssl_patches (unlabeled
    only)."""
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    # 2 samples: one labeled, one unlabeled, one labeled
    s1 = _sample(img, label=True, rope_rescale=1.0)
    s2 = _sample(img, label=False, rope_rescale=1.0)
    s3 = _sample(img, label=True, rope_rescale=0.7)
    ds = _make_multi_sample_batch(tmp_path, [s1, s2, s3], pipeline)
    batch = ds[0]

    n_prefix = ds.n_prefix
    n_p = s1['n_patches']  # all samples have the same n_patches here
    total_patches = batch.patches.shape[0]  # full array holds ALL samples
    full_offset_0 = 0                       # s1's full-array offset
    full_offset_2 = 2 * n_p                 # s3's full-array offset (s1 + s2)
    total_seg_len = 2 * (n_prefix + n_p)
    assert batch.total_seg_len == total_seg_len
    assert len(batch.seg_patch_mask) == total_seg_len
    assert batch.seg_patch_mask.sum().item() == 2 * n_p == len(batch.seg_patch_gather_idx)
    # gather idx stays within the FULL patch array (labeled samples are
    # interspersed with unlabeled, so max can exceed n_ssl_patches).
    assert batch.seg_patch_gather_idx.max().item() < total_patches
    # gather idx is the concat of the two labeled samples' full-array slots
    expected = torch.cat([torch.arange(full_offset_0, full_offset_0 + n_p),
                          torch.arange(full_offset_2, full_offset_2 + n_p)])
    assert torch.equal(batch.seg_patch_gather_idx, expected)


def test_seg_coords_use_per_sample_rope_rescale_and_prefix_zero(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s1 = _sample(img, label=True, rope_rescale=1.0)
    s3 = _sample(img, label=True, rope_rescale=0.7)
    s2 = _sample(img, label=False)
    ds = _make_multi_sample_batch(tmp_path, [s1, s2, s3], pipeline)
    batch = ds[0]
    n_prefix = ds.n_prefix
    n_p = s1['n_patches']
    # prefix coords are zero (first n_prefix of each labeled sample)
    assert torch.allclose(batch.seg_coords[:n_prefix], torch.zeros(n_prefix, 3))
    # Per-sample rope_rescale shows up in the coord ranges. _sample_coords is
    # linear in r: max abs = ((D-1)/D, (H-1)/H, (W-1)/W).max() * r. For grid
    # (1,4,4) the max is 0.75 * r (0.75 from the H/W dims, D=1 gives 0).
    seg0 = batch.seg_coords[n_prefix:n_prefix + n_p]
    seg1 = batch.seg_coords[(n_prefix + n_p) + n_prefix:(n_prefix + n_p) + n_prefix + n_p]
    assert abs(seg0.abs().max().item() - 0.75) < 1e-5          # r=1.0
    assert abs(seg1.abs().max().item() - 0.75 * 0.7) < 1e-5    # r=0.7
    # ratio check (robust to grid shape): seg1/seg0 == 0.7
    assert abs(seg1.abs().max().item() / seg0.abs().max().item() - 0.7) < 1e-5


def test_layout_invariant_patches_holds_all_samples_once(tmp_path, pipeline):
    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s1 = _sample(img, label=True); s2 = _sample(img, label=False)
    ds = _make_multi_sample_batch(tmp_path, [s1, s2], pipeline)
    batch = ds[0]
    # patches holds ALL samples once (no second copy of labeled patches);
    # n_ssl_patches counts UNLABELED patches only, so the two diverge when a
    # labeled sample is present.
    assert batch.patches.shape[0] == s1['n_patches'] + s2['n_patches']
    assert batch.n_ssl_patches == s2['n_patches']  # unlabeled only


def test_zero_labeled_batch_yields_seg_none(tmp_path, pipeline):
    img = _write_image(tmp_path)
    s1 = _sample(img, label=False); s2 = _sample(img, label=False)
    ds = _make_multi_sample_batch(tmp_path, [s1, s2], pipeline)
    batch = ds[0]
    assert batch.total_seg_len == 0
    assert batch.seg_patch_gather_idx is None
    assert batch.n_ssl_patches > 0  # SSL half still complete


def test_end_to_end_through_ucpt_model(tmp_path, pipeline, monkeypatch):
    """The dataset's UCPTBatch drives UCPTModel.forward (student+teacher+seg)
    without shape/index errors and returns finite losses."""
    from pumit.model.vit import ViT, ViTConfig
    from pumit.ucpt.model import UCPTModel, SegDecoderStack
    from pumit.ucpt.ssl.heads import (
        ReconDecoder, PatchDistillDecoder, ClsPredictor,
    )
    from tests.ucpt.conftest import cast_ucpt_batch_dtype, use_cpu_attention_reference

    use_cpu_attention_reference(monkeypatch)

    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    s1 = _sample(img, label=True, rope_rescale=1.0)
    s2 = _sample(img, label=False)
    stream_dir, latent_dir = _make_stream(tmp_path, [s1, s2])
    shard = [{'step_idx': 0, 'samples': [s1, s2]}]
    with open(stream_dir / 'shard_00000.msgpack', 'wb') as f:
        msgpack.pack({'batches': shard}, f)
    with open(stream_dir / 'meta.yaml', 'w') as f:
        yaml.dump({'stream_complete': True, 'n_shards': 1, 'batches_per_shard': 1, 'seed': 42}, f)
    ds = UCPTReplayDataset(
        stream_dir=stream_dir, latent_dir=latent_dir, rank=0, world_size=1,
        pipeline=pipeline, data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    batch = ds[0]

    # e2e test must exercise the seg forward (labeled sample present); the
    # SSL-only path would still produce finite loss and silently mask a dropped
    # labeled sample.
    assert batch.seg_patch_gather_idx is not None and batch.total_seg_len > 0, \
        'e2e test must exercise the seg forward (labeled sample present)'

    EMBED_DIM = 256
    # latent_channels=32 matches the dataset's latents (FLUX2 AE codec dim;
    # _make_stream writes 32-dim latents), not the model test fixture's 16.
    vit = ViT(ViTConfig(hidden_size=EMBED_DIM, num_hidden_layers=6, num_attention_heads=4,
                        intermediate_size=512, patch_size=16, num_register_tokens=4, grad_ckpt=True))
    model = UCPTModel(
        vit=vit,
        recon_decoder=ReconDecoder(encoder_dim=EMBED_DIM, decoder_dim=128, depth=1, num_heads=4, latent_channels=32),
        patch_distill_decoder=PatchDistillDecoder(encoder_dim=EMBED_DIM, decoder_dim=128, depth=1, num_heads=4, output_dim=EMBED_DIM),
        cls_predictor=ClsPredictor(embed_dim=EMBED_DIM, hidden_dim=512),
        seg=SegDecoderStack(embed_dim=EMBED_DIM),
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()
    out = model(batch)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.seg_loss)
    assert out.seg_loss.item() != 0.0, 'seg loss is zero — labeled sample was silently dropped from the seg forward'
    out.loss.backward()


def test_seg_loss_all_negative_sample_is_finite():
    """An all-negative labeled sample yields a finite loss (no empty stack, no NaN)."""
    from pumit.ucpt.seg.loss import seg_loss
    K, D, H, W = 3, 4, 8, 8
    logits = torch.randn(K, 1, D, H, W)
    targets = torch.zeros(K, 1, D, H, W)          # negatives -> zero targets
    is_positive = torch.zeros(K, dtype=torch.bool)  # all-negative
    out = seg_loss(logits, targets, is_positive)
    assert torch.isfinite(out['focal'])
    assert torch.isfinite(out['dice_sum'])
    assert out['n_pos'] == 0          # all-negative: Dice contributes nothing
    assert float(out['dice_sum']) == 0.0


def test_collate_ssl_tensors_cover_unlabeled_only():
    """Mixed batch: 2 unlabeled + 1 labeled (order: unlab, lab, unlab). SSL
    masks/latents/coords cover only the 2 unlabeled samples' patches;
    num_blocks == 2*n_unlabeled; teacher_patch_gather_idx indexes the unlabeled
    patches in the full array; seg covers the one labeled sample."""
    import torch
    from pumit.ucpt.replay_dataset import _collate

    n_prefix = 5
    n_patches_list = [24, 24, 24]          # full-array order
    labeled_flags = [False, True, False]
    shapes = [(6, 2, 2), (6, 2, 2), (6, 2, 2)]
    total_patches = sum(n_patches_list)     # 72
    n_unlab_patches = 48

    all_patches = torch.randn(total_patches, 3, 16, 16, 16)
    latent = torch.randn(n_unlab_patches, 32)   # UNLABELED patches only
    rescales = [1.0, 1.0, 1.0]
    seg_payloads = [None,
                    {'patch_grid': (6, 2, 2), 'da': 2,
                     'text_embeddings': torch.randn(2, 24, 1152),
                     'text_valid_mask': torch.arange(24).unsqueeze(0).expand(2, 24) < 6,
                     'is_positive': torch.tensor([True, False]),
                     'target_masks': torch.randn(2, 6, 2, 2)},
                    None]

    b = _collate(
        all_patches, latent, n_patches_list, shapes, rescales,
        seg_payloads, mask_rng=np.random.default_rng(1),
        da_list=[2, 2, 2],
        view_specs=[
            {'strategy': 'random', 'ratio_2d': (0.70, 0.80), 'ratio_3d': (0.75, 0.85)},
            {'strategy': 'block', 'ratio_2d': (0.70, 0.80), 'ratio_3d': (0.75, 0.85)},
        ],
        n_prefix=n_prefix,
        labeled_flags=labeled_flags,
    )

    assert b.n_ssl_patches == n_unlab_patches
    assert b.n_views == 2
    # Visible/masked indices partition V full-grid decoder blocks.
    decoder_total = b.n_views * n_unlab_patches
    assert b.view_visible_idx.numel() + b.view_masked_idx.numel() == decoder_total
    assert torch.equal(
        torch.cat([b.view_visible_idx, b.view_masked_idx]).sort().values,
        torch.arange(decoder_total),
    )
    assert b.view_decoder_coords.shape[0] == b.n_views * n_unlab_patches
    assert b.latents.shape[0] == n_unlab_patches
    assert b.num_blocks == 2 * 2  # V=2 views * 2 unlabeled samples
    # teacher gathers unlabeled sample 0 [0,24) and sample 2 [48,72); labeled (1) excluded.
    expected = torch.cat([torch.arange(0, 24), torch.arange(48, 72)])
    assert torch.equal(b.teacher_patch_gather_idx, expected)
    assert b.total_seg_len == n_prefix + 24
    # sample_is_labeled marks the middle sample.
    assert torch.equal(b.sample_is_labeled, torch.tensor([False, True, False]))
    # student gather indices point into the FULL array (max index < total_patches,
    # and none fall inside the labeled sample's [24,48) range).
    gi = b.student_patch_gather_idx
    assert gi.max().item() < total_patches
    assert not (((gi >= 24) & (gi < 48)).any())
    # view_target_gather_idx addresses unlabeled-LOCAL patch space (< n_unlab_patches).
    assert b.view_target_gather_idx.max().item() < n_unlab_patches


def test_collate_rejects_payloadless_labeled_sample():
    from pumit.ucpt.replay_dataset import _collate

    n_patches_list = [8, 8, 8]
    payload = {
        'patch_grid': (2, 2, 2),
        'da': 2,
        'text_embeddings': torch.randn(1, 24, 1152),
        'text_valid_mask': torch.ones(1, 24, dtype=torch.bool),
        'is_positive': torch.tensor([False]),
        'target_masks': torch.zeros(1, 2, 2, 2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match='payload presence'):
        _collate(
            torch.randn(24, 3, 16, 16, 16),
            torch.randn(8, 32),
            n_patches_list,
            [(2, 2, 2)] * 3,
            [1.0] * 3,
            [None, None, payload],
            mask_rng=np.random.default_rng(1),
            da_list=[2, 2, 2],
            view_specs=default_view_specs(),
            n_prefix=5,
            labeled_flags=[False, True, True],
        )


def test_end_to_end_mixed_batch_forward(tmp_path, pipeline, monkeypatch):
    """A stream batch with 2 unlabeled + 1 labeled sample (order unlab, lab,
    unlab): dataset __getitem__ builds a UCPTBatch (SSL over unlabeled only, seg
    over labeled), and a tiny model forward produces finite losses. Proves the
    collate->model seam under the unlabeled-only-recon design: latents are
    stored unlabeled-only and the latent slice from _load_shard matches
    n_ssl_patches, recon runs over unlabeled patches, seg over the labeled one.

    Scope: this is a seam/finiteness check (misaligned gathers -> NaN/inf).
    Precise teacher/student gather-index correctness (exact indices, no leak
    into the labeled range) is asserted separately by
    test_collate_ssl_tensors_cover_unlabeled_only.
    """
    from tests.ucpt.test_ucpt_model import _build_model
    from tests.ucpt.conftest import cast_ucpt_batch_dtype, use_cpu_attention_reference

    use_cpu_attention_reference(monkeypatch)

    img = _write_image(tmp_path)
    _write_mask(tmp_path, 'ds', 'key', 'src', 'liver', (16, 64, 64))
    _write_text_cache(tmp_path / 'text.pt')
    # Order [unlab, lab, unlab]: SSL covers the two unlabeled samples, seg the
    # middle labeled one. latent_dim=16 matches _build_model's recon_decoder
    # (latent_channels=16); unlabeled_only_latents writes rows for s1+s3 only.
    s1 = _sample(img, label=False, rope_rescale=1.0)
    s2 = _sample(img, label=True, rope_rescale=0.7)
    s3 = _sample(img, label=False, rope_rescale=1.0)
    samples = [s1, s2, s3]
    stream_dir, latent_dir = _make_stream(
        tmp_path, samples, unlabeled_only_latents=True, latent_dim=16,
    )
    # Pack all 3 samples into a single batch, preserving the [unlab, lab, unlab] order.
    shard = [{'step_idx': 0, 'samples': samples}]
    with open(stream_dir / 'shard_00000.msgpack', 'wb') as f:
        msgpack.pack({'batches': shard}, f)
    with open(stream_dir / 'meta.yaml', 'w') as f:
        yaml.dump({'stream_complete': True, 'n_shards': 1, 'batches_per_shard': 1, 'seed': 42}, f)

    ds = UCPTReplayDataset(
        stream_dir=stream_dir, latent_dir=latent_dir, rank=0, world_size=1,
        pipeline=pipeline, data_root=tmp_path,
        text_cache_path=tmp_path / 'text.pt',
        class_captions_dir=tmp_path / 'captions',
    )
    batch = ds[0]

    n_unlab_patches = s1['n_patches'] + s3['n_patches']  # 16 + 16 = 32
    assert batch.n_ssl_patches == n_unlab_patches
    assert batch.latents.shape[0] == n_unlab_patches
    assert batch.total_seg_len > 0  # labeled sample present -> seg forward exercised

    model = _build_model()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()
    out = model(batch)
    assert torch.isfinite(out.loss)
    for name in (
        'recon_loss', 'image_distill_loss', 'patch_distill_loss',
        'seg_loss', 'seg_focal_loss', 'seg_dice_loss',
    ):
        v = getattr(out, name)
        assert torch.isfinite(v), f'{name} not finite: {v}'
    # seg_loss must be nonzero: a labeled sample present but silently dropped
    # would still yield finite (zero) seg_loss, masking the regression.
    assert out.seg_loss.item() != 0.0, \
        'seg_loss is zero — labeled sample was dropped from the seg forward'
    out.loss.backward()
