# tests/ucpt/test_viz.py
"""UCPT viz tests. Unit tests use synthetic masks/params (no real dataset)."""
from pathlib import Path
import io

import numpy as np
import pytest
import torch

from pumit.segmentation_mask import pack_binary_mask


def test_apply_affine_to_mask_crops_and_resizes():
    """A raw mask, cropped + affine-applied, lands at spatial_size=patch_size."""
    from pumit.ucpt.mask import _apply_affine_to_mask

    # 40x256x256 raw volume; crop [10:30, 50:150, 50:150] -> affine to [20,100,100]
    shape_3d = (40, 256, 256)
    raw_mask = np.zeros(shape_3d, dtype=bool)
    raw_mask[10:30, 50:150, 50:150] = True

    params = [{
        'patch_size': [20, 100, 100],
        'affine': np.eye(4).ravel().tolist(),
        'load_slice_start': [10, 50, 50],
        'load_slice_stop': [30, 150, 150],
    }]

    out = _apply_affine_to_mask(raw_mask, params)
    assert out.shape == (1, 20, 100, 100)
    assert out.dtype == torch.float32
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    # the cropped region was all-1 -> output is all-1
    assert out.sum() == out.numel()


def test_load_mask_returns_none_when_dir_absent(tmp_path):
    from pumit.ucpt.mask import _load_mask
    assert _load_mask(tmp_path, 'd', 'k', 'src', 'cls', (1, 1, 1)) is None


def test_load_mask_loads_zst_file(tmp_path):
    """A written packbits+zst mask loads back as the unpacked (D,H,W) bool array."""
    import io
    import zstandard as zstd
    from pumit.ucpt.mask import _load_mask

    shape_3d = (2, 2, 3)  # 12 voxels
    mask = np.zeros(shape_3d, dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 2] = True

    label_dir = tmp_path / 'd' / 'labels' / 'k' / 'src'
    label_dir.mkdir(parents=True)
    packed = pack_binary_mask(mask)
    buf = io.BytesIO()
    np.save(buf, packed)
    cctx = zstd.ZstdCompressor()
    (label_dir / 'masks__cls.npy.zst').write_bytes(cctx.compress(buf.getvalue()))

    out = _load_mask(tmp_path, 'd', 'k', 'src', 'cls', shape_3d)
    assert out.shape == shape_3d
    assert out.dtype == bool
    assert np.array_equal(out, mask)
    out_u8 = _load_mask(tmp_path, 'd', 'k', 'src', 'cls', shape_3d, as_bool=False)
    assert out_u8.dtype == np.uint8
    assert np.array_equal(out_u8, mask)


def _write_synthetic_sample(tmp_path):
    """Write a tiny .npy image + a .npy.zst mask; return a sample dict + data_root."""
    import zstandard as zstd
    img_vol = np.random.RandomState(0).random((1, 20, 64, 64)).astype(np.float16)
    img_path = tmp_path / 'img.npy'
    np.save(img_path, img_vol)

    data_root = tmp_path
    label_dir = data_root / 'ds' / 'labels' / 'key' / 'src'
    label_dir.mkdir(parents=True)
    mask = np.zeros((20, 64, 64), dtype=bool)
    mask[5:15, 20:40, 20:40] = True
    packed = pack_binary_mask(mask)
    mask_buf = io.BytesIO()
    np.save(mask_buf, packed)
    cctx = zstd.ZstdCompressor()
    (label_dir / 'm__cls.npy.zst').write_bytes(cctx.compress(mask_buf.getvalue()))

    sample = {
        'img': str(img_path),
        'da_enc': 0,
        'n_patches': 20 * 4 * 4,
        'depth': 20,
        'spacing_label': [2.0, 1.0, 1.0],
        'rope_rescale': 1.0,
        'labeled': True,
        'params': [{
            'patch_size': [20, 64, 64],
            'affine': np.eye(4).ravel().tolist(),
            'load_slice_start': [0, 0, 0],
            'load_slice_stop': [20, 64, 64],
        }],
        'dataset': 'ds', 'key': 'key', 'modality': 'CT',
        'classes': [
            {'source': 'src', 'name': 'cls', 'is_positive': True, 'target_voxels': 4000},
        ],
    }
    return sample, data_root


def test_render_sample_contract(tmp_path):
    """render_sample returns image/raw/masks/raw_masks with correct shapes/dtypes."""
    from pumit.ucpt.transforms import build_ucpt_pipeline
    from pumit.ucpt.viz import render_sample

    sample, data_root = _write_synthetic_sample(tmp_path)
    pipeline = build_ucpt_pipeline(
        size_xy_choices=[64], size_xy_choices_2d=[64],
        max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12}, max_da=4,
    )
    state = {'shape': np.array([20, 64, 64]), 'spacing': np.array([2.0, 1.0, 1.0])}
    params = pipeline.sample_params(state, np.random.default_rng(0))
    assert params is not None
    sample['params'] = params

    out = render_sample(pipeline, sample, data_root=data_root)
    assert out['image'].dtype == np.uint8
    assert out['image'].ndim == 3 and out['image'].shape[0] == 24
    assert out['raw'].dtype == np.uint8 and out['raw'].shape[0] == 20
    assert 'src:cls' in out['masks'] and out['masks']['src:cls'].dtype == bool
    assert 'src:cls' in out['raw_masks'] and out['raw_masks']['src:cls'].dtype == bool
    assert out['raw_masks']['src:cls'].shape == out['raw'].shape
    assert 'label_classes' not in out['meta'] and out['meta']['labeled']


def test_render_sample_missing_positive_mask_raises(tmp_path):
    from pumit.ucpt.transforms import build_ucpt_pipeline
    from pumit.ucpt.viz import render_sample

    sample, data_root = _write_synthetic_sample(tmp_path)
    sample['classes'] = [
        {'source': 'NOPE', 'name': 'cls', 'is_positive': True, 'target_voxels': 4000},
    ]
    pipeline = build_ucpt_pipeline(
        size_xy_choices=[64], size_xy_choices_2d=[64],
        max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12}, max_da=4,
    )
    state = {'shape': np.array([20, 64, 64]), 'spacing': np.array([2.0, 1.0, 1.0])}
    params = pipeline.sample_params(state, np.random.default_rng(0))
    sample['params'] = params
    with pytest.raises(FileNotFoundError, match='missing positive mask'):
        render_sample(pipeline, sample, data_root=data_root)


def test_phase_indices_no_drops_and_clamps():
    """Every slice appears; shorter parts clamp at the index; no IndexError."""
    from pumit.ucpt.viz import phase_indices
    # D=128, grid 3x3 -> 9 parts [15,15,14,14,14,14,14,14,14], max_len=15
    pages = phase_indices(128, grid_h=3, grid_w=3)
    shown = sorted({s for pg in pages for s in pg})
    assert shown == list(range(128))   # no drops
    assert len(pages) == 15            # max_part_len phases
    assert all(len(pg) == 9 for pg in pages)


def test_phase_indices_thin_and_2d():
    from pumit.ucpt.viz import phase_indices
    # D < grid_h*grid_w -> n_cells = D, each part len 1
    assert phase_indices(3, grid_h=3, grid_w=3) == [[0, 1, 2]]
    # 2D
    assert phase_indices(1, grid_h=1, grid_w=1) == [[0]]


def test_phase_indices_non_square_grid():
    from pumit.ucpt.viz import phase_indices
    # 3x4 grid, D=10 -> n_cells=10, one phase
    pages = phase_indices(10, grid_h=3, grid_w=4)
    assert len(pages) == 1
    assert sorted(pages[0]) == list(range(10))
