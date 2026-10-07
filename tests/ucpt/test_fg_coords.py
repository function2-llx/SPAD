# tests/ucpt/test_fg_coords.py
"""Tests for fg_coords: placement helper, cache reader, precompute asserts."""
from __future__ import annotations

import numpy as np
import pytest

from pumit.segmentation_mask import pack_binary_mask
from pumit.ucpt.fg_coords import place_crop_start


class TestPlaceCropStart:
    @staticmethod
    def _place(voxel, load_size, shape, rng, *, jitter_fraction=0):
        return place_crop_start(
            voxel,
            load_size,
            shape,
            load_size,
            np.eye(3),
            rng,
            jitter_fraction=jitter_fraction,
        )

    def test_centers_on_voxel_interior(self):
        # voxel well inside; zero jitter -> crop centered on voxel
        shape = np.array([100, 200, 200])
        load_size = np.array([16, 64, 64])
        voxel = np.array([50, 100, 100])
        rng = np.random.default_rng(0)
        start = self._place(voxel, load_size, shape, rng)
        # voxel inside [start, start+load_size)
        assert np.all(start <= voxel)
        assert np.all(voxel < start + load_size)
        # centered: start == voxel - load_size//2
        assert np.array_equal(start, voxel - load_size // 2)

    def test_clamps_at_low_edge(self):
        shape = np.array([100, 200, 200])
        load_size = np.array([16, 64, 64])
        voxel = np.array([0, 0, 0])
        rng = np.random.default_rng(0)
        start = self._place(voxel, load_size, shape, rng)
        assert np.array_equal(start, np.array([0, 0, 0]))

    def test_clamps_at_high_edge(self):
        shape = np.array([100, 200, 200])
        load_size = np.array([16, 64, 64])
        voxel = np.array([99, 199, 199])
        rng = np.random.default_rng(0)
        start = self._place(voxel, load_size, shape, rng)
        assert np.array_equal(start, shape - load_size)

    @pytest.mark.parametrize('output_to_source', [
        np.diag([1.0, 0.75, 0.75]),
        np.diag([1.0, -4 / 3, 4 / 3]),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2**-0.5, -2**-0.5],
            [0.0, 2**-0.5, 2**-0.5],
        ]),
        np.array([
            [0.98, 0.12, 0.05],
            [-0.10, 0.94, -0.24],
            [-0.08, 0.23, 0.95],
        ]),
    ])
    def test_jitter_is_bounded_in_post_affine_coordinates(self, output_to_source):
        shape = np.array([512, 512, 512])
        crop_size = np.array([64, 128, 128])
        load_size = np.ceil(np.abs(output_to_source) @ crop_size).astype(np.int64)
        voxel = np.array([256, 256, 256])
        rng = np.random.default_rng(1)
        recovered = []
        for _ in range(200):
            start = place_crop_start(
                voxel,
                load_size,
                shape,
                crop_size,
                output_to_source,
                rng,
                jitter_fraction=1 / 4,
            )
            source_center = start + load_size // 2
            recovered.append(np.linalg.solve(output_to_source, voxel - source_center))
        recovered = np.asarray(recovered)
        tolerance = np.linalg.norm(np.linalg.inv(output_to_source), ord=np.inf)
        assert np.abs(recovered[:, 0]).max() <= tolerance
        assert np.abs(recovered[:, 1:]).max() <= crop_size[1] / 4 + tolerance
        assert np.ptp(recovered[:, 1:], axis=0).min() > crop_size[1] / 3
        assert np.abs(recovered[:, 1:].mean(axis=0)).max() < crop_size[1] / 32

    def test_same_output_jitter_distribution_across_scale(self):
        shape = np.array([512, 512, 512])
        crop_size = np.array([64, 128, 128])
        voxel = np.array([256, 256, 256])
        recovered = []
        for scale in (0.75, 1.0, 4 / 3):
            output_to_source = np.diag([1.0, scale, scale])
            load_size = np.ceil(np.abs(output_to_source) @ crop_size).astype(np.int64)
            start = place_crop_start(
                voxel,
                load_size,
                shape,
                crop_size,
                output_to_source,
                np.random.default_rng(10),
                jitter_fraction=1 / 4,
            )
            source_center = start + load_size // 2
            recovered.append(np.linalg.solve(output_to_source, voxel - source_center))
        assert np.allclose(recovered, recovered[0], atol=1)

    def test_actual_resampling_preserves_output_space_jitter(self):
        import torch

        from pumit.transforms.loader import get_rotation_matrix
        from pumit.ucpt.affine import _affine_resample, _realized_output_to_source

        shape = np.array([96, 128, 128])
        crop_size = np.array([32, 64, 64])
        voxel = np.array([48, 64, 64])
        axes = torch.meshgrid(*(torch.arange(int(n)) for n in shape), indexing='ij')
        distance_sq = sum((axis - int(center)) ** 2 for axis, center in zip(axes, voxel, strict=True))
        volume = torch.exp(-distance_sq / (2 * 2.0**2)).unsqueeze(0)

        axial = get_rotation_matrix((1, 0, 0), np.pi / 4) @ np.diag([1.0, 0.9, 1.2])
        oblique_axis = np.array([0.98, 0.14, 0.14])
        oblique_axis /= np.linalg.norm(oblique_axis)
        oblique = get_rotation_matrix(oblique_axis, 1.1) @ np.diag([1.1, 0.8, 1.2])
        oblique[:, 2] *= -1
        transforms = [
            np.diag([1.0, 0.75, 0.75]),
            np.diag([1.0, -4 / 3, 4 / 3]),
            axial,
            oblique,
        ]

        centroids = []
        for affine_3x3 in transforms:
            load_size = np.ceil(np.abs(affine_3x3) @ crop_size).astype(np.int64)
            output_to_source = _realized_output_to_source(affine_3x3, crop_size)
            start = place_crop_start(
                voxel,
                load_size,
                shape,
                crop_size,
                output_to_source,
                np.random.default_rng(10),
                jitter_fraction=1 / 4,
            )
            stop = start + load_size
            crop = volume[:, start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            affine = np.eye(4)
            affine[:3, :3] = affine_3x3
            result = _affine_resample(
                crop,
                affine,
                crop_size.tolist(),
                interp_mode='trilinear',
                grid_mode='bilinear',
            )[0].double()
            output_axes = torch.meshgrid(
                *(torch.arange(int(n), dtype=torch.double) for n in crop_size),
                indexing='ij',
            )
            centroids.append(np.array([(result * axis).sum().item() / result.sum().item() for axis in output_axes]))

        centroids = np.asarray(centroids)
        assert np.ptp(centroids, axis=0).max() < 2
        assert np.abs(centroids[:, 0] - (crop_size[0] - 1) / 2).max() < 2
        assert np.all(centroids[:, 1:] >= crop_size[1:] / 4 - 2)
        assert np.all(centroids[:, 1:] <= 3 * crop_size[1:] / 4 + 2)

    def test_asserts_load_size_le_shape(self):
        shape = np.array([10, 10, 10])
        load_size = np.array([16, 64, 64])  # exceeds shape
        voxel = np.array([5, 5, 5])
        rng = np.random.default_rng(0)
        with pytest.raises(AssertionError):
            self._place(voxel, load_size, shape, rng)


import orjson


def _write_sidecar(tmp_path, dataset, pairs):
    """pairs: dict[(key, source, cls)] -> (M,3) int16 array. Writes one dataset.

    On-disk index.json uses the v3 envelope written by the precompute command.
    """
    d = tmp_path / dataset / 'fg_coords'
    d.mkdir(parents=True)
    records = []
    blocks = []
    off = 0
    for (key, source, cls), coords in pairs.items():
        bbox_start = coords.min(axis=0).tolist()
        bbox_stop = (coords.max(axis=0) + 1).tolist()
        records.append([key, source, cls, off, len(coords), bbox_start, bbox_stop])
        blocks.append(coords.astype(np.int16))
        off += len(coords)
    arr = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 3), np.int16)
    np.save(d / 'coords.npy', arr)
    (d / 'index.json').write_bytes(orjson.dumps({
        'version': 3,
        'coord_cap': 1024,
        'records': records,
    }))


class TestFgCoordCache:
    def test_sample_voxel_returns_stored_coord(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache
        coords = np.array([[5, 10, 20], [6, 11, 21]], dtype=np.int16)
        _write_sidecar(tmp_path, 'd1', {('k1', 's1', 'liver'): coords})
        cache = FgCoordCache(tmp_path, datasets=['d1'])
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(50):
            v = cache.sample_voxel('d1', 'k1', 's1', 'liver', rng)
            seen.add(tuple(v))
        assert seen <= {(5, 10, 20), (6, 11, 21)}
        assert len(seen) == 2  # both eventually drawn

    def test_rejects_v1_index(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache
        d = tmp_path / 'd1' / 'fg_coords'
        d.mkdir(parents=True)
        np.save(d / 'coords.npy', np.array([[5, 10, 20]], dtype=np.int16))
        (d / 'index.json').write_bytes(orjson.dumps([['k1', 's1', 'liver', 0, 1]]))
        with pytest.raises(ValueError, match='envelope'):
            FgCoordCache(tmp_path, datasets=['d1'])

    @pytest.mark.parametrize(('field', 'value', 'match'), [
        ('version', 1, 'version'),
        ('coord_cap', 256, 'cap'),
    ])
    def test_rejects_wrong_v3_contract(self, tmp_path, field, value, match):
        from pumit.ucpt.fg_coords import FgCoordCache
        _write_sidecar(
            tmp_path,
            'd1',
            {('k1', 's1', 'liver'): np.array([[5, 10, 20]], dtype=np.int16)},
        )
        path = tmp_path / 'd1' / 'fg_coords' / 'index.json'
        index = orjson.loads(path.read_bytes())
        index[field] = value
        path.write_bytes(orjson.dumps(index))
        with pytest.raises(ValueError, match=match):
            FgCoordCache(tmp_path, datasets=['d1'])

    def test_bbox_returns_exact_half_open_bounds(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache

        coords = np.array([[1, 4, 9], [3, 8, 10]], dtype=np.int16)
        _write_sidecar(tmp_path, 'd1', {('k1', 's1', 'liver'): coords})
        cache = FgCoordCache(tmp_path, datasets=['d1'])

        start, stop = cache.bbox('d1', 'k1', 's1', 'liver')

        assert np.array_equal(start, [1, 4, 9])
        assert np.array_equal(stop, [4, 9, 11])

    def test_has_pair(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache
        _write_sidecar(tmp_path, 'd1', {('k1', 's1', 'liver'): np.array([[1, 2, 3]], np.int16)})
        cache = FgCoordCache(tmp_path, datasets=['d1'])
        assert cache.has_pair('d1', 'k1', 's1', 'liver')
        assert not cache.has_pair('d1', 'k1', 's1', 'spleen')

    def test_multi_pair_offset_slicing(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache
        liver = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int16)
        spleen = np.array([[7, 7, 7]], dtype=np.int16)
        _write_sidecar(tmp_path, 'd1', {
            ('k1', 's1', 'liver'): liver,
            ('k1', 's1', 'spleen'): spleen,
        })
        cache = FgCoordCache(tmp_path, datasets=['d1'])
        rng = np.random.default_rng(0)
        # spleen has a nonzero row offset; its only coord must be (7,7,7)
        for _ in range(20):
            v = cache.sample_voxel('d1', 'k1', 's1', 'spleen', rng)
            assert tuple(v) == (7, 7, 7)
        # liver draws only from its own two rows, never spleen's
        liver_seen = {tuple(cache.sample_voxel('d1', 'k1', 's1', 'liver', rng)) for _ in range(20)}
        assert liver_seen == {(1, 1, 1), (2, 2, 2)}


class TestPrecomputeHelpers:
    def test_resolve_mask_file_prefixed(self, tmp_path):
        from pumit.ucpt.fg_coords.__main__ import resolve_mask_file
        label_dir = tmp_path / 'labels' / 'k1' / 's1'
        label_dir.mkdir(parents=True)
        (label_dir / '00__liver.npy.zst').write_bytes(b'x')
        got = resolve_mask_file(label_dir, 'liver')
        assert got.name == '00__liver.npy.zst'

    def test_resolve_mask_file_sanitizes_slash(self, tmp_path):
        from pumit.ucpt.fg_coords.__main__ import resolve_mask_file
        label_dir = tmp_path / 'labels' / 'k1' / 's1'
        label_dir.mkdir(parents=True)
        (label_dir / '03__T10_T11.npy.zst').write_bytes(b'x')
        got = resolve_mask_file(label_dir, 'T10/T11')
        assert got.name == '03__T10_T11.npy.zst'

    def test_injectivity_violation_raises(self, tmp_path):
        from pumit.ucpt.fg_coords.__main__ import assert_injective
        resolved = {('k1', 's1', 'T10/T11'): tmp_path / 'f.zst',
                    ('k1', 's1', 'T10_T11'): tmp_path / 'f.zst'}
        with pytest.raises(AssertionError, match='inject'):
            assert_injective(resolved)

    def test_subsample_caps_at_1024(self):
        from pumit.ucpt.fg_coords.__main__ import subsample_coords
        mask = np.zeros((10, 100, 100), dtype=bool)
        mask[5, :, :] = True  # 10000 fg voxels
        rng = np.random.default_rng(0)
        coords = subsample_coords(mask, cap=1024, rng=rng)
        assert coords.shape == (1024, 3)
        assert coords.dtype == np.int16
        assert mask[coords[:, 0], coords[:, 1], coords[:, 2]].all()

    def test_subsample_stores_all_when_below_cap(self):
        from pumit.ucpt.fg_coords.__main__ import subsample_coords
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1, 2, 3] = True
        mask[0, 0, 0] = True
        rng = np.random.default_rng(0)
        coords = subsample_coords(mask, cap=256, rng=rng)
        assert coords.shape == (2, 3)

    def test_subsample_rejects_empty_positive_mask(self):
        from pumit.ucpt.fg_coords.__main__ import subsample_coords

        with pytest.raises(ValueError, match='positive mask is empty'):
            subsample_coords(
                np.zeros((4, 4, 4), dtype=bool),
                cap=1024,
                rng=np.random.default_rng(0),
            )

    def test_foreground_bbox_returns_half_open_bounds(self):
        from pumit.ucpt.fg_coords.__main__ import foreground_bbox

        mask = np.zeros((4, 5, 6), dtype=bool)
        mask[1:3, 2:5, 4:6] = True

        start, stop = foreground_bbox(mask)

        assert np.array_equal(start, [1, 2, 4])
        assert np.array_equal(stop, [3, 5, 6])


class TestForcedCentering:
    def _labeled_state_3d(self):
        return {
            'shape': np.array([100, 256, 256]),
            'spacing': np.array([1.0, 1.0, 1.0]),
            'labeled_draw': True,
            'dataset': 'd1', 'key': 'k1',
        }

    def test_forced_draw_box_contains_center_voxel(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache
        from pumit.ucpt.transforms import AffinePatchLoader
        coords = np.array([[50, 128, 128]], dtype=np.int16)
        _write_sidecar(tmp_path, 'd1', {('k1', 's1', 'liver'): coords})
        cache = FgCoordCache(tmp_path, datasets=['d1'])
        loader = AffinePatchLoader(
            size_xy_choices=[128], size_xy_choices_2d=[128],
            max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12}, max_da=4,
            fg_cache=cache, force_fraction=1.0,
        )
        state = self._labeled_state_3d()
        state['_center_class'] = ('s1', 'liver')
        rng = np.random.default_rng(0)
        p = loader.sample_params(state, rng)
        start = np.array(p['load_slice_start'])
        stop = np.array(p['load_slice_stop'])
        v = np.array([50, 128, 128])
        assert np.all(start <= v) and np.all(v < stop)

    @pytest.mark.parametrize('is_2d', [False, True])
    def test_forced_jitter_is_bounded_after_sampled_affine(self, tmp_path, is_2d):
        from pumit.ucpt.fg_coords import FgCoordCache
        from pumit.ucpt.affine import _realized_output_to_source
        from pumit.ucpt.transforms import AffinePatchLoader

        voxel = np.array([0 if is_2d else 128, 256, 256])
        _write_sidecar(tmp_path, 'd1', {('k1', 's1', 'liver'): voxel[None].astype(np.int16)})
        cache = FgCoordCache(tmp_path, datasets=['d1'])
        loader = AffinePatchLoader(
            size_xy_choices=[128],
            size_xy_choices_2d=[128],
            max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12},
            max_da=4,
            rotate_prob=1,
            rotate_axis_prob=1,
            scale_xy_p=1,
            scale_z_p=1,
            fg_cache=cache,
            force_fraction=1,
        )
        state = {
            'shape': np.array([1 if is_2d else 256, 512, 512]),
            'spacing': np.array([np.nan if is_2d else 1.0, 1.0, 1.0]),
            'labeled_draw': True,
            'dataset': 'd1',
            'key': 'k1',
            '_center_class': ('s1', 'liver'),
        }

        for seed in range(50):
            params = loader.sample_params(state, np.random.default_rng(seed))
            crop_size = np.asarray(params['crop_size'])
            affine = np.asarray(params['affine']).reshape(4, 4)[:3, :3]
            output_to_source = _realized_output_to_source(affine, crop_size)
            start = np.asarray(params['load_slice_start'])
            stop = np.asarray(params['load_slice_stop'])
            source_center = start + (stop - start) // 2
            output_offset = np.linalg.solve(output_to_source, voxel - source_center)
            tolerance = np.linalg.norm(np.linalg.inv(output_to_source), ord=np.inf)
            assert abs(output_offset[0]) <= tolerance
            assert np.abs(output_offset[1:]).max() <= crop_size[1] / 4 + tolerance

    def test_no_cache_is_deterministic(self):
        from pumit.ucpt.transforms import AffinePatchLoader
        loader = AffinePatchLoader(
            size_xy_choices=[128], size_xy_choices_2d=[128],
            max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12}, max_da=4,
        )
        state = {'shape': np.array([100, 256, 256]), 'spacing': np.array([1.0, 1.0, 1.0])}
        p1 = loader.sample_params(state, np.random.default_rng(7))
        p2 = loader.sample_params(state, np.random.default_rng(7))
        assert p1['load_slice_start'] == p2['load_slice_start']

    def test_forced_2d_draw_box_contains_center_voxel(self, tmp_path):
        from pumit.ucpt.fg_coords import FgCoordCache
        from pumit.ucpt.transforms import AffinePatchLoader
        # 2D: single slice (d=0), in-plane voxel
        coords = np.array([[0, 100, 100]], dtype=np.int16)
        _write_sidecar(tmp_path, 'd2', {('k2', 's1', 'lesion'): coords})
        cache = FgCoordCache(tmp_path, datasets=['d2'])
        loader = AffinePatchLoader(
            size_xy_choices=[128], size_xy_choices_2d=[128],
            max_depth_per_da={0: 128, 1: 64, 2: 32, 3: 24, 4: 12}, max_da=4,
            fg_cache=cache, force_fraction=1.0,
        )
        state = {
            'shape': np.array([1, 384, 384]),
            'spacing': np.array([float('nan'), 1.0, 1.0]),
            'labeled_draw': True, 'dataset': 'd2', 'key': 'k2',
            '_center_class': ('s1', 'lesion'),
        }
        rng = np.random.default_rng(0)
        p = loader.sample_params(state, rng)
        start = np.array(p['load_slice_start'])
        stop = np.array(p['load_slice_stop'])
        v = np.array([0, 100, 100])
        assert np.all(start <= v) and np.all(v < stop)


class TestProcessDataset:
    """End-to-end _process_dataset over a fake on-disk dataset."""

    def _write_dataset(self, root, dataset, meta, masks):
        """masks: dict[(key, source, stem)] -> (D,H,W) bool array -> zstd+packbits file."""
        import io

        import orjson
        import zstandard as zstd

        ds_dir = root / dataset
        ds_dir.mkdir(parents=True)
        (ds_dir / 'meta.json').write_bytes(orjson.dumps(meta))
        cctx = zstd.ZstdCompressor()
        for (key, source, stem), vol in masks.items():
            out = ds_dir / 'labels' / key / source
            out.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            np.save(buf, pack_binary_mask(vol))
            (out / f'{stem}.npy.zst').write_bytes(cctx.compress(buf.getvalue()))

    def test_null_label_classes_entry_is_skipped(self, tmp_path):
        """An entry with label_classes=None (unlabeled image in a mixed dataset)
        must be skipped, not crash. Regression: `.get('label_classes', {})`
        returns None for an explicit null, so `or {}` is required. Drives the
        full main() flow (resolve -> flat decode -> write)."""
        from pumit.ucpt.fg_coords.__main__ import _resolve_dataset, main

        vol = np.zeros((4, 8, 8), dtype=bool)
        vol[1, 2, 3] = True
        meta = {
            'labeled_key': {
                'shape': [1, 4, 8, 8],
                'label_classes': {'src1': {'positive': ['liver'], 'negative': []}},
            },
            'unlabeled_key': {
                'shape': [1, 4, 8, 8],
                'label_classes': None,  # the crash case
            },
        }
        self._write_dataset(tmp_path, 'D', meta, {('labeled_key', 'src1', '00__liver'): vol})

        # unit: resolve skips the null entry, keeps the labeled pair
        resolved, shapes = _resolve_dataset('D', tmp_path)
        assert set(resolved.keys()) == {('labeled_key', 'src1', 'liver')}
        assert set(shapes.keys()) == {'labeled_key', 'unlabeled_key'}

        # end-to-end: main() writes the sidecar with just the one pair
        import sys
        argv = sys.argv
        sys.argv = ['fg_coords', '--data-root', str(tmp_path), '--datasets', 'D', '--workers', '1']
        try:
            main()
        finally:
            sys.argv = argv

        index = orjson.loads((tmp_path / 'D' / 'fg_coords' / 'index.json').read_bytes())
        assert index['version'] == 3
        assert index['coord_cap'] == 1024
        records = index['records']
        assert [(r[0], r[1], r[2]) for r in records] == [('labeled_key', 'src1', 'liver')]

        assert _resolve_dataset('D', tmp_path) == 'skip'
        forced, _ = _resolve_dataset('D', tmp_path, force=True)
        assert set(forced) == {('labeled_key', 'src1', 'liver')}
