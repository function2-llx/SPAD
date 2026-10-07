import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pumit.ucpt.affine import (
    _affine_load_extent,
    _affine_resample,
    _canonical_inplane_affine,
    _grid_resample_3d,
    _grid_resample_inplane,
    _quantize_scale_to_load_extent,
    _realized_output_to_source,
    _uses_interpolate,
    replay_affine_patch,
    stream_crop_size,
)
from pumit.ucpt.transforms import AffinePatchLoader, build_ucpt_pipeline


def _cube(C, D, H, W):
    return torch.from_numpy(np.random.RandomState(0).random((C, D, H, W)).astype(np.float32))


def _rotation_z(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    M = np.eye(4)
    M[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return M


def _centroid(t):
    idx = np.nonzero(t.squeeze(0).numpy())
    return [float(i.mean()) for i in idx]


def test_identity_downsamples_not_center_crops():
    """The anti-center-crop guard: a smaller output_size DOWNSAMPLES the whole
    FOV (not a center crop). A 32^3 volume with a bright periphery, resampled
    to 8^3 nearest-exact, must still carry periphery content -> not all-zero at edges."""
    crop = torch.zeros(1, 32, 32, 32)
    crop[:, :, :, :4] = 1.0  # bright in the first coarse cell at the periphery
    out = _affine_resample(
        crop, np.eye(4), [8, 8, 8], interp_mode='nearest-exact', grid_mode='nearest',
    )
    assert out.shape == (1, 8, 8, 8)
    # Full-FOV downsampling keeps the first coarse cell at out[..., 0].
    assert out[0, :, :, 0].any(), 'periphery content was center-cropped away'


def test_nearest_preserves_binary():
    crop = torch.zeros(1, 16, 16, 16)
    crop[:, 4:12, 4:12, 4:12] = 1.0
    out = _affine_resample(
        crop, np.eye(4), [8, 8, 8], interp_mode='nearest-exact', grid_mode='nearest',
    )
    assert out.shape == (1, 8, 8, 8)
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_rotation_preserves_shape():
    """A non-diagonal affine (rotation) takes the grid_sample path and preserves shape."""
    theta = np.eye(4)
    theta[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    crop = _cube(1, 16, 16, 16)
    out = _affine_resample(
        crop, theta, [8, 8, 8], interp_mode='nearest-exact', grid_mode='nearest',
    )
    assert out.shape == (1, 8, 8, 8)


def test_interpolate_fast_path_requires_an_exact_diagonal():
    diagonal = np.diag([1.0, -0.8, 1.2])
    near_diagonal = diagonal.copy()
    near_diagonal[1, 2] = np.nextafter(0.0, 1.0)

    assert _uses_interpolate(diagonal)
    assert not _uses_interpolate(near_diagonal)


def test_legacy_inplane_roundoff_is_canonicalized_without_touching_oblique_terms():
    legacy = _rotation_z(37)[:3, :3]
    legacy[0] = [-1.0, 1.13e-16, -3.26e-17]
    canonical = _canonical_inplane_affine(legacy)

    assert canonical is not None
    np.testing.assert_array_equal(canonical[0], [-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(canonical[1:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(canonical[1:, 1:], legacy[1:, 1:])

    oblique = legacy.copy()
    oblique[0, 1] = 1e-12
    assert _canonical_inplane_affine(oblique) is None


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_affine_resample_rejects_non_finite_metadata(value):
    affine = np.eye(4)
    affine[0, 1] = value

    with pytest.raises(ValueError, match='affine matrix must contain only finite values'):
        _affine_resample(
            torch.zeros(1, 8, 8, 8),
            affine,
            [8, 8, 8],
            interp_mode='trilinear',
            grid_mode='bilinear',
        )


def test_scale_is_quantized_to_its_integer_load_extent():
    reference = [32, 128, 128]
    scale = [0.8, 0.75, 0.8]

    realized = _quantize_scale_to_load_extent(scale, reference)

    np.testing.assert_array_equal(realized * reference, [26, 96, 103])
    np.testing.assert_allclose(realized, [26 / 32, 0.75, 103 / 128], rtol=0, atol=0)


def test_image_path_unchanged_bilinear_trilinear():
    """Image call (trilinear diag / bilinear rot) still produces a finite,
    correctly-shaped output on a realistic diagonal affine."""
    crop = _cube(3, 16, 64, 64)
    affine = np.diag([1.0, 0.5, 0.5, 1.0])
    affine[3, :3] = 0.0
    out = _affine_resample(
        crop, affine, [16, 32, 32], interp_mode='trilinear', grid_mode='bilinear',
    )
    assert out.shape == (3, 16, 32, 32)
    assert torch.isfinite(out).all()


# --- reference_size regression tests (2026-07-22 seg-target FOV bug) ---

PATCH = [128, 128, 128]
TARGET = [32, 32, 32]


def _rotated_crop_with_blob(deg=30):
    """Load box for a pure in-plane rotation of a 128^3 patch, with an off-center
    12^3 blob inside the rotated support and image FOV but outside the central
    quarter window the buggy theta used to sample."""
    M = _rotation_z(deg)
    load_size = np.ceil(np.abs(M[:3, :3]) @ np.array(PATCH)).astype(int)
    crop = torch.zeros(1, *load_size.tolist())
    c = [load_size[0] // 2 - 6, int(0.72 * load_size[1]), int(0.72 * load_size[2])]
    crop[:, c[0]:c[0] + 12, c[1]:c[1] + 12, c[2]:c[2] + 12] = 1.0
    return crop, M


def test_rotation_mask_reference_covers_image_fov():
    """The mask call (coarse output, image-sized reference) must cover the SAME
    FOV as the image call: an off-center blob lands at the block-downsampled
    image position, not dropped (bug: theta scaled by output sampled only the
    central target/reference window)."""
    crop, M = _rotated_crop_with_blob()
    img = _affine_resample(crop.clone(), M, PATCH,
                           interp_mode='nearest-exact', grid_mode='nearest')
    assert img.bool().sum() > 0, 'blob must be visible in the image FOV'
    expected = [x / 4 for x in _centroid(img)]

    out = _affine_resample(crop.clone(), M, TARGET,
                           interp_mode='nearest-exact', grid_mode='nearest',
                           reference_size=PATCH)
    assert out.bool().sum() > 0, 'reference-sized theta must not drop the off-center blob'
    actual = _centroid(out)
    assert all(abs(a - e) < 1.0 for a, e in zip(actual, expected)), (
        f'mask blob centroid {actual} != image-block centroid {expected}')


@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
def test_strict_inplane_operator_matches_generic_3d_reference(mode):
    crop, affine = _rotated_crop_with_blob(37)
    matrix = affine[:3, :3]
    output_size = [64, 96, 80]

    strict_2d = _affine_resample(
        crop,
        affine,
        output_size,
        interp_mode='nearest-exact' if mode == 'nearest' else 'trilinear',
        grid_mode=mode,
        reference_size=PATCH,
    )
    generic_3d = _grid_resample_3d(
        crop.float(),
        matrix,
        output_size,
        PATCH,
        mode=mode,
    )

    if mode == 'nearest':
        assert torch.equal(strict_2d, generic_3d)
    else:
        torch.testing.assert_close(strict_2d, generic_3d, rtol=0, atol=2e-6)


def test_strict_inplane_operator_never_mixes_depth_slices():
    affine = _rotation_z(31)
    affine[0, 0] = -1
    load_extent = _affine_load_extent(affine[:3, :3], [8, 32, 32])
    crop = torch.arange(load_extent[0], dtype=torch.float32).view(1, -1, 1, 1)
    crop = crop.expand(1, *load_extent).clone()

    actual = _affine_resample(
        crop,
        affine,
        [8, 32, 32],
        interp_mode='trilinear',
        grid_mode='bilinear',
    )

    for depth, expected in enumerate(reversed(range(8))):
        torch.testing.assert_close(
            actual[0, depth],
            torch.full((32, 32), expected, dtype=torch.float32),
            rtol=0,
            atol=1e-6,
        )


def test_strict_inplane_operator_returns_contiguous_tensor():
    affine = _rotation_z(31)
    reference_size = [8, 32, 32]
    load_extent = _affine_load_extent(affine[:3, :3], reference_size)
    crop = torch.rand(1, *load_extent.tolist())

    actual = _grid_resample_inplane(
        crop,
        affine[:3, :3],
        reference_size,
        reference_size,
        mode='bilinear',
    )

    assert actual.is_contiguous()


def _full_inplane_replay(tmp_path, *, contrast: bool, gamma: bool) -> torch.Tensor:
    affine = _rotation_z(37)
    crop_size = [8, 16, 16]
    load_extent = _affine_load_extent(affine[:3, :3], crop_size)
    image_path = tmp_path / 'image.npy'
    np.save(
        image_path,
        np.random.default_rng(0).random((1, *load_extent.tolist())).astype(np.float16),
    )
    pipeline = build_ucpt_pipeline(
        size_xy_choices=[16],
        size_xy_choices_2d=[16],
        max_depth_per_da={0: 16},
    )
    params = [
        {
            'da_enc': 0,
            'crop_size': crop_size,
            'affine': affine.ravel().tolist(),
            'load_slice_start': [0, 0, 0],
            'load_slice_stop': load_extent.tolist(),
            'n_patches': 1,
            'spacing_label': [1.0, 1.0, 1.0],
        },
        {'enabled': False},
        {'enabled': contrast, **({'factor': 1.1} if contrast else {})},
        {'enabled': gamma, **({'gammas': [1.1], 'invert': False} if gamma else {})},
        {},
    ]
    return pipeline.replay({'img': str(image_path)}, params)['img']


def test_full_inplane_replay_accepts_adjust_contrast(tmp_path):
    result = _full_inplane_replay(tmp_path, contrast=True, gamma=False)

    assert result.shape == (3, 8, 16, 16)
    assert torch.isfinite(result).all()


def test_full_inplane_replay_accepts_gamma_correction(tmp_path):
    result = _full_inplane_replay(tmp_path, contrast=False, gamma=True)

    assert result.shape == (3, 8, 16, 16)
    assert torch.isfinite(result).all()


def test_legacy_oversized_inplane_requires_metadata_migration(tmp_path):
    output_size = [128, 16, 16]
    canonical = _rotation_z(37)
    canonical[0, 0] = -1.0
    legacy = canonical.copy()
    legacy[0, 1:3] = [1.13e-16, -3.26e-17]
    legacy[1:3, 0] = [-6.2e-17, 4.1e-17]
    image_path = tmp_path / 'image.npy'
    np.save(image_path, np.zeros((1, 129, 16, 16), dtype=np.float32))

    for affine in (legacy, canonical):
        with pytest.raises(ValueError, match='migrate the frozen metadata'):
            replay_affine_patch(
                {'img': str(image_path)},
                da_enc=0,
                affine=affine.ravel().tolist(),
                load_slice_start=[0, 0, 0],
                load_slice_stop=[129, 16, 16],
                n_patches=1,
                spacing_label=[1.0, 1.0, 1.0],
                crop_size=output_size,
            )


def test_oversized_true_3d_crop_requires_metadata_migration(tmp_path):
    theta = np.deg2rad(31)
    affine = np.eye(4)
    affine[:3, :3] = [
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
    output_size = [16, 16, 16]
    load_extent = _affine_load_extent(affine[:3, :3], output_size)
    oversized = load_extent.copy()
    oversized[0] += 1
    image_path = tmp_path / 'image.npy'
    np.save(image_path, np.zeros((1, *oversized.tolist()), dtype=np.float32))

    with pytest.raises(ValueError, match='exceeds its load extent'):
        replay_affine_patch(
            {'img': str(image_path)},
            da_enc=0,
            affine=affine.ravel().tolist(),
            load_slice_start=[0, 0, 0],
            load_slice_stop=oversized.tolist(),
            n_patches=1,
            spacing_label=[1.0, 1.0, 1.0],
            crop_size=output_size,
        )


def test_replay_rejects_load_slice_outside_full_volume(tmp_path):
    image_path = tmp_path / 'image.npy'
    np.save(image_path, np.zeros((1, 4, 4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match='outside volume shape'):
        replay_affine_patch(
            {'img': str(image_path)},
            da_enc=0,
            affine=np.eye(4).ravel().tolist(),
            load_slice_start=[0, 0, 0],
            load_slice_stop=[5, 4, 4],
            n_patches=1,
            spacing_label=[1.0, 1.0, 1.0],
            crop_size=[4, 4, 4],
        )


def test_rotation_mask_default_reference_shrinks_fov():
    """Omitting reference_size retains the low-level resampler's output-FOV semantics."""
    crop, M = _rotated_crop_with_blob()
    out = _affine_resample(
        crop.clone(),
        M,
        TARGET,
        interp_mode='nearest-exact',
        grid_mode='nearest',
    )
    assert out.bool().sum() == 0


@pytest.mark.parametrize('out_d', [32, 64, 128])
def test_rotation_mask_depth_coverage(out_d):
    """Depth FOV must be full at any output depth (the bug windowed D only when
    the da-dependent D_out/D ratio was < 1; the fix covers D at all ratios).
    Blob off-center in D at 0.72 of the load box."""
    M = _rotation_z(30)
    load_size = np.ceil(np.abs(M[:3, :3]) @ np.array(PATCH)).astype(int)
    crop = torch.zeros(1, *load_size.tolist())
    c = [int(0.72 * load_size[0]), load_size[1] // 2 - 6, load_size[2] // 2 - 6]
    crop[:, c[0]:c[0] + 12, c[1]:c[1] + 12, c[2]:c[2] + 12] = 1.0
    img = _affine_resample(crop.clone(), M, PATCH,
                           interp_mode='nearest-exact', grid_mode='nearest')
    assert img.bool().sum() > 0
    expected_d = _centroid(img)[0] / (PATCH[0] / out_d)

    out = _affine_resample(crop.clone(), M, [out_d, 32, 32],
                           interp_mode='nearest-exact', grid_mode='nearest',
                           reference_size=PATCH)
    assert out.bool().sum() > 0, f'blob dropped at output depth {out_d}'
    assert abs(_centroid(out)[0] - expected_d) < 1.0


def test_diagonal_boundary_clamp_pads_like_image():
    """Diagonal + volume-clamped crop: image pads to the requested load before
    resampling; the mask call must pad identically (L_req from reference_size),
    not stretch the clamped crop over its own FOV."""
    reference = 128
    crop_len = 100  # requested 128, clamped to 100 at the volume bound
    crop = torch.zeros(1, crop_len, crop_len, crop_len)
    crop[:, 90:96, 50:56, 50:56] = 1.0  # marker near the far (unclamped) edge
    affine = np.eye(4)

    img = _affine_resample(crop.clone(), affine, [reference] * 3,
                           interp_mode='nearest-exact', grid_mode='nearest')
    img_pos = _centroid(img)
    mask = _affine_resample(crop.clone(), affine, [32, 32, 32],
                            interp_mode='nearest-exact', grid_mode='nearest',
                            reference_size=[reference] * 3)
    mask_pos = _centroid(mask)
    expected = [p * 32 / reference for p in img_pos]
    assert all(abs(m - e) < 1.0 for m, e in zip(mask_pos, expected)), (
        f'mask marker {mask_pos} != image-relative {expected}: clamped crop '
        f'was stretched instead of padded')


def test_nearest_lattice_is_continuous_across_resample_branches():
    """An infinitesimal off-diagonal term must not change the coarse-grid sampling phase."""
    crop = torch.zeros(1, 32, 32, 32)
    crop[..., 2] = 1.0  # center of the first 4-voxel output cell
    diagonal = _affine_resample(
        crop, np.eye(4), [8, 8, 8], interp_mode='nearest-exact', grid_mode='nearest',
        reference_size=[32, 32, 32],
    )
    near_diagonal = np.eye(4)
    near_diagonal[1, 2] = 2e-6  # enter grid_sample without moving the W sampling coordinate
    rotated = _affine_resample(
        crop, near_diagonal, [8, 8, 8], interp_mode='nearest-exact', grid_mode='nearest',
        reference_size=[32, 32, 32],
    )
    assert diagonal.bool().sum() > 0
    assert rotated.bool().sum() > 0
    assert torch.equal(diagonal, rotated)


def test_rotation_boundary_clamp_pads_to_requested_load_extent():
    reference = [32, 32, 32]
    affine = _rotation_z(30)
    load_extent = _affine_load_extent(affine[:3, :3], reference)
    depth_ramp = torch.arange(31, dtype=torch.float32).view(1, 31, 1, 1)
    crop = depth_ramp.expand(1, 31, int(load_extent[1]), int(load_extent[2])).clone()

    actual = _affine_resample(
        crop,
        affine,
        reference,
        interp_mode='trilinear',
        grid_mode='bilinear',
        reference_size=reference,
    )
    explicitly_padded = F.pad(crop.float().unsqueeze(0), (0, 0, 0, 0, 0, 1), mode='replicate').squeeze(0)
    expected = _affine_resample(
        explicitly_padded,
        affine,
        reference,
        interp_mode='trilinear',
        grid_mode='bilinear',
        reference_size=reference,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize('seed', range(16))
def test_generated_diagonal_affine_equals_its_realized_map(seed):
    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=0,
        scale_z=(0.8, 0.9),
        scale_z_p=1,
        scale_xy=(0.8, 0.9),
        scale_xy_p=1,
    )
    params = loader.sample_params(
        {'shape': [64, 64, 64], 'spacing': [1.0, 1.0, 1.0]},
        np.random.default_rng(seed),
    )
    matrix = np.asarray(params['affine']).reshape(4, 4)[:3, :3]

    assert _uses_interpolate(matrix)
    np.testing.assert_allclose(
        _realized_output_to_source(matrix, params['crop_size']),
        matrix,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize('seed', range(4))
def test_axial_rotation_has_exactly_separable_depth(seed):
    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=1,
        rotate_axis_prob=0,
        scale_z_p=0,
        scale_xy_p=0,
    )
    params = loader.sample_params(
        {'shape': [31, 64, 64], 'spacing': [1.0, 1.0, 1.0]},
        np.random.default_rng(seed),
    )
    affine = np.asarray(params['affine']).reshape(4, 4)[:3, :3]

    assert np.equal(affine[0, 1:], 0).all()
    assert np.equal(affine[1:, 0], 0).all()
    assert np.abs(affine[1:, 1:] - np.diag(np.diag(affine[1:, 1:]))).max() > 1e-6
    assert _affine_load_extent(affine, params['crop_size'])[0] == 32


def test_axial_rotation_only_draws_its_rotation_angle():
    seed = 7
    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=1,
        rotate_axis_prob=0,
        scale_z_p=0,
        scale_xy_p=0,
    )
    params = loader.sample_params(
        {'shape': [64, 64, 64], 'spacing': [1.0, 1.0, 1.0]},
        np.random.default_rng(seed),
    )

    expected_rng = np.random.default_rng(seed)
    expected_rng.random()  # scale_xy probability
    expected_rng.random()  # scale_z probability
    expected_rng.random()  # rotation probability
    expected_rng.random()  # oblique-axis probability
    theta = expected_rng.uniform(0, 2 * np.pi)
    expected_xy = np.abs(np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ]))
    affine = np.asarray(params['affine']).reshape(4, 4)[:3, :3]

    np.testing.assert_allclose(np.abs(affine[1:, 1:]), expected_xy, rtol=0, atol=1e-15)


def test_oblique_rotation_can_mix_depth_and_inplane():
    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=1,
        rotate_axis_prob=1,
        scale_z_p=0,
        scale_xy_p=0,
    )
    params = loader.sample_params(
        {'shape': [64, 64, 64], 'spacing': [1.0, 1.0, 1.0]},
        np.random.default_rng(0),
    )
    affine = np.asarray(params['affine']).reshape(4, 4)[:3, :3]

    assert np.abs(affine[0, 1:]).max() > 1e-6
    assert np.abs(affine[1:, 0]).max() > 1e-6


@pytest.mark.parametrize('spacing', [[np.nan, np.nan, np.nan], [np.nan, 0.5, 0.5]])
def test_labeled_2d_rotation_replays_image_and_mask_with_padding(tmp_path, spacing):
    from pumit.ucpt.seg.data import _resample_mask_crops

    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=1,
        scale_xy_p=0,
    )
    source = np.zeros((1, 1, 32, 32), dtype=np.float32)
    source[:, :, 10:19, 14:23] = 1
    path = tmp_path / 'image.npy'
    np.save(path, source)
    params = loader.sample_params(
        {'shape': [1, 32, 32], 'spacing': spacing, 'labeled_draw': True},
        np.random.default_rng(0),
    )
    affine = np.asarray(params['affine']).reshape(4, 4)
    np.testing.assert_array_equal(affine[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(affine[1:, 0], [0, 0, 0])
    assert abs(affine[1, 2]) > 1e-6
    assert np.all(_affine_load_extent(affine[:3, :3], params['crop_size'])[1:] > 32)
    image = loader({'img': str(path)}, **params)['img']
    target = _resample_mask_crops(source.astype(bool), affine, params['crop_size'])

    assert image.shape == target.shape == (1, 1, 32, 32)
    assert target.any()
    assert image[target].min() > 0
    assert target[image > 0.99].all()
    assert params['da_enc'] is None
    assert params['n_patches'] == 4


def test_labeled_2d_flips_replay_image_and_mask(tmp_path):
    from pumit.ucpt.seg.data import _resample_mask_crops

    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=0,
        scale_xy_p=0,
    )
    source = np.arange(32 * 32, dtype=np.float32).reshape(1, 1, 32, 32)
    path = tmp_path / 'image.npy'
    np.save(path, source)
    params = loader.sample_params(
        {'shape': [1, 32, 32], 'spacing': [np.nan] * 3, 'labeled_draw': True},
        np.random.default_rng(0),
    )
    image = loader({'img': str(path)}, **params)['img']
    mask = (source > 200) & (source < 400)
    target = _resample_mask_crops(mask, np.asarray(params['affine']).reshape(4, 4), params['crop_size'])

    assert torch.equal(image, torch.from_numpy(source).flip((2, 3)))
    assert torch.equal(target, torch.from_numpy(mask).flip((2, 3)))


def test_unlabeled_2d_keeps_crop_resize_sampling():
    loader = AffinePatchLoader(
        size_xy_choices=[32],
        size_xy_choices_2d=[32],
        max_depth_per_da={0: 32},
        rotate_prob=1,
        scale_xy_p=0,
    )
    params = loader.sample_params(
        {'shape': [1, 32, 32], 'spacing': [np.nan] * 3, 'labeled_draw': False},
        np.random.default_rng(0),
    )

    np.testing.assert_array_equal(np.asarray(params['affine']).reshape(4, 4), np.eye(4))
    assert params['load_slice_start'] == [0, 0, 0]
    assert params['load_slice_stop'] == [1, 32, 32]


def test_stream_crop_size_key_compat():
    assert stream_crop_size({'crop_size': [1, 2, 3]}) == [1, 2, 3]
    assert stream_crop_size({'patch_size': [4, 5, 6]}) == [4, 5, 6]
    with pytest.raises(KeyError):
        stream_crop_size({})
