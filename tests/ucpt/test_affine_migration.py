import math

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from pumit.ucpt.stream.latent_backend import (
    LatentEncodeWork,
    _build_encode_work,
    _materialize_latent_parts,
    _write_latent_part,
)
from pumit.ucpt.stream.migration import (
    apply_spacing_correction,
    migration_rng,
    requires_affine_v3_upgrade,
    spatial_migration_decision,
)
from pumit.ucpt.affine import _affine_load_extent, _affine_resample, _column_norm_spacing


def _spatial(
    matrix: np.ndarray,
    *,
    reference: tuple[int, int, int] = (32, 32, 32),
    actual: tuple[int, int, int] | None = None,
) -> dict:
    affine = np.eye(4)
    affine[:3, :3] = matrix
    actual = reference if actual is None else actual
    return {
        'crop_size': list(reference),
        'affine': affine.reshape(-1).tolist(),
        'load_slice_start': [0, 0, 0],
        'load_slice_stop': list(actual),
    }


def _rotation_z(degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    matrix = np.eye(3)
    matrix[1:, 1:] = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]
    return matrix


def _old_resample_grid(crop: torch.Tensor, affine: np.ndarray, output_size: list[int]) -> torch.Tensor:
    matrix = affine[:3, :3]
    matrix_whd = matrix[::-1, ::-1].copy()
    load_dhw = list(crop.shape[1:])
    load_whd = [load_dhw[2], load_dhw[1], load_dhw[0]]
    reference_whd = [output_size[2], output_size[1], output_size[0]]
    theta = matrix_whd * np.asarray(reference_whd)[None, :] / np.asarray(load_whd)[:, None]
    theta_3x4 = torch.zeros(1, 3, 4, dtype=torch.float32)
    theta_3x4[0, :, :3] = torch.from_numpy(theta.astype(np.float32))
    grid = F.affine_grid(theta_3x4, [1, crop.shape[0], *output_size], align_corners=False)
    return F.grid_sample(
        crop.float().unsqueeze(0),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=False,
    ).squeeze(0).to(crop.dtype)


def test_grid_without_new_padding_is_unaffected():
    matrix = _rotation_z(30)
    extent = _affine_load_extent(matrix, [32, 32, 32])
    decision = spatial_migration_decision(
        _spatial(matrix, actual=tuple(int(value) for value in extent))
    )

    assert not decision.required
    assert decision.reasons == ()
    assert decision.old_padding == decision.new_padding == ((0, 0),) * 3


def test_grid_with_clamped_axis_requires_migration():
    matrix = _rotation_z(30)
    extent = _affine_load_extent(matrix, [32, 32, 32])
    actual = extent.copy()
    actual[0] -= 1
    decision = spatial_migration_decision(
        _spatial(matrix, actual=tuple(int(value) for value in actual))
    )

    assert decision.required
    assert decision.reasons == ('grid_padding',)


def test_diagonal_negative_odd_padding_requires_migration():
    matrix = np.diag([-1.0, 1.0, 1.0])
    decision = spatial_migration_decision(_spatial(matrix, actual=(31, 32, 32)))

    assert decision.required
    assert decision.reasons == ('diagonal_flip_pad_order',)
    assert decision.old_padding[0] == (0, 1)
    assert decision.new_padding[0] == (1, 0)


def test_diagonal_negative_even_padding_is_unaffected():
    matrix = np.diag([-1.0, 1.0, 1.0])
    decision = spatial_migration_decision(_spatial(matrix, actual=(30, 32, 32)))

    assert not decision.required
    assert decision.old_padding == decision.new_padding


def test_near_integer_diagonal_extent_is_not_snapped_down():
    reference = (1, 192, 192)
    matrix = np.diag([1.0, 153 / 192 + 1e-16, 153 / 192 + 1e-16])
    sample = {
        'params': [_spatial(matrix, reference=reference, actual=(1, 153, 153))],
        'spacing_label': [1.0, 0.8020833333333334, 0.8020833333333334],
    }
    decision = spatial_migration_decision(sample)
    corrected = apply_spacing_correction(sample, decision)

    assert _affine_load_extent(matrix, reference).tolist() == [1, 154, 154]
    assert not decision.required
    assert decision.reasons == ()
    assert corrected is sample


def test_spacing_is_unchanged_when_frozen_crop_covers_ceil_extent():
    reference = (1, 192, 192)
    matrix = np.diag([1.0, 153 / 192 + 1e-16, 153 / 192 + 1e-16])
    sample = {
        'params': [_spatial(matrix, reference=reference, actual=(1, 154, 154))],
        'spacing_label': [1.0, 0.8020833333333334, 0.8020833333333334],
    }

    decision = spatial_migration_decision(sample)
    corrected = apply_spacing_correction(sample, decision)

    assert not decision.required
    assert decision.spacing_scale == (1.0, 1.0, 1.0)
    assert corrected is sample


def test_legacy_near_diagonal_sample_switches_to_grid_and_updates_spacing():
    reference = (32, 32, 32)
    matrix = np.diag([0.8, 0.8, 0.8])
    matrix[0, 1] = 5e-7
    source_spacing = np.array([2.0, 1.0, 1.0])
    old_realized = np.diag([26 / 32, 26 / 32, 26 / 32])
    sample = {
        'params': [_spatial(matrix, reference=reference, actual=(26, 26, 26))],
        'spacing_label': _column_norm_spacing(old_realized, source_spacing),
    }

    decision = spatial_migration_decision(sample)
    corrected = apply_spacing_correction(sample, decision)

    assert decision.required
    assert decision.reasons == ('near_diagonal_grid',)
    np.testing.assert_allclose(
        corrected['spacing_label'],
        _column_norm_spacing(matrix, source_spacing),
        rtol=0,
        atol=1e-15,
    )
    assert requires_affine_v3_upgrade(sample)


def test_frozen_old_and_fixed_grid_operators_match_when_unaffected():
    matrix = _rotation_z(30)
    reference = [32, 32, 32]
    extent = _affine_load_extent(matrix, reference)
    crop = torch.arange(
        int(np.prod(extent)),
        dtype=torch.int32,
    ).reshape(1, *extent)
    affine = np.eye(4)
    affine[:3, :3] = matrix

    old = _old_resample_grid(crop, affine, reference)
    fixed = _affine_resample(
        crop,
        affine,
        reference,
        interp_mode='nearest-exact',
        grid_mode='nearest',
    )

    assert torch.equal(old, fixed)


def test_migration_rng_is_addressed_by_namespace_and_ordinal():
    expected = migration_rng(42, 7, 2, 11).integers(0, 2**31, size=8)

    assert np.array_equal(expected, migration_rng(42, 7, 2, 11).integers(0, 2**31, size=8))
    assert not np.array_equal(expected, migration_rng(42, 7, 2, 12).integers(0, 2**31, size=8))


def test_latent_encode_work_uses_shape_uniform_batches():
    unlabeled = [
        {
            'da_enc': da,
            'depth': depth,
            'n_patches': depth,
            'params': [{'patch_size': [depth, size, size]}],
        }
        for da, depth, size in [
            (None, 1, 64),
            (0, 8, 64),
            (None, 1, 64),
            (0, 8, 64),
            (1, 4, 96),
        ]
    ]

    work = _build_encode_work(
        unlabeled,
        [0, 1, 2, 3, 4],
        shard_id=7,
        memory_budget_gb=0.01,
    )

    assert sorted(index for item in work for index in item.selected_indices) == [0, 1, 2, 3, 4]
    assert all(item.shard_id == 7 for item in work)
    for item in work:
        keys = {
            (
                unlabeled[index]['da_enc'],
                unlabeled[index]['depth'],
                unlabeled[index]['params'][0]['patch_size'][1],
            )
            for index in item.selected_indices
        }
        assert len(keys) == 1


def test_latent_parts_are_scattered_into_distinct_complete_output(tmp_path):
    source = tmp_path / 'source.safetensors'
    output = tmp_path / 'latents' / 'shard_00000.safetensors'
    output.parent.mkdir()
    source_latents = torch.arange(8 * 32, dtype=torch.float16).reshape(8, 32)
    save_file({'latents': source_latents}, source)
    unlabeled = [
        {'n_patches': 2},
        {'n_patches': 3},
        {'n_patches': 1},
        {'n_patches': 2},
    ]
    work = [
        LatentEncodeWork(0, 0, (1,), 3, 1),
        LatentEncodeWork(0, 1, (3,), 2, 1),
    ]
    replacement = torch.full((3, 32), 11, dtype=torch.float16)
    suffix = torch.full((2, 32), 17, dtype=torch.float16)
    _write_latent_part(tmp_path, work[0], replacement)
    _write_latent_part(tmp_path, work[1], suffix)

    _materialize_latent_parts(
        source,
        unlabeled,
        source_prefix_samples=3,
        work=work,
        stream_dir=tmp_path,
        output_path=output,
    )

    actual = load_file(output)['latents']
    assert output.stat().st_ino != source.stat().st_ino
    assert torch.equal(actual[:2], source_latents[:2])
    assert torch.equal(actual[2:5], replacement)
    assert torch.equal(actual[5:6], source_latents[5:6])
    assert torch.equal(actual[6:], suffix)


def test_latent_parts_can_materialize_a_full_shard_without_source(tmp_path):
    output = tmp_path / 'latents' / 'shard_00000.safetensors'
    output.parent.mkdir()
    unlabeled = [{'n_patches': 2}, {'n_patches': 1}]
    work = [LatentEncodeWork(0, 0, (0, 1), 3, 1)]
    encoded = torch.arange(3 * 32, dtype=torch.float16).reshape(3, 32)
    _write_latent_part(tmp_path, work[0], encoded)

    _materialize_latent_parts(
        None,
        unlabeled,
        source_prefix_samples=0,
        work=work,
        stream_dir=tmp_path,
        output_path=output,
    )

    assert torch.equal(load_file(output)['latents'], encoded)
