"""Tests for SpatialRotaryEmbedding3D: 2D equivalence with DINOv3 and 3D additive properties."""
import math

import pytest
import torch

from pumit.model.rope import SpatialRotaryEmbedding3D, rotate_half, apply_rope, build_rope


class TestRotateHalf:
    def test_split_half_convention(self):
        """rotate_half uses split-half: [-x2, x1] where x1=first half, x2=second half."""
        x = torch.arange(8, dtype=torch.float).unsqueeze(0)  # [1, 8]
        result = rotate_half(x)
        expected = torch.tensor([[-4., -5., -6., -7., 0., 1., 2., 3.]])
        assert torch.equal(result, expected)


class TestDINOv3Equivalence:
    """2D RoPE must be bit-identical to DINOv3's DINOv3ViTRopePositionEmbedding."""

    @pytest.fixture
    def rope(self):
        return SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)

    @pytest.fixture
    def dinov3_reference(self):
        head_dim = 64
        base = 100.0
        inv_freq = 1 / base ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32)

        def compute_2d(H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
            coords_h = torch.arange(0.5, H, dtype=torch.float32) / H
            coords_w = torch.arange(0.5, W, dtype=torch.float32) / W
            coords = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1
            ).flatten(0, 1)
            coords = 2.0 * coords - 1.0
            angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            return torch.cos(angles), torch.sin(angles)

        return compute_2d

    def test_2d_equivalence_14x14(self, rope, dinov3_reference):
        cos_ref, sin_ref = dinov3_reference(14, 14)
        result = rope.compute(spatial_shape=(1, 14, 14), training=False)  # [196, 2, 64]
        assert torch.equal(result[:, 0], cos_ref)
        assert torch.equal(result[:, 1], sin_ref)

    def test_2d_equivalence_7x7(self, rope, dinov3_reference):
        cos_ref, sin_ref = dinov3_reference(7, 7)
        result = rope.compute(spatial_shape=(1, 7, 7), training=False)
        assert torch.equal(result[:, 0], cos_ref)
        assert torch.equal(result[:, 1], sin_ref)

    def test_2d_equivalence_asymmetric(self, rope, dinov3_reference):
        cos_ref, sin_ref = dinov3_reference(10, 14)
        result = rope.compute(spatial_shape=(1, 10, 14), training=False)
        assert torch.equal(result[:, 0], cos_ref)
        assert torch.equal(result[:, 1], sin_ref)


class TestAdditiveDepth:

    @pytest.fixture
    def rope(self):
        return SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)

    def test_depth1_shape(self, rope):
        result = rope.compute(spatial_shape=(1, 14, 14), training=False)
        assert result.shape == (196, 2, 64)

    def test_3d_output_shape(self, rope):
        result = rope.compute(spatial_shape=(4, 8, 8), training=False)
        assert result.shape == (4 * 8 * 8, 2, 64)

    def test_depth_changes_angles(self, rope):
        result = rope.compute(spatial_shape=(4, 8, 8), training=False)
        assert not torch.allclose(result[0, 0], result[64, 0])

    def test_different_depths(self, rope):
        r4 = rope.compute(spatial_shape=(4, 8, 8), training=False)
        r8 = rope.compute(spatial_shape=(8, 8, 8), training=False)
        assert r4.shape == (256, 2, 64)
        assert r8.shape == (512, 2, 64)


class TestDistinctDepthBase:
    """Depth frequency family must use a distinct base to break the (c, c, -c) null direction."""

    @pytest.fixture
    def rope(self):
        return SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)

    def test_depth_base_defaults_to_sqrt_theta(self, rope):
        assert rope.theta_depth == 10.0
        expected = 1.0 / 10.0 ** torch.arange(0, 1, 4 / 64)
        torch.testing.assert_close(rope.inv_freq_depth, expected)

    def test_equal_bases_rejected(self):
        with pytest.raises(ValueError):
            SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, theta_depth=100.0)

    def test_fp32_rounding_equal_bases_rejected(self):
        """theta_depth differing from theta by < fp32 epsilon (1e-9) passes a scalar `!=` check but yields
        an fp32 inv_freq_depth identical to inv_freq — the validation must reject the actual family."""
        with pytest.raises(ValueError):
            SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, theta_depth=100.0 * (1 + 1e-9))

    def test_no_null_direction_collision(self, rope):
        """On a 4x16x16 grid the depth coord step is 4x the in-plane step, so with a shared frequency
        family (l+1, i, j) and (l, i+4, j+4) produce identical angles. The offset must break this."""
        result = rope.compute(spatial_shape=(4, 16, 16), training=False)  # [1024, 2, 64]

        def flat_idx(l: int, i: int, j: int) -> int:
            return l * 256 + i * 16 + j

        a = result[flat_idx(1, 0, 0)]
        b = result[flat_idx(0, 4, 4)]
        assert not torch.allclose(a, b, atol=1e-5)

    def test_min_distance_at_2d_floor(self, rope):
        """With a distinct depth base, adding depth must not push any pair below the in-plane floor
        (the min pairwise distance of the pure 2D encoding, ~1.65 for a 16x16 grid)."""
        result = rope.compute(spatial_shape=(4, 16, 16), training=False)
        flat = result.reshape(result.shape[0], -1)
        dist = torch.cdist(flat, flat)
        dist.fill_diagonal_(float('inf'))
        result_2d = rope.compute(spatial_shape=(1, 16, 16), training=False)
        flat_2d = result_2d.reshape(result_2d.shape[0], -1)
        dist_2d = torch.cdist(flat_2d, flat_2d)
        dist_2d.fill_diagonal_(float('inf'))
        torch.testing.assert_close(dist.min(), dist_2d.min(), atol=1e-4, rtol=1e-4)

    def test_compute_from_coords_matches_compute(self, rope):
        """The freq_matrix path must agree with the meshgrid path, including the depth offset."""
        D, H, W = 4, 8, 8
        coords_d = (2 * torch.arange(0.5, D) / D) - 1
        coords_h = (2 * torch.arange(0.5, H) / H) - 1
        coords_w = (2 * torch.arange(0.5, W) / W) - 1
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'), dim=-1
        ).flatten(0, 2)
        from_coords = rope.compute_from_coords(coords)
        from_shape = rope.compute(spatial_shape=(D, H, W), training=False)
        torch.testing.assert_close(from_coords, from_shape, atol=1e-6, rtol=1e-5)


class TestDisableDepth:
    """disable_depth must behave identically across all three angle paths and respect the lru_cache."""

    def test_normal_path_shape_and_depth_agnostic(self):
        rope = SpatialRotaryEmbedding3D(head_dim=64, disable_depth=True)
        result = rope.compute(spatial_shape=(4, 8, 8), training=False)
        assert result.shape == (4 * 64, 2, 64)
        # All depth slices carry the same (2D) angles.
        per_slice = result.reshape(4, 64, 2, 64)
        for l in range(1, 4):
            torch.testing.assert_close(per_slice[l], per_slice[0])

    def test_augmented_path_shape(self):
        rope = SpatialRotaryEmbedding3D(head_dim=64, rescale=2.0, disable_depth=True)
        result = rope.compute(spatial_shape=(4, 8, 8), training=True)
        assert result.shape == (4 * 64, 2, 64), 'augmented path must repeat 2D angles over depth, not drop them'

    def test_compute_from_coords_zeroes_depth(self):
        rope = SpatialRotaryEmbedding3D(head_dim=64, disable_depth=True)
        coords = torch.tensor([[0.5, 0.25, -0.25], [-0.5, 0.25, -0.25], [0.0, 0.25, -0.25]])
        result = rope.compute_from_coords(coords)
        # Same (h, w), different depth -> identical output when depth is disabled.
        torch.testing.assert_close(result[1], result[0])
        torch.testing.assert_close(result[2], result[0])

    def test_cache_respects_toggle(self):
        """Toggling disable_depth after a cached call must not serve the stale entry."""
        rope = SpatialRotaryEmbedding3D(head_dim=64)
        with_depth = rope.compute(spatial_shape=(4, 8, 8), training=False)
        rope.disable_depth = True
        without_depth = rope.compute(spatial_shape=(4, 8, 8), training=False)
        assert not torch.allclose(with_depth, without_depth)
        rope.disable_depth = False
        again = rope.compute(spatial_shape=(4, 8, 8), training=False)
        torch.testing.assert_close(again, with_depth)


class TestBaseValidation:

    def test_rejects_theta_depth_one(self):
        with pytest.raises(ValueError):
            SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, theta_depth=1.0)

    def test_rejects_nonfinite_and_nonpositive(self):
        for bad in (float('inf'), float('nan'), 0.0, -10.0):
            with pytest.raises(ValueError):
                SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, theta_depth=bad)


class TestVisibleIdxGathering:

    @pytest.fixture
    def rope(self):
        return SpatialRotaryEmbedding3D(head_dim=64, theta=100.0, rescale=2.0)

    def test_gather_shape(self, rope):
        visible_idx = torch.randint(0, 196, (2, 100))
        result = rope.compute((1, 14, 14), visible_idx=visible_idx, training=False)
        assert result.shape == (2, 100, 2, 64)

    def test_gather_consistency(self, rope):
        full = rope.compute((1, 14, 14), training=False)  # [196, 2, 64]
        idx = torch.tensor([[0, 5, 100, 195]])
        gathered = rope.compute((1, 14, 14), visible_idx=idx, training=False)  # [1, 4, 2, 64]
        for i, j in enumerate(idx[0]):
            torch.testing.assert_close(gathered[0, i], full[j])


class TestBuildRope:

    def test_prepends_identity(self):
        patch_rope = torch.randn(2, 10, 2, 64)
        result = build_rope(patch_rope, n_prefix=5)
        assert result.shape == (2, 15, 2, 64)
        # Prefix: cos=1, sin=0
        torch.testing.assert_close(result[:, :5, 0, :], torch.ones(2, 5, 64))
        torch.testing.assert_close(result[:, :5, 1, :], torch.zeros(2, 5, 64))
        # Patch part unchanged
        torch.testing.assert_close(result[:, 5:], patch_rope)

    def test_unbatched(self):
        patch_rope = torch.randn(10, 2, 64)
        result = build_rope(patch_rope, n_prefix=5)
        assert result.shape == (15, 2, 64)
        torch.testing.assert_close(result[:5, 0, :], torch.ones(5, 64))
