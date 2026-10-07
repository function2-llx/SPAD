"""Tests for SPAD U-Net geometry utilities."""
import math
import torch
import pytest
from pumit.spad_unet.geometry import (
    SUPPORTED_FGC_STAGES,
    compute_continuous_da,
    decompose_continuous_da,
    sample_da,
    select_da_pair,
    compute_da_schedule,
    compute_fgc_geometry,
    compute_stride_schedule,
)


class TestComputeContinuousDA:
    def test_isotropic(self):
        assert compute_continuous_da((1.0, 1.0, 1.0)) == 0.0

    def test_depth_less_than_inplane(self):
        assert compute_continuous_da((0.5, 1.0, 1.0)) == 0.0

    def test_kits(self):
        da = compute_continuous_da((1.0, 0.78, 0.78))
        assert abs(da - math.log2(1.0 / 0.78)) < 1e-6

    def test_acdc(self):
        da = compute_continuous_da((5.0, 1.56, 1.56))
        assert abs(da - math.log2(5.0 / 1.56)) < 1e-6

    def test_btcv(self):
        da = compute_continuous_da((3.0, 0.76, 0.76))
        assert abs(da - math.log2(3.0 / 0.76)) < 1e-6

    def test_uses_finest_inplane_spacing(self):
        da = compute_continuous_da((4.0, 1.0, 2.0))
        assert da == 2.0


class TestSampleDA:
    def test_integer_da_returns_same(self):
        assert sample_da(0.0) == 0
        assert sample_da(1.0) == 1
        assert sample_da(2.0) == 2

    def test_fractional_returns_floor_or_ceil(self):
        torch.manual_seed(42)
        results = [sample_da(1.5) for _ in range(100)]
        assert set(results) == {1, 2}
        assert 30 < results.count(2) < 70


class TestDecomposeContinuousDA:
    def test_fractional_da(self):
        assert decompose_continuous_da(1.25) == (1, 2, 0.25)

    def test_integer_da_has_one_endpoint(self):
        assert decompose_continuous_da(2.0) == (2, 2, 0.0)

    @pytest.mark.parametrize('value', [-1.0, float('inf'), float('nan'), True])
    def test_invalid_da_is_rejected(self, value):
        with pytest.raises(ValueError, match='finite non-negative real'):
            decompose_continuous_da(value)


class TestSelectDAPair:
    def test_stochastic_tied_uses_one_draw(self, monkeypatch):
        calls = []

        def sample(value):
            calls.append(value)
            return 1

        monkeypatch.setattr('pumit.spad_unet.geometry.sample_da', sample)

        assert select_da_pair(1.6, cross=False) == (1, 1)
        assert calls == [1.6]

    def test_stochastic_cross_uses_independent_draws(self, monkeypatch):
        draws = iter((1, 2))
        monkeypatch.setattr(
            'pumit.spad_unet.geometry.sample_da',
            lambda value: next(draws),
        )

        assert select_da_pair(1.6, cross=True) == (1, 2)


class TestComputeDASchedule:
    def test_da0(self):
        assert compute_da_schedule(0, 6) == [0, 0, 0, 0, 0, 0]

    def test_da1(self):
        assert compute_da_schedule(1, 6) == [1, 1, 0, 0, 0, 0]

    def test_da2(self):
        assert compute_da_schedule(2, 6) == [2, 2, 1, 0, 0, 0]

    def test_da3(self):
        assert compute_da_schedule(3, 6) == [3, 3, 2, 1, 0, 0]


class TestStrideSchedule:
    def test_stage0_always_no_stride(self):
        for da in range(4):
            ss = compute_stride_schedule(da, 7, (128, 128, 128))
            assert ss[0] == (1, 1, 1)

    def test_isotropic_large(self):
        """256^3 at da=0: all stages downsample fully."""
        ss = compute_stride_schedule(0, 7, (256, 256, 256))
        assert ss == [(1, 1, 1)] + [(2, 2, 2)] * 6

    def test_da_forces_depth_stride_1(self):
        """DA>=1 forces depth stride to 1."""
        ss = compute_stride_schedule(2, 7, (128, 256, 256))
        # da_schedule: [2, 2, 1, 0, 0, 0, 0]
        assert ss[1] == (1, 2, 2)  # da=2
        assert ss[2] == (1, 2, 2)  # da=1
        assert ss[3] == (2, 2, 2)  # da=0, first full 3D downsample

    def test_min_bottleneck_4(self):
        """Clamping uses min_bottleneck=4 (nnU-Net default)."""
        ss = compute_stride_schedule(0, 7, (32, 32, 32), min_bottleneck=4)
        # 32 -> 16 -> 8 -> 4 (stop). Stages 1-3 stride, 4-6 clamped.
        assert ss[1:4] == [(2, 2, 2)] * 3
        assert ss[4:] == [(1, 1, 1)] * 3

    def test_quantized_values_only(self):
        """Only (1,1,1), (1,2,2), or (2,2,2) produced."""
        for da in range(5):
            for patch in [(20, 256, 224), (96, 96, 96), (160, 224, 192)]:
                ss = compute_stride_schedule(da, 7, patch)
                for s in ss:
                    assert s in ((1, 1, 1), (1, 2, 2), (2, 2, 2)), f"Bad stride {s} for da={da}, patch={patch}"

    def test_depth_clamping_at_da0(self):
        """When depth is small but in-plane large, depth clamps to (1,2,2)."""
        ss = compute_stride_schedule(0, 7, (8, 256, 256))
        # depth 8: can do 8->4 (stage 1), then 4->? clamped (stage 2+)
        assert ss[1] == (2, 2, 2)
        assert ss[2] == (1, 2, 2)  # depth clamped at 4


class TestFeatureGridCanonicalizationGeometry:
    @pytest.mark.parametrize(
        ('spacing_ratio', 'patch_depth', 'canonical_depths'),
        [
            (1.25, 160, (192, 96, 48, 24, 12)),
            (2.5, 80, (96, 96, 48, 24, 12)),
            (5.0, 40, (48, 48, 48, 24, 12)),
            (10.0, 20, (24, 24, 24, 24, 12)),
        ],
    )
    @pytest.mark.parametrize('stage', SUPPORTED_FGC_STAGES)
    def test_placement_sweep_for_every_supported_floor_state(
        self,
        spacing_ratio,
        patch_depth,
        canonical_depths,
        stage,
    ):
        geometry = compute_fgc_geometry(
            math.log2(spacing_ratio),
            (patch_depth, 192, 192),
            stage=stage,
        )

        assert geometry.stage == stage
        assert geometry.canonical_shape == (
            canonical_depths[stage - 1],
            192 // 2 ** (stage - 1),
            192 // 2 ** (stage - 1),
        )
        assert geometry.feature_shapes[-1] == (6, 6, 6)

    def test_five_mm_common_patch(self):
        geometry = compute_fgc_geometry(
            math.log2(5.0),
            (40, 192, 192),
            stage=2,
        )

        assert geometry.route_da == 2
        assert geometry.stage == 2
        assert geometry.native_bridge_shape == (40, 96, 96)
        assert geometry.da_schedule == (2, 2, 1, 0, 0, 0)
        assert geometry.stride_schedule == (
            (1, 1, 1),
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        )
        assert geometry.canonical_shape == (48, 96, 96)
        assert geometry.feature_shapes == (
            (40, 192, 192),
            (40, 96, 96),
            (48, 48, 48),
            (24, 24, 24),
            (12, 12, 12),
            (6, 6, 6),
        )

    @pytest.mark.parametrize(
        ('continuous_da', 'patch_size', 'canonical_shape'),
        [
            (0.0, (192, 192, 192), (96, 96, 96)),
            (1.0, (96, 192, 192), (96, 96, 96)),
            (2.0, (48, 192, 192), (48, 96, 96)),
            (3.0, (24, 192, 192), (24, 96, 96)),
        ],
    )
    def test_dyadic_spacing_is_identity(
        self,
        continuous_da,
        patch_size,
        canonical_shape,
    ):
        geometry = compute_fgc_geometry(
            continuous_da,
            patch_size,
            stage=2,
        )

        assert geometry.canonical_shape == canonical_shape
        assert geometry.feature_shapes[-1] == (6, 6, 6)

    def test_uses_round_half_up_for_shape_closure(self):
        geometry = compute_fgc_geometry(
            math.log2(5.2),
            (40, 192, 192),
            stage=2,
        )

        assert geometry.canonical_shape == (56, 96, 96)

    @pytest.mark.parametrize(
        ('spacing_ratio', 'patch_size', 'canonical_shape'),
        [
            (1.25, (160, 192, 192), (96, 96, 96)),
            (2.5, (80, 192, 192), (96, 96, 96)),
            (5.0, (40, 192, 192), (48, 96, 96)),
            (10.0, (20, 192, 192), (24, 96, 96)),
        ],
    )
    def test_fractional_closure_for_every_supported_floor_state(
        self,
        spacing_ratio,
        patch_size,
        canonical_shape,
    ):
        geometry = compute_fgc_geometry(
            math.log2(spacing_ratio),
            patch_size,
            stage=2,
        )

        assert geometry.canonical_shape == canonical_shape

    def test_half_up_is_stable_at_float_tie(self):
        geometry = compute_fgc_geometry(
            math.log2(13 / 3),
            (48, 128, 128),
            stage=2,
        )

        assert geometry.canonical_shape == (56, 64, 64)

    def test_rejects_non_common_patch_depth(self):
        with pytest.raises(ValueError, match='full-route-compatible'):
            compute_fgc_geometry(
                math.log2(5.0),
                (39, 192, 192),
                stage=2,
            )

    @pytest.mark.parametrize(
        'patch_size',
        [
            (40.5, 192, 192),
            (True, 192, 192),
        ],
    )
    def test_rejects_non_integer_patch_shape(self, patch_size):
        with pytest.raises(ValueError, match='three positive integers'):
            compute_fgc_geometry(
                math.log2(5.0),
                patch_size,
                stage=2,
            )

    def test_rejects_size_clamped_geometry(self):
        with pytest.raises(ValueError, match='size-clamped'):
            compute_fgc_geometry(0.0, (8, 16, 16), stage=2)

    def test_ceil_route_canonicalizes_its_own_bridge(self):
        geometry = compute_fgc_geometry(
            math.log2(5.0),
            (40, 192, 192),
            stage=2,
            route_da=3,
        )

        assert geometry.route_da == 3
        assert geometry.da_schedule == (3, 3, 2, 1, 0, 0)
        assert geometry.native_bridge_shape == (40, 96, 96)
        # The ceil suffix keeps one more z stride, so closure rounds 40 * 1.25 to a multiple of 4.
        assert geometry.canonical_shape == (52, 96, 96)
        assert geometry.feature_shapes == (
            (40, 192, 192),
            (40, 96, 96),
            (52, 48, 48),
            (52, 24, 24),
            (26, 12, 12),
            (13, 6, 6),
        )

    def test_ceil_route_keeps_upsampling_direction_when_bridges_differ(self):
        floor_geometry = compute_fgc_geometry(
            math.log2(1.5),
            (128, 192, 192),
            stage=2,
        )
        ceil_geometry = compute_fgc_geometry(
            math.log2(1.5),
            (128, 192, 192),
            stage=2,
            route_da=1,
        )

        assert floor_geometry.native_bridge_shape == (64, 96, 96)
        assert floor_geometry.canonical_shape == (96, 96, 96)
        assert ceil_geometry.native_bridge_shape == (128, 96, 96)
        assert ceil_geometry.canonical_shape == (192, 96, 96)
        assert ceil_geometry.feature_shapes[-1] == (12, 6, 6)

    @pytest.mark.parametrize('stage', SUPPORTED_FGC_STAGES)
    def test_every_placement_supports_the_ceil_route(self, stage):
        geometry = compute_fgc_geometry(
            math.log2(5.0),
            (40, 192, 192),
            stage=stage,
            route_da=3,
        )

        assert geometry.canonical_shape[0] >= geometry.native_bridge_shape[0]
        assert geometry.canonical_shape[1:] == geometry.native_bridge_shape[1:]

    def test_integer_spacing_accepts_only_its_single_route(self):
        geometry = compute_fgc_geometry(
            2.0,
            (48, 192, 192),
            stage=2,
            route_da=2,
        )

        assert geometry.canonical_shape == (48, 96, 96)
        with pytest.raises(ValueError, match='endpoint DA states'):
            compute_fgc_geometry(2.0, (48, 192, 192), stage=2, route_da=3)

    @pytest.mark.parametrize('route_da', [1, 4])
    def test_rejects_route_outside_the_case_endpoints(self, route_da):
        with pytest.raises(ValueError, match='endpoint DA states'):
            compute_fgc_geometry(
                math.log2(5.0),
                (40, 192, 192),
                stage=2,
                route_da=route_da,
            )


