"""Spatial geometry and deep-supervision utilities for SPAD U-Net."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch


SUPPORTED_FGC_STAGES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class FeatureGridCanonicalizationGeometry:
    """Static mixed-grid geometry for one FGC placement on one integer DA route."""

    stage: int
    route_da: int
    da_schedule: tuple[int, ...]
    stride_schedule: tuple[tuple[int, int, int], ...]
    native_bridge_shape: tuple[int, int, int]
    canonical_shape: tuple[int, int, int]
    feature_shapes: tuple[tuple[int, int, int], ...]


def _round_half_up(value: float) -> int:
    """Round a non-negative float half-up with an ulp-scale tie tolerance."""
    lower = math.floor(value)
    fraction = value - lower
    if math.isclose(
        fraction,
        0.5,
        rel_tol=0.0,
        abs_tol=8 * math.ulp(value),
    ):
        return lower + 1
    return math.floor(value + 0.5)


def compute_continuous_da(target_spacing: tuple[float, float, float]) -> float:
    """Compute continuous DA from target spacing [depth, height, width].

    DA = log2(depth_spacing / finest_inplane_spacing). Returns 0.0 if depth <= inplane.
    """
    depth_spacing = target_spacing[0]
    inplane_spacing = min(target_spacing[1], target_spacing[2])
    if depth_spacing <= inplane_spacing:
        return 0.0
    return math.log2(depth_spacing / inplane_spacing)


def decompose_continuous_da(continuous_da: float) -> tuple[int, int, float]:
    """Return floor state, ceil state, and ceil weight for a continuous DA."""
    if (
        isinstance(continuous_da, bool)
        or not isinstance(continuous_da, Real)
        or not math.isfinite(continuous_da)
        or continuous_da < 0
    ):
        raise ValueError(
            f'continuous_da must be a finite non-negative real, got {continuous_da!r}'
        )
    continuous_da = float(continuous_da)
    floor_da = math.floor(continuous_da)
    ceil_weight = continuous_da - floor_da
    ceil_da = floor_da if ceil_weight == 0 else floor_da + 1
    return floor_da, ceil_da, ceil_weight


def sample_da(continuous_da: float) -> int:
    """Stochastic DA: Bernoulli sample floor or ceil based on fractional part."""
    floor_da, ceil_da, ceil_weight = decompose_continuous_da(continuous_da)
    if floor_da == ceil_da:
        return floor_da
    return floor_da + int(torch.bernoulli(torch.tensor(ceil_weight)).item())


def select_da_pair(
    continuous_da: float,
    *,
    cross: bool,
) -> tuple[int, int]:
    """Stochastically select tied or independent encoder and decoder DA states."""
    decompose_continuous_da(continuous_da)
    encoder_da = sample_da(continuous_da)
    decoder_da = sample_da(continuous_da) if cross else encoder_da
    return encoder_da, decoder_da


def compute_da_schedule(sample_da: int, n_stages: int) -> list[int]:
    """Compute per-stage DA schedule. DA decreases by 1 at each downsampling stage."""
    schedule = []
    remaining_da = sample_da
    for i in range(n_stages):
        schedule.append(remaining_da)
        if i > 0 and remaining_da > 0:
            remaining_da -= 1
    return schedule


def compute_stride_schedule(
    sample_da: int,
    n_stages: int,
    patch_size: tuple[int, int, int],
    min_bottleneck: int = 4,
) -> list[tuple[int, int, int]]:
    """Compute per-stage strides with DA adaptation and per-dimension size clamping.

    Stride at each stage is determined by:
      1. DA schedule (primary): DA>=1 forces depth stride=1
      2. Size clamping (override): if current_size[dim] / 2 < min_bottleneck, stride=1

    Only produces strides compatible with K3S2: (1,1,1), (1,2,2), or (2,2,2).
    When depth is clamped but in-plane is OK: (1,2,2).
    When any in-plane dim is also clamped: (1,1,1).

    min_bottleneck=4 matches nnU-Net's UNet_featuremap_min_edge_length.

    Args:
        sample_da: integer DA for this sample
        n_stages: number of encoder stages
        patch_size: input patch size (D, H, W)
        min_bottleneck: minimum spatial size before stride is clamped to 1

    Returns:
        list of (stride_d, stride_h, stride_w) per stage
    """
    da_schedule = compute_da_schedule(sample_da, n_stages)
    strides = []
    current_size = list(patch_size)

    for i in range(n_stages):
        if i == 0:
            strides.append((1, 1, 1))
            continue

        da = da_schedule[i]
        # DA-determined intended stride
        depth_stride = 1 if da >= 1 else 2
        h_stride = 2
        w_stride = 2

        # Per-dimension size clamping
        if current_size[0] // depth_stride < min_bottleneck:
            depth_stride = 1
        if current_size[1] // h_stride < min_bottleneck:
            h_stride = 1
        if current_size[2] // w_stride < min_bottleneck:
            w_stride = 1

        # Quantize to K3S2-compatible strides: (1,1,1), (1,2,2), or (2,2,2)
        if h_stride == 1 or w_stride == 1:
            stride = (1, 1, 1)
        elif depth_stride == 1:
            stride = (1, 2, 2)
        else:
            stride = (2, 2, 2)

        strides.append(stride)
        current_size[0] //= stride[0]
        current_size[1] //= stride[1]
        current_size[2] //= stride[2]

    return strides


def compute_stage_shapes(
    shape: tuple[int, int, int],
    stride_schedule: Sequence[tuple[int, int, int]],
    *,
    first_stage: int = 0,
    label: str = 'feature',
) -> list[tuple[int, int, int]]:
    """Walk a shape through consecutive per-stage strides, requiring exact divisibility."""
    shapes = []
    current_shape = tuple(shape)
    for offset, stride in enumerate(stride_schedule):
        if any(
            size % step
            for size, step in zip(current_shape, stride, strict=True)
        ):
            raise ValueError(
                f'{label} shape {current_shape} is not divisible by '
                f'stage-{first_stage + offset} stride {stride}'
            )
        current_shape = tuple(
            size // step
            for size, step in zip(current_shape, stride, strict=True)
        )
        shapes.append(current_shape)
    return shapes


def compute_fgc_geometry(
    continuous_da: float,
    patch_size: tuple[int, int, int],
    *,
    stage: int,
    n_stages: int = 6,
    min_bottleneck: int = 4,
    route_da: int | None = None,
) -> FeatureGridCanonicalizationGeometry:
    """Build one mixed native/canonical hierarchy for Sample-Native-Z FGC.

    The experiment keeps the existing full-route-compatible input patch,
    canonicalizes the feature before ``stage``, and rejects geometries whose
    native route relies on runtime size clamping.

    Args:
        route_da: integer operator DA whose hierarchy hosts the canonical suffix, restricted to the
            case's floor or ceil endpoint. Defaults to floor. Each route removes the same fractional
            correction from its own native bridge, so a route that reaches ``stage`` with fewer z
            strides canonicalizes onto a correspondingly finer grid.
    """
    patch_size = tuple(patch_size)
    if len(patch_size) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
        for value in patch_size
    ):
        raise ValueError(
            f'patch_size must contain three positive integers, got {patch_size}'
        )
    patch_size = tuple(int(value) for value in patch_size)
    if n_stages != 6:
        raise ValueError(f'FGC requires six stages, got {n_stages}')
    if (
        isinstance(stage, bool)
        or not isinstance(stage, Integral)
        or stage not in SUPPORTED_FGC_STAGES
    ):
        raise ValueError(
            f'FGC stage must be one of {SUPPORTED_FGC_STAGES}, '
            f'got {stage!r}'
        )
    stage = int(stage)

    floor_da, ceil_da, fractional_da = decompose_continuous_da(continuous_da)
    if floor_da > 3:
        raise ValueError(
            f'FGC supports floor DA states 0 through 3, got {floor_da}'
        )
    if route_da is None:
        route_da = floor_da
    elif route_da not in (floor_da, ceil_da):
        raise ValueError(
            f'FGC route must be one of the case endpoint DA states '
            f'{(floor_da, ceil_da)}, got {route_da!r}'
        )
    da_schedule = tuple(compute_da_schedule(route_da, n_stages))
    stride_schedule = tuple(compute_stride_schedule(
        route_da,
        n_stages,
        patch_size,
        min_bottleneck,
    ))
    intended_stride_schedule = tuple(
        (1, 1, 1)
        if stage_idx == 0
        else ((1, 2, 2) if da_schedule[stage_idx] >= 1 else (2, 2, 2))
        for stage_idx in range(n_stages)
    )
    if stride_schedule != intended_stride_schedule:
        raise ValueError(
            'FGC does not support a size-clamped stride schedule: '
            f'expected {intended_stride_schedule}, got {stride_schedule}'
        )
    full_route_divisor = tuple(
        math.prod(stride[axis] for stride in stride_schedule)
        for axis in range(3)
    )
    if any(
        size % divisor
        for size, divisor in zip(
            patch_size,
            full_route_divisor,
            strict=True,
        )
    ):
        raise ValueError(
            f'FGC requires a common full-route-compatible patch: '
            f'{patch_size} is not divisible by {full_route_divisor}'
        )

    feature_shapes = compute_stage_shapes(
        patch_size,
        stride_schedule[:stage],
        label='native prefix',
    )
    native_bridge_shape = feature_shapes[-1]

    suffix_depth_divisor = math.prod(
        stride[0]
        for stride in stride_schedule[stage:]
    )
    ideal_depth = native_bridge_shape[0] * 2**fractional_da
    canonical_units = _round_half_up(ideal_depth / suffix_depth_divisor)
    canonical_depth = suffix_depth_divisor * max(1, canonical_units)
    if canonical_depth < native_bridge_shape[0]:
        raise ValueError(
            f'FGC must not downsample the common-patch feature depth: '
            f'{native_bridge_shape[0]} -> {canonical_depth}'
        )
    canonical_shape = (
        canonical_depth,
        native_bridge_shape[1],
        native_bridge_shape[2],
    )

    feature_shapes.extend(compute_stage_shapes(
        canonical_shape,
        stride_schedule[stage:],
        first_stage=stage,
        label='canonical suffix',
    ))

    return FeatureGridCanonicalizationGeometry(
        stage=stage,
        route_da=route_da,
        da_schedule=da_schedule,
        stride_schedule=stride_schedule,
        native_bridge_shape=native_bridge_shape,
        canonical_shape=canonical_shape,
        feature_shapes=tuple(feature_shapes),
    )


