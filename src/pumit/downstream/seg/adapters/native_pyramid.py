"""P2-P5 view of a native pyramid encoder for the trunk-only readout.

The CNN and Swin encoders (STU-Net-L, SAT-Pro, UniMiSS+, VoCo) emit every plan stage at their pretrained widths.
``NativePyramidBackbone`` presents their P2-P5 levels as a multi-level backbone so ``PlanAlignedPyramidEncoder3D``
can project them onto the plan's channel schedule; the encoder runs unchanged (P0/P1 are still computed) and the
released checkpoints load through the wrapped encoder.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from torch import Tensor, nn

from pumit.downstream.seg.adapters.pyramid import (
    _PYRAMID_START_STAGE,
    PlanAlignedPyramidEncoder3D,
    pyramid_options,
)
from pumit.downstream.seg.plan import EncoderPlan


class NativePyramidBackbone(nn.Module):
    """Expose stages P2 onward of a plan-aligned native pyramid encoder."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.feature_channels = tuple(int(value) for value in encoder.output_channels[_PYRAMID_START_STAGE:])
        self.feature_strides = tuple(
            tuple(int(value) for value in stride)
            for stride in encoder.cumulative_strides[_PYRAMID_START_STAGE:]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.encoder(x)[_PYRAMID_START_STAGE:])

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return self.encoder.backbone.parameter_layers()


def native_plan(plan: EncoderPlan, native_channels: tuple[int, ...]) -> EncoderPlan:
    """The downstream plan at the encoder's pretrained widths (the wrapped encoder validates the stage count)."""
    return replace(plan, output_channels=tuple(native_channels[:len(plan.output_channels)]))


def build_native_pyramid_encoder(
    make_encoder: Callable[[EncoderPlan], nn.Module],
    plan: EncoderPlan,
    native_channels: tuple[int, ...],
    *,
    input_channels: int,
    config: Mapping[str, object],
) -> nn.Module:
    """Build a native pyramid encoder for the plan, or its trunk-only P2-P5 projection when the stem is dropped.

    ``make_encoder`` constructs the encoder on the plan it receives; without the stem that plan carries the
    pretrained widths and the levels are projected onto the downstream plan's channel schedule.
    """
    options = pyramid_options(config)
    if options['with_high_resolution_stem']:
        return make_encoder(plan)
    return PlanAlignedPyramidEncoder3D(
        NativePyramidBackbone(make_encoder(native_plan(plan, native_channels))),
        plan,
        input_channels=input_channels,
        **options,
    )


def native_encoder(encoder: nn.Module) -> nn.Module:
    """The native pyramid encoder behind a projected readout, or the encoder itself."""
    if isinstance(encoder, PlanAlignedPyramidEncoder3D) and isinstance(encoder.backbone, NativePyramidBackbone):
        return encoder.backbone.encoder
    return encoder
