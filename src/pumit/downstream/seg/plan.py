"""nnU-Net encoder-plan contracts and spatial shape calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import nn


def compute_cumulative_strides(
    strides: Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
    """Convert nnU-Net's relative stage strides to cumulative input strides."""
    cumulative = (1, 1, 1)
    result = []
    for stage, stage_stride in enumerate(strides):
        stride = tuple(int(value) for value in stage_stride)
        if len(stride) != 3 or any(value <= 0 for value in stride):
            raise ValueError(f'invalid three-dimensional stride at stage {stage}: {stride}')
        cumulative = tuple(a * b for a, b in zip(cumulative, stride))
        result.append(cumulative)
    return tuple(result)


def compute_stage_shapes(
    input_shape: Sequence[int],
    cumulative_strides: Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
    """Compute plan-stage feature shapes for one input patch."""
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f'input_shape must contain three positive dimensions, got {shape}')

    stage_shapes = []
    for stage, stage_stride in enumerate(cumulative_strides):
        stride = tuple(int(value) for value in stage_stride)
        if len(stride) != 3 or any(value <= 0 for value in stride):
            raise ValueError(f'invalid cumulative stride at stage {stage}: {stride}')
        if any(size % step for size, step in zip(shape, stride)):
            raise ValueError(f'shape {shape} is not divisible by cumulative stride {stride} at stage {stage}')
        stage_shapes.append(tuple(size // step for size, step in zip(shape, stride)))
    return tuple(stage_shapes)


@dataclass(frozen=True)
class EncoderPlan:
    """The complete nnU-Net encoder-stage contract consumed by its stock decoder."""

    output_channels: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, int, int], ...]
    strides: tuple[tuple[int, int, int], ...]
    conv_op: type[nn.Module]
    conv_bias: bool
    norm_op: type[nn.Module] | None
    norm_op_kwargs: dict | None
    dropout_op: type[nn.Module] | None
    dropout_op_kwargs: dict | None
    nonlin: type[nn.Module] | None
    nonlin_kwargs: dict | None
    n_blocks_per_stage: tuple[int, ...]
    n_conv_per_stage_decoder: tuple[int, ...]

    @property
    def cumulative_strides(self) -> tuple[tuple[int, int, int], ...]:
        return compute_cumulative_strides(self.strides)


def encoder_plan_from_architecture_kwargs(
    *,
    features_per_stage: Sequence[int],
    kernel_sizes: Sequence[Sequence[int]],
    strides: Sequence[Sequence[int]],
    n_blocks_per_stage: Sequence[int],
    n_conv_per_stage_decoder: Sequence[int],
    conv_op: type[nn.Module],
    conv_bias: bool,
    norm_op: type[nn.Module] | None,
    norm_op_kwargs: Mapping[str, object] | None,
    dropout_op: type[nn.Module] | None,
    dropout_op_kwargs: Mapping[str, object] | None,
    nonlin: type[nn.Module] | None,
    nonlin_kwargs: Mapping[str, object] | None,
) -> EncoderPlan:
    """Build the encoder-decoder contract from native nnU-Net architecture kwargs."""
    output_channels = tuple(int(value) for value in features_per_stage)
    kernel_sizes = tuple(
        tuple(int(value) for value in values)
        for values in kernel_sizes
    )
    strides = tuple(
        tuple(int(value) for value in values)
        for values in strides
    )
    encoder_blocks = tuple(int(value) for value in n_blocks_per_stage)
    decoder_convs = tuple(int(value) for value in n_conv_per_stage_decoder)
    if not len(output_channels) == len(kernel_sizes) == len(strides) == len(encoder_blocks):
        raise ValueError(
            'features_per_stage, kernel_sizes, strides, and n_blocks_per_stage must describe '
            'the same encoder stages'
        )
    if len(output_channels) <= 3:
        raise ValueError('nnU-Net encoder plan must contain more than 3 stages')
    if len(decoder_convs) != len(output_channels) - 1:
        raise ValueError(
            f'n_conv_per_stage_decoder has {len(decoder_convs)} entries for '
            f'{len(output_channels)} encoder stages'
        )
    if conv_op is not nn.Conv3d:
        raise ValueError(f'downstream segmentation requires nn.Conv3d, got {conv_op}')

    return EncoderPlan(
        output_channels=output_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        conv_op=conv_op,
        conv_bias=bool(conv_bias),
        norm_op=norm_op,
        norm_op_kwargs=dict(norm_op_kwargs) if norm_op_kwargs is not None else None,
        dropout_op=dropout_op,
        dropout_op_kwargs=dict(dropout_op_kwargs) if dropout_op_kwargs is not None else None,
        nonlin=nonlin,
        nonlin_kwargs=dict(nonlin_kwargs) if nonlin_kwargs is not None else None,
        n_blocks_per_stage=encoder_blocks,
        n_conv_per_stage_decoder=decoder_convs,
    )
