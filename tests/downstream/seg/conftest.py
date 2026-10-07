"""Shared fixtures for the downstream segmentation tests."""

from collections.abc import Sequence

from torch import nn

from pumit.downstream.seg.plan import EncoderPlan, encoder_plan_from_architecture_kwargs

# nnU-Net ResEnc layer types shared by every test plan; the geometry varies per test.
PLAN_LAYERS = {
    'conv_op': nn.Conv3d,
    'conv_bias': True,
    'norm_op': nn.InstanceNorm3d,
    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
    'dropout_op': None,
    'dropout_op_kwargs': None,
    'nonlin': nn.LeakyReLU,
    'nonlin_kwargs': {'inplace': True},
}
ISOTROPIC_STRIDES = [[1, 1, 1]] + [[2, 2, 2]] * 5


def architecture_kwargs(
    features_per_stage: Sequence[int],
    strides: Sequence[Sequence[int]] | None = None,
    *,
    n_blocks_per_stage: Sequence[int] | None = None,
    n_conv_per_stage_decoder: Sequence[int] | None = None,
) -> dict:
    """nnU-Net architecture kwargs for a test plan: one block per stage and one decoder conv unless given."""
    n_stages = len(features_per_stage)
    return {
        'features_per_stage': list(features_per_stage),
        'kernel_sizes': [[3, 3, 3]] * n_stages,
        'strides': [list(stride) for stride in (strides if strides is not None else ISOTROPIC_STRIDES[:n_stages])],
        'n_blocks_per_stage': list(n_blocks_per_stage) if n_blocks_per_stage is not None else [1] * n_stages,
        'n_conv_per_stage_decoder': (
            list(n_conv_per_stage_decoder) if n_conv_per_stage_decoder is not None else [1] * (n_stages - 1)
        ),
        **PLAN_LAYERS,
    }


def make_plan(
    features_per_stage: Sequence[int],
    strides: Sequence[Sequence[int]] | None = None,
    *,
    n_blocks_per_stage: Sequence[int] | None = None,
    n_conv_per_stage_decoder: Sequence[int] | None = None,
) -> EncoderPlan:
    return encoder_plan_from_architecture_kwargs(
        **architecture_kwargs(
            features_per_stage,
            strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        )
    )
