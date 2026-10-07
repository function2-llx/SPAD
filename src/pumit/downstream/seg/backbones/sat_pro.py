"""SAT-Pro U-Net encoder adapter for dense segmentation.

SAT (zhaoziheng/SAT, npj Digital Medicine 2025) is a text-prompted universal segmenter trained
supervised on SAT-DS (72 datasets, 497 classes, CT/MR/PET — including KiTS23, AMOS, and stroke
ATLAS R2, so its exposure on those benchmarks is in its favor). Its SAT-Pro vision backbone is a
plain six-stage 3D U-Net; the text encoder, query transformer, projections, and U-Net decoder are
dropped here, mirroring how PUMIT transfers its trunk without the UCPT text/query head.

The encoder half of their vendored dynamic-network-architectures fork is behavior-identical to the
installed upstream ``PlainConvEncoder`` (verified key-for-key against the released checkpoint), so
this module instantiates the library class directly at the UNET-L configuration: channels
128/128/256/512/1024/1536, three convs per stage, InstanceNorm + LeakyReLU, downsampling as each
stage's first strided conv — re-strided to the plan nnU-Net style, changing no weight. SAT feeds
single-channel volumes repeated to three channels; summing the first conv over its input dim is
exactly equivalent on one channel, so the checkpoint is collapsed to the plan's channel count.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.native_pyramid import build_native_pyramid_encoder, native_encoder
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_NATIVE_CHANNELS = (128, 128, 256, 512, 1024, 1536)
_CONVS_PER_STAGE = 3
_RELEASED_INPUT_CHANNELS = 3
_ENCODER_PREFIX = 'module.backbone.encoder.'
# The stage-0 first conv appears twice in the state dict: as itself and via the
# ConvDropoutNormReLU ``all_modules`` Sequential aliasing the same storage.
_INPUT_WEIGHT_KEYS = (
    'stages.0.0.convs.0.conv.weight',
    'stages.0.0.convs.0.all_modules.0.weight',
)
# Non-encoder scopes inside the released checkpoint: the U-Net decoder half and the
# text-query machinery (projections and cross-attention transformer).
_ALLOWED_UNUSED_PREFIXES = (
    'module.backbone.decoder.',
    'module.projection_layer.',
    'module.mask_embed_proj.',
    'module.mid_mask_embed_proj.',
    'module.transformer_decoder.',
)
CONFIG_KEYS = frozenset()
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch'})


class SATProPyramidBackbone(nn.Module):
    """Expose the six pretrained SAT-Pro encoder stages at the plan's strides."""

    def __init__(self, input_channels: int, stage_strides: tuple[tuple[int, int, int], ...]):
        super().__init__()
        if not len(_NATIVE_CHANNELS) - 1 <= len(stage_strides) <= len(_NATIVE_CHANNELS):
            raise ValueError(
                f'expected {len(_NATIVE_CHANNELS) - 1} or {len(_NATIVE_CHANNELS)} '
                f'stage strides, got {len(stage_strides)}'
            )
        n_stages = len(stage_strides)
        self.encoder = PlainConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=_NATIVE_CHANNELS[:n_stages],
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=stage_strides,
            n_conv_per_stage=(_CONVS_PER_STAGE,) * n_stages,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs=None,
            return_skips=True,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.encoder(x))

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the six convolutional stages from shallow to deep."""
        return tuple(tuple(stage.parameters()) for stage in self.encoder.stages)


class SATProEncoder3D(nn.Module):
    """Serve the pretrained six-stage SAT-Pro hierarchy as a native-depth plan encoder."""

    def __init__(self, plan: EncoderPlan, *, input_channels: int):
        super().__init__()
        n_stages = len(plan.output_channels)
        if not len(_NATIVE_CHANNELS) - 1 <= n_stages <= len(_NATIVE_CHANNELS):
            raise ValueError(f'SAT-Pro requires a five- or six-stage plan, got {n_stages} stages')
        if tuple(plan.output_channels) != _NATIVE_CHANNELS[:n_stages]:
            raise ValueError(
                f'SAT-Pro requires plan channels {_NATIVE_CHANNELS[:n_stages]}, '
                f'got {tuple(plan.output_channels)}'
            )
        strides = tuple(tuple(int(v) for v in s) for s in plan.strides)
        if len(strides) != n_stages or strides[0] != (1, 1, 1):
            raise ValueError(
                f'SAT-Pro requires a plan starting at stride (1, 1, 1), got {strides}'
            )
        # A five-stage plan builds and consumes the shallowest five pretrained stages.
        self.backbone = SATProPyramidBackbone(input_channels, strides)
        self.input_channels = input_channels
        adopt_plan_contract(self, plan)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )

        features = list(self.backbone(x))
        target_shapes = compute_stage_shapes(tuple(x.shape[2:]), self.cumulative_strides)
        for stage, (feature, channels, shape) in enumerate(
            zip(features, self.output_channels, target_shapes)
        ):
            expected = (x.shape[0], channels, *shape)
            if tuple(feature.shape) != expected:
                raise RuntimeError(
                    f'encoder stage {stage} shape {tuple(feature.shape)} does not match {expected}'
                )
        return features


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate options for the pinned SAT-Pro encoder."""
    if weights is None:
        raise ValueError('SAT-Pro requires the released SAT_Pro.pth checkpoint')
    if checkpoint_format is not None:
        raise ValueError('SAT-Pro does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('SAT-Pro does not currently expose gradient checkpointing')
    return {}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> nn.Module:
    """Construct the plan-aligned SAT-Pro encoder; without the stem its P2-P5 are projected onto the plan schedule."""
    return build_native_pyramid_encoder(
        lambda encoder_plan: SATProEncoder3D(encoder_plan, input_channels=input_channels),
        plan,
        _NATIVE_CHANNELS,
        input_channels=input_channels,
        config=config,
    )


def collapse_repeated_input_weight(weight: Tensor) -> Tensor:
    """Sum the released three-channel stem kernel over its input dim.

    SAT repeats a single-channel volume to three channels, so ``conv(repeat(x)) ==
    conv_summed(x)`` exactly.
    """
    if weight.ndim != 5 or weight.shape[1] != _RELEASED_INPUT_CHANNELS:
        raise ValueError(
            f'expected a five-dimensional {_RELEASED_INPUT_CHANNELS}-channel stem kernel, '
            f'got shape {tuple(weight.shape)}'
        )
    return weight.sum(dim=1, keepdim=True)


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the U-Net encoder stages; the decoder half and text-query machinery are dropped."""
    if weights is None:
        raise ValueError('SAT-Pro requires the released SAT_Pro.pth checkpoint')
    encoder = native_encoder(encoder)
    if not isinstance(encoder, SATProEncoder3D):
        raise TypeError(f'expected SATProEncoder3D, got {type(encoder).__name__}')

    import numpy

    # The released training checkpoint pickles numpy scalars (data-only types) alongside the
    # optimizer state; allowlist exactly those instead of dropping weights_only. The alias entry
    # covers the pre-numpy-2 module path recorded in the pickle.
    with torch.serialization.safe_globals([
        (numpy._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
        numpy.dtype,
        numpy.dtypes.Float64DType,
        numpy.dtypes.Float32DType,
        numpy.dtypes.Int64DType,
    ]):
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    encoder_state: dict[str, Tensor] = {}
    dropped: list[str] = []
    for key, value in checkpoint.items():
        if key.startswith(_ENCODER_PREFIX):
            encoder_state[key.removeprefix(_ENCODER_PREFIX)] = value
        else:
            dropped.append(key)
    bad_dropped = [key for key in dropped if not key.startswith(_ALLOWED_UNUSED_PREFIXES)]
    if bad_dropped:
        raise RuntimeError(f'SAT-Pro checkpoint has unexpected non-encoder keys: {bad_dropped}')

    input_channels = encoder.backbone.encoder.stages[0][0].convs[0].conv.in_channels
    for key in _INPUT_WEIGHT_KEYS:
        stem = collapse_repeated_input_weight(encoder_state[key])
        if input_channels != 1:
            stem = repeat_single_channel_weight(stem, input_channels)
        encoder_state[key] = stem
    # A five-stage build consumes the shallowest five pretrained stages; deeper release keys drop here.
    built_stages = len(encoder.backbone.encoder.stages)
    encoder_state = {
        key: value
        for key, value in encoder_state.items()
        if not key.startswith('stages.') or int(key.split('.')[1]) < built_stages
    }
    encoder.backbone.encoder.load_state_dict(encoder_state, strict=True)
