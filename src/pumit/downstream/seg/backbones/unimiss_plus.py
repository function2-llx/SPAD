"""Plan-stride UniMiSS+ hierarchical encoder adapter for dense segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import Tensor, nn

from pumit.downstream.cls.backbones.unimiss_plus import (
    _2D_BRANCH_PREFIXES,
    _build_encoder,
    _load_encoder_state_dict,
)
from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.native_pyramid import build_native_pyramid_encoder, native_encoder
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_NATIVE_CHANNELS = (32, 64, 128, 256, 320, 320)
_INPUT_WEIGHT_KEY = 'ConvBlock3D0.0.conv.weight'
_POSITION_KEYS = (
    'pos_embed3D1',
    'pos_embed3D2',
    'pos_embed3D3',
    'pos_embed3D4',
)
CONFIG_KEYS = frozenset()
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch'})


class UniMissPlusPyramidBackbone(nn.Module):
    """Expose six pretrained MiT stages using the downstream plan's stride schedule."""

    feature_channels = _NATIVE_CHANNELS

    def __init__(
        self,
        input_channels: int,
        stage_strides: tuple[tuple[int, int, int], ...],
    ):
        super().__init__()
        if len(stage_strides) != len(_NATIVE_CHANNELS):
            raise ValueError(
                f'UniMiSS+ requires {len(_NATIVE_CHANNELS)} stage strides, got {len(stage_strides)}'
            )
        if stage_strides[0] != (1, 1, 1):
            raise ValueError(f'UniMiSS+ stage 0 stride must be (1, 1, 1), got {stage_strides[0]}')
        self.model = _build_encoder(input_channels, use_cls_tokens=False)

        # Stride is parameter-free. Keep the checkpoint-native position-table grids while making
        # every spatial reduction follow the shared nnU-Net plan.
        self.model.ConvBlock3D1[0].conv.stride = stage_strides[1]
        patch_embeds = (
            self.model.patch_embed3D1,
            self.model.patch_embed3D2,
            self.model.patch_embed3D3,
            self.model.patch_embed3D4,
        )
        for patch_embed, stride in zip(patch_embeds, stage_strides[2:]):
            patch_embed.patch_size = stride
            patch_embed.proj.conv.stride = stride

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return self.model.forward3d_pyramid(x)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return convolutional and Transformer parameters from shallow to deep."""
        model = self.model
        return (
            tuple(model.ConvBlock3D0.parameters()),
            tuple(model.ConvBlock3D1.parameters()),
            (
                *model.patch_embed3D1.parameters(),
                model.pos_embed3D1,
            ),
            *(tuple(block.parameters()) for block in model.block1),
            (
                *model.patch_embed3D2.parameters(),
                model.pos_embed3D2,
            ),
            *(tuple(block.parameters()) for block in model.block2),
            (
                *model.patch_embed3D3.parameters(),
                model.pos_embed3D3,
            ),
            *(tuple(block.parameters()) for block in model.block3),
            (
                *model.patch_embed3D4.parameters(),
                model.pos_embed3D4,
            ),
            *(tuple(block.parameters()) for block in model.block4),
        )


class UniMissPlusEncoder3D(nn.Module):
    """Run the pretrained UniMiSS+ hierarchy on one nnU-Net encoder plan."""

    def __init__(self, plan: EncoderPlan, *, input_channels: int):
        super().__init__()
        if plan.output_channels != _NATIVE_CHANNELS:
            raise ValueError(
                f'UniMiSS+ channels {_NATIVE_CHANNELS} do not match '
                f'nnU-Net plan channels {plan.output_channels}'
            )
        self.backbone = UniMissPlusPyramidBackbone(input_channels, plan.strides)
        self.input_channels = input_channels
        adopt_plan_contract(self, plan)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )

        input_shape = tuple(int(value) for value in x.shape[2:])
        target_shapes = compute_stage_shapes(input_shape, self.cumulative_strides)
        features = self.backbone(x)
        if len(features) != len(_NATIVE_CHANNELS):
            raise RuntimeError(
                f'UniMiSS+ returned {len(features)} stages, expected {len(_NATIVE_CHANNELS)}'
            )

        for stage, (feature, channels, shape) in enumerate(
            zip(features, self.output_channels, target_shapes)
        ):
            expected = (x.shape[0], channels, *shape)
            if tuple(feature.shape) != expected:
                raise RuntimeError(
                    f'encoder stage {stage} shape {tuple(feature.shape)} does not match {expected}'
                )
        return list(features)


def prepare_config(
    weights: Path | None,
    checkpoint_format: str | None,
    gradient_checkpointing: bool,
) -> dict[str, object]:
    """Validate the pinned UniMiSS+ architecture options."""
    if weights is None:
        raise ValueError('UniMiSS+ requires the released UniMissPlus.pth checkpoint')
    if checkpoint_format is not None:
        raise ValueError('UniMiSS+ does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('UniMiSS+ does not currently expose gradient checkpointing')
    return {}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> nn.Module:
    """Construct the plan-aligned UniMiSS+ encoder; without the stem its P2-P5 are projected onto the plan schedule."""
    return build_native_pyramid_encoder(
        lambda encoder_plan: UniMissPlusEncoder3D(encoder_plan, input_channels=input_channels),
        plan,
        _NATIVE_CHANNELS,
        input_channels=input_channels,
        config=config,
    )


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load the official UniMiSS+ student branch into the 3D encoder."""
    if weights is None:
        raise ValueError('UniMiSS+ requires the released UniMissPlus.pth checkpoint')
    encoder = native_encoder(encoder)
    if not isinstance(encoder, UniMissPlusEncoder3D):
        raise TypeError(f'expected UniMissPlusEncoder3D, got {type(encoder).__name__}')

    state_dict = _load_encoder_state_dict(str(weights))
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith('cls_tokens')
    }
    for key in _POSITION_KEYS:
        state_dict[key] = state_dict[key][:, 1:]
    input_channels = encoder.backbone.model.ConvBlock3D0[0].conv.in_channels
    if input_channels != 1:
        state_dict = dict(state_dict)
        state_dict[_INPUT_WEIGHT_KEY] = repeat_single_channel_weight(
            state_dict[_INPUT_WEIGHT_KEY],
            input_channels,
        )

    missing, unexpected = encoder.backbone.model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f'UniMiSS+ checkpoint is missing encoder keys: {missing}')
    bad_unexpected = [
        key
        for key in unexpected
        if not key.startswith(_2D_BRANCH_PREFIXES) and '.sr2D.' not in key
    ]
    if bad_unexpected:
        raise RuntimeError(
            f'UniMiSS+ checkpoint has unexpected non-2D-branch keys: {bad_unexpected}'
        )
