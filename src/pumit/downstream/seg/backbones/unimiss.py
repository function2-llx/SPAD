"""Plan-stride UniMiSS hierarchical encoder adapter for dense segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import Tensor, nn

from dynamic_network_architectures.architectures.unet import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)

from pumit.downstream.cls.backbones.unimiss import (
    _2D_BRANCH_PREFIXES,
    _build_encoder,
    _load_encoder_state_dict,
)
from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_NATIVE_CHANNELS = (32, 48, 128, 256, 512)
_INPUT_WEIGHT_KEY = 'patch_embed3D0.conv.weight'
CONFIG_KEYS = frozenset()


def _make_projection(
    in_channels: int,
    out_channels: int,
    plan: EncoderPlan,
) -> nn.Module:
    if in_channels == out_channels:
        return nn.Identity()
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=plan.conv_bias)


class UniMissPyramidBackbone(nn.Module):
    """Expose the five pretrained MiT stages using plan stages P1 through P5."""

    def __init__(
        self,
        input_channels: int,
        stage_strides: tuple[tuple[int, int, int], ...],
    ):
        super().__init__()
        if len(stage_strides) != len(_NATIVE_CHANNELS):
            raise ValueError(
                f'UniMiSS requires {len(_NATIVE_CHANNELS)} native stage strides, '
                f'got {len(stage_strides)}'
            )
        self.model = _build_encoder(input_channels)
        del self.model.norm_new
        del self.model.head_new
        self.model.patch_embed3D0.conv.stride = stage_strides[0]
        patch_embeds = (
            self.model.patch_embed3D1,
            self.model.patch_embed3D2,
            self.model.patch_embed3D3,
            self.model.patch_embed3D4,
        )
        for patch_embed, stride in zip(patch_embeds, stage_strides[1:]):
            patch_embed.patch_size = stride
            patch_embed.proj.conv.stride = stride

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return self.model.forward3d_pyramid(x)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return convolutional and Transformer parameters from shallow to deep."""
        model = self.model
        return (
            tuple(model.patch_embed3D0.parameters()),
            (
                *model.patch_embed3D1.parameters(),
                model.pos_embed3D1,
                model.cls_tokens1,
            ),
            *(tuple(block.parameters()) for block in model.block1),
            (
                *model.patch_embed3D2.parameters(),
                model.pos_embed3D2,
                *model.cls_tokens2.parameters(),
            ),
            *(tuple(block.parameters()) for block in model.block2),
            (
                *model.patch_embed3D3.parameters(),
                model.pos_embed3D3,
                *model.cls_tokens3.parameters(),
            ),
            *(tuple(block.parameters()) for block in model.block3),
            (
                *model.patch_embed3D4.parameters(),
                model.pos_embed3D4,
                *model.cls_tokens4.parameters(),
            ),
            *(tuple(block.parameters()) for block in model.block4),
        )


class UniMissEncoder3D(nn.Module):
    """Add scratch P0 and adapt the pretrained hierarchy to plan stages P1 through P5."""

    def __init__(self, plan: EncoderPlan, *, input_channels: int):
        super().__init__()
        if len(plan.output_channels) != len(_NATIVE_CHANNELS) + 1:
            raise ValueError(
                f'UniMiSS expects a six-stage nnU-Net plan, got {len(plan.output_channels)} stages'
            )
        self.backbone = UniMissPyramidBackbone(input_channels, plan.strides[1:])
        self.input_channels = input_channels
        adopt_plan_contract(self, plan)
        self.high_resolution_stem = ResidualEncoder(
            input_channels=input_channels,
            n_stages=1,
            features_per_stage=plan.output_channels[:1],
            conv_op=plan.conv_op,
            kernel_sizes=plan.kernel_sizes[:1],
            strides=plan.strides[:1],
            n_blocks_per_stage=plan.n_blocks_per_stage[:1],
            conv_bias=plan.conv_bias,
            norm_op=plan.norm_op,
            norm_op_kwargs=plan.norm_op_kwargs,
            dropout_op=plan.dropout_op,
            dropout_op_kwargs=plan.dropout_op_kwargs,
            nonlin=plan.nonlin,
            nonlin_kwargs=plan.nonlin_kwargs,
            return_skips=True,
        )
        self.stage_projections = nn.ModuleList(
            _make_projection(source, target, plan)
            for source, target in zip(_NATIVE_CHANNELS, plan.output_channels[1:])
        )
        initializer = InitWeights_He(1e-2)
        self.high_resolution_stem.apply(initializer)
        self.high_resolution_stem.apply(init_last_bn_before_add_to_0)
        self.stage_projections.apply(initializer)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f'expected [B, C, D, H, W] input, got shape {tuple(x.shape)}')
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f'input has {x.shape[1]} channels, but the nnU-Net plan expects {self.input_channels}'
            )

        native_features = self.backbone(x)
        features = [
            *self.high_resolution_stem(x),
            *(
                projection(feature)
                for projection, feature in zip(self.stage_projections, native_features)
            ),
        ]

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
    """Validate the pinned UniMiSS architecture options."""
    if weights is None:
        raise ValueError('UniMiSS requires the released UniMiss_small.pth checkpoint')
    if checkpoint_format is not None:
        raise ValueError('UniMiSS does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('UniMiSS does not currently expose gradient checkpointing')
    return {}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> UniMissEncoder3D:
    """Construct the plan-aligned UniMiSS encoder without reading its weights."""
    return UniMissEncoder3D(plan, input_channels=input_channels)


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load the official UniMiSS student branch into the native five-stage hierarchy."""
    if weights is None:
        raise ValueError('UniMiSS requires the released UniMiss_small.pth checkpoint')
    if not isinstance(encoder, UniMissEncoder3D):
        raise TypeError(f'expected UniMissEncoder3D, got {type(encoder).__name__}')

    state_dict = _load_encoder_state_dict(str(weights))
    input_channels = encoder.backbone.model.patch_embed3D0.conv.in_channels
    if input_channels != 1:
        state_dict = dict(state_dict)
        state_dict[_INPUT_WEIGHT_KEY] = repeat_single_channel_weight(
            state_dict[_INPUT_WEIGHT_KEY],
            input_channels,
        )

    missing, unexpected = encoder.backbone.model.load_state_dict(state_dict, strict=False)
    missing_critical = [
        key
        for key in missing
        if not key.startswith(('norm_new', 'head_new'))
    ]
    if missing_critical:
        raise RuntimeError(f'UniMiSS checkpoint is missing encoder keys: {missing_critical}')
    bad_unexpected = [
        key
        for key in unexpected
        if not key.startswith(_2D_BRANCH_PREFIXES) and '.sr2D.' not in key
    ]
    if bad_unexpected:
        raise RuntimeError(
            f'UniMiSS checkpoint has unexpected non-2D-branch keys: {bad_unexpected}'
        )
