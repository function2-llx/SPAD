"""STU-Net-L encoder adapter for dense segmentation.

Vendors the encoder half of STU-Net (uni-medical/STU-Net, Apache-2.0; the standalone class in
nnUNet-2.2/.../STUNetTrainer.py): six stages of two residual blocks each at channels
64/128/256/512/1024/1024, all 3x3x3 kernels, InstanceNorm + LeakyReLU. The released
``large_ep4k.model`` was trained supervised on TotalSegmentator v1 (104 classes, independent Basel
hospital data). The network is consumed at its native depth through a six-stage plan; downsampling
is the strided first conv (plus 1x1x1 shortcut) of each stage, so re-striding it to the plan's
strides nnU-Net style reinterprets where the pretrained kernels are applied without touching any
weight. InstanceNorm carries no batch state, so the frozen trunk is checkpoint-faithful per sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from pumit.downstream.seg.adapters.checkpoint import repeat_single_channel_weight
from pumit.downstream.seg.adapters.native_pyramid import build_native_pyramid_encoder, native_encoder
from pumit.downstream.seg.adapters.pyramid import adopt_plan_contract
from pumit.downstream.seg.plan import EncoderPlan, compute_stage_shapes

_NATIVE_CHANNELS = (64, 128, 256, 512, 1024, 1024)
_BLOCKS_PER_STAGE = 2
_ENCODER_PREFIX = 'conv_blocks_context.'
_INPUT_WEIGHT_KEYS = (
    'conv_blocks_context.0.0.conv1.weight',
    'conv_blocks_context.0.0.conv3.weight',
)
# Non-encoder scopes inside the released checkpoint: the decoder half and its heads.
_ALLOWED_UNUSED_PREFIXES = ('upsample_layers.', 'conv_blocks_localization.', 'seg_outputs.')
CONFIG_KEYS = frozenset()
OPTIONAL_CONFIG_KEYS = frozenset({'with_high_resolution_stem', 'pyramid_branch'})


class BasicResBlock(nn.Module):
    """STU-Net residual block: conv3-IN-LReLU, conv3-IN, shortcut add, LReLU (vendored semantics)."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        use_1x1conv: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(input_channels, output_channels, 1, stride=stride) if use_1x1conv else None

    def forward(self, x: Tensor) -> Tensor:
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.act2(y + x)


class STUNetPyramidBackbone(nn.Module):
    """Expose the six pretrained STU-Net-L encoder stages at the plan's strides.

    The module tree mirrors the release's ``conv_blocks_context`` naming so the checkpoint loads
    without key remapping.
    """

    def __init__(self, input_channels: int, stage_strides: tuple[tuple[int, int, int], ...]):
        super().__init__()
        if not len(_NATIVE_CHANNELS) - 2 <= len(stage_strides) <= len(_NATIVE_CHANNELS) - 1:
            raise ValueError(
                f'expected {len(_NATIVE_CHANNELS) - 2} or {len(_NATIVE_CHANNELS) - 1} '
                f'inter-stage strides, got {len(stage_strides)}'
            )
        stages = [
            nn.Sequential(
                BasicResBlock(input_channels, _NATIVE_CHANNELS[0], use_1x1conv=True),
                *[
                    BasicResBlock(_NATIVE_CHANNELS[0], _NATIVE_CHANNELS[0])
                    for _ in range(_BLOCKS_PER_STAGE - 1)
                ],
            )
        ]
        for depth, stride in enumerate(stage_strides, start=1):
            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        _NATIVE_CHANNELS[depth - 1],
                        _NATIVE_CHANNELS[depth],
                        stride=stride,
                        use_1x1conv=True,
                    ),
                    *[
                        BasicResBlock(_NATIVE_CHANNELS[depth], _NATIVE_CHANNELS[depth])
                        for _ in range(_BLOCKS_PER_STAGE - 1)
                    ],
                )
            )
        self.conv_blocks_context = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        features = []
        for stage in self.conv_blocks_context:
            x = stage(x)
            features.append(x)
        return tuple(features)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the six convolutional stages from shallow to deep."""
        return tuple(tuple(stage.parameters()) for stage in self.conv_blocks_context)


class STUNetEncoder3D(nn.Module):
    """Serve the pretrained six-stage STU-Net-L hierarchy as a native-depth plan encoder."""

    def __init__(self, plan: EncoderPlan, *, input_channels: int):
        super().__init__()
        n_stages = len(plan.output_channels)
        if not len(_NATIVE_CHANNELS) - 1 <= n_stages <= len(_NATIVE_CHANNELS):
            raise ValueError(f'STU-Net-L requires a five- or six-stage plan, got {n_stages} stages')
        if tuple(plan.output_channels) != _NATIVE_CHANNELS[:n_stages]:
            raise ValueError(
                f'STU-Net-L requires plan channels {_NATIVE_CHANNELS[:n_stages]}, '
                f'got {tuple(plan.output_channels)}'
            )
        strides = tuple(tuple(int(v) for v in s) for s in plan.strides)
        if len(strides) != n_stages or strides[0] != (1, 1, 1):
            raise ValueError(
                f'STU-Net-L requires a plan starting at stride (1, 1, 1), got {strides}'
            )
        # A five-stage plan builds and consumes the shallowest five pretrained stages.
        self.backbone = STUNetPyramidBackbone(input_channels, strides[1:])
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
    """Validate options for the pinned STU-Net-L encoder."""
    if weights is None:
        raise ValueError('STU-Net-L requires the released large_ep4k.model checkpoint')
    if checkpoint_format is not None:
        raise ValueError('STU-Net-L does not accept checkpoint_format')
    if gradient_checkpointing:
        raise ValueError('STU-Net-L does not currently expose gradient checkpointing')
    return {}


def build_encoder(
    plan: EncoderPlan,
    input_channels: int,
    config: Mapping[str, object],
) -> nn.Module:
    """Construct the plan-aligned STU-Net-L encoder; without the stem its P2-P5 are projected onto the plan schedule."""
    return build_native_pyramid_encoder(
        lambda encoder_plan: STUNetEncoder3D(encoder_plan, input_channels=input_channels),
        plan,
        _NATIVE_CHANNELS,
        input_channels=input_channels,
        config=config,
    )


def load_pretrained(
    encoder: nn.Module,
    weights: Path | None,
) -> None:
    """Load only the ``conv_blocks_context`` stages; the decoder half and seg heads are dropped."""
    if weights is None:
        raise ValueError('STU-Net-L requires the released large_ep4k.model checkpoint')
    encoder = native_encoder(encoder)
    if not isinstance(encoder, STUNetEncoder3D):
        raise TypeError(f'expected STUNetEncoder3D, got {type(encoder).__name__}')

    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    encoder_state: dict[str, Tensor] = {}
    dropped: list[str] = []
    for key, value in state_dict.items():
        stripped = key.removeprefix('module.')
        if stripped.startswith(_ENCODER_PREFIX):
            encoder_state[stripped] = value
        else:
            dropped.append(stripped)
    bad_dropped = [key for key in dropped if not key.startswith(_ALLOWED_UNUSED_PREFIXES)]
    if bad_dropped:
        raise RuntimeError(f'STU-Net checkpoint has unexpected non-encoder keys: {bad_dropped}')

    input_channels = encoder.backbone.conv_blocks_context[0][0].conv1.in_channels
    if input_channels != 1:
        for key in _INPUT_WEIGHT_KEYS:
            encoder_state[key] = repeat_single_channel_weight(encoder_state[key], input_channels)
    # A five-stage build consumes the shallowest five pretrained stages; deeper release keys drop here.
    built_stages = len(encoder.backbone.conv_blocks_context)
    encoder_state = {
        key: value
        for key, value in encoder_state.items()
        if int(key.removeprefix(_ENCODER_PREFIX).split('.', 1)[0]) < built_stages
    }
    # The backbone owns the keys under its own name; load at the module that matches the release tree.
    missing, unexpected = encoder.backbone.load_state_dict(encoder_state, strict=True)
    assert not missing and not unexpected
