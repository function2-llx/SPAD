"""Full SuPreM U-Net transfer with synchronized BatchNorm and task-specific output heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from .backbones.suprem_unet import LUConv
from .plan import encoder_plan_from_architecture_kwargs
from .registry import build_encoder, load_pretrained


class UpTransition(nn.Module):
    """Released SuPreM upsampling and skip fusion, with plan-aligned spatial strides."""

    def __init__(self, in_channels: int, out_channels: int, stride: Sequence[int]):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=tuple(stride), stride=tuple(stride),
        )
        self.ops = nn.SyncBatchNorm.convert_sync_batchnorm(
            nn.Sequential(
                LUConv(in_channels + out_channels, out_channels),
                LUConv(out_channels, out_channels),
            ),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.ops(torch.cat((self.up_conv(x), skip), dim=1))


class SupremDecoder(nn.Module):
    """Three pretrained decoder stages and scratch deep-supervision heads."""

    def __init__(self, strides: Sequence[Sequence[int]], num_classes: int, deep_supervision: bool):
        super().__init__()
        self.up_tr256 = UpTransition(512, 256, strides[3])
        self.up_tr128 = UpTransition(256, 128, strides[2])
        self.up_tr64 = UpTransition(128, 64, strides[1])
        self.seg_layers = nn.ModuleList(nn.Conv3d(channels, num_classes, 1) for channels in (256, 128, 64))
        self.seg_layers.apply(InitWeights_He(1e-2))
        self.deep_supervision = deep_supervision
        if not deep_supervision:
            for head in self.seg_layers[:-1]:
                head.requires_grad_(False)

    def forward(self, skips: Sequence[Tensor]) -> Tensor | list[Tensor]:
        x = skips[-1]
        predictions = []
        stages = (self.up_tr256, self.up_tr128, self.up_tr64)
        for index, (stage, skip, head) in enumerate(zip(stages, reversed(skips[:-1]), self.seg_layers)):
            x = stage(x, skip)
            if self.deep_supervision or index == len(stages) - 1:
                predictions.append(head(x))
        return list(reversed(predictions)) if self.deep_supervision else predictions[-1]


class SupremSegmentationNetwork(nn.Module):
    """Transfer SuPreM's encoder and decoder under the shared nnU-Net training protocol."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        backbone_name: str,
        backbone_config: Mapping[str, object],
        deep_supervision: bool,
        **architecture_kwargs,
    ):
        super().__init__()
        if backbone_name != 'suprem-unet':
            raise ValueError(f'full SuPreM transfer requires suprem-unet, got {backbone_name}')
        plan = encoder_plan_from_architecture_kwargs(**architecture_kwargs)
        if any(value not in (1, 2) for stride in plan.strides[1:] for value in stride):
            raise ValueError(f'SuPreM supports pool strides 1 or 2, got {plan.strides}')
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(
            build_encoder(backbone_name, plan, input_channels, backbone_config),
        )
        self.decoder = SupremDecoder(plan.strides, num_classes, deep_supervision)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        return self.decoder(self.encoder(x))

    def load_pretrained(self, weights: Path | None) -> None:
        """Initialize the full encoder and decoder, retaining fresh task output heads."""
        load_pretrained('suprem-unet', self.encoder, weights)
        self.load_pretrained_decoder(weights)

    def load_pretrained_decoder(self, weights: Path) -> None:
        """Load released decoder weights while leaving task output heads randomly initialized.

        A stride-one axis sums the two released transpose-kernel slices, preserving their total contribution when that upsampling axis is removed.
        """
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        decoder_state = {
            key.removeprefix('module.backbone.'): value
            for key, value in checkpoint['net'].items()
            if key.startswith('module.backbone.up_tr')
        }
        target_state = {key: value for key, value in self.decoder.state_dict().items() if key.startswith('up_tr')}
        if decoder_state.keys() != target_state.keys():
            missing = target_state.keys() - decoder_state.keys()
            unexpected = decoder_state.keys() - target_state.keys()
            raise RuntimeError(f'SuPreM decoder checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        for key, target in target_state.items():
            value = decoder_state[key]
            if key.endswith('up_conv.weight'):
                if value.shape[2:] != (2, 2, 2):
                    raise RuntimeError(f'expected released 2x2x2 transpose kernel for {key}, got {value.shape}')
                for axis, size in enumerate(target.shape[2:], start=2):
                    if size == 1:
                        value = value.sum(dim=axis, keepdim=True)
            if value.shape != target.shape:
                raise RuntimeError(f'SuPreM decoder shape mismatch for {key}: {value.shape} != {target.shape}')
            decoder_state[key] = value
        for name in ('up_tr256', 'up_tr128', 'up_tr64'):
            getattr(self.decoder, name).load_state_dict(
                {key.removeprefix(f'{name}.'): value for key, value in decoder_state.items() if key.startswith(f'{name}.')},
                strict=True,
            )
