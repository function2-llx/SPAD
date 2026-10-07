"""3D U-Net baselines: Models Genesis and SuPreM (frozen-probe and finetune).

Contract: models_genesis/suprem(*, dims, device, weights, img_size, trainable) -> nn.Module with
forward(x)->(global,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

Both releases (MrGiovanni's group) ship the same 3D U-Net, so one encoder serves both and the two
arms differ only in pretraining: Models Genesis learns by image restoration on LUNA16 chest CT,
SuPreM by supervised organ segmentation on AbdomenAtlas. Their checkpoints differ only in layout,
which `_Release` records.

Readout follows the official Models Genesis classification recipe (TargetNet in
MrGiovanni/ModelsGenesis): global average pooling over the encoder bottleneck `out512`, the stage-4
output at stride 8 with 512 channels. The network is fully convolutional, so img_size only sets the
bottleneck grid; 96 puts it at 12^3, the grid the ViT arms hold at /16.

Architecture from the official unet3d.py, encoder only: four stages of two 3^3 conv+BN+ReLU
(64/128/256/512 channels), 2x max-pool after the first three. The reference `ContBatchNorm3d`
normalizes with batch statistics even at inference; standard BatchNorm3d in eval mode uses the
checkpoint's running statistics instead, so frozen features are per-sample deterministic.

Input is single-channel [0,1]: both pipelines window CT intensities into that range (Genesis
[-1000, 1000] HU, SuPreM [-175, 250] HU); MedMNIST uint8 maps by /255.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

BOTTLENECK_STRIDE = 8
BOTTLENECK_CHANNELS = 512
_ENCODER_PREFIXES = ('down_tr64.', 'down_tr128.', 'down_tr256.', 'down_tr512.')


@dataclass(frozen=True)
class _Release:
    """Where one released checkpoint keeps the encoder, and what else it is allowed to contain.

    `tolerated` lists the non-encoder key prefixes as they read *after* `prefix` is stripped, so an
    unrecognized key raises instead of being dropped on the floor.
    """
    container: str
    prefix: str
    tolerated: tuple[str, ...]


GENESIS = _Release(
    container='state_dict',
    prefix='module.',
    tolerated=('up_tr256.', 'up_tr128.', 'up_tr64.', 'out_tr.'),
)
SUPREM = _Release(
    container='net',
    prefix='module.backbone.',
    # decoder, plus the CLIP-driven segmentation head that replaces Genesis's out_tr
    tolerated=('up_tr256.', 'up_tr128.', 'up_tr64.',
               'module.organ_embedding', 'module.precls_conv.', 'module.GAP.',
               'module.controller.', 'module.text_to_vision.'),
)


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} in [0,1] at native res (CPU). Shape-blind."""
    if not is_3d:
        raise ValueError('the 3D U-Net baselines cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0
    return {'x': x.unsqueeze(1)}


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    # child names match the official LUConv so the checkpoint keys load unchanged
    return nn.Sequential(OrderedDict(
        conv1=nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        bn1=nn.BatchNorm3d(out_ch),
        activation=nn.ReLU(),
    ))


class _DownTransition(nn.Module):
    """Official DownTransition: two conv blocks, then 2x max-pool except at the bottleneck."""

    def __init__(self, in_ch: int, depth: int, *, pool: bool):
        super().__init__()
        mid = 32 * 2 ** depth
        self.ops = nn.Sequential(_conv_block(in_ch, mid), _conv_block(mid, 2 * mid))
        self.maxpool = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool(self.ops(x))


class UNetEncoder(nn.Module):
    """The official UNet3D encoder path; forward returns the bottleneck (B,512,D/8,H/8,W/8)."""

    def __init__(self):
        super().__init__()
        self.down_tr64 = _DownTransition(1, 0, pool=True)
        self.down_tr128 = _DownTransition(64, 1, pool=True)
        self.down_tr256 = _DownTransition(128, 2, pool=True)
        self.down_tr512 = _DownTransition(256, 3, pool=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_tr512(self.down_tr256(self.down_tr128(self.down_tr64(x))))


def load_encoder_state_dict(path: str, release: _Release) -> dict[str, Tensor]:
    """Encoder subtree of a released checkpoint; any key the release does not declare raises."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    sd = {k.removeprefix(release.prefix): v for k, v in ck[release.container].items()}
    leftovers = [k for k in sd if not k.startswith(_ENCODER_PREFIXES + release.tolerated)]
    if leftovers:
        raise RuntimeError(f'unexpected keys in {path}: {leftovers[:5]}')
    encoder = {k: v for k, v in sd.items() if k.startswith(_ENCODER_PREFIXES)}
    if not encoder:
        raise RuntimeError(f'no encoder keys found in {path}')
    return encoder


class UNet3DEncoder(nn.Module):
    """A released 3D U-Net encoder as a MedMNIST3D baseline; 3D-only.

    forward(x) with x: (B,1,D,H,W) in [0,1] at native res -> resized to img_size^3 -> bottleneck
    (B,512,g,g,g), g = img_size/8 -> (global via GAP, patch_tokens flattened).
    """

    def __init__(self, weights: str, img_size: int, release: _Release):
        super().__init__()
        if img_size % BOTTLENECK_STRIDE:
            raise ValueError(
                f'img_size {img_size} not divisible by {BOTTLENECK_STRIDE} (the bottleneck stride)'
            )
        self.img_size = img_size
        self.unet = UNetEncoder()
        # strict: every encoder parameter and BN statistic must come from the checkpoint
        self.unet.load_state_dict(load_encoder_state_dict(weights, release), strict=True)
        self.embed_dim = BOTTLENECK_CHANNELS

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return the four convolutional stages from shallow to deep."""
        return tuple(
            tuple(stage.parameters())
            for stage in (
                self.unet.down_tr64, self.unet.down_tr128,
                self.unet.down_tr256, self.unet.down_tr512,
            )
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[-3:] != (self.img_size,) * 3:
            x = F.interpolate(x, size=(self.img_size,) * 3, mode='trilinear', align_corners=False)
        bottleneck = self.unet(x)
        global_features = bottleneck.mean(dim=(2, 3, 4))
        patch_tokens = bottleneck.flatten(2).transpose(1, 2)   # (B, g^3, 512)
        return global_features, patch_tokens


def _build(release: _Release, *, dims: int, device: str, weights: str, img_size: int,
           trainable: bool) -> UNet3DEncoder:
    if dims != 3:
        raise ValueError('the 3D U-Net baselines are 3D-only; pass dims=3')
    enc = UNet3DEncoder(weights=weights, img_size=img_size, release=release).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc


def models_genesis(
    *, dims: int, device: str, weights: str, img_size: int,
    trainable: bool = False, drop_path_rate: float = 0.0,
) -> UNet3DEncoder:
    """Models Genesis (restoration SSL on LUNA16 chest CT) from its released checkpoint.

    `weights` and `img_size` are required: the checkpoint is a local file, and img_size sets the
    bottleneck grid (96 -> 12^3). 3D-only: dims must be 3.
    """
    if drop_path_rate != 0:
        raise ValueError('Models Genesis has no residual branches; drop_path_rate must be zero')
    return _build(
        GENESIS, dims=dims, device=device, weights=weights, img_size=img_size,
        trainable=trainable,
    )


def suprem(*, dims: int, device: str, weights: str, img_size: int,
           trainable: bool = False) -> UNet3DEncoder:
    """SuPreM (supervised organ segmentation on AbdomenAtlas) from its released U-Net checkpoint.

    Same encoder and arguments as `models_genesis`; only the pretraining differs.
    """
    return _build(SUPREM, dims=dims, device=device, weights=weights, img_size=img_size,
                  trainable=trainable)
