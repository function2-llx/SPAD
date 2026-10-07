"""SPAD-ViT baseline/probe as a backbone module (DINOv3 init or UCPT teacher).

Contract: dinov3/ucpt(*, dims, device, vit_config, weights, trainable) -> nn.Module with
forward(x)->(cls,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

forward owns the GPU resize to the fixed cube and `da` (=max_adapt for a 2D single-slice
so the native-3D patch conv collapses depth; 0 for 3D), hidden from the framework.
transform_batch is shape-blind; see eva02.py for the worker/forward shape-rule.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import Tensor, nn

from .encoder import _finalize, load_dinov3_encoder, load_encoder
from ..data import normalize_to_encoder, resize_volume, to_3ch_volume
from ..optim import make_parameter_layers
from pumit.model.vit import ViT, ViTConfig

IMG_SIZE = 256  # Default spatial edge for the encoder wrapper.

# DINOv3 uses ImageNet stats; transform_batch retains legacy PUMIT [-1,1].
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 batch -> {'x': (B,3,D,H,W)} in [-1,1] at native res (CPU). Shape-blind."""
    return {'x': normalize_to_encoder(images, is_3d=is_3d)}


def transform_batch_imagenet(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 batch -> {'x': (B,3,D,H,W)} with ImageNet normalization at native resolution (CPU)."""
    x = to_3ch_volume(images, is_3d=is_3d)
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(1, 3, 1, 1, 1)
    return {'x': ((x - mean) / std).contiguous()}


class SpadEncoder(nn.Module):
    """Wrap a SPAD ViT so forward owns the resize + `da`, exposing (cls, patch).

    forward(x) with x: (B,3,D,H,W) at native res. 2D sets arrive as D==1 and use
    da=max_adapt (single-slice); 3D sets use da=0 (isotropic).
    """

    def __init__(self, vit: ViT, dims: int, img_size: int = IMG_SIZE):
        super().__init__()
        if dims not in (2, 3):
            raise ValueError(f'dims must be 2 or 3, got {dims}')
        self.vit = vit
        self.dims = dims
        self.img_size = img_size
        self.embed_dim = vit.config.hidden_size
        self.da = vit.embeddings.patch_embeddings.max_adapt if dims == 2 else 0

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = resize_volume(x, self.img_size, is_3d=(self.dims == 3))
        return self.vit.encode_image(x, da=self.da)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        """Return ViT parameters shallow-to-deep for LLRD (final norm folded into the top block)."""
        return make_parameter_layers(
            self.vit.embeddings.parameters(), self.vit.layer, self.vit.norm.parameters()
        )


def _build(*, is_ckpt: bool, dims: int, device: str, vit_config: ViTConfig, img_size: int,
           weights: str | None = None, trainable: bool = False, **_) -> SpadEncoder:
    """Construct a SPAD-ViT encoder on `device` (arch + weights, atomic).

    `img_size` is required: it sets the token grid (192 -> 12^3 = 1728 at patch 16), so the
    caller must state the sequence length rather than inherit a module default. `weights` is the
    path to either a UCPT checkpoint (is_ckpt=True, slices teacher_vit.*) or official DINOv3
    safetensors (is_ckpt=False, 2D->3D inflated on load); omitting it gives random init (the
    "does pretraining help" ablation control). RoPE positional aug is disabled
    (pos_embed_rescale=None) for reproducible eval. `**_` swallows kwargs meant for other
    backbones (e.g. size).
    """
    if img_size % vit_config.patch_size:
        raise ValueError(
            f'img_size {img_size} is not divisible by patch {vit_config.patch_size}; the token '
            'grid would be silently floor-divided'
        )
    cfg = dataclasses.replace(vit_config, pos_embed_rescale=None)
    if weights is None:
        vit = _finalize(ViT(cfg), device, trainable)
    elif is_ckpt:
        vit = load_encoder(weights, cfg, device, trainable=trainable)
    else:
        vit = load_dinov3_encoder(weights, cfg, device, trainable=trainable)
    # vit is already on `device` via _finalize; SpadEncoder holds no other params.
    return SpadEncoder(vit, dims=dims, img_size=img_size)


def dinov3(*, dims: int, device: str, vit_config: ViTConfig, img_size: int,
           weights: str | None = None, trainable: bool = False, **_) -> SpadEncoder:
    """SPAD-ViT from official DINOv3 safetensors (2D->3D inflated on load)."""
    return _build(is_ckpt=False, dims=dims, device=device, vit_config=vit_config,
                  img_size=img_size, weights=weights, trainable=trainable, **_)


def ucpt(*, dims: int, device: str, vit_config: ViTConfig, img_size: int,
         weights: str | None = None, trainable: bool = False, **_) -> SpadEncoder:
    """SPAD-ViT from a UCPT checkpoint (slices teacher_vit.* EMA teacher)."""
    return _build(is_ckpt=True, dims=dims, device=device, vit_config=vit_config,
                  img_size=img_size, weights=weights, trainable=trainable, **_)
