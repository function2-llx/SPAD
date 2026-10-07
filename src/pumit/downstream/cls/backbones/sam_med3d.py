"""SAM-Med3D baseline as a backbone module (3D-only, segmentation-pretrained ViT).

Contract: sam_med3d(*, dims=3, device, img_size, trainable) -> nn.Module with
forward(x)->(global,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

SAM-Med3D (uni-medical/SAM-Med3D, ECCV BIC 2024) is a 3D adaptation of SAM for medical
segmentation. The image encoder is a true 3D ViT (patch16, 768d/12L, 12 heads) with windowed +
global attention; the `turbo` checkpoint is the `vit_b_ori` config trained at 128^3
(pos_embed 8^3, window_size=14, global_attn_indexes=[2,5,8,11]). This is the only public
3D medical ViT besides M3D-CLIP, and crucially a SEGMENTATION-pretrained one (vs M3D's
contrastive) -- a distinct pretraining family for the baseline roster.

The wrapper keeps the complete vendored ImageEncoderViT3D in self.encoder (the official forward
is used as-is) and applies only: resize 64^3 -> img_size^3 then per-sample full-volume
Z-normalization in forward (matching the official order: CropOrPad then TorchIO ZNormalization
before image_encoder; the foreground rule degenerates on binary-mask MedMNIST sets, so
full-volume is the documented adaptation), and readout = GAP over the post-neck image embedding
(SAM-Med3D has no cls token; its `neck` Conv3d 768->384 produces the SAM image embedding, the
native global representation).

`img_size` selects the token grid rather than inheriting the pretraining size; the checkpoint's
pos_embed and rel_pos tables are resampled onto it at load (see `adapt_state_dict`), and the
window span is clamped to the grid so window attention pads nothing (see `build_encoder`).

3D-only; single-channel CT, full-volume Z-normalization (NOT the generic `123.675/58.395`
ImageNet constants, which live in `Sam3D.preprocess` and are bypassed by the training pipeline).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from ._sam_med3d_encoder import ImageEncoderViT3D
from ..optim import make_parameter_layers

SAM_MED3D_REPO = 'blueyo0/SAM-Med3D'
SAM_MED3D_WEIGHTS = 'sam_med3d_turbo.pth'
SAM_MED3D_REVISION = 'fc482a040ea69cdc9ae576a1d6bd9db02ab1994c'   # verified HF HEAD

NATIVE_IMG_SIZE = 128   # pretraining size; the checkpoint pos_embed is the resulting 8^3 grid
PATCH = 16
OUT_CHANS = 384         # SAM image-embedding dim (neck output)
NATIVE_WINDOW = 14      # pretrained window span (windowed blocks carry 2*14-1 = 27 rel_pos entries)
GLOBAL_ATTN_BLOCKS = (2, 5, 8, 11)   # blocks that attend over the whole grid instead of a window


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} as float, *not* normalized (CPU, native res).

    Normalization is deferred to `forward` so it runs AFTER the resize -- matching the official
    order (train.py: CropOrPad, then TorchIO ZNormalization immediately before image_encoder).
    Computing Z-norm stats at 64³ and then resizing would smooth the volume and lower its std
    (trilinear is a low-pass op), so the encoder would not receive unit-std input.
    3D-only; raises on 2D.
    """
    if not is_3d:
        raise ValueError('SAM-Med3D is a 3D-only baseline; cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float()   # (N,D,H,W)
    return {'x': x.unsqueeze(1)}                                 # (N,1,D,H,W)


def _znormalize(x: Tensor) -> Tensor:
    """Per-sample full-volume Z-normalization (zero mean, unit std) on (B,1,D,H,W).

    The official rule is foreground Z-norm (`masking_method=lambda x: x > 0`), which degenerates
    (std=0) on the binary-mask sets AdrenalMNIST3D/VesselMNIST3D where all positive voxels share
    one value; full-volume is the documented MedMNIST adaptation that stays well-defined across all
    six tasks. A constant volume (zero std) raises -- TorchIO v0.19.6 raises on the same; no
    reviewed MedMNIST3D volume is constant.
    """
    stats = x.flatten(1).std(dim=1, unbiased=False)              # (B,)
    if not torch.all(stats > 0):
        raise ValueError('SAM-Med3D Z-normalization encountered a constant (zero-std) volume')
    mean = x.flatten(1).mean(dim=1).view(-1, 1, 1, 1, 1)
    std = stats.view(-1, 1, 1, 1, 1)
    return (x - mean) / std


def build_encoder(input_channels: int = 1, img_size: int = NATIVE_IMG_SIZE,
                  window_size: int | None = None) -> ImageEncoderViT3D:
    """Construct the SAM-Med3D `vit_b_ori` image encoder (random init; weights loaded separately).

    `window_size` defaults to min(NATIVE_WINDOW, grid): window attention pads the grid up to a
    multiple of the window, so a 12^3 grid under the pretrained span of 14 would run on 14^3 with
    37% of the positions padding. Clamping the span to the grid removes the padding, and costs no
    pretrained weight -- rel_pos is indexed by relative distance and is resampled to the span in
    `adapt_state_dict`, the same interpolation `get_rel_pos` performs per call.
    """
    grid = img_size // PATCH
    return ImageEncoderViT3D(
        depth=12,
        embed_dim=768,
        img_size=img_size,
        in_chans=input_channels,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=PATCH,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=list(GLOBAL_ATTN_BLOCKS),
        window_size=window_size if window_size is not None else min(NATIVE_WINDOW, grid),
        out_chans=OUT_CHANS,
    )


def adapt_state_dict(state_dict: dict[str, Tensor], model: ImageEncoderViT3D) -> dict[str, Tensor]:
    """Resample the checkpoint's resolution-dependent tables onto `model`'s declared shapes.

    Two kinds move: the absolute `pos_embed`, trilinearly over its 3D grid, and every `rel_pos`,
    linearly over signed relative distance (entry i means offset i-(span-1), so a span-s axis needs
    2s-1 entries). `get_rel_pos` already interpolates on the fly, but `nn.Parameter` shapes are
    fixed at construction, so `load_state_dict` would reject the checkpoint before that runs.
    Everything else is copied unchanged.
    """
    adapted = dict(state_dict)
    target = model.state_dict()
    grid = model.pos_embed.shape[1]
    pos = einops.rearrange(state_dict['pos_embed'], '1 d h w c -> 1 c d h w')
    pos = F.interpolate(pos.float(), size=(grid,) * 3, mode='trilinear', align_corners=False)
    adapted['pos_embed'] = einops.rearrange(pos, '1 c d h w -> 1 d h w c').to(
        state_dict['pos_embed'].dtype)
    for key, value in state_dict.items():
        if 'rel_pos' not in key:
            continue
        length = target[key].shape[0]
        if value.shape[0] == length:
            continue
        resized = F.interpolate(value.t().unsqueeze(0).float(), size=length, mode='linear',
                                align_corners=False)
        adapted[key] = resized.squeeze(0).t().to(value.dtype)
    return adapted


def _load_state_dict(weights: str | Path | None = None) -> dict[str, Tensor]:
    """Load the local or pinned SAM-Med3D checkpoint and return the image-encoder state dict.

    The checkpoint is a full Sam3D dump under `model_state_dict`; only the `image_encoder.*`
    keys are retained (the prompt/mask decoders are dropped).
    """
    path = weights
    if path is None:
        path = hf_hub_download(SAM_MED3D_REPO, SAM_MED3D_WEIGHTS, revision=SAM_MED3D_REVISION)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    full = sd['model_state_dict']
    return {
        k.removeprefix('image_encoder.'): v
        for k, v in full.items()
        if k.startswith('image_encoder.')
    }


class SAMMed3DEncoder(nn.Module):
    """SAM-Med3D image encoder as a MedMNIST3D baseline; 3D-only.

    forward(x) with x: (B,1,D,H,W) at native res -> resized to img_size^3 -> the SAM image
    embedding (B,384,g,g,g) -> (global[B,384], patch_tokens[B,g^3,384]) via GAP + flatten.

    `img_size` sets the token grid (img_size/16 per axis); the checkpoint's pos_embed and rel_pos
    tables are resampled onto it at load.
    """

    def __init__(self, img_size: int = NATIVE_IMG_SIZE):
        super().__init__()
        if img_size % PATCH:
            raise ValueError(f'img_size {img_size} is not divisible by patch {PATCH}')
        self.img_size = img_size
        self.encoder = build_encoder(img_size=img_size)
        self.encoder.load_state_dict(adapt_state_dict(_load_state_dict(), self.encoder),
                                     strict=True)
        self.embed_dim = OUT_CHANS

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return make_parameter_layers(
            (*self.encoder.patch_embed.parameters(), self.encoder.pos_embed),
            self.encoder.blocks,
            self.encoder.neck.parameters(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Official order: resize 64³ -> img_size³ FIRST, then full-volume Z-normalization (so the
        # encoder receives unit-std input; normalizing before resize would be undone by interpolation).
        x = F.interpolate(x, size=(self.img_size,) * 3, mode='trilinear', align_corners=False)
        x = _znormalize(x)
        embedding = self.encoder(x)                            # (B,384,g,g,g), g = img_size/16
        global_features = embedding.mean(dim=(2, 3, 4))        # GAP over the spatial grid
        patch_tokens = embedding.flatten(2).transpose(1, 2)   # (B,g^3,384)
        return global_features, patch_tokens


def sam_med3d(
    *, dims: int, device: str, img_size: int,
    trainable: bool = False, drop_path_rate: float = 0.0,
):
    """Construct the SAM-Med3D 3D encoder on `device` (image-encoder weights from HF hub, atomic).

    `img_size` is required: at patch 16 it sets the token grid (192 -> 12^3 = 1728), so the caller
    states the sequence length rather than inheriting the 128^3 pretraining size. `weights` is not
    declared -- the pinned HF checkpoint is loaded inside the constructor. 3D-only: dims must be 3.
    """
    if dims != 3:
        raise ValueError('SAM-Med3D is a 3D-only baseline; pass dims=3')
    if drop_path_rate != 0:
        raise ValueError('SAM-Med3D has no native drop path; drop_path_rate must be 0')
    enc = SAMMed3DEncoder(img_size=img_size).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc
