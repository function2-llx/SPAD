"""3DINO baseline as a backbone module (3D-only, DINOv2-style self-supervised ViT-L).

Contract: three_dino(*, dims=3, device, weights, trainable) -> nn.Module with forward(x)->(global,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

3DINO (AICONSlab/3DINO, npj Digital Medicine 2025) adapts DINOv2 self-distillation to 3D medical volumes natively (no 2D inflation; retrained from scratch in 3D).
ViT-L (1024d/24L, patch16, single-channel), pretrained at 96/112^3 (highres teacher at 112^3 = 7^3 grid).
The only public 3D medical SSL ViT besides M3D-CLIP -- a self-supervised peer to M3D's contrastive and SAM-Med3D's segmentation pretraining, completing the pretraining-family coverage of the 3D baseline roster.

The vendored DinoVisionTransformer3d (under _3dino/) retains the released topology.
MemEffAttention.forward uses F.scaled_dot_product_attention instead of xformers (the 3DINO-pinned xformers 0.0.18 is incompatible with our torch; 0.0.35's flash kernels reject fp32).
The drop-path schedule preserves Python endpoint precision to honor the native block's branching threshold.
The encoder loads the gated HF teacher checkpoint ({'teacher': {'backbone.<k>': v}}), stripping the teacher/module/backbone prefixes.

3D-only; single-channel.
3DINO pretraining uses MONAI percentile scaling to [-1,1] on raw CT; MedMNIST3D is already scaled to uint8, so min-max [-1,1] (the same target range) is the native approximation.
Readout = the normed CLS token (1024-dim, the native forward_head pre-logits; 3DINO's linear probe optionally conccats averaged patch tokens, but the cls readout uses CLS).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from ._3dino.models.vision_transformer import vit_large_3d
from ..optim import make_parameter_layers

THREEDINO_REPO = 'AICONSlab/3DINO-ViT'
THREEDINO_WEIGHTS = '3dino_vit_weights.pth'
THREEDINO_REVISION = '8a00a2bb14becb3fbe955064837322a3217f7ac0'   # verified HF HEAD (gated)

NATIVE_IMG_SIZE = 112   # 3DINO highres-stage pretraining size (checkpoint pos_embed is 7^3)
PATCH = 16
BLOCK_CHUNKS = 4         # the checkpoint's FSDP block chunking (4 chunks of 6 blocks each)
INIT_VALUES = 1e-5       # official layerscale init (ssl3d_default_config.yaml); the trained ls*.gamma load with this


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} with 3DINO's official per-volume percentile scaling.

    Matches the official classification transform: ScaleIntensityPercentiled(lower=0.05,
    upper=99.95, b_min=-1, b_max=1, clip=True) -- clip to the per-volume 0.05th/99.95th percentile
    range, then map that range to [-1,1]. Shape-blind (per-volume stats; resize/RoPE in forward).
    3D-only; raises on 2D.
    """
    if not is_3d:
        raise ValueError('3DINO is a 3D-only baseline; cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float()          # (N,D,H,W)
    flat = x.flatten(1)                                                 # (N, D*H*W)
    lo = torch.quantile(flat, 0.0005, dim=1).view(-1, 1, 1, 1)         # 0.05th percentile
    hi = torch.quantile(flat, 0.9995, dim=1).view(-1, 1, 1, 1)         # 99.95th percentile
    # A volume sparse enough that both percentiles land on the same value (vesselmnist3d has one
    # with 76/262144 nonzero voxels) collapses the range. MONAI's ScaleIntensityRange, which the
    # official transform wraps, warns and returns `img - a_min + b_min` there; matching that keeps
    # the degenerate case on the official path instead of failing the run.
    degenerate = (hi - lo) <= 0
    scale = torch.where(degenerate, torch.ones_like(hi), hi - lo)
    x = torch.where(degenerate, x - lo - 1.0, ((x - lo) / scale) * 2 - 1)
    return {'x': x.clamp(-1, 1).unsqueeze(1)}                         # (N,1,D,H,W)


def build_encoder(input_channels: int = 1, *, drop_path_rate: float = 0.0) -> nn.Module:
    """Construct the 3DINO ViT-L 3D encoder at the 112^3 highres stage (random init; weights separate).

    Always built at NATIVE_IMG_SIZE so the pos_embed parameter keeps the checkpoint's 7^3 shape and loads strictly; running at another input size resamples that table per forward instead.
    `init_values=1e-5` enables LayerScale, matching the official SSL config; the trained `blocks.*.ls{1,2}.gamma` (48 params, mean abs ~0.13) load only with this set.
    Drop path uses the native depth-linear schedule.
    """
    return vit_large_3d(
        patch_size=PATCH, in_chans=input_channels, img_size=NATIVE_IMG_SIZE,
        block_chunks=BLOCK_CHUNKS, init_values=INIT_VALUES,
        drop_path_rate=drop_path_rate,
    )


def _load_state_dict(weights: str | Path | None = None) -> dict[str, Tensor]:
    """Load the local or gated 3DINO teacher checkpoint and return the bare encoder state dict.

    The checkpoint is `{'teacher': {'backbone.<k>': v, 'dino_head.*': ..., 'ibot_head.*': ...},
    'epoch': ...}`; the `teacher`/`module.`/`backbone.` wrappers are stripped (exact prefix
    removal, not unguarded substring replace) so the keys match the encoder. When `weights`
    is omitted, the gated repo requires an HF token.
    """
    path = weights
    if path is None:
        path = hf_hub_download(THREEDINO_REPO, THREEDINO_WEIGHTS, revision=THREEDINO_REVISION,
                               token=os.environ.get('HF_TOKEN'))
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if 'teacher' not in sd:
        raise RuntimeError(
            f'{THREEDINO_REPO}: expected a DINO dump with a `teacher` branch, got keys {list(sd)[:4]}'
        )
    sd = sd['teacher']
    return {k[len('backbone.'):] if k.startswith('backbone.') else
            k[len('module.backbone.'):] if k.startswith('module.backbone.') else k: v
            for k, v in sd.items()}


# DINO/iBOT projection heads present in the checkpoint but not in the encoder; allowlisted as
# the only acceptable unexpected keys after a strict-ish load.
_HEAD_PREFIXES = ('dino_head.', 'ibot_head.')


class ThreeDinoEncoder(nn.Module):
    """3DINO ViT-L as a MedMNIST3D baseline; 3D-only.

    forward(x) with x: (B,1,D,H,W) at native res -> resized to img_size^3 -> forward_features ->
    (global[B,1024], patch_tokens[B,N,1024]); global = the normed CLS token (native readout).

    `img_size` sets the token grid (img_size/16 per axis). The checkpoint's pos_embed describes
    the 7^3 pretraining grid; the vendored forward_features resamples it to whatever grid the
    input produces (interpolate_pos_encoding3D, trilinear), the same mechanism DINOv2/v3 use to
    change resolution.
    """

    def __init__(self, img_size: int = NATIVE_IMG_SIZE, drop_path_rate: float = 0.0):
        super().__init__()
        if img_size % PATCH:
            raise ValueError(f'img_size {img_size} is not divisible by patch {PATCH}')
        self.img_size = img_size
        self.model = build_encoder(drop_path_rate=drop_path_rate)
        sd = _load_state_dict()
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f'3DINO checkpoint is missing encoder keys: {missing}')
        bad_unexpected = [k for k in unexpected if not k.startswith(_HEAD_PREFIXES)]
        if bad_unexpected:
            raise RuntimeError(
                f'3DINO checkpoint has unexpected non-head keys (topology drift): {bad_unexpected}'
            )
        self.embed_dim = self.model.embed_dim   # 1024

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        blocks = (
            [block for chunk in self.model.blocks for block in chunk if not isinstance(block, nn.Identity)]
            if self.model.chunked_blocks else self.model.blocks
        )
        return make_parameter_layers(
            (
                *self.model.patch_embed.parameters(), self.model.cls_token,
                self.model.pos_embed, self.model.mask_token,
            ),
            blocks,
            self.model.norm.parameters(),
        )

    def _resize(self, x: Tensor) -> Tensor:
        return F.interpolate(x, size=(self.img_size,) * 3, mode='trilinear', align_corners=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._resize(x)
        feat = self.model.forward_features(x)
        return feat['x_norm_clstoken'], feat['x_norm_patchtokens']


def three_dino(
    *, dims: int, device: str, img_size: int,
    trainable: bool = False, drop_path_rate: float = 0.0,
):
    """Construct the 3DINO 3D encoder on `device` (gated HF weights, atomic).

    `img_size` is required: at patch 16 it sets the token grid (192 -> 12^3 = 1728), so the
    caller states the sequence length rather than inheriting the 112^3 pretraining size.
    `weights` is not declared -- 3DINO loads the pinned gated HF checkpoint inside the
    constructor (set HF_TOKEN after accepting the gate). 3D-only: dims must be 3.
    """
    if dims != 3:
        raise ValueError('3DINO is a 3D-only baseline; pass dims=3')
    enc = ThreeDinoEncoder(img_size=img_size, drop_path_rate=drop_path_rate).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc
