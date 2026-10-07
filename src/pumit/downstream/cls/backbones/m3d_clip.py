"""M3D-CLIP 3D baseline module (medical 3D CT ViT, contrastive-pretrained on Radiopaedia CT-report pairs).

The only public 3D-native medical ViT -> the direct 3D peer for PUMIT on the 6 MedMNIST3D sets.
3D-only (a CT volumetric model; no 2D path). Loaded via transformers `AutoModel` (trust_remote_code)
from `GoodBaiBai88/M3D-CLIP` at the pinned M3D_REVISION.

MUST run in the `m3d` pixi env (py3.11 + monai==1.3 + transformers==4.44.2): M3D's remote ViT uses
the removed `pos_embed='perceptron'` MONAI API (gone in monai>=1.4, and monai<=1.3 needs py<=3.11),
and its modeling class is 4.x-era (transformers 5.x breaks it).

Contract: m3d(*, dims=3, device, weights, trainable) -> nn.Module with forward(x)->(cls,patch);
transform_batch(images, *, is_3d) -> forward kwargs (CPU, native res, shape-blind).

Input: 64^3 MedMNIST3D volumes resize to (64,256,256) -> patch (4,16,16) -> grid (16,16,16) = 4096
tokens. The token-aligned cls comparison runs every backbone at a 12^3 grid, which this adapter
does not expose -- its input shape is pinned, so M3D is not part of that table. M3D's
learned pos_embed is a fixed (8,16,16)=2048 table (its native 32x256x256 grid); we interpolate it
to (16,16,16) on load -- the standard learned-abs-pos_embed resolution adaptation (here 3D->3D).
monai 1.3's attention is O(N^2) einsum+softmax; all SABlocks are swapped for the SDPA duplicate
(monai_sablock_sdpa.py, identical math) so the unified bs=8 fits.

The wrapper owns ONLY the vision side (vision_encoder + vision_projection): the remote
encode_image() is exactly `vision_encoder -> mm_vision_proj -> L2 normalize`, so the text tower,
language projection and logit scale (110M dead parameters) are dropped at load instead of being
frozen -- registered dead state would still move across device/mode and enter snapshots.
"""
from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .monai_sablock_sdpa import SABlock as SDPASABlock


def swap_sdpa_attention(root: nn.Module) -> int:
    """Replace every monai SABlock under `root` with the SDPA duplicate (same weights).

    monai 1.3's einsum+softmax attention materializes the 4096^2 matrix (OOM at bs=8); the
    duplicate computes the identical function with a flash kernel. Each replacement preserves
    the source block's device, dtype, train/eval mode, per-parameter requires_grad, strict
    state-dict keys/values and save_attn flag. Returns the swap count.
    """
    from monai.networks.blocks.selfattention import SABlock as MonaiSABlock

    targets = [(mod, attr, child) for mod in root.modules() for attr, child in mod.named_children()
               if type(child) is MonaiSABlock]
    for mod, attr, old in targets:
        new = SDPASABlock(hidden_size=old.qkv.in_features, num_heads=old.num_heads,
                          dropout_rate=old.drop_weights.p, qkv_bias=old.qkv.bias is not None,
                          save_attn=old.save_attn)
        new.load_state_dict(old.state_dict())
        for old_p, new_p in zip(old.parameters(), new.parameters(), strict=True):
            new_p.requires_grad_(old_p.requires_grad)
        new = new.to(device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
        new.train(old.training)
        setattr(mod, attr, new)
    return len(targets)


M3D_NAME = 'GoodBaiBai88/M3D-CLIP'
M3D_REVISION = 'ae091d89a0ef38b533ecc4ed21426f7658853963'
INPUT_SHAPE = (64, 256, 256)   # resize target: patch (4,16,16) -> grid (16,16,16) = 4096 tokens
NATIVE_GRID = (8, 16, 16)      # M3D's pretrained pos_embed grid (native 32x256x256 / patch)
TARGET_GRID = (16, 16, 16)     # our grid at INPUT_SHAPE


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 3D batch -> {'x': (B,1,D,H,W)} in [0,1] at native res (CPU). Shape-blind.

    M3D is single-channel CT (in_channels=1) with min-max [0,1] normalization -- no 3-channel
    replicate, no mean/std. 3D-only; raises on 2D (M3D has no 2D path).
    """
    if not is_3d:
        raise ValueError('M3D is a 3D-only baseline; cannot transform 2D batches')
    x = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0  # [0,1] min-max
    x = x.unsqueeze(1)                                      # (N,D,H,W) -> (N,1,D,H,W)
    return {'x': x}


def _load_remote_vision() -> tuple[nn.Module, nn.Module]:
    """Load the revision-pinned remote model and detach the vision side.

    The full remote model (incl. the text tower) is still constructed transiently on CPU;
    only vision_encoder + mm_vision_proj are retained, the multimodal parent leaves scope.
    """
    from transformers import AutoModel
    remote = AutoModel.from_pretrained(M3D_NAME, trust_remote_code=True, revision=M3D_REVISION)
    return remote.vision_encoder, remote.mm_vision_proj


def resize_position_embeddings(vision_encoder: nn.Module) -> None:
    """Interpolate M3D's learned (1, 8*16*16, C) pos_embed to the (16,16,16) target grid, in place.

    Standard learned-abs-pos_embed resolution adaptation (3D->3D here): trilinear resize when
    the patch grid changes.
    """
    pe_param = vision_encoder.patch_embedding.position_embeddings  # (1, 2048, C)
    pe = pe_param.data
    grid = einops.rearrange(pe, '1 (d h w) c -> 1 c d h w',
                            d=NATIVE_GRID[0], h=NATIVE_GRID[1], w=NATIVE_GRID[2])
    grid = F.interpolate(grid, size=TARGET_GRID, mode='trilinear', align_corners=False)
    new_pe = einops.rearrange(grid, '1 c d h w -> 1 (d h w) c')
    vision_encoder.patch_embedding.position_embeddings = nn.Parameter(new_pe.contiguous())


class M3DEncoder(nn.Module):
    """Vision-only M3D-CLIP wrapper: forward(x) -> (cls[B,C], patch_tokens[B,N,C]).

    forward resizes the input to INPUT_SHAPE (grid 16^3) and replays the exact three ops of the
    pinned remote encode_image(): vision_encoder -> vision_projection -> L2 normalize. The cls
    token is prepended inside the remote vision ViT (single-prefix contract, validated by the
    reviewed remote topology).
    """

    def __init__(self):
        super().__init__()
        self.vision_encoder, self.vision_projection = _load_remote_vision()
        self.embed_dim = self.vision_projection.out_features
        resize_position_embeddings(self.vision_encoder)
        n = swap_sdpa_attention(self.vision_encoder)
        expected = len(self.vision_encoder.blocks)
        if n != expected:
            raise RuntimeError(f'expected {expected} monai SABlocks in the remote ViT, swapped {n}')

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B,1,D,H,W) native -> (64,256,256); grid 16^3 matches the interpolated pos_embed
        x = F.interpolate(x, size=INPUT_SHAPE, mode='trilinear', align_corners=False)
        seq, _ = self.vision_encoder(x)
        seq = self.vision_projection(seq)
        seq = F.normalize(seq, dim=-1)
        return seq[:, 0], seq[:, 1:]


def m3d(*, dims: int, device: str, trainable: bool = False):
    """Construct the M3D-CLIP 3D encoder on `device` (vision weights from HF hub, atomic).

    Declares neither `weights` nor `img_size`: M3D loads from the HF hub inside the constructor
    and resizes to a fixed INPUT_SHAPE, so either argument would be a silent no-op. Passing one
    is a TypeError. 3D-only: dims must be 3.
    """
    if dims != 3:
        raise ValueError('M3D is a 3D-only baseline; pass dims=3 (use eva02/biomedclip for 2D)')
    enc = M3DEncoder().to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc
