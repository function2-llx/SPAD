"""BiomedCLIP baseline as a backbone module (2D native, 3D uniform-inflated + interp abs pos).

Contract: biomedclip(*, dims, device, weights, trainable) -> nn.Module with
forward(x)->(cls,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

Medical-domain 2D ViT-B/16 foil (pretrained on 15M PMC image-text pairs) vs the
natural-image EVA/DINOv3 2D baselines. The wrapper keeps ONE complete open_clip vision
object in self.visual for both modes (a timm ViT trunk: abs pos_embed, NO RoPE,
768d/12L/patch16/224, plus the official 768->512 CLIP projection); the PubMedBERT text
tower is dropped at load. dims=3 replaces only trunk.patch_embed (shared
FixedGridPatchEmbed3d, uniform-inflated) and trunk.pos_embed (trilinear 14^2 -> grid^3 lift;
depth-blind, every depth slice gets the same 2D position -- we do NOT inject 3D RoPE,
BiomedCLIP was pretrained without rotary positions). Execution reuses native
trunk.forward_features()/forward_head(pre_logits=True); the adapter forward applies
visual.head to both global and patch outputs (the downstream contract returns the
official projected representations).

Loading is revision-pinned: create_model_from_pretrained('hf-hub:...') cannot pin a
revision, so _load_visual snapshots the pinned commit and loads via the local-dir schema
(config and weights from the same snapshot; no moving-main fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

import einops
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import Tensor, nn

from ._timm_patch_embed import FixedGridPatchEmbed3d
from ..data import resize_volume, to_3ch_volume
from ..optim import make_parameter_layers

BIOMEDCLIP_REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
BIOMEDCLIP_REVISION = '9f341de24bfb00180f1b847274256e9b65a3a32e'
REVIEWED_OPEN_CLIP_VERSION = '3.3.0'

# OpenAI CLIP stats (BiomedCLIP's training normalization).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 MedMNIST batch -> {'x': (B,3,D,H,W)} with CLIP normalization (CPU, native res).

    Shape-blind (normalize + channel-replicate at native res); resize + patchify live in
    forward. Owns this experiment's normalization (CLIP mean/std, not PUMIT [-1,1]).
    """
    x = to_3ch_volume(images, is_3d=is_3d)
    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype).view(1, 3, 1, 1, 1)
    return {'x': ((x - mean) / std).contiguous()}


def assert_reviewed_open_clip() -> None:
    """Fail fast if open_clip is not the reviewed version (local-dir schema, TimmModel topology)."""
    if open_clip.__version__ != REVIEWED_OPEN_CLIP_VERSION:
        raise RuntimeError(
            f'the BiomedCLIP adapter was reviewed against open_clip {REVIEWED_OPEN_CLIP_VERSION}, '
            f'found {open_clip.__version__}; re-validate before running'
        )


def _load_visual(drop_path_rate: float = 0.0) -> nn.Module:
    """Load the revision-pinned BiomedCLIP vision tower (config + weights from one snapshot)."""
    path = snapshot_download(
        BIOMEDCLIP_REPO,
        revision=BIOMEDCLIP_REVISION,
        allow_patterns=['open_clip_config.json', 'open_clip_pytorch_model.bin'],
    )
    vision_cfg = json.loads((Path(path) / 'open_clip_config.json').read_text())['model_cfg']['vision_cfg']
    vision_cfg['timm_drop_path'] = drop_path_rate
    model, _ = open_clip.create_model_from_pretrained(
        f'local-dir:{path}', vision_cfg=vision_cfg,
    )
    return model.visual


def _validate_head(head: nn.Module, trunk_dim: int) -> None:
    """The official projection must be exactly token-wise Dropout(0) + Linear(trunk_dim, out)."""
    if not (isinstance(head, nn.Sequential) and len(head) == 2
            and isinstance(head[0], nn.Dropout) and head[0].p == 0
            and isinstance(head[1], nn.Linear) and head[1].bias is None
            and head[1].in_features == trunk_dim):
        raise RuntimeError(
            f'unexpected BiomedCLIP visual projection {head!r}; the adapter must be revalidated'
        )


def _lift_pos_embed(pe: Tensor, grid: int) -> Tensor:
    """Native (1, 1+s^2, C) 2D pos embed -> (1, 1+g^3, C): trilinear s->g resample plus depth lift.

    Depth-blind: a 2D-pretrained PE has no depth axis, so the same plane replicates across depth.
    """
    cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
    s = round(grid_pe.shape[1] ** 0.5)
    if s * s != grid_pe.shape[1]:
        raise RuntimeError(f'non-square native pos grid: {grid_pe.shape[1]} tokens')
    grid2d = einops.rearrange(grid_pe, '1 (h w) c -> 1 c 1 h w', h=s, w=s)
    grid3d = F.interpolate(grid2d, size=(grid, grid, grid), mode='trilinear', align_corners=False)
    grid3d = einops.rearrange(grid3d, '1 c d h w -> 1 (d h w) c')
    return torch.cat([cls_pe, grid3d], dim=1)


def _adapt_biomedclip_to_3d(visual: nn.Module, image_size_3d: int, patch: int) -> nn.Module:
    """Replace trunk.patch_embed/pos_embed of a CPU-loaded visual with the 3D counterparts.

    Validates the reviewed trunk topology before and after replacement; any upstream drift
    fails here, before training. Runs before device transfer, optimizer and DDP construction.
    """
    trunk = visual.trunk

    def bad(reason: str) -> RuntimeError:
        return RuntimeError(
            f'unexpected BiomedCLIP trunk topology ({reason}); the 3D adapter must be revalidated'
        )

    if trunk.dynamic_img_size:
        raise bad('dynamic_img_size')
    if trunk.num_prefix_tokens != 1 or trunk.reg_token is not None or trunk.no_embed_class:
        raise bad('prefix topology is not a single embedded cls token')
    if trunk.global_pool != 'token':
        raise bad(f'global_pool={trunk.global_pool!r}, expected token')
    if not isinstance(trunk.fc_norm, nn.Identity):
        raise bad('fc_norm is not Identity')
    if trunk.head_drop.p != 0 or trunk.pos_drop.p != 0:
        raise bad('head_drop/pos_drop rates must be 0')
    if not isinstance(trunk.patch_drop, nn.Identity):
        raise bad('patch_drop is not Identity')
    if not isinstance(trunk.norm_pre, nn.Identity):
        raise bad('norm_pre is not Identity')

    proj = trunk.patch_embed.proj
    native_patch = proj.kernel_size[0]
    if proj.kernel_size != (native_patch, native_patch) or proj.stride != (native_patch, native_patch):
        raise bad(f'patch projection {proj}')
    if patch != native_patch:
        raise ValueError(f'patch {patch} does not match the trunk patch {native_patch}')
    if image_size_3d % patch:
        raise ValueError(f'image_size_3d {image_size_3d} not divisible by patch {patch}')
    grid = image_size_3d // patch

    trunk.patch_embed = FixedGridPatchEmbed3d.from_2d(trunk.patch_embed, image_size_3d)
    trunk.pos_embed = nn.Parameter(_lift_pos_embed(trunk.pos_embed.data, grid))

    if trunk.patch_embed.num_patches != grid ** 3:
        raise bad('adapted patch grid mismatch')
    if trunk.num_prefix_tokens + grid ** 3 != trunk.pos_embed.shape[1]:
        raise bad('adapted pos_embed length mismatch')
    return visual


class BiomedCLIPEncoder(nn.Module):
    """BiomedCLIP vision tower as a MedMNIST baseline; dims=2 native, dims=3 adapted in-place.

    forward(x): 2D -> x (B,3,H,W); 3D -> x (B,3,D,H,W), each resized to the fixed size.
    Returns (cls[B,C], patch_tokens[B,N,C]), both through the official CLIP projection.
    """

    def __init__(
        self, dims: int = 3, img_size: int | None = None, patch: int = 16,
        drop_path_rate: float = 0.0,
    ):
        # img_size=None keeps the per-dims native default for direct construction (tests, 2D);
        # the factory requires an explicit value so experiment code cannot inherit a grid size.
        super().__init__()
        assert_reviewed_open_clip()
        if dims not in (2, 3):
            raise ValueError(f'dims must be 2 or 3, got {dims}')
        self.dims = dims
        self.img_size = img_size if img_size is not None else (256 if dims == 3 else 224)
        self.patch = patch
        self.visual = _load_visual(drop_path_rate=drop_path_rate)
        _validate_head(self.visual.head, self.visual.trunk.embed_dim)
        self.embed_dim = self.visual.head[1].out_features
        if dims == 3:
            _adapt_biomedclip_to_3d(self.visual, self.img_size, patch)

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        trunk = self.visual.trunk
        return make_parameter_layers(
            (*trunk.patch_embed.parameters(), trunk.cls_token, trunk.pos_embed),
            trunk.blocks,
            (*trunk.norm.parameters(), *self.visual.head.parameters()),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = resize_volume(x, self.img_size, is_3d=self.dims == 3)
        if self.dims == 2:
            x = x[:, :, 0]                               # (B,3,1,H,W) -> (B,3,H,W)
        trunk = self.visual.trunk
        seq = trunk.forward_features(x)
        global_features = self.visual.head(trunk.forward_head(seq, pre_logits=True))
        patch_tokens = self.visual.head(seq[:, trunk.num_prefix_tokens:])
        return global_features, patch_tokens


def _build(
    *, dims: int, device: str, img_size: int,
    trainable: bool = False, patch: int = 16, drop_path_rate: float = 0.0, **_,
) -> BiomedCLIPEncoder:
    """Construct a BiomedCLIP encoder on `device` (arch + weights, atomic).

    `img_size` is required: at patch 16 it sets the token grid (192 -> 12^3 = 1728, 256 ->
    16^3 = 4096), so the caller must state the sequence length rather than inherit it.
    `weights` is not declared -- the pinned HF snapshot is loaded internally, not from a file.
    """
    enc = BiomedCLIPEncoder(
        dims=dims, img_size=img_size, patch=patch, drop_path_rate=drop_path_rate,
    ).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc


def biomedclip(
    *, dims: int, device: str, img_size: int,
    trainable: bool = False, drop_path_rate: float = 0.0, **_,
) -> BiomedCLIPEncoder:
    """BiomedCLIP ViT-B/16 factory. See _build for the contract."""
    return _build(
        dims=dims, device=device, img_size=img_size,
        trainable=trainable, drop_path_rate=drop_path_rate, **_,
    )
