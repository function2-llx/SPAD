"""EVA-02 baseline as a backbone module (dims=2 native, dims=3 2D->3D inflated).

Contract: eva02_base/eva02_large(*, dims, device, weights, trainable) -> nn.Module with
forward(x)->(rep,patch); transform_batch(images, *, is_3d) -> forward kwargs (CPU).

The wrapper keeps ONE complete timm EVA model in self.model for both modes and reuses native
forward_features()/forward_head(pre_logits=True); rep is EVA-02's native classification readout
fc_norm(avg_pool(patch)) (verified == timm forward_head pre_logits). dims=3 replaces only three
components inside that model: patch_embed (shared FixedGridPatchEmbed3d, uniform-inflated),
pos_embed (depth-replicated; the 2D source carries no depth information) and rope
(_FixedEvaRoPE3d: native in-plane phase plus an additive centered depth phase in the same
adjacent-pair layout, zero at D=1 and canceling within a slice, preserving EVA's pretrained 2D
reduction and same-slice relative rotations).

transform_batch is shape-blind (normalize only, at native res); resize + RoPE live in
forward, derived from the post-resize size.
"""
from __future__ import annotations

import math

import einops
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._timm_patch_embed import FixedGridPatchEmbed3d
from ..data import normalize_to_encoder, resize_volume, to_3ch_volume
from ..optim import make_parameter_layers

EVA_MODELS = {
    'base': 'eva02_base_patch14_224',
    'large': 'eva02_large_patch14_224',
}
EVA_ROPE_BASE = 10000.0
EVA_DEPTH_ROPE_BASE = EVA_ROPE_BASE / 10

# The selected timm EVA-02 MIM checkpoints use OpenAI CLIP normalization statistics.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def transform_batch(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Raw uint8 batch -> {'x': (B,3,D,H,W)} with EVA's native CLIP normalization (CPU, native res).

    Shape-blind (normalize + channel-replicate; resize/RoPE in forward).
    """
    x = to_3ch_volume(images, is_3d=is_3d)
    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype).view(1, 3, 1, 1, 1)
    return {'x': ((x - mean) / std).contiguous()}


def transform_batch_pm1(images: np.ndarray, *, is_3d: bool) -> dict[str, Tensor]:
    """Legacy [-1,1] (PUMIT pretrain) normalization -- off-distribution for EVA. Archival only."""
    return {'x': normalize_to_encoder(images, is_3d=is_3d)}


def _extend_eva_rope_3d(rope_2d: Tensor, depth: int, depth_base: float) -> Tensor:
    """Fold a centered depth phase into timm EVA's adjacent-pair 2D RoPE table.

    Args:
        rope_2d: EVA table with shape (H*W, 2*head_dim), laid out as concatenated [sin, cos].
        depth: Number of depth slices in the target token grid.
        depth_base: Geometric frequency base for the additive depth phase.

    Returns:
        EVA-layout [sin, cos] coefficients with shape (D*H*W, 2*head_dim).
    """
    if rope_2d.ndim != 2 or rope_2d.shape[1] % 8:
        raise ValueError(
            f'expected EVA RoPE table (H*W, 2*head_dim) with head_dim % 4 == 0, got {rope_2d.shape}'
        )
    if depth < 1:
        raise ValueError(f'depth must be positive, got {depth}')
    if not math.isfinite(depth_base) or depth_base <= 1:
        raise ValueError(f'depth_base must be finite and greater than 1, got {depth_base}')
    if depth == 1:
        return rope_2d

    sin_2d, cos_2d = rope_2d.chunk(2, dim=-1)
    head_dim = sin_2d.shape[-1]
    num_bands = head_dim // 4
    calc_dtype = torch.float32
    depth_coords = (
        2 * math.pi
        * ((2 * torch.arange(0.5, depth, device=rope_2d.device, dtype=calc_dtype) / depth) - 1)
    )
    depth_freqs = depth_base ** (
        -torch.arange(num_bands, device=rope_2d.device, dtype=calc_dtype) / num_bands
    )
    # EVA lays out H and W as two 16-band blocks, then repeats each angle for adjacent channel pairs.
    depth_angles = torch.outer(depth_coords, depth_freqs).repeat(1, 2).repeat_interleave(2, dim=-1)
    sin_depth = depth_angles.sin().to(dtype=rope_2d.dtype)
    cos_depth = depth_angles.cos().to(dtype=rope_2d.dtype)

    sin_3d = (
        sin_2d.unsqueeze(0) * cos_depth[:, None]
        + cos_2d.unsqueeze(0) * sin_depth[:, None]
    )
    cos_3d = (
        cos_2d.unsqueeze(0) * cos_depth[:, None]
        - sin_2d.unsqueeze(0) * sin_depth[:, None]
    )
    return torch.cat([sin_3d, cos_3d], dim=-1).flatten(0, 1)


class _FixedEvaRoPE3d(nn.Module):
    """Fixed additive-depth 3D EVA RoPE table standing in for timm's rope module.

    The table is deterministic (sinusoidal from the validated native config), so it is a
    non-persistent buffer; timm's non-dynamic _pos_embed calls get_embed() with no arguments.
    """

    def __init__(self, rope_2d: Tensor, depth: int, depth_base: float):
        super().__init__()
        self.register_buffer('table', _extend_eva_rope_3d(rope_2d, depth, depth_base),
                             persistent=False)

    def get_embed(self, shape: tuple[int, int] | None = None) -> Tensor:
        return self.table


def _resize_pos_grid(pe: Tensor, *, native: int, target: int) -> Tensor:
    """(1, 1+native^2, C) -> (1, 1+target^2, C): keep the cls token, bicubically resize the grid.

    The pretrained table describes a `native` x `native` layout; a different token budget needs the
    same positional field sampled at a new resolution. Standard ViT practice, and what the
    BiomedCLIP adapter already does.
    """
    cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
    if grid_pe.shape[1] != native * native:
        raise RuntimeError(f'expected a {native}x{native} native pos grid, got {grid_pe.shape[1]} tokens')
    if native == target:
        return pe
    grid2d = einops.rearrange(grid_pe, '1 (h w) c -> 1 c h w', h=native, w=native)
    grid2d = F.interpolate(grid2d.float(), size=(target, target), mode='bicubic',
                           align_corners=False).to(pe.dtype)
    return torch.cat([cls_pe, einops.rearrange(grid2d, '1 c h w -> 1 (h w) c')], dim=1)


def _inflate_pos_embed(pe: Tensor, grid: int) -> Tensor:
    """(1, 1+g^2, C) 2D pos embed -> (1, 1+g^3, C) via depth replication of the grid."""
    cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
    if grid_pe.shape[1] != grid * grid:
        raise RuntimeError(f'expected a {grid}x{grid} native pos grid, got {grid_pe.shape[1]} tokens')
    grid2d = einops.rearrange(grid_pe, '1 (h w) c -> 1 c h w', h=grid, w=grid)
    # lift 2D grid to a 3D cube by replicating along depth (no depth info in the source)
    grid3d = grid2d.unsqueeze(2).repeat(1, 1, grid, 1, 1)
    grid3d = einops.rearrange(grid3d, '1 c d h w -> 1 (d h w) c')
    return torch.cat([cls_pe, grid3d], dim=1)           # (1, 1+g^3, C)


def _adapt_eva_to_3d(model: nn.Module, image_size_3d: int) -> nn.Module:
    """Replace patch_embed/pos_embed/rope of a CPU-loaded timm EVA with the 3D counterparts.

    Validates the reviewed EVA-02 topology before and after replacement; any upstream drift
    fails here, before training. Runs before device transfer, optimizer and DDP construction.
    """
    def bad(reason: str) -> RuntimeError:
        return RuntimeError(f'unexpected EVA topology ({reason}); the 3D adapter must be revalidated')

    if model.dynamic_img_size:
        raise bad('dynamic_img_size interprets the patch output as a 2D grid')
    if getattr(model, 'rope_mixed', False):
        raise bad('rope_mixed')
    if model.num_prefix_tokens != 1 or model.reg_token is not None or model.no_embed_class:
        raise bad('prefix topology is not a single embedded cls token')
    if model.global_pool != 'avg':
        raise bad(f'global_pool={model.global_pool!r}, expected avg')
    if model.patch_drop is not None:
        raise bad('patch_drop is not part of the reviewed adapter contract')
    if not isinstance(model.norm_pre, nn.Identity):
        raise bad('norm_pre is not Identity')
    if model.pos_drop.p != 0 or model.head_drop.p != 0:
        raise bad('pos_drop/head_drop rates must be 0')

    proj = model.patch_embed.proj
    patch = proj.kernel_size[0]
    if proj.kernel_size != (patch, patch) or proj.stride != (patch, patch):
        raise bad(f'patch projection {proj}')
    if image_size_3d % patch:
        raise ValueError(f'image_size_3d {image_size_3d} not divisible by patch {patch}')
    grid = image_size_3d // patch                       # TARGET token grid per axis
    native_grid = model.patch_embed.grid_size[0]        # what the pretrained weights describe
    if model.patch_embed.grid_size != (native_grid, native_grid):
        raise bad(f'native patch grid {model.patch_embed.grid_size} is not square')
    if model.pos_embed.shape[1] != 1 + native_grid * native_grid:
        raise bad(f'native pos table {tuple(model.pos_embed.shape)} is not 1 + {native_grid}^2')

    rope = model.rope
    if (
        rope is None
        or rope.dim != model.blocks[0].attn.head_dim
        or rope.temperature != EVA_ROPE_BASE
        or rope.in_pixels
        or tuple(rope.feat_shape) != (native_grid, native_grid)
        or tuple(rope.ref_feat_shape) != (native_grid, native_grid)
    ):
        raise bad('RoPE configuration changed; the 3D extension must be revalidated')

    # RoPE is sinusoidal -- nothing is learned, so a different grid is rebuilt rather than
    # resampled (interpolating sin/cos values is not the same as evaluating the angles at new
    # coordinates). ref_feat_shape stays at the native grid so the angular scale remains on the
    # pretrained reference instead of being compressed into the smaller grid.
    if grid != native_grid:
        rope = type(rope)(dim=rope.dim, temperature=rope.temperature, in_pixels=rope.in_pixels,
                          feat_shape=(grid, grid), ref_feat_shape=(native_grid, native_grid))
    rope_2d = rope.get_embed()
    pos = _resize_pos_grid(model.pos_embed.data, native=native_grid, target=grid)
    model.patch_embed = FixedGridPatchEmbed3d.from_2d(model.patch_embed, image_size_3d)
    model.pos_embed = nn.Parameter(_inflate_pos_embed(pos, grid))
    model.rope = _FixedEvaRoPE3d(rope_2d, grid, EVA_DEPTH_ROPE_BASE)

    patch_count = grid ** 3
    if model.patch_embed.num_patches != patch_count:
        raise bad('adapted patch grid mismatch')
    if model.num_prefix_tokens + patch_count != model.pos_embed.shape[1]:
        raise bad('adapted pos_embed length mismatch')
    if model.rope.get_embed().shape[0] != patch_count:
        raise bad('adapted RoPE table length mismatch')
    return model


class Eva02Encoder(nn.Module):
    """Pretrained EVA-02 as a MedMNIST baseline; dims=2 native, dims=3 adapted in-place.

    forward(x) with x: (B,3,D,H,W) at native res -> resized to img_size internally ->
    (rep[B,C], patch_tokens[B,N,C]); rep = native forward_head pre-logits (fc_norm(avg_pool)).
    """

    def __init__(
        self, size: str = 'base', dims: int = 3, img_size: int = 224,
        patch: int = 14, pretrained: bool = True, drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if dims not in (2, 3):
            raise ValueError(f'dims must be 2 or 3, got {dims}')
        self.dims = dims
        self.img_size = img_size          # fixed edge (GPU resize target in forward)
        m = timm.create_model(
            EVA_MODELS[size], pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        native_patch = m.patch_embed.proj.kernel_size[0]
        if patch != native_patch:
            raise ValueError(f'patch {patch} does not match the model patch {native_patch}')
        self.patch = patch
        self.embed_dim = m.embed_dim
        self.model = _adapt_eva_to_3d(m, img_size) if dims == 3 else m

    def parameter_layers(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return make_parameter_layers(
            (*self.model.patch_embed.parameters(), self.model.cls_token, self.model.pos_embed),
            self.model.blocks,
            (*self.model.norm.parameters(), *self.model.fc_norm.parameters()),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = resize_volume(x, self.img_size, is_3d=self.dims == 3)
        if self.dims == 2:
            x = x[:, :, 0]                               # (B,3,1,H,W) -> (B,3,H,W)
        seq = self.model.forward_features(x)
        global_features = self.model.forward_head(seq, pre_logits=True)
        patch_tokens = seq[:, self.model.num_prefix_tokens:]
        return global_features, patch_tokens


def _build(
    *, size: str, dims: int, device: str, img_size: int,
    trainable: bool = False, patch: int = 14, drop_path_rate: float = 0.0, **_,
) -> Eva02Encoder:
    """Construct an EVA-02 baseline encoder on `device` (arch + weights, atomic).

    `img_size` is required: at patch 14 the token grid is img_size/14 per axis, so the caller
    must state it rather than inherit a default that silently changes the sequence length (168
    -> 12^3 = 1728, the native 224 -> 16^3 = 4096). `weights` is not declared -- EVA's pretrained
    weights come from the timm hub, not a file. The 2D->3D uniform inflation happens inside
    Eva02Encoder at construction. `**_` swallows kwargs meant for other backbones (e.g. vit_config).
    """
    enc = Eva02Encoder(
        size=size, dims=dims, img_size=img_size, patch=patch,
        pretrained=True, drop_path_rate=drop_path_rate,
    ).to(device)
    if trainable:
        enc.train()
    else:
        enc.eval()
        enc.requires_grad_(False)
    return enc


def eva02_base(
    *, dims: int, device: str, img_size: int,
    trainable: bool = False, drop_path_rate: float = 0.0, **_,
) -> Eva02Encoder:
    """EVA-02 base (768d) variant factory. See _build for the contract."""
    return _build(
        size='base', dims=dims, device=device, img_size=img_size,
        trainable=trainable, drop_path_rate=drop_path_rate, **_,
    )


def eva02_large(
    *, dims: int, device: str, img_size: int,
    trainable: bool = False, drop_path_rate: float = 0.0, **_,
) -> Eva02Encoder:
    """EVA-02 large (1024d) variant factory. See _build for the contract."""
    return _build(
        size='large', dims=dims, device=device, img_size=img_size,
        trainable=trainable, drop_path_rate=drop_path_rate, **_,
    )
