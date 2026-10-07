"""Vendored SegVol image encoder (MONAI 0.9.0 ViT lineage, Apache-2.0 / SegVol MIT).

SegVol (BAAI-DCAI/SegVol) instantiates ``monai.networks.nets.ViT`` from MONAI 0.9.0: hidden 768,
depth 12, heads 12, MLP 3072, pre-LN GELU blocks whose fused qkv projection carries no bias, a
'perceptron' patch embedding (einops Rearrange + Linear) over patches of (4, 16, 16), and a learned
absolute position embedding for the fixed (8, 16, 16) token grid of its (32, 256, 256) input.

This vendored copy makes two representation changes that preserve the computation exactly:

- The Rearrange+Linear patch embedding becomes an equivalent ``nn.Conv3d`` (the released Linear
  weight is reshaped at load; the Rearrange flattens each patch as (p1 p2 p3 c) with channel
  innermost, and the token order is the row-major (D', H', W') grid, which is also the row-major
  flatten of the conv output).
- The flat (1, 2048, 768) position embedding is held as its (1, 8, 16, 16, 768) grid so it can be
  trilinearly resampled onto other token grids at forward time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NATIVE_PATCH = (4, 16, 16)
NATIVE_GRID = (8, 16, 16)
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_DIM = 3072


class SABlock(nn.Module):
    """MONAI 0.9.0 self-attention: fused bias-free qkv, biased output projection."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        x = (attention @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out_proj(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.fn(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int):
        super().__init__()
        self.mlp = MLPBlock(hidden_size, mlp_dim)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SegVolViT(nn.Module):
    """The SegVol trunk: conv patch embedding, grid-resampled position embedding, 12 blocks, norm.

    ``forward`` is intentionally absent; the ViT-Adapter backbone drives ``embed_tokens`` and the
    blocks directly.
    """

    def __init__(self, input_channels: int):
        super().__init__()
        self.patch_embed = nn.Conv3d(
            input_channels,
            EMBED_DIM,
            NATIVE_PATCH,
            stride=NATIVE_PATCH,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, *NATIVE_GRID, EMBED_DIM))
        self.blocks = nn.ModuleList(
            TransformerBlock(EMBED_DIM, MLP_DIM, NUM_HEADS) for _ in range(DEPTH)
        )
        self.norm = nn.LayerNorm(EMBED_DIM)

    def embed_tokens(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """Patchify and add the (resampled) position embedding; tokens are the row-major grid."""
        features = self.patch_embed(x)
        grid = tuple(int(value) for value in features.shape[2:])
        position = self.pos_embed
        if tuple(position.shape[1:4]) != grid:
            position = F.interpolate(
                position.permute(0, 4, 1, 2, 3),
                size=grid,
                mode='trilinear',
                align_corners=False,
            ).permute(0, 2, 3, 4, 1)
        tokens = features.flatten(2).transpose(1, 2) + position.flatten(1, 3)
        return tokens, grid


def convert_patch_embedding_weight(linear_weight: Tensor, input_channels: int) -> Tensor:
    """Reshape the released Linear patch-projection weight into the equivalent Conv3d kernel."""
    expected = NATIVE_PATCH[0] * NATIVE_PATCH[1] * NATIVE_PATCH[2] * input_channels
    if tuple(linear_weight.shape) != (EMBED_DIM, expected):
        raise ValueError(
            f'patch projection weight has shape {tuple(linear_weight.shape)}, '
            f'expected ({EMBED_DIM}, {expected})'
        )
    return (
        linear_weight.view(EMBED_DIM, *NATIVE_PATCH, input_channels)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )
