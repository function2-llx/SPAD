"""Shared fixed-grid timm 3D patch embedding.

FixedGridPatchEmbed3d is the ONLY boundary shared by the EVA-02 and BiomedCLIP 3D adapters: both trunks are fixed-grid timm models whose PatchEmbed is a non-overlapping square Conv2d returning flattened NLC tokens.
The component derives its entire configuration from the source module (from_2d is the only constructor) and owns nothing beyond the Conv3d, the inflated weights, the source normalization and fixed-grid metadata.
Position inflation, RoPE, prefix composition and readout stay model-local.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from pumit.spadop.conv import uniform_inflator

class FixedGridPatchEmbed3d(nn.Module):
    """3D uniform-inflated counterpart of a fixed-grid timm PatchEmbed; forward returns [B, DHW, C].

    Attributes:
        grid_size: tokens per axis (d, h, w) at the configured 3D image size.
        num_patches: d*h*w.
        embed_dim: output channel count, copied from the source projection.
    """

    def __init__(self, proj: nn.Conv3d, norm: nn.Module, grid_size: tuple[int, int, int]):
        super().__init__()
        self.proj = proj
        self.norm = norm
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        self.embed_dim = proj.out_channels

    @classmethod
    def from_2d(cls, source: timm.layers.PatchEmbed, image_size_3d: int) -> 'FixedGridPatchEmbed3d':
        """Build the 3D patch embedding from an existing timm 2D module (single source of truth).

        Uniform-inflates the source Conv2d kernel and copies its bias; the source normalization
        module is carried over unchanged. Stride == kernel here (asserted below), so a center
        inflation would zero out every non-central depth slice of the patch.
        """
        proj2d = source.proj
        kh, kw = proj2d.kernel_size
        if kh != kw:
            raise ValueError(f'expected a square 2D patch kernel, got {proj2d.kernel_size}')
        if proj2d.stride != proj2d.kernel_size:
            raise ValueError(f'expected non-overlapping stride == kernel, got stride {proj2d.stride}')
        if not getattr(source, 'flatten', True):
            raise ValueError('expected a flattened (NLC) source patch embed')
        if image_size_3d % kh:
            raise ValueError(f'image_size_3d {image_size_3d} not divisible by patch {kh}')
        g = image_size_3d // kh

        proj3d = nn.Conv3d(proj2d.in_channels, proj2d.out_channels, kh, stride=kh,
                           bias=proj2d.bias is not None)
        w3d = uniform_inflator(proj2d.weight.detach(), kh).permute(1, 2, 0, 3, 4)
        proj3d.weight = nn.Parameter(w3d.contiguous())
        if proj2d.bias is not None:
            proj3d.bias = nn.Parameter(proj2d.bias.detach().clone())
        return cls(proj3d, source.norm, (g, g, g))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, D, H, W) on the fixed grid -> (B, D*H*W, embed_dim), d-major NLC."""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
