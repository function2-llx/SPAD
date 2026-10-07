"""PUMIT 3D Spatial Rotary Position Embedding (additive depth extension of 2D RoPE).

Depth angles use a distinct frequency base (default theta_depth = sqrt(theta)) and are added onto both
in-plane angle blocks; see SpatialRotaryEmbedding3D.__init__ for why sharing or constant-offsetting the
in-plane family aliases. Exact 2D reduction at D == 1 comes from the centered coordinate map (the single
slice sits at coordinate 0), independent of the frequency family.
"""
from functools import lru_cache
import math

import torch
from torch import nn, Tensor

import einops


__all__ = ['SpatialRotaryEmbedding3D', 'rotate_half', 'apply_rope', 'build_rope']


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin


def build_rope(patch_rope: Tensor, n_prefix: int) -> Tensor:
    """Prepend identity RoPE (cos=1, sin=0) for n_prefix tokens.

    Identity RoPE means prefix tokens are not rotated but still participate
    in attention with all other tokens (position-agnostic global aggregators).

    Args:
        patch_rope: [B, num_patches, 2, head_dim] or [num_patches, 2, head_dim]
        n_prefix: number of prefix tokens (CLS + registers)

    Returns:
        [B, n_prefix + num_patches, 2, head_dim] or [n_prefix + num_patches, 2, head_dim]
    """
    prefix_shape = (*patch_rope.shape[:-3], n_prefix, 2, patch_rope.shape[-1])
    prefix = torch.zeros(prefix_shape, device=patch_rope.device, dtype=patch_rope.dtype)
    prefix[..., 0, :] = 1.0  # cos = 1
    return torch.cat([prefix, patch_rope], dim=-3)


class SpatialRotaryEmbedding3D(nn.Module):
    inv_freq: Tensor
    inv_freq_depth: Tensor

    def __init__(
        self,
        head_dim: int,
        theta: float = 100.0,
        theta_depth: float | None = None,
        rescale: float = 2.0,
        disable_depth: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        # Depth must use a DISTINCT frequency base (default sqrt(theta): same geometric family at half the
        # log-rate). Sharing the in-plane family makes every angle a function of (h + d, w + d) only, so
        # positions differing by (c, c, -c) collide exactly wherever the coordinate steps are commensurate
        # (true on real grids, e.g. D=4 vs H=W=16). A constant per-band offset (e.g. the half-step exponent
        # theta^{-2/head_dim}) removes exact collisions but keeps the band-INDEPENDENT frequency ratio, so
        # one displacement still nearly cancels in every band at once. A distinct base makes the ratio
        # (theta/theta_depth)^{4k/head_dim} sweep with k, so no single displacement is near-commensurate
        # across bands. Band k=0 always coincides (ratio 1); the remaining bands do the separating.
        self.theta_depth = math.sqrt(theta) if theta_depth is None else theta_depth
        if not math.isfinite(self.theta_depth) or self.theta_depth <= 0:
            raise ValueError(f'theta_depth must be finite and positive, got {self.theta_depth}')
        self.rescale = rescale
        self.disable_depth = disable_depth
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        inv_freq_depth = 1.0 / self.theta_depth ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32)
        self.register_buffer('inv_freq_depth', inv_freq_depth, persistent=False)
        # Validate the actual fp32 families, not just the input scalars: a depth base that differs from theta
        # by < ~1e-7 (e.g. theta_depth = theta * (1 + 1e-9)) passes a scalar `!=` check but rounds to an
        # fp32 inv_freq_depth identical to inv_freq, silently reintroducing the shared-family collision.
        # Likewise theta_depth -> 1 collapses the depth family to a single frequency (all ones).
        if not inv_freq_depth.isfinite().all() or not (inv_freq_depth > 0).all():
            raise ValueError(f'theta_depth={self.theta_depth} yields a non-finite or non-positive frequency')
        if torch.equal(inv_freq_depth, inv_freq):
            raise ValueError(
                f'theta_depth={self.theta_depth} yields an fp32 depth frequency family identical to the '
                f'in-plane family (theta={theta}); this reintroduces exact (c, c, -c) collisions. '
                f'Use a depth base that differs enough that the fp32 families separate.'
            )
        if torch.equal(inv_freq_depth, inv_freq_depth[0].expand_as(inv_freq_depth)):
            raise ValueError(
                f'theta_depth={self.theta_depth} collapses the depth family to a single frequency '
                f'(structural aliasing endpoint); use a base > 1.'
            )

        # freq_matrix: (3, head_dim) encodes the axis-to-frequency tiling convention
        # angles = coords @ freq_matrix reproduces the meshgrid + tile + add pattern
        freq_matrix = torch.zeros(3, head_dim, dtype=torch.float32)
        freq_matrix[0] = inv_freq_depth.tile(4)     # depth: all slots
        freq_matrix[1, :head_dim // 4] = inv_freq   # height: slots 0-15, 32-47
        freq_matrix[1, head_dim // 2:3 * head_dim // 4] = inv_freq
        freq_matrix[2, head_dim // 4:head_dim // 2] = inv_freq  # width: slots 16-31, 48-63
        freq_matrix[2, 3 * head_dim // 4:] = inv_freq
        freq_matrix *= 2 * math.pi
        self.register_buffer('freq_matrix', freq_matrix, persistent=False)

    def _compute_angles(self, spatial_shape: tuple[int, int, int], device: torch.device) -> Tensor:
        # disable_depth is mutable (probes toggle it post-construction), so it must be part of the cache key.
        return self._compute_angles_cached(spatial_shape, device, self.disable_depth)

    @lru_cache(maxsize=32)
    def _compute_angles_cached(
        self, spatial_shape: tuple[int, int, int], device: torch.device, disable_depth: bool
    ) -> Tensor:
        """Returns stacked [num_patches, 2, head_dim] where dim -2 is (cos, sin)."""
        D, H, W = spatial_shape
        inv_freq = self.inv_freq.to(device)

        coords_h = (2 * torch.arange(0.5, H, device=device, dtype=torch.float32) / H) - 1
        coords_w = (2 * torch.arange(0.5, W, device=device, dtype=torch.float32) / W) - 1
        coords_2d = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1
        ).flatten(0, 1)

        angles_2d = (2 * math.pi * coords_2d[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)

        if D > 1 and not disable_depth:
            coords_d = (2 * torch.arange(0.5, D, device=device, dtype=torch.float32) / D) - 1
            angles_d = (2 * math.pi * coords_d[:, None] * self.inv_freq_depth.to(device)[None, :]).tile(4)
            angles = (angles_d[:, None, :] + angles_2d[None, :, :]).reshape(D * H * W, self.head_dim)
        elif D > 1:
            angles = einops.repeat(angles_2d, 'hw hd -> (d hw) hd', d=D)
        else:
            angles = angles_2d

        return torch.stack([angles.cos(), angles.sin()], dim=-2)  # [num_patches, 2, hd]

    def compute(
        self,
        spatial_shape: tuple[int, int, int],
        visible_idx: Tensor | None = None,
        training: bool = False,
    ) -> Tensor:
        """Compute RoPE for the given spatial shape.

        Returns:
            [num_patches, 2, head_dim] without visible_idx
            [B, num_visible, 2, head_dim] with visible_idx
        """
        device = self.inv_freq.device

        if training and self.rescale is not None:
            rope = self._compute_angles_augmented(spatial_shape, device)
        else:
            rope = self._compute_angles(spatial_shape, device)

        if visible_idx is not None:
            batch_size, num_visible = visible_idx.shape
            gather_idx = einops.repeat(visible_idx, 'n l -> n l r d', r=2, d=self.head_dim)
            rope = einops.repeat(rope, 'l r d -> n l r d', n=batch_size).gather(1, gather_idx)

        return rope

    def _compute_angles_augmented(self, spatial_shape: tuple[int, int, int], device: torch.device) -> Tensor:
        D, H, W = spatial_shape
        inv_freq = self.inv_freq.to(device)

        rescale_range = math.log(self.rescale)
        r = torch.empty(1, device=device, dtype=torch.float32).uniform_(-rescale_range, rescale_range).exp()

        coords_h = ((2 * torch.arange(0.5, H, device=device, dtype=torch.float32) / H) - 1) * r
        coords_w = ((2 * torch.arange(0.5, W, device=device, dtype=torch.float32) / W) - 1) * r
        coords_2d = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1
        ).flatten(0, 1)

        angles_2d = (2 * math.pi * coords_2d[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)

        if D > 1 and not self.disable_depth:
            coords_d = ((2 * torch.arange(0.5, D, device=device, dtype=torch.float32) / D) - 1) * r
            angles_d = (2 * math.pi * coords_d[:, None] * self.inv_freq_depth.to(device)[None, :]).tile(4)
            angles = (angles_d[:, None, :] + angles_2d[None, :, :]).reshape(D * H * W, self.head_dim)
        elif D > 1:
            # disable_depth with a 3D grid: same token count as the depth path, depth-agnostic angles.
            angles = einops.repeat(angles_2d, 'hw hd -> (d hw) hd', d=D)
        else:
            angles = angles_2d

        return torch.stack([angles.cos(), angles.sin()], dim=-2)

    def compute_from_coords(self, coords: Tensor) -> Tensor:
        """Vectorized RoPE from precomputed (seq_len, 3) coordinates.

        Returns (seq_len, 2, head_dim) stacked [cos, sin].
        """
        if self.disable_depth:
            # Match the shape-path semantics: depth contributes no rotation (freq_matrix row 0 is depth).
            coords = coords * coords.new_tensor([0.0, 1.0, 1.0])
        angles = coords.to(self.freq_matrix.dtype) @ self.freq_matrix
        return torch.stack([angles.cos(), angles.sin()], dim=-2)
