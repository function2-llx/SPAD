"""Vendored VoCo SwinUNETR-v2 encoder (MONAI 1.3.0 lineage, Apache-2.0).

Faithful 3D-only extraction of ``monai.networks.nets.swin_unetr.SwinTransformer`` with ``use_v2=True`` residual conv stages plus the ``UnetrBasicBlock``/``UnetResBlock`` skip encoders, matching VoCo's released encoder checkpoint keys (Luffy03/Large-Scale-Medical, ``Self-supervised/models/voco_head.py``).
The classic ``PatchMerging`` retains the duplicated slices used during VoCo pretraining under MONAI 1.3.0; later MONAI releases reordered them and compute different stage features with the same weights.
Optional stochastic depth and activation recomputation support classification finetuning without changing the default segmentation behavior or checkpoint keys.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError(f'expected three values, got {values}')
    return values


class _ConvOnly(nn.Sequential):
    """MONAI ``Convolution`` with act/norm disabled: one child named ``conv``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
    ):
        super().__init__()
        kernel = _triple(kernel_size)
        self.add_module(
            'conv',
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel,
                stride=_triple(stride),
                padding=tuple((k - 1) // 2 for k in kernel),
                bias=False,
            ),
        )


class UnetResBlock(nn.Module):
    """MONAI dynunet residual block with instance norm (no affine) and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
    ):
        super().__init__()
        self.conv1 = _ConvOnly(in_channels, out_channels, kernel_size, stride)
        self.conv2 = _ConvOnly(out_channels, out_channels, kernel_size, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.downsample = in_channels != out_channels or any(s != 1 for s in _triple(stride))
        if self.downsample:
            self.conv3 = _ConvOnly(in_channels, out_channels, 1, stride)
            self.norm3 = nn.InstanceNorm3d(out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        residual = inp
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.norm2(self.conv2(out))
        if self.downsample:
            residual = self.norm3(self.conv3(residual))
        out = out + residual
        return self.lrelu(out)


class UnetrBasicBlock(nn.Module):
    """MONAI UNETR basic block in its residual configuration (``res_block=True``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
    ):
        super().__init__()
        self.layer = UnetResBlock(in_channels, out_channels, kernel_size, stride)

    def forward(self, inp: Tensor) -> Tensor:
        return self.layer(inp)


def window_partition(x: Tensor, window_size: Sequence[int]) -> Tensor:
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    return (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size[0] * window_size[1] * window_size[2], c)
    )


def window_reverse(windows: Tensor, window_size: Sequence[int], dims: Sequence[int]) -> Tensor:
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)


def get_window_size(
    x_size: Sequence[int],
    window_size: Sequence[int],
    shift_size: Sequence[int] | None = None,
):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(
    dims: Sequence[int],
    window_size: Sequence[int],
    shift_size: Sequence[int],
    device: torch.device,
) -> Tensor:
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d_slice in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h_slice in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w_slice in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        self.attention_chunk_size = attention_chunk_size
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size[0]),
                torch.arange(window_size[1]),
                torch.arange(window_size[2]),
                indexing='ij',
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        if self.attention_chunk_size is None:
            return self._attention(x, mask)
        # Shifted windows stay grouped by image so the original mask broadcasts unchanged.
        chunk_size = mask.shape[0] if mask is not None else self.attention_chunk_size
        outputs = []
        for chunk in x.split(chunk_size):
            if self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._attention, chunk, mask, use_reentrant=False))
            else:
                outputs.append(self._attention(chunk, mask))
        return torch.cat(outputs, dim=0)

    def _attention(self, x: Tensor, mask: Tensor | None) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.fn(self.linear1(x)))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        use_checkpoint: bool = False,
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, attention_chunk_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def _attention(self, x: Tensor, mask_matrix: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

    def forward(self, x: Tensor, mask_matrix: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            x = x + self.drop_path(checkpoint(self._attention, x, mask_matrix, use_reentrant=False))
            return x + checkpoint(self._mlp, x, use_reentrant=False)
        x = x + self.drop_path(self._attention(x, mask_matrix))
        return x + self._mlp(x)

    def _mlp(self, x: Tensor) -> Tensor:
        return self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Classic (v0.9.0) merging: the duplicated x5/x6 slices are pretraining-faithful."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        if (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        downsample: bool,
        drop_path: Sequence[float] | None = None,
        use_checkpoint: bool = False,
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for _ in window_size)
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                drop_path=drop_path[i] if drop_path is not None else 0.0,
                use_checkpoint=use_checkpoint,
                attention_chunk_size=attention_chunk_size,
            )
            for i in range(depth)
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(x, 'b d h w c -> b c d h w')


class PatchEmbed(nn.Module):
    """Swin patch embedding: strided conv with input padded to patch multiples, no norm."""

    def __init__(self, patch_size: Sequence[int], in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, self.patch_size, stride=self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        _, _, d, h, w = x.shape
        if w % self.patch_size[2]:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1]:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0]:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))
        return self.proj(x)


class SwinTransformer(nn.Module):
    """SwinViT with the v2 residual conv stage prologues, returning the five stage outputs."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int] = (7, 7, 7),
        patch_size: Sequence[int] = (2, 2, 2),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.layers1c = nn.ModuleList()
        self.layers2c = nn.ModuleList()
        self.layers3c = nn.ModuleList()
        self.layers4c = nn.ModuleList()
        stage_lists = (self.layers1, self.layers2, self.layers3, self.layers4)
        conv_lists = (self.layers1c, self.layers2c, self.layers3c, self.layers4c)
        drop_path = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2**i_layer)
            stage_lists[i_layer].append(
                BasicLayer(
                    dim=dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    downsample=True,
                    drop_path=drop_path[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    attention_chunk_size=attention_chunk_size,
                )
            )
            conv_lists[i_layer].append(UnetrBasicBlock(dim, dim, kernel_size=3, stride=1))

    @staticmethod
    def _proj_out(x: Tensor) -> Tensor:
        ch = x.shape[1]
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = F.layer_norm(x, [ch])
        return rearrange(x, 'n d h w c -> n c d h w')

    def forward(self, x: Tensor) -> list[Tensor]:
        x0 = self.patch_embed(x)
        x0_out = self._proj_out(x0)
        x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self._proj_out(x1)
        x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self._proj_out(x2)
        x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self._proj_out(x3)
        x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self._proj_out(x4)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]
