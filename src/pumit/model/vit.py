"""Vision Transformer with DINOv3 HF-compatible key naming and SPAD patch embedding.

Key naming matches the HuggingFace DINOv3 checkpoint format exactly:
    embeddings.cls_token, embeddings.register_tokens
    embeddings.patch_embeddings.weight, embeddings.patch_embeddings.bias
    layer.{i}.norm1.{weight,bias}, layer.{i}.norm2.{weight,bias}
    layer.{i}.attention.{q,k,v,o}_proj.{weight,bias}
    layer.{i}.layer_scale1.lambda1, layer.{i}.layer_scale2.lambda1
    layer.{i}.mlp.up_proj.{weight,bias}, layer.{i}.mlp.down_proj.{weight,bias}
    norm.{weight,bias}
"""

from dataclasses import dataclass

import einops
from timm.layers import DropPath
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils import checkpoint as torch_checkpoint
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from pumit.spadop.conv import uniform_inflator, Inflator
from pumit.types import NoWeightDecayParameter
from .rope import SpatialRotaryEmbedding3D, rotate_half

from monai.utils import ensure_tuple_rep


@dataclass
class ViTConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    patch_size: int = 16
    num_register_tokens: int = 4
    rope_theta: float = 100.0
    pos_embed_rescale: float = 2.0
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    drop_path_rate: float = 0.0
    grad_ckpt: bool = False
    # None checkpoints every layer when grad_ckpt is enabled.
    grad_ckpt_first_n_layers: int | None = None
    in_channels: int = 3


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.lambda1


class Mlp(nn.Module):
    def __init__(self, dim: int, intermediate_size: int, bias: bool = True):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=bias)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(intermediate_size, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim, _rem = divmod(dim, num_heads)
        assert _rem == 0, f'dim={dim} not divisible by num_heads={num_heads}'

        self.q_proj = nn.Linear(dim, dim, bias=query_bias)
        self.k_proj = nn.Linear(dim, dim, bias=key_bias)
        self.v_proj = nn.Linear(dim, dim, bias=value_bias)
        self.o_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor, rope: Tensor, attn_bias=None) -> Tensor:
        B, N, _ = x.shape
        assert rope.shape[-1] == self.head_dim, f'rope head_dim {rope.shape[-1]} != {self.head_dim}'

        q = einops.rearrange(self.q_proj(x), 'b n (h d) -> b n h d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), 'b n (h d) -> b n h d', h=self.num_heads)
        v = einops.rearrange(self.v_proj(x), 'b n (h d) -> b n h d', h=self.num_heads)

        # rope: [..., 2, head_dim]. Unpack cos/sin, add head broadcast dim.
        cos = rope[..., 0, :].unsqueeze(-2)  # [..., 1, hd]
        sin = rope[..., 1, :].unsqueeze(-2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        x = xops.memory_efficient_attention(q.type_as(v), k.type_as(v), v, attn_bias=attn_bias)
        x = einops.rearrange(x, 'b n h d -> b n (h d)')
        return self.o_proj(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_size: int,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(
            dim, num_heads,
            query_bias=query_bias, key_bias=key_bias,
            value_bias=value_bias, proj_bias=proj_bias,
        )
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, intermediate_size, bias=mlp_bias)
        self.layer_scale2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _apply_drop_path(
        self,
        x: Tensor,
        token_to_sequence: Tensor | None,
        num_sequences: int,
    ) -> Tensor:
        if token_to_sequence is None or not self.training or isinstance(self.drop_path, nn.Identity):
            return self.drop_path(x)
        mask = self.drop_path(x.new_ones((num_sequences, 1)))
        return x * mask[token_to_sequence].unsqueeze(0)

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        attn_bias=None,
        token_to_sequence: Tensor | None = None,
        num_sequences: int = 0,
    ) -> Tensor:
        x = x + self._apply_drop_path(
            self.layer_scale1(self.attention(self.norm1(x), rope, attn_bias)),
            token_to_sequence, num_sequences,
        )
        x = x + self._apply_drop_path(
            self.layer_scale2(self.mlp(self.norm2(x))), token_to_sequence, num_sequences,
        )
        return x


class SPADPatchEmbed(nn.Module):
    """Patch embedding with SPAD depth adaptation and 2D->3D weight inflation.

    Uses raw nn.Parameter weight/bias (not nn.Conv3d sub-module) with F.conv3d.

    Inflation defaults to uniform (mean): kernel size == stride here, so unlike the codec's
    overlapping 3x3s1 convs, a center-inflated patch would read only the middle depth slices
    and discard the rest of the patch.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        inflator: Inflator = uniform_inflator,
    ):
        super().__init__()
        self.patch_size: tuple[int, int, int] = ensure_tuple_rep(patch_size, 3)
        assert self.patch_size[0] & (self.patch_size[0] - 1) == 0, 'depth patch_size must be power of 2'
        # Raw parameters, not nn.Conv3d
        self.weight = nn.Parameter(
            torch.empty(embed_dim, in_channels, *self.patch_size)
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
        self.inflator = inflator

    @property
    def max_adapt(self) -> int:
        return self.patch_size[0].bit_length() - 1

    def forward(self, x: Tensor, da: int = 0) -> Tensor:
        adapt = min(da, self.max_adapt)
        if adapt == 0:
            return F.conv3d(x, self.weight, self.bias, stride=self.patch_size)
        new_depth = self.patch_size[0] >> adapt
        weight = einops.reduce(
            self.weight,
            'co ci (dr dc) kh kw -> co ci dr kh kw',
            'sum', dr=new_depth,
        )
        stride = (new_depth, self.patch_size[1], self.patch_size[2])
        return F.conv3d(x, weight, self.bias, stride=stride, padding=0)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight_key = f'{prefix}weight'
        if (w := state_dict.get(weight_key)) is not None and w.ndim == 4:
            # 2D [Co, Ci, H, W] -> interpolate if HW mismatch -> inflate to 3D
            if w.shape[2:] != self.patch_size[1:]:
                w = F.interpolate(w.float(), self.patch_size[1:], mode='bicubic')
            d = self.patch_size[0]
            w3d = self.inflator(w, d)  # (D, Co, Ci, H, W)
            state_dict[weight_key] = einops.rearrange(w3d, 'd co ci kh kw -> co ci d kh kw')
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Embeddings(nn.Module):
    """Container for CLS token, register tokens, and patch embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_register_tokens: int,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        self.cls_token = NoWeightDecayParameter(torch.empty(1, 1, embed_dim))
        self.register_tokens = NoWeightDecayParameter(
            torch.empty(1, num_register_tokens, embed_dim)
        )
        self.patch_embeddings = SPADPatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def forward(self, x: Tensor, da: int = 0) -> Tensor:
        """Apply patch embedding and flatten to tokens.

        Returns:
            [B, n_patches, embed_dim]
        """
        out = self.patch_embeddings(x, da=da)
        return einops.rearrange(out, 'b c ... -> b (...) c')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # DINOv3 HF checkpoint includes mask_token; we don't use it
        state_dict.pop(f'{prefix}mask_token', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ViT(nn.Module):
    """Vision Transformer with DINOv3 HF-compatible architecture.

    Key naming matches HuggingFace DINOv3 checkpoint format exactly.
    No key remapping needed for loading.
    """

    def __init__(self, config: ViTConfig, skip_embed: bool = False):
        super().__init__()
        self.config = config
        self.skip_embed = skip_embed
        hidden_size = config.hidden_size

        self._grad_ckpt = config.grad_ckpt
        self._grad_ckpt_first_n_layers = config.grad_ckpt_first_n_layers
        if self._grad_ckpt_first_n_layers is not None:
            if not self._grad_ckpt:
                raise ValueError('grad_ckpt_first_n_layers requires grad_ckpt=True')
            if not 0 <= self._grad_ckpt_first_n_layers <= config.num_hidden_layers:
                raise ValueError(
                    f'grad_ckpt_first_n_layers must be in [0, {config.num_hidden_layers}], '
                    f'got {self._grad_ckpt_first_n_layers}',
                )

        if not skip_embed:
            self.embeddings = Embeddings(
                embed_dim=hidden_size,
                num_register_tokens=config.num_register_tokens,
                patch_size=config.patch_size,
                in_channels=config.in_channels,
            )

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layer = nn.ModuleList([
            Block(
                dim=hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                query_bias=config.query_bias,
                key_bias=config.key_bias,
                value_bias=config.value_bias,
                proj_bias=config.proj_bias,
                mlp_bias=config.mlp_bias,
                drop_path=dpr[i],
            )
            for i in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        head_dim = hidden_size // config.num_attention_heads
        self.rope = SpatialRotaryEmbedding3D(
            head_dim, theta=config.rope_theta, rescale=config.pos_embed_rescale,
        )

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

    @property
    def n_prefix(self) -> int:
        return 1 + self.config.num_register_tokens

    @property
    def patch_embed(self) -> SPADPatchEmbed:
        return self.embeddings.patch_embeddings

    def encode_image(
        self,
        x: Tensor,
        da: int = 0,
        visible_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Convenience: image in, (cls_token, patch_tokens) out.

        Wraps patch embedding, prefix prepending, RoPE computation, and forward().
        For custom pipelines (e.g., multi-view packing), call the components directly.
        """
        from .rope import build_rope

        patch_size = self.config.patch_size
        max_adapt = self.embeddings.patch_embeddings.max_adapt
        depth_stride = patch_size >> min(da, max_adapt)
        _, _, D_in, H_in, W_in = x.shape
        spatial_shape = (D_in // depth_stride, H_in // patch_size, W_in // patch_size)

        patch_tokens = self.embeddings(x, da=da)
        B = patch_tokens.shape[0]

        if visible_idx is not None:
            gather_idx = einops.repeat(visible_idx, 'n l -> n l d', d=self.embed_dim)
            patch_tokens = patch_tokens.gather(dim=1, index=gather_idx)

        prefix = torch.cat([
            self.embeddings.cls_token.expand(B, -1, -1),
            self.embeddings.register_tokens.expand(B, -1, -1),
        ], dim=1)
        x = torch.cat([prefix, patch_tokens], dim=1)

        patch_rope = self.rope.compute(spatial_shape, visible_idx, training=self.training)
        rope = build_rope(patch_rope, n_prefix=self.n_prefix)
        if rope.ndim == 3:
            rope = rope.unsqueeze(0).expand(B, -1, -1, -1)

        out = self.forward(x, rope)
        return out[:, 0], out[:, self.n_prefix:]

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        attn_bias=None,
        return_hidden: bool = False,
        hidden_layers: set[int] | None = None,
    ):
        """Run transformer blocks on token sequences.

        Args:
            x: [B, L, C] token sequence.
            rope: [B, L, 2, head_dim] stacked (cos, sin). Identity (cos=1, sin=0)
                for positions that should not be rotated.
            attn_bias: Optional attention bias (e.g., xformers BlockDiagonalMask).
            return_hidden: If True, return all intermediate layer hidden states.
                (Prefer hidden_layers for memory-sensitive callers.)
            hidden_layers: If set, return only the hidden states at these 1-indexed
                layer numbers (e.g. {16, 20}). Cheaper than return_hidden under
                grad_ckpt because only the requested intermediates are retained.
                Mutually exclusive with return_hidden=True.

        Returns:
            If neither return_hidden nor hidden_layers: [B, L, C] normalized output.
            If return_hidden: (normed_output, tuple of ALL per-layer hidden states).
            If hidden_layers: (normed_output, tuple of requested hidden states,
                in ascending layer order).
        """
        assert rope.shape[1] == x.shape[1], f'rope L={rope.shape[1]} != x L={x.shape[1]}'
        assert not (return_hidden and hidden_layers is not None), \
            'return_hidden and hidden_layers are mutually exclusive'
        if hidden_layers is not None:
            assert len(hidden_layers) == 0 or max(hidden_layers) <= len(self.layer), \
                f'hidden_layers {hidden_layers} out of range for {len(self.layer)} layers'

        if return_hidden or hidden_layers is not None:
            hidden_states = []
        else:
            hidden_states = None

        drop_path_args = ()
        if self.training and self.config.drop_path_rate > 0 and isinstance(attn_bias, BlockDiagonalMask):
            # Packed batches have B=1. Share a mask within each sequence, including its prefix tokens.
            seqstart = attn_bias.q_seqinfo.seqstart
            token_to_sequence = torch.bucketize(
                torch.arange(x.shape[1], device=x.device), seqstart[1:], right=True,
            )
            drop_path_args = (token_to_sequence, seqstart.shape[0] - 1)

        for i, block in enumerate(self.layer):
            should_checkpoint = self.training and self._grad_ckpt and (
                self._grad_ckpt_first_n_layers is None or i < self._grad_ckpt_first_n_layers
            )
            if should_checkpoint:
                x = torch_checkpoint.checkpoint(
                    block, x, rope, attn_bias, *drop_path_args,
                    use_reentrant=False,
                )
            else:
                x = block(x, rope, attn_bias, *drop_path_args)
            if hidden_states is not None:
                if return_hidden or (hidden_layers is not None and (i + 1) in hidden_layers):
                    hidden_states.append(x)

        out = self.norm(x)
        if hidden_states is not None:
            return out, tuple(hidden_states)
        return out

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args, **kwargs):
        # Pop keys that exist in HF checkpoint but not in our model
        state_dict.pop(f'{prefix}embeddings.mask_token', None)
        state_dict.pop(f'{prefix}rope_embeddings.inv_freq', None)
        # Pop old rope buffers if present
        for k in list(state_dict):
            if k.startswith(f'{prefix}rope.'):
                state_dict.pop(k)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
