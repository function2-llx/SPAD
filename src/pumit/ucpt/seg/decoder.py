"""SAM 3 semantic path in 3D: fusion encoder + semantic head (K classes batched).

Module/attribute names mirror HF Sam3* so pretrained weights key-match.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pumit import spadop
from pumit.ucpt.seg.schedule import neck_da_schedule

class Attention(nn.Module):
    """Separate-projection MHA, names mirror Sam3Attention (q/k/v/o_proj)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        b, nq, _ = query.shape
        nk = key.shape[1]
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(query).view(b, nq, h, d).transpose(1, 2)
        k = self.k_proj(key).view(b, nk, h, d).transpose(1, 2)
        v = self.v_proj(value).view(b, nk, h, d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(b, nq, h * d)
        return self.o_proj(out)

class MLP(nn.Module):
    """MLP whose parameter names mirror ``Sam3MLP``."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))

class FusionEncoderLayer(nn.Module):
    """Self-attention, text cross-attention, and MLP layer mirroring ``Sam3DetrEncoderLayer``."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads)
        self.cross_attn = Attention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        vision: Tensor,
        text: Tensor,
        pos: Tensor,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        residual = vision
        h = self.layer_norm1(vision)
        h_pos = h + pos
        h = self.self_attn(query=h_pos, key=h_pos, value=h)
        h = h + residual
        residual = h
        h2 = self.layer_norm2(h)
        h2 = self.cross_attn(query=h2, key=text, value=text, attn_bias=attn_bias)
        h = h2 + residual
        residual = h
        h3 = self.layer_norm3(h)
        h3 = self.mlp(h3)
        return h3 + residual

class FusionEncoder(nn.Module):
    """Stack vision self-attention and text cross-attention layers."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        num_layers: int = 6,
    ):
        """
        Args:
            hidden_size: Token feature width.
            num_heads: Attention-head count.
            intermediate_size: MLP hidden width.
            num_layers: Encoder-layer count.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FusionEncoderLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ],
        )

    def forward(
        self,
        vision: Tensor,
        text: Tensor,
        pos: Tensor,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        """Fuse vision tokens with text prompts.

        Args:
            vision: Vision tokens shaped ``(B, N, C)``.
            text: Text tokens shaped ``(B, L, C)``.
            pos: Vision position encoding shaped ``(N, C)``.
            attn_bias: Additive mask on text keys shaped ``(B, 1, 1, L)``,
                or None to leave cross-attn unmasked.

        Returns:
            Fused vision tokens shaped ``(B, N, C)``.
        """
        for layer in self.layers:
            vision = layer(vision, text, pos, attn_bias)
        return vision

class PixelDecoder(nn.Module):
    """Coarse-to-fine FPN mirroring ``Sam3PixelDecoder``."""

    def __init__(self, hidden_size: int = 256, num_upsampling_stages: int = 3):
        """
        Args:
            hidden_size: Pyramid feature width.
            num_upsampling_stages: Number of allocated convolution stages.
        """
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                spadop.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
                for _ in range(num_upsampling_stages)
            ],
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_size) for _ in range(num_upsampling_stages)])
        self.out_channels = hidden_size

    def forward(self, start_1_16: Tensor, levels: dict[str, Tensor], da: int | None) -> Tensor:
        """Decode fused features from 1/16 to 1/4 resolution.

        Args:
            start_1_16: Fusion output shaped ``(B, 256, D, H, W)``.
            levels: Neck skip features keyed by pyramid scale.
            da: Input SPAD depth-adaptation level, or ``None`` for 2D.

        Returns:
            Decoded 1/4-scale feature map.
        """
        schedule = neck_da_schedule(da)
        order = ['1/8', '1/4']
        prev = start_1_16
        for stage_idx, key in enumerate(order):
            skip = levels[key]
            scale_d = 2 if schedule[stage_idx].upsample_depth else 1
            prev = F.interpolate(
                prev,
                size=(prev.shape[2] * scale_d, skip.shape[3], skip.shape[4]),
                mode='nearest',
            )
            prev = prev + skip
            prev = self.conv_layers[stage_idx](prev, da=schedule[stage_idx].da)
            prev = F.relu(self.norms[stage_idx](prev))
        return prev

class SemanticHead(nn.Module):
    """Semantic path of ``Sam3MaskDecoder``, batched over concepts."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 8, num_upsampling_stages: int = 3):
        """
        Args:
            hidden_size: Feature width.
            num_heads: Prompt cross-attention head count.
            num_upsampling_stages: Number of allocated pixel-decoder stages.
        """
        super().__init__()
        self.prompt_cross_attn = Attention(hidden_size, num_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size)
        self.pixel_decoder = PixelDecoder(hidden_size, num_upsampling_stages)
        self.semantic_projection = nn.Conv3d(hidden_size, 1, kernel_size=1).to(
            memory_format=torch.channels_last_3d,
        )

    def forward(
        self,
        fused_1_16: Tensor,
        levels: dict[str, Tensor],
        text_k: Tensor,
        attn_bias: Tensor | None,
        da: int | None,
    ) -> Tensor:
        """Predict independent mask logits for a batch of concepts.

        Args:
            fused_1_16: Concept-batched fusion features shaped ``(K, 256, D, H, W)``.
            levels: Single-sample neck pyramid broadcast across concepts.
            text_k: Per-concept prompt tokens shaped ``(K, L, 256)``.
            attn_bias: Additive mask on text keys shaped ``(K, 1, 1, L)``,
                or None to leave prompt cross-attn unmasked.
            da: Input SPAD depth-adaptation level, or ``None`` for 2D.

        Returns:
            Mask logits shaped ``(K, 1, D_out, H_out, W_out)`` at 1/4 scale.
        """
        b, c, d, h, w = fused_1_16.shape
        pix = fused_1_16.flatten(2).transpose(1, 2)
        normed = self.prompt_cross_attn_norm(pix)
        attn = self.prompt_cross_attn(query=normed, key=text_k, value=text_k, attn_bias=attn_bias)
        pix = pix + attn
        start = pix.transpose(1, 2).view(b, c, d, h, w)
        pixel_embed = self.pixel_decoder(start, levels, da)
        return self.semantic_projection(pixel_embed)
