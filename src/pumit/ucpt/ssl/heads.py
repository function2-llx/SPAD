"""UCPT-owned SSL decoders, predictors, and post-encoder losses.

The implementation is copied from ``pumit.ssl`` so UCPT can evolve its LR groups and EMA scope independently.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pumit.model.vit import ViT, ViTConfig


# ---------------------------------------------------------------------------
# Recon decoder (copied from pumit.ssl.MaskedDecoder)
# ---------------------------------------------------------------------------

class ReconDecoder(nn.Module):
    """Masked-token decoder. Copy of pumit.ssl.MaskedDecoder."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        *,
        depth: int = 4,
        num_heads: int = 6,
        latent_channels: int | None = 16,
        output_dim: int | None = None,
    ):
        """
        Args:
            encoder_dim: Input encoder-feature width.
            decoder_dim: Decoder token width.
            depth: Decoder ViT depth.
            num_heads: Decoder attention-head count.
            latent_channels: Reconstruction output width, or ``None`` to disable ``pred_head``.
            output_dim: Feature-projection width, or ``None`` to disable ``output_proj``.
        """
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        decoder_config = ViTConfig(
            hidden_size=decoder_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=decoder_dim * 4,
            rope_theta=100.0,
            pos_embed_rescale=None,
        )
        self.vit = ViT(decoder_config, skip_embed=True)

        self.pred_head = (
            nn.Linear(decoder_dim, latent_channels)
            if latent_channels is not None
            else None
        )
        self.output_proj = (
            nn.Linear(decoder_dim, output_dim)
            if output_dim is not None
            else None
        )

    def forward(
        self,
        visible_emb: Tensor,
        visible_idx: Tensor,
        masked_idx: Tensor,
        mim_decoder_coords: Tensor,
        attn_bias,
    ) -> Tensor:
        """Decode masked positions from packed visible tokens.

        Args:
            visible_emb: Visible encoder features shaped ``(num_visible, encoder_dim)``.
            visible_idx: Decoder-space positions for ``visible_emb``.
            masked_idx: Decoder-space positions to predict.
            mim_decoder_coords: Patch coordinates shaped ``(num_patches, 3)``.
            attn_bias: Block-diagonal attention bias for packed samples.

        Returns:
            Predictions at masked positions.
        """
        x = self.encoder_proj(visible_emb)

        total_patches = mim_decoder_coords.shape[0]
        full_tokens = (
            self.mask_token.squeeze(0)
            .expand(total_patches, -1)
            .to(x.dtype)
            .clone()
        )
        full_tokens.index_copy_(0, visible_idx, x)

        rope = self.vit.rope.compute_from_coords(mim_decoder_coords)

        packed_tokens = full_tokens.unsqueeze(0)
        packed_rope = rope.unsqueeze(0)
        out = self.vit.forward(packed_tokens, packed_rope, attn_bias).squeeze(0)

        masked_out = out.index_select(0, masked_idx)
        if self.pred_head is not None:
            return self.pred_head(masked_out)
        if self.output_proj is not None:
            return self.output_proj(masked_out)
        return masked_out


class PatchDistillDecoder(ReconDecoder):
    """Feature-projection decoder for JEPA / patch-level distillation."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        *,
        depth: int = 8,  # Defaults match the production patch-distill config (scripts/ssl/train.py).
        num_heads: int = 8,
        output_dim: int | None = None,
    ):
        """
        Args:
            encoder_dim: Input encoder-feature width.
            decoder_dim: Decoder token width.
            depth: Decoder ViT depth.
            num_heads: Decoder attention-head count.
            output_dim: Predicted teacher-feature width.
        """
        super().__init__(
            encoder_dim,
            decoder_dim,
            depth=depth,
            num_heads=num_heads,
            latent_channels=None,
            output_dim=output_dim,
        )


# ---------------------------------------------------------------------------
# CLS predictor (extracted from ViTForSSL.__init__)
# ---------------------------------------------------------------------------

class ClsPredictor(nn.Module):
    """Student-only BYOL-style CLS predictor."""

    def __init__(self, embed_dim: int, hidden_dim: int = 4096):
        """
        Args:
            embed_dim: Input and output feature width.
            hidden_dim: Predictor hidden width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ssl_post — lifted from ViTForSSL.forward steps 5-10
# ---------------------------------------------------------------------------

def ssl_post(
    *,
    recon_decoder: ReconDecoder,
    patch_distill_decoder: PatchDistillDecoder,
    cls_predictor: ClsPredictor,
    student_out: Tensor,
    teacher_cls: Tensor,
    teacher_patch_feats: Tensor,
    latents: Tensor,
    student_patch_mask: Tensor,
    view_visible_idx: Tensor,
    view_masked_idx: Tensor,
    view_target_gather_idx: Tensor,
    view_decoder_coords: Tensor,
    view_decoder_attn_bias,
    n_prefix: int,
    n_views: int,
) -> dict[str, Tensor]:
    """Compute reconstruction and I-JEPA distillation losses over all masked views.

    Each student block is one masked view (prefix + visible tokens), encoded once; both decoders consume that
    same visible output. The recon decoder predicts the masked codec latents at the view's masked positions;
    the patch-distill decoder predicts the channel-normalized EMA teacher features at those same positions;
    the CLS predictor distills each view's CLS from the (per-sample) teacher CLS. Distillation uses raw
    student predictions, LayerNorm'd EMA targets, and smooth-L1. Teacher features are passed as tensors,
    keeping the online encoder unrepresentable as the teacher.

    Both decoders share one packing: V blocks per sample (one per view), each the full grid, in
    sample-major/view-minor order. ``view_visible_idx`` and ``view_masked_idx`` partition that decoder
    space; ``view_target_gather_idx`` maps each masked token to unlabeled-local patch index, indexing BOTH
    the latent targets and the (per-sample, view-broadcast) teacher-feature targets.

    Args:
        recon_decoder: Masked latent decoder.
        patch_distill_decoder: Masked teacher-feature predictor.
        cls_predictor: Student CLS predictor.
        student_out: Packed student encoder output (V blocks per sample).
        teacher_cls: No-grad teacher CLS targets, one row per unlabeled sample.
        teacher_patch_feats: No-grad teacher patch targets, unlabeled-local patch order.
        latents: Reconstruction targets for unlabeled patches, unlabeled-local order.
        student_patch_mask: Patch-token mask in packed student output (False on prefix rows).
        view_visible_idx: Decoder-space visible positions.
        view_masked_idx: Decoder-space masked positions.
        view_target_gather_idx: Masked-token -> unlabeled-local index, both decoders' targets.
        view_decoder_coords: Shared decoder coordinates.
        view_decoder_attn_bias: Shared decoder attention bias.
        n_prefix: Prefix-token count per student block.
        n_views: Views per sample (V).

    Returns:
        Reconstruction, image-distillation, and patch-distillation losses as fp32 SUMS (the caller divides
        each by its DDP-global element count).
    """
    # Per-view CLS: one row per student block (= one per view). Prefix rows are the non-patch positions;
    # [::n_prefix] strides to each block's CLS. Order is sample-major/view-minor.
    view_cls = student_out[~student_patch_mask][::n_prefix]

    # Image-level CLS distillation: each view's CLS predicts its sample's teacher CLS. teacher_cls has one row
    # per sample; repeat_interleave to broadcast across that sample's V views (matching the CLS block order).
    pred_cls = cls_predictor(view_cls)
    teacher_cls_views = teacher_cls.repeat_interleave(n_views, dim=0)
    image_distill_loss = F.smooth_l1_loss(
        pred_cls.float(),
        F.layer_norm(teacher_cls_views, (teacher_cls_views.shape[-1],)).float(),
        reduction='sum',
    )

    # Visible patch tokens across all views, in decoder (sample, view)-major order.
    visible_emb = student_out[student_patch_mask]

    # Reconstruction: predict masked latents at each view's masked positions.
    recon_predictions = recon_decoder(
        visible_emb=visible_emb,
        visible_idx=view_visible_idx,
        masked_idx=view_masked_idx,
        mim_decoder_coords=view_decoder_coords,
        attn_bias=view_decoder_attn_bias,
    )
    recon_targets = latents[view_target_gather_idx]
    recon_loss = F.smooth_l1_loss(
        recon_predictions.float(), recon_targets.float(), beta=1.0, reduction='sum',
    )

    # Patch-level JEPA distillation: predict LayerNorm'd teacher features at the same masked positions.
    student_patch_feats = patch_distill_decoder(
        visible_emb=visible_emb,
        visible_idx=view_visible_idx,
        masked_idx=view_masked_idx,
        mim_decoder_coords=view_decoder_coords,
        attn_bias=view_decoder_attn_bias,
    )
    target = teacher_patch_feats.detach()[view_target_gather_idx]
    target = F.layer_norm(target, (target.shape[-1],))
    patch_distill_loss = F.smooth_l1_loss(
        student_patch_feats.float(), target.float(), reduction='sum',
    )

    return {
        'recon': recon_loss,
        'image_distill': image_distill_loss,
        'patch_distill': patch_distill_loss,
    }
