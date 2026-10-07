"""Joint self-supervised and text-conditioned segmentation for UCPT.

The online modules stay flat. ``teacher_vit`` provides EMA targets during training, while ``ema_seg`` is maintained
only for downstream export. Their momentum schedules are controlled independently.

The training entry point compiles ``vit``, ``teacher_vit``, ``ssl_post``, the fusion encoder, ``PixelDecoder``, and the
dense segmentation loss. Batch packing and per-sample segmentation orchestration remain eager, and this module does
not apply ``torch.compile`` itself.
"""

from __future__ import annotations as _

from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils import checkpoint as torch_checkpoint

from pumit.model.vit import ViT
from pumit.ucpt.batch import UCPTBatch
from pumit.ucpt.ddp_mean import (
    finish_global_count_reduce,
    reduce_detached_sums,
    start_global_count_reduce,
)
from pumit.ucpt.seg import SPADNeck, FusionEncoder, SemanticHead, seg_loss, TextEncoder
from pumit.ucpt.seg.pos_embed import sine_pos_embed_3d
from pumit.ucpt.ssl.heads import (
    ReconDecoder, PatchDistillDecoder,
    ClsPredictor, ssl_post,
)


class TeacherTargets(NamedTuple):
    """No-grad targets produced by the EMA teacher.

    Attributes:
        teacher_cls: Image-level CLS features.
        teacher_patch_feats: Raw backbone features at masked patch positions;
            ``ssl_post`` normalizes them before the smooth-L1 loss.
    """

    teacher_cls: Tensor
    teacher_patch_feats: Tensor


@dataclass
class UCPTOutput:
    """Joint loss and detached per-task metrics.

    Attributes:
        loss: Differentiable total loss.
        recon_loss: Latent reconstruction loss.
        image_distill_loss: Image-level distillation loss.
        patch_distill_loss: Patch-level distillation loss.
        seg_loss: Segmentation loss.
        seg_focal_loss: Focal component, averaged over all concepts.
        seg_dice_loss: Dice component, averaged over positive concepts.
    """

    loss: Tensor
    recon_loss: Tensor
    image_distill_loss: Tensor
    patch_distill_loss: Tensor
    seg_loss: Tensor
    seg_focal_loss: Tensor
    seg_dice_loss: Tensor


def _prefix_tokens(vit: ViT) -> Tensor:
    """Concatenate the CLS and register tokens for one packed block.

    Args:
        vit: ViT that owns the prefix tokens.

    Returns:
        Prefix tokens shaped ``(n_prefix, C)``.
    """
    return torch.cat(
        [
            vit.embeddings.cls_token[0],
            vit.embeddings.register_tokens[0],
        ], dim=0,
    )


class SegDecoderStack(nn.Module):
    """Text-conditioned decoder copied and exported as one EMA unit.

    Groups the text projection, feature neck, fusion encoder, and semantic head under the same EMA and export boundary.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        text_embed_dim: int = 1152,
        hidden_size: int = 256,
        fusion_grad_ckpt: bool = False,
    ):
        """
        Args:
            embed_dim: ViT patch-feature width.
            text_embed_dim: Input text-embedding width.
            hidden_size: Decoder feature width.
            fusion_grad_ckpt: Whether to activation-checkpoint the fusion encoder.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_grad_ckpt = fusion_grad_ckpt
        self.text_encoder = TextEncoder(text_embed_dim=text_embed_dim, hidden_size=hidden_size)
        self.neck = SPADNeck(in_channels=embed_dim, hidden_size=hidden_size)
        self.fusion = FusionEncoder(
            hidden_size=hidden_size, num_heads=8,
            intermediate_size=2048, num_layers=6,
        )
        self.head = SemanticHead(
            hidden_size=hidden_size, num_heads=8, num_upsampling_stages=3,
        )
        # Freeze the dead pixel-decoder 3rd conv+norm pair (load-3-run-2 SAM-3 mirror)
        pd = self.head.pixel_decoder
        for p in pd.conv_layers[2].parameters():
            p.requires_grad_(False)
        for p in pd.norms[2].parameters():
            p.requires_grad_(False)

    def forward(
        self,
        feat_3d: Tensor,
        text_raw: Tensor,
        valid_mask: Tensor,
        da: int | None,
    ) -> Tensor:
        """Decode all concepts for one image.

        Args:
            feat_3d: Patch features shaped ``(1, C, D, H, W)``.
            text_raw: Text token embeddings shaped ``(K, L, text_embed_dim)``.
            valid_mask: Bool token mask shaped ``(K, L)``.
            da: Sample SPAD depth-adaptation level, or ``None`` for 2D.

        Returns:
            Per-concept mask logits shaped ``(K, 1, D_out, H_out, W_out)``.
        """
        levels = self.neck(feat_3d, da)
        l16 = levels['1/16']
        _, c, d16, h16, w16 = l16.shape
        vision = l16.reshape(1, c, d16 * h16 * w16).transpose(1, 2)  # (1, N, C)

        # Full-width 3D sine PE: depth folded into the H/W angles (own freq family), so c (=seg hidden,
        # 256) need only be divisible by 4 and fills exactly with no zero-pad.
        pe = sine_pos_embed_3d(d16, h16, w16, c, vision.device).to(vision.dtype)
        text_kv = self.text_encoder(text_raw)  # (K, L, C)
        k = text_kv.shape[0]
        # Additive attn bias: 0 on real keys, -inf on pad. (K,1,1,L) broadcasts
        # over heads and all N queries (padding is a text property, not per-query).
        attn_bias = torch.zeros(
            valid_mask.shape[0], 1, 1, valid_mask.shape[1],
            device=text_kv.device, dtype=text_kv.dtype,
        ).masked_fill_(~valid_mask[:, None, None, :], float('-inf'))

        # Fusion batched over K classes
        vision_k = vision.expand(k, -1, -1)  # (K, N, C)
        if self.training and self.fusion_grad_ckpt:
            fused = torch_checkpoint.checkpoint(
                self.fusion,
                vision_k,
                text_kv,
                pe,
                attn_bias,
                use_reentrant=False,
            )
        else:
            fused = self.fusion(vision_k, text_kv, pe, attn_bias)
        fused_3d = fused.transpose(1, 2).reshape(k, c, d16, h16, w16)

        # Semantic head batched over K; levels stay B=1 and broadcast.
        return self.head(fused_3d, levels, text_kv, attn_bias, da)  # (K, 1, ...)


class UCPTModel(nn.Module):
    """Shared ViT with SSL and segmentation heads plus frozen EMA twins.

    ``teacher_vit`` supplies self-supervised targets during training.
    ``ema_seg`` is updated for downstream export and is not used in forward.
    """

    def __init__(
        self,
        *,
        vit: ViT,
        recon_decoder: ReconDecoder,
        patch_distill_decoder: PatchDistillDecoder,
        cls_predictor: ClsPredictor,
        seg: SegDecoderStack,
        recon_weight: float = 1.0,
        image_distill_weight: float = 1.0,
        patch_distill_weight: float = 1.0,
        seg_weight: float = 1.0,
        teacher_momentum: float = 0.996,
        artifact_momentum: float = 0.996,
    ):
        """
        Args:
            vit: Online ViT shared by SSL and segmentation.
            recon_decoder: Masked latent reconstruction decoder.
            patch_distill_decoder: Masked patch-feature predictor.
            cls_predictor: Student-side CLS predictor.
            seg: Text-conditioned segmentation decoder.
            recon_weight: Latent reconstruction loss weight.
            image_distill_weight: Image-level distillation loss weight.
            patch_distill_weight: Patch-level distillation loss weight.
            seg_weight: Segmentation loss weight.
            teacher_momentum: EMA momentum for the training-time ViT teacher.
            artifact_momentum: EMA momentum for the downstream segmentation artifact.
        """
        super().__init__()
        self.vit = vit
        self.recon_decoder = recon_decoder
        self.patch_distill_decoder = patch_distill_decoder
        self.cls_predictor = cls_predictor
        self.seg = seg

        self.recon_weight = recon_weight
        self.image_distill_weight = image_distill_weight
        self.patch_distill_weight = patch_distill_weight
        self.seg_weight = seg_weight
        self.teacher_momentum = teacher_momentum
        self.artifact_momentum = artifact_momentum

        # Pretrained weights must be loaded before these copies are created.
        self.teacher_vit = deepcopy(vit).requires_grad_(False)
        self.ema_seg = deepcopy(seg).requires_grad_(False)

        self._assert_pair_invariants()

    def _ema_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        """Return the online and EMA modules in update order.

        Returns:
            ``(online, ema)`` module pairs.
        """
        return [
            (self.vit, self.teacher_vit),
            (self.seg, self.ema_seg),
        ]

    def _ema_specs(self) -> list[tuple[nn.Module, nn.Module, float]]:
        """Return each online/EMA pair with its momentum."""
        return [
            (self.vit, self.teacher_vit, self.teacher_momentum),
            (self.seg, self.ema_seg, self.artifact_momentum),
        ]

    def _assert_pair_invariants(self):
        for s, t in self._ema_pairs():
            s_names = [n for n, _ in s.named_parameters()]
            t_names = [n for n, _ in t.named_parameters()]
            if s_names != t_names:
                if len(s_names) == len(t_names):
                    divergence = next(
                        (a, b) for a, b in zip(s_names, t_names) if a != b
                    )
                else:
                    divergence = (len(s_names), len(t_names))
                raise AssertionError(
                    f'EMA pair param names misaligned between student and twin '
                    f'({type(s).__name__}); positional lerp would silently mix '
                    f'weights. First divergence: {divergence}',
                )
            for m, side in ((s, 'student'), (t, 'twin')):
                persistent = set(m.state_dict()) & {n for n, _ in m.named_buffers()}
                assert not persistent, (
                    f'EMA pair {side} ({type(m).__name__}) has persistent '
                    f'buffers {sorted(persistent)}; parameters()-only EMA '
                    f'would silently fossilize them in the exported artifact'
                )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_vit.eval()
        self.ema_seg.eval()
        return self

    @torch.no_grad()
    def ema_update(self):
        """Update trainable parameters in both EMA twins."""
        # Student-side filter only: twins are wholly requires_grad_(False),
        # an AND-of-both-sides gate would silently no-op the entire EMA.
        for s, t, momentum in self._ema_specs():
            for p_s, p_t in zip(s.parameters(), t.parameters()):
                if p_s.requires_grad:
                    p_t.lerp_(p_s, 1 - momentum)

    @torch.no_grad()
    def copy_student_to_twins(self):
        """Copy all student parameters to their EMA twins."""
        for s, t in self._ema_pairs():
            for p_s, p_t in zip(s.parameters(), t.parameters()):
                p_t.copy_(p_s)

    def no_weight_decay(self) -> set[str]:
        """Return parameter names excluded from weight decay.

        Returns:
            Names of biases, normalization parameters, and prefix tokens.
        """
        names: set[str] = set()
        for name, param in self.named_parameters():
            if param.ndim <= 1:
                names.add(name)
        names.add('vit.embeddings.cls_token')
        names.add('vit.embeddings.register_tokens')
        return names

    def state_dict(self, *args, **kwargs):
        # strip torch.compile prefix so checkpoints load uncompiled (matches SSL pattern)
        full = super().state_dict(*args, **kwargs)
        return {k.replace('_orig_mod.', ''): v for k, v in full.items()}

    def ssl_post(
        self,
        student_out: Tensor,
        **kwargs,
    ):
        """Run the SSL heads on encoded student tokens.

        Args:
            student_out: Packed student ViT output shaped ``(L, C)``.
            **kwargs: Targets, masks, coordinates, and attention biases accepted by
                :func:`pumit.ucpt.ssl.heads.ssl_post`.

        Returns:
            Reconstruction and distillation loss sums.
        """
        return ssl_post(
            recon_decoder=self.recon_decoder,
            patch_distill_decoder=self.patch_distill_decoder,
            cls_predictor=self.cls_predictor,
            student_out=student_out,
            **kwargs,
        )

    def seg_loss_sample(
        self,
        mask_logits: Tensor,
        targets: Tensor,
        is_positive: Tensor | bool,
    ) -> tuple[Tensor, Tensor]:
        """Compute dense focal and Dice terms for one labeled sample."""
        mask_logits = mask_logits.float()
        targets = targets.float()
        if mask_logits.shape[-3:] != targets.shape[-3:]:
            mask_logits = F.interpolate(
                mask_logits,
                size=targets.shape[-3:],
                mode='trilinear',
                align_corners=False,
            )
        losses = seg_loss(mask_logits, targets, is_positive)
        return losses['focal'], losses['dice_sum']

    def seg_decode(
        self,
        seg_features: Tensor,
        batch: UCPTBatch,
    ) -> tuple[list[Tensor], int]:
        """Decode labeled samples while batching concepts within each sample.

        Python batch metadata and per-sample orchestration stay eager.

        Args:
            seg_features: Packed segmentation ViT output shaped ``(1, L, C)``.
            batch: UCPT batch containing sample shapes, depth-adaptation levels, and prompts.

        Returns:
            A pair containing one ``(K_i, 1, D, H, W)`` logits tensor per labeled sample and the total number of
            concepts.
        """
        seg_logits: list[Tensor] = []
        total_k = 0
        embed_dim = self.vit.embed_dim

        patch_feats = seg_features[0][batch.seg_patch_mask]  # (total_seg_patches, C)
        offset = 0
        for shape, da, text_raw, valid_mask in zip(
            batch.seg_sample_shapes, batch.seg_das,
            batch.text_embeddings, batch.text_valid_masks,
        ):
            D, H, W = shape
            n_p = D * H * W
            feat_1d = patch_feats[offset:offset + n_p]
            offset += n_p
            feat_3d = feat_1d.reshape(1, D, H, W, embed_dim).permute(0, 4, 1, 2, 3)
            seg_logits.append(self.seg(feat_3d, text_raw, valid_mask, da))
            total_k += text_raw.shape[0]

        return seg_logits, total_k

    def forward(self, batch: UCPTBatch, reduce_metrics: bool = True) -> UCPTOutput:
        """Compute joint SSL and segmentation losses.

        Every cross-rank-varying term is represented as a local sum plus an element count. The count
        all-reduce starts before the model body and overlaps with forward compute. Scaling each local sum by
        ``W / global_count`` makes DDP's own 1/W gradient averaging collapse to
        ``grad(global_sum / global_count)``.

        Args:
            batch: Packed UCPT training batch.
            reduce_metrics: Reduce detached loss numerators for globally identical metric values. Training
                disables this on non-logging steps; it does not affect the loss gradient.

        Returns:
            Value-corrected total loss and detached per-task metrics. Metric values are DDP-global when
            ``reduce_metrics`` is true and rank-local otherwise; the loss gradient is identical either way.
        """
        device = batch.patches.device
        n_prefix = self.vit.n_prefix
        embed_dim = self.vit.embed_dim
        n_views = batch.n_views
        total_k = sum(text.shape[0] for text in batch.text_embeddings)
        total_pos = sum((positive.sum() for positive in batch.is_positive),
                        start=torch.zeros([], device=device, dtype=torch.int64))
        unlabeled_count = (~batch.sample_is_labeled).sum()
        # recon and patch-distill share one per-view masked-target gather (both predict each view's masked
        # tokens): recon over latent channels, distill over embed_dim. image_distill is V CLS rows/sample.
        n_view_targets = batch.view_target_gather_idx.numel()
        local_counts = [
            n_view_targets * batch.latents.shape[-1],
            n_views * unlabeled_count * embed_dim,
            n_view_targets * embed_dim,
            total_k,
            total_pos,
        ]
        pending_counts = start_global_count_reduce(local_counts, device)

        patch_embeds = self.vit.patch_embed(batch.patches, da=0).reshape(-1, embed_dim)
        prefix = _prefix_tokens(self.vit).to(patch_embeds.dtype)
        fp = torch.float32  # losses accumulate in fp32 regardless of the autocast compute dtype

        # Both subsets keep DDP grad coverage complete; guards support single-task use.

        # --- SSL half (teacher + student masked ViT + ssl_post), unlabeled samples only ---
        if batch.n_ssl_patches > 0:
            targets = self.teacher_vit_forward(batch)
            student_out = self.vit_forward(patch_embeds, batch)

            # --- SSL post-encoder (Region 2) ---
            ssl_losses = self.ssl_post(
                student_out=student_out,
                teacher_cls=targets.teacher_cls,
                teacher_patch_feats=targets.teacher_patch_feats,
                latents=batch.latents,
                n_prefix=n_prefix,
                n_views=n_views,
                student_patch_mask=batch.student_patch_mask,
                view_visible_idx=batch.view_visible_idx,
                view_masked_idx=batch.view_masked_idx,
                view_target_gather_idx=batch.view_target_gather_idx,
                view_decoder_coords=batch.view_decoder_coords,
                view_decoder_attn_bias=batch.view_decoder_attn_bias,
            )
            # ssl_post returns fp32 SUMS; counts are the elementwise denominators of the old means.
            recon_sum = ssl_losses['recon']
            image_distill_sum = ssl_losses['image_distill']
            patch_distill_sum = ssl_losses['patch_distill']
        else:
            # No unlabeled samples this batch: SSL contributes nothing to the loss.
            recon_sum = image_distill_sum = patch_distill_sum = torch.zeros([], device=device, dtype=fp)

        # --- Seg forward + decoder (Region 3, eager) ---
        if batch.total_seg_len > 0:
            packed_seg = torch.empty(
                batch.total_seg_len, embed_dim,
                device=device, dtype=patch_embeds.dtype,
            )
            packed_seg[batch.seg_patch_mask] = patch_embeds[batch.seg_patch_gather_idx]
            packed_seg[~batch.seg_patch_mask] = prefix.repeat(len(batch.seg_sample_shapes), 1)
            seg_rope = self.vit.rope.compute_from_coords(batch.seg_coords)
            # seg_out is (1, total_seg_len, C): .unsqueeze(0) adds the batch dim
            # inside forward. seg_decode indexes seg_out[0] to drop it.
            seg_out = self.vit(
                packed_seg.unsqueeze(0),
                seg_rope.unsqueeze(0),
                batch.seg_attn_bias,
            )
            seg_logits_list, _ = self.seg_decode(seg_out, batch)
            # Focal is a per-class mean over ALL classes; Dice is a mean over
            # POSITIVE classes only (negatives' Dice grad vanishes with volume).
            # Accumulate both numerators; total_k / total_pos are their denominators.
            seg_focal_sum = torch.zeros([], device=device, dtype=fp)
            seg_dice_sum = torch.zeros([], device=device, dtype=fp)
            for i, seg_logits_i in enumerate(seg_logits_list):
                tgt = batch.target_masks[i].unsqueeze(1).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )  # (K_i, 1, D, H, W)
                focal, dice_sum = self.seg_loss_sample(seg_logits_i, tgt, batch.is_positive[i])
                seg_focal_sum = seg_focal_sum + seg_logits_i.shape[0] * focal
                seg_dice_sum = seg_dice_sum + dice_sum
        else:
            # No labeled samples this batch: segmentation contributes nothing to the loss.
            seg_focal_sum = seg_dice_sum = torch.zeros([], device=device, dtype=fp)

        # Denominators have been in flight since the start of forward. Detached numerators are globally reduced
        # only when their values are consumed for logging; they never participate in the backward gradient.
        n_local, n_global, world = finish_global_count_reduce(pending_counts)
        detached_sums = [recon_sum, image_distill_sum, patch_distill_sum, seg_focal_sum, seg_dice_sum]
        if reduce_metrics:
            metric_counts = n_global
            metric_sums = reduce_detached_sums(detached_sums)
        else:
            metric_counts = n_local
            metric_sums = torch.stack([s.detach().to(torch.float64) for s in detached_sums])

        # Scale each differentiable local sum by W / global_count; cast the fp64 factor back to the loss
        # dtype so the backward graph stays fp32 (an fp64 factor would silently promote it).
        def _scale(local_sum: Tensor, idx: int) -> Tensor:
            return local_sum * (world / n_global[idx]).to(fp)

        ssl_total = (
            self.recon_weight * _scale(recon_sum, 0)
            + self.image_distill_weight * _scale(image_distill_sum, 1)
            + self.patch_distill_weight * _scale(patch_distill_sum, 2)
        )
        seg_total = _scale(seg_focal_sum, 3) + _scale(seg_dice_sum, 4)
        backward_loss = ssl_total + self.seg_weight * seg_total

        # Global per-term means (detached) for metrics and the value-corrected total.
        recon_mean = metric_sums[0] / metric_counts[0]
        image_distill_mean = metric_sums[1] / metric_counts[1]
        patch_distill_mean = metric_sums[2] / metric_counts[2]
        seg_focal_mean = metric_sums[3] / metric_counts[3]
        seg_dice_mean = metric_sums[4] / metric_counts[4]
        seg_mean = seg_focal_mean + seg_dice_mean
        global_obj = (
            self.recon_weight * recon_mean
            + self.image_distill_weight * image_distill_mean
            + self.patch_distill_weight * patch_distill_mean
            + self.seg_weight * seg_mean
        ).to(fp)

        # Value-corrected surrogate: grad(loss) == grad(backward_loss), loss.value == global_obj
        # (identical on all ranks, safe to log).
        loss = backward_loss + (global_obj - backward_loss.detach())
        return UCPTOutput(
            loss=loss,
            recon_loss=recon_mean.to(fp),
            image_distill_loss=image_distill_mean.to(fp),
            patch_distill_loss=patch_distill_mean.to(fp),
            seg_loss=seg_mean.to(fp),
            seg_focal_loss=seg_focal_mean.to(fp),
            seg_dice_loss=seg_dice_mean.to(fp),
        )

    @torch.no_grad()
    def teacher_vit_forward(self, batch: UCPTBatch) -> TeacherTargets:
        """Encode unmasked unlabeled tokens with the EMA ViT.

        The teacher runs its own EMA patch-embed stem (a full EMA copy, deepcopied and lerp'd every step), so the target
        frontend lags the online model rather than tracking it instantly. ``patch_embed`` is per-patch independent, so
        only the teacher's gathered patches are embedded.

        Args:
            batch: Packed UCPT batch with teacher masks, coordinates, and raw patches.

        Returns:
            CLS and masked-position patch targets.
        """
        embed_dim = self.vit.embed_dim
        n_prefix = self.vit.n_prefix
        device = batch.patches.device
        teacher_patches = batch.patches[batch.teacher_patch_gather_idx]
        ssl_tokens = self.teacher_vit.patch_embed(teacher_patches, da=0).reshape(-1, embed_dim)
        packed_teacher = torch.empty(
            batch.total_teacher_len, embed_dim,
            device=device, dtype=ssl_tokens.dtype,
        )
        packed_teacher[batch.teacher_patch_mask] = ssl_tokens
        t_prefix = _prefix_tokens(self.teacher_vit).to(ssl_tokens.dtype)
        # One teacher block per unlabeled sample (student has n_views blocks per sample).
        num_teacher_blocks = batch.num_blocks // batch.n_views
        packed_teacher[~batch.teacher_patch_mask] = t_prefix.repeat(
            num_teacher_blocks, 1,
        )
        teacher_rope = self.teacher_vit.rope.compute_from_coords(batch.teacher_coords,)
        teacher_out = self.teacher_vit(
            packed_teacher.unsqueeze(0),
            teacher_rope.unsqueeze(0),
            batch.teacher_attn_bias,
        ).squeeze(0)
        teacher_cls = teacher_out[~batch.teacher_patch_mask][::n_prefix]
        # Full-grid patch features in unlabeled-local order; ssl_post gathers each view's masked
        # targets via view_target_gather_idx (a token masked in multiple views is a target per view).
        teacher_patch_feats = teacher_out[batch.teacher_patch_mask]
        return TeacherTargets(teacher_cls, teacher_patch_feats)

    def vit_forward(self, patch_embeds: Tensor, batch: UCPTBatch) -> Tensor:
        """Encode packed distillation and reconstruction views.

        Args:
            patch_embeds: Online patch embeddings for every sample, shaped ``(num_patches, C)``.
            batch: Packed UCPT batch with student masks and coordinates.

        Returns:
            Packed student ViT output shaped ``(total_student_len, C)``.
        """
        device = patch_embeds.device
        packed_student = torch.empty(
            batch.total_student_len, self.vit.embed_dim,
            device=device, dtype=patch_embeds.dtype,
        )
        packed_student[batch.student_patch_mask] = patch_embeds[batch.student_patch_gather_idx]
        prefix = _prefix_tokens(self.vit).to(patch_embeds.dtype)
        packed_student[~batch.student_patch_mask] = prefix.repeat(batch.num_blocks, 1)
        student_rope = self.vit.rope.compute_from_coords(batch.student_coords)
        return self.vit(
            packed_student.unsqueeze(0),
            student_rope.unsqueeze(0),
            batch.student_attn_bias,
        ).squeeze(0)
