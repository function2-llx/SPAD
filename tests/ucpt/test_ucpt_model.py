import dataclasses

import torch
import pytest
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

import pumit.ucpt.model as model_module
from pumit.model.vit import ViT, ViTConfig
from pumit.ucpt.batch import UCPTBatch, seg_output_grid, seg_supervision_grid
from pumit.ucpt.model import UCPTModel, UCPTOutput, SegDecoderStack
from pumit.ucpt.ssl.heads import (
    ReconDecoder, PatchDistillDecoder,
    ClsPredictor,
)
from tests.ucpt.conftest import cast_ucpt_batch_dtype, use_cpu_attention_reference


EMBED_DIM = 256
N_PREFIX = 5  # 1 CLS + 4 registers
N_LAYERS = 6
TEXT_LEN = 24  # fixed L for text token sequences (K, L, 1152)


def _build_vit():
    cfg = ViTConfig(
        hidden_size=EMBED_DIM, num_hidden_layers=N_LAYERS, num_attention_heads=4,
        intermediate_size=512, patch_size=16, num_register_tokens=4, grad_ckpt=True,
    )
    return ViT(cfg)


def _dtype():
    # B300 (cap 10.3) xformers backends don't support float32 attention
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _build_model():
    vit = _build_vit()
    return UCPTModel(
        vit=vit,
        recon_decoder=ReconDecoder(encoder_dim=EMBED_DIM, decoder_dim=128, depth=1, num_heads=4, latent_channels=16),
        patch_distill_decoder=PatchDistillDecoder(encoder_dim=EMBED_DIM, decoder_dim=128, depth=1, num_heads=4, output_dim=EMBED_DIM),
        cls_predictor=ClsPredictor(embed_dim=EMBED_DIM, hidden_dim=512),
        seg=SegDecoderStack(embed_dim=EMBED_DIM),
    )


def _coords(shape, r):
    D, H, W = shape
    cd = ((2 * (torch.arange(D) + 0.5) / D) - 1) * r
    ch = ((2 * (torch.arange(H) + 0.5) / H) - 1) * r
    cw = ((2 * (torch.arange(W) + 0.5) / W) - 1) * r
    gd, gh, gw = torch.meshgrid(cd, ch, cw, indexing='ij')
    return torch.stack([gd.flatten(), gh.flatten(), gw.flatten()], dim=1)


def _make_batch(n_labeled=1, k=2, n_ssl=1):
    """Hand-build a tiny UCPTBatch: n_ssl (0 or 1) SSL samples (V=2 masked views each) +
    n_labeled seg samples. n_ssl=0 builds a pure-labeled batch (all SSL fields empty),
    mirroring _collate's empty-unlabeled degradation.

    Each SSL sample produces two views over the same (6,4,4)=96 grid; view 0 keeps 48 visible,
    view 1 keeps 24 visible (distinct counts exercise variable per-view packing). Both views feed
    both decoders; each view's masked tokens are its recon/distill targets."""
    assert n_ssl in (0, 1)
    n_prefix = N_PREFIX
    n_views = 2
    n_patches_ssl = 96 * n_ssl  # 6*4*4
    v0_vis = 48 * n_ssl
    v1_vis = 24 * n_ssl
    v0_masked = n_patches_ssl - v0_vis   # 48
    v1_masked = n_patches_ssl - v1_vis   # 72
    total_student_len = (2 * n_prefix + v0_vis + v1_vis) * n_ssl
    total_teacher_len = (n_prefix + n_patches_ssl) * n_ssl
    num_blocks = n_views * n_ssl

    # Student: two view blocks, each [prefix, visible]. Visible = first n_vis patches of the grid.
    student_patch_mask = torch.zeros(total_student_len, dtype=torch.bool)
    student_patch_mask[n_prefix:n_prefix + v0_vis] = True                 # view 0 visible
    student_patch_mask[2 * n_prefix + v0_vis:] = True                     # view 1 visible
    student_patch_gather_idx = torch.cat([
        torch.arange(v0_vis),
        torch.arange(v1_vis),
    ])
    teacher_patch_mask = torch.zeros(total_teacher_len, dtype=torch.bool)
    teacher_patch_mask[n_prefix:] = True

    ssl_shape = (6, 4, 4)
    ssl_coords = _coords(ssl_shape, 1.0) if n_ssl else torch.zeros(0, 3)
    student_coords = torch.cat([
        torch.zeros(n_prefix * n_ssl, 3), ssl_coords[:v0_vis],
        torch.zeros(n_prefix * n_ssl, 3), ssl_coords[:v1_vis],
    ])
    teacher_coords = torch.cat([torch.zeros(n_prefix * n_ssl, 3), ssl_coords])

    # Shared decoder packing: V blocks, each full grid. Integer indices partition decoder space; targets
    # use the complementary masked tokens (unlabeled-local index).
    if n_ssl:
        view_visible_idx = torch.cat([
            torch.arange(v0_vis),
            torch.arange(n_patches_ssl, n_patches_ssl + v1_vis),
        ])
        view_masked_idx = torch.cat([
            torch.arange(v0_vis, n_patches_ssl),
            torch.arange(n_patches_ssl + v1_vis, 2 * n_patches_ssl),
        ])
        view_decoder_coords = torch.cat([ssl_coords, ssl_coords])
        view_target_gather_idx = torch.cat([
            torch.arange(v0_vis, n_patches_ssl),   # view 0 masked
            torch.arange(v1_vis, n_patches_ssl),   # view 1 masked
        ])
        view_decoder_seqlens = [n_patches_ssl, n_patches_ssl]
    else:
        view_visible_idx = torch.zeros(0, dtype=torch.long)
        view_masked_idx = torch.zeros(0, dtype=torch.long)
        view_decoder_coords = torch.zeros(0, 3)
        view_target_gather_idx = torch.zeros(0, dtype=torch.long)
        view_decoder_seqlens = []

    # seg: n_labeled samples, each shape (D,H,W), K=k classes
    seg_shapes = [(6, 4, 4) for _ in range(n_labeled)]
    seg_n_patches = [D * H * W for (D, H, W) in seg_shapes]
    total_seg_len = sum(n_prefix + n for n in seg_n_patches) if n_labeled > 0 else 0
    # Disjoint layout: unlabeled (SSL) patches at [0, n_patches_ssl), labeled
    # (seg) patches at [n_patches_ssl, n_patches_ssl + sum(seg_n_patches)).
    seg_patch_gather_idx = (
        torch.arange(n_patches_ssl, n_patches_ssl + sum(seg_n_patches))
        if n_labeled > 0 else torch.zeros(0, dtype=torch.long)
    )
    seg_patch_mask = torch.zeros(total_seg_len, dtype=torch.bool)
    off = 0
    for n in seg_n_patches:
        seg_patch_mask[off + n_prefix:off + n_prefix + n] = True
        off += n_prefix + n
    if n_labeled > 0:
        seg_coords = torch.cat([
            torch.cat([torch.zeros(n_prefix, 3), _coords(s, 1.0)])
            for s in seg_shapes
        ])
        seg_attn_bias = BlockDiagonalMask.from_seqlens(
            [n_prefix + n for n in seg_n_patches],
        )
    else:
        seg_coords = torch.zeros(0, 3)
        seg_attn_bias = BlockDiagonalMask.from_seqlens([])
    seg_das = [0 for _ in range(n_labeled)]
    text_embeddings = [torch.randn(k, TEXT_LEN, 1152) for _ in range(n_labeled)]
    text_valid_masks = []
    for _ in range(n_labeled):
        m = torch.zeros(k, TEXT_LEN, dtype=torch.bool)
        m[:, :6] = True  # first 6 positions real, rest padding
        text_valid_masks.append(m)
    is_positive = [
        torch.tensor([True, False][:k] + [True] * max(0, k - 2), dtype=torch.bool)
        for _ in range(n_labeled)
    ]
    target_masks = []
    for s, positive in zip(seg_shapes, is_positive):
        D_out, H_out, W_out = seg_output_grid(0, s)
        target = torch.randint(0, 2, (k, D_out, H_out, W_out), dtype=torch.bool)
        target[~positive] = False
        target_masks.append(target)

    # Disjoint patch array: SSL patches then labeled seg patches (no overlap).
    total_patches = n_patches_ssl + (sum(seg_n_patches) if n_labeled > 0 else 0)
    # Unlabeled patches are the prefix [0, n_patches_ssl) in this fixture.
    teacher_patch_gather_idx = torch.arange(0, n_patches_ssl)
    # n_ssl unlabeled SSL samples + n_labeled labeled samples.
    sample_is_labeled = torch.tensor(
        [False] * n_ssl + [True] * n_labeled, dtype=torch.bool,
    )

    return UCPTBatch(
        patches=torch.randn(total_patches, 3, 16, 16, 16),
        latents=torch.randn(n_patches_ssl, 16),
        student_attn_bias=BlockDiagonalMask.from_seqlens(
            [n_prefix + v0_vis, n_prefix + v1_vis] if n_ssl else [],
        ),
        teacher_attn_bias=BlockDiagonalMask.from_seqlens(
            [n_prefix + n_patches_ssl] if n_ssl else [],
        ),
        view_decoder_attn_bias=BlockDiagonalMask.from_seqlens(view_decoder_seqlens),
        student_coords=student_coords, teacher_coords=teacher_coords,
        view_decoder_coords=view_decoder_coords,
        student_patch_mask=student_patch_mask,
        student_patch_gather_idx=student_patch_gather_idx,
        teacher_patch_mask=teacher_patch_mask,
        view_visible_idx=view_visible_idx,
        view_masked_idx=view_masked_idx,
        view_target_gather_idx=view_target_gather_idx,
        total_student_len=total_student_len, total_teacher_len=total_teacher_len,
        num_blocks=num_blocks,
        n_views=n_views,
        teacher_patch_gather_idx=teacher_patch_gather_idx,
        sample_is_labeled=sample_is_labeled,
        seg_patch_gather_idx=seg_patch_gather_idx, seg_patch_mask=seg_patch_mask,
        seg_coords=seg_coords, seg_attn_bias=seg_attn_bias,
        seg_sample_shapes=seg_shapes, seg_das=seg_das,
        text_embeddings=text_embeddings, is_positive=is_positive,
        text_valid_masks=text_valid_masks,
        target_masks=target_masks,
        total_seg_len=total_seg_len,
        n_ssl_patches=n_patches_ssl,
    )


def test_forward_backward_smoke():
    model = _build_model()
    batch = _make_batch(n_labeled=1, k=2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.seg.fusion_grad_ckpt = True
    model.train()
    # JEPA: forward(batch) only — no teacher_temp, no precomputed_targets.
    out = model(batch)
    assert isinstance(out, UCPTOutput)
    assert torch.isfinite(out.loss)
    # prototype path is gone: no patch_teacher_logits / patch_student_logits fields.
    out_fields = {f.name for f in dataclasses.fields(UCPTOutput)}
    assert 'patch_teacher_logits' not in out_fields
    assert 'patch_student_logits' not in out_fields
    # patch_distill_loss is the JEPA SmoothL1 scalar.
    assert torch.isfinite(out.patch_distill_loss)
    out.loss.backward()
    n_grad_student = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    n_grad_twin = sum(
        1
        for twin in (model.teacher_vit, model.ema_seg)
        for p in twin.parameters()
        if p.grad is not None
    )
    assert n_grad_student > 0
    assert n_grad_twin == 0
    seg_param = next(p for p in model.seg.head.parameters() if p.requires_grad)
    assert seg_param.grad is not None, 'seg loss did not reach seg head (joint training broken)'
    text_encoder_grad = model.seg.text_encoder.resizer.weight.grad
    assert text_encoder_grad is not None, 'seg loss did not reach text_encoder.resizer'
    # JEPA: the patch_distill_decoder must receive grad.
    pd_dec_param = next(p for p in model.patch_distill_decoder.parameters() if p.requires_grad)
    assert pd_dec_param.grad is not None, 'patch_distill_loss did not reach patch_distill_decoder (JEPA path broken)'


def test_forward_interpolates_seg_logits_to_full_voxel_target():
    model = _build_model()
    batch = _make_batch(n_labeled=1, k=2)
    target_shape = seg_supervision_grid(0, (6, 4, 4), stride=1)
    target = torch.zeros(2, *target_shape)
    target[0, 16:48, 16:48, 16:48] = 1
    batch.target_masks = [target]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    out = model(batch)
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert model.seg.head.semantic_projection.weight.grad is not None


def test_seg_weight_scales_only_total_loss():
    model = _build_model()
    model.seg_weight = 0.25
    batch = _make_batch(n_labeled=1, k=2, n_ssl=0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()

    out = model(batch)

    expected = model.seg_weight * out.seg_loss
    assert torch.allclose(out.loss.detach(), expected)
    assert out.seg_loss.item() != 0.0
    torch.testing.assert_close(out.seg_loss, out.seg_focal_loss + out.seg_dice_loss)


def test_zero_ssl_component_weights_preserve_metrics_but_remove_ssl_gradients():
    model = _build_model()
    model.recon_weight = 0.0
    model.image_distill_weight = 0.0
    model.patch_distill_weight = 0.0
    batch = _make_batch(n_labeled=1, k=2, n_ssl=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()

    out = model(batch)
    expected = model.seg_weight * out.seg_loss
    torch.testing.assert_close(out.loss.detach(), expected)
    assert out.recon_loss.item() != 0.0
    assert out.image_distill_loss.item() != 0.0
    assert out.patch_distill_loss.item() != 0.0

    out.loss.backward()
    for module in (model.recon_decoder, model.patch_distill_decoder, model.cls_predictor):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and not g.any() for g in grads)
    seg_param = next(p for p in model.seg.head.parameters() if p.requires_grad)
    assert seg_param.grad is not None and seg_param.grad.any()
    vit_param = model.vit.layer[0].attention.q_proj.weight
    assert vit_param.grad is not None and vit_param.grad.any()


def test_ema_update_uses_independent_momenta():
    model = _build_model()
    model.teacher_momentum = 0.5
    model.artifact_momentum = 0.9
    teacher_student = model.vit.layer[0].attention.q_proj.weight
    teacher_twin = model.teacher_vit.layer[0].attention.q_proj.weight
    artifact_student = model.seg.neck.proj10_1.weight
    artifact_twin = model.ema_seg.neck.proj10_1.weight
    with torch.no_grad():
        teacher_student.add_(0.2)
        artifact_student.add_(0.4)
    teacher_before = teacher_twin.clone()
    artifact_before = artifact_twin.clone()
    expected_teacher = torch.lerp(teacher_before, teacher_student, 1 - model.teacher_momentum)
    expected_artifact = torch.lerp(artifact_before, artifact_student, 1 - model.artifact_momentum)

    model.ema_update()

    assert torch.allclose(teacher_twin, expected_teacher)
    assert torch.allclose(artifact_twin, expected_artifact)


def test_frozen_pixel_decoder_pair_unaffected_by_ema():
    """ema_update skips frozen params (requires_grad=False); the teacher's
    frozen 3rd conv should not be lerp'd. Positive control: a non-frozen
    teacher param DID move, confirming ema_update actually ran."""
    model = _build_model()
    # Perturb the student side so the twins have something to move toward
    with torch.no_grad():
        for s, _ in model._ema_pairs():
            for p in s.parameters():
                p.add_(0.1)
    frozen_before = model.ema_seg.head.pixel_decoder.conv_layers[2].weight.clone()
    moved_before = model.ema_seg.neck.proj10_1.weight.clone()
    model.ema_update()
    frozen_after = model.ema_seg.head.pixel_decoder.conv_layers[2].weight
    moved_after = model.ema_seg.neck.proj10_1.weight
    assert torch.equal(frozen_before, frozen_after), \
        'frozen teacher 3rd conv should not be touched by EMA'
    assert not torch.equal(moved_before, moved_after), \
        'non-frozen teacher param should have moved under EMA'


def test_empty_labeled_guard():
    """0-labeled batch -> seg loss 0.0, backward still works."""
    model = _build_model()
    batch = _make_batch(n_labeled=0, k=0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()
    out = model(batch)
    assert torch.isfinite(out.loss)
    assert out.seg_loss.item() == 0.0
    assert out.seg_focal_loss.item() == 0.0
    assert out.seg_dice_loss.item() == 0.0
    out.loss.backward()


def test_empty_unlabeled_guard():
    """Pure-labeled batch (0 unlabeled) -> SSL losses 0.0, no teacher pass,
    segmentation still runs and backward works. Training batches never hit
    this (packer floors) — it's the bench/eval path."""
    model = _build_model()
    batch = _make_batch(n_labeled=2, k=2, n_ssl=0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()
    out = model(batch)
    assert torch.isfinite(out.loss)
    assert out.recon_loss.item() == 0.0
    assert out.image_distill_loss.item() == 0.0
    assert out.patch_distill_loss.item() == 0.0
    assert out.seg_loss.item() != 0.0
    out.loss.backward()
    seg_param = next(p for p in model.seg.head.parameters() if p.requires_grad)
    assert seg_param.grad is not None


def test_seg_block_equals_standalone_forward():
    """Option B regression: a seg block packed alongside a junk distill block
    produces the same output as the seg block alone (block-diagonal independence)."""
    cfg = ViTConfig(
        hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
        intermediate_size=256, patch_size=16, num_register_tokens=2,
    )
    vit = ViT(cfg).eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    vit = vit.to(device=device, dtype=dt)
    n_prefix = vit.n_prefix
    n_p = 8 * 4 * 4
    seg_patches = torch.randn(n_p, 128, device=device, dtype=dt)
    prefix_x = torch.randn(n_prefix, 128, device=device, dtype=dt)
    head_dim = 128 // 4
    prefix_rope = torch.zeros(n_prefix, 2, head_dim, device=device, dtype=dt)
    prefix_rope[:, 0, :] = 1.0
    seg_rope = torch.zeros(n_p, 2, head_dim, device=device, dtype=dt)
    seg_rope[:, 0, :] = 1.0
    seg_block_rope = torch.cat([prefix_rope, seg_rope], 0)

    b_x = torch.cat([prefix_x, seg_patches], 0).unsqueeze(0)
    b_bias = BlockDiagonalMask.from_seqlens([n_prefix + n_p])
    with torch.no_grad():
        b_out = vit(b_x, seg_block_rope.unsqueeze(0), b_bias)
    b_seg = b_out[0, -n_p:, :]

    dvis = n_p // 2
    d_patches = torch.randn(dvis, 128, device=device, dtype=dt)
    d_rope = torch.cat([prefix_rope, seg_rope[:dvis]], 0)
    d_block = torch.cat([prefix_x, d_patches], 0)
    a_x = torch.cat([d_block, torch.cat([prefix_x, seg_patches], 0)], 0).unsqueeze(0)
    a_rope = torch.cat([d_rope, seg_block_rope], 0).unsqueeze(0)
    a_bias = BlockDiagonalMask.from_seqlens([n_prefix + dvis, n_prefix + n_p])
    with torch.no_grad():
        a_out = vit(a_x, a_rope, a_bias)
    a_seg = a_out[0, -n_p:, :]

    assert torch.allclose(a_seg, b_seg, atol=1e-3), \
        f'block-diagonal independence violated: max|diff|={(a_seg - b_seg).abs().max():.3e}'


def test_seg_head_k_batched_equals_per_class_loop():
    """The semantic head batched over K classes equals the per-class loop
    (per-class attention keys; convs/GroupNorm batch-parallel). Guards the
    Region 3 per-sample callee refactor across all da schedule variants."""
    torch.manual_seed(0)
    seg = SegDecoderStack(embed_dim=EMBED_DIM).eval()
    k = 5
    for da in (None, 0, 1, 2, 3, 4):
        depth = 1 if da is None else 4
        feat_3d = torch.randn(1, EMBED_DIM, depth, 6, 5)
        with torch.no_grad():
            levels = seg.neck(feat_3d, da)
            l16 = levels['1/16']
            _, c, d16, h16, w16 = l16.shape
            fused_3d = torch.randn(k, c, d16, h16, w16)
            text_k = torch.randn(k, 1, c)
            attn_bias = torch.zeros(k, 1, 1, 1)
            batched = seg.head(fused_3d, levels, text_k, attn_bias, da)
            looped = torch.cat([
                seg.head(fused_3d[j:j + 1], levels, text_k[j:j + 1], attn_bias[j:j + 1], da)
                for j in range(k)
            ])
        assert batched.shape == looped.shape
        assert torch.allclose(batched, looped, atol=1e-5), \
            f'da={da}: max|diff|={(batched - looped).abs().max():.3e}'


def test_seg_decode_matches_per_class_reference():
    """seg_decode (eager loop + per-sample callee, K-batched head) reproduces
    the pre-refactor per-class-loop implementation. Pins the Region 3 boundary
    move as pure code motion."""
    from pumit.ucpt.seg.pos_embed import sine_pos_embed_3d
    torch.manual_seed(0)
    model = _build_model().eval()
    batch = _make_batch(n_labeled=2, k=3)
    seg_features = torch.randn(1, batch.total_seg_len, EMBED_DIM)

    with torch.no_grad():
        got_logits, got_total_k = model.seg_decode(seg_features, batch)

    # Reference: the pre-refactor implementation (per-class head loop).
    seg = model.seg
    ref_logits: list[torch.Tensor] = []
    ref_total_k = 0
    with torch.no_grad():
        patch_feats = seg_features[0][batch.seg_patch_mask]
        offset = 0
        for i, (shape, da) in enumerate(
            zip(batch.seg_sample_shapes, batch.seg_das)
        ):
            D, H, W = shape
            n_p = D * H * W
            feat_1d = patch_feats[offset:offset + n_p]
            offset += n_p
            feat_3d = feat_1d.reshape(1, D, H, W, EMBED_DIM).permute(0, 4, 1, 2, 3)
            levels = seg.neck(feat_3d, da)
            l16 = levels['1/16']
            _, c, d16, h16, w16 = l16.shape
            vision = l16.reshape(1, c, d16 * h16 * w16).transpose(1, 2)
            pe = sine_pos_embed_3d(d16, h16, w16, c, vision.device).to(vision.dtype)
            text_kv = seg.text_encoder(batch.text_embeddings[i])  # (K_i, L, C)
            K_i = text_kv.shape[0]
            ref_total_k += K_i
            vmask = batch.text_valid_masks[i]
            attn_bias = torch.zeros(
                vmask.shape[0], 1, 1, vmask.shape[1],
                device=text_kv.device, dtype=text_kv.dtype,
            ).masked_fill_(~vmask[:, None, None, :], float('-inf'))
            vision_k = vision.expand(K_i, -1, -1)
            fused = seg.fusion(vision_k, text_kv, pe, attn_bias)
            fused_3d = fused.transpose(1, 2).reshape(K_i, c, d16, h16, w16)
            per_class = [
                seg.head(fused_3d[j:j + 1], levels, text_kv[j:j + 1], attn_bias[j:j + 1], da).squeeze(0)
                for j in range(K_i)
            ]
            ref_logits.append(torch.stack(per_class, dim=0))

    assert got_total_k == ref_total_k == 6
    assert len(got_logits) == len(ref_logits) == 2
    for got, ref in zip(got_logits, ref_logits):
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-5), \
            f'max|diff|={(got - ref).abs().max():.3e}'


def test_text_conditioning_is_query_dependent():
    """With L>1 real tokens, seg output depends on the vision query and the
    fusion cross-attn query projection receives gradient. Guards against the
    old degenerate single-pooled-token path where Q had no effect."""
    torch.manual_seed(0)
    model = _build_model().train()
    K, D, H, W = 3, 6, 4, 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    da = 0

    feat_a = torch.randn(1, EMBED_DIM, D, H, W, device=device, dtype=dt)
    feat_b = torch.randn(1, EMBED_DIM, D, H, W, device=device, dtype=dt)
    text = torch.randn(K, TEXT_LEN, 1152, device=device, dtype=dt)
    mask = torch.zeros(K, TEXT_LEN, dtype=torch.bool, device=device)
    mask[:, :6] = True

    with torch.no_grad():
        out_a = model.seg(feat_a, text, mask, da)
        out_b = model.seg(feat_b, text, mask, da)
    # Different vision features -> different mask logits (query matters).
    assert not torch.allclose(out_a, out_b, atol=1e-5), \
        'seg output is independent of the vision query (degenerate cross-attn)'

    # Backward through one pass: the fusion cross-attn Q proj must receive grad.
    out = model.seg(feat_a, text, mask, da)
    out.sum().backward()
    q_grad = model.seg.fusion.layers[0].cross_attn.q_proj.weight.grad
    assert q_grad is not None and q_grad.norm() > 1e-4, \
        'fusion cross_attn q_proj received no gradient (query path is dead)'


def test_compile_smoke():
    """Compile the training regions, run two steps, and assert finite loss.

    The per-sample segmentation loop stays eager while the fusion encoder and PixelDecoder are compiled as modules.
    """
    if not torch.cuda.is_available():
        pytest.skip('compile smoke needs CUDA')
    import torch._dynamo as dyn
    dyn.config.recompile_limit = 128

    model = _build_model()
    batch = _make_batch(n_labeled=1, k=2)
    device = torch.device('cuda:0')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.seg.fusion_grad_ckpt = True
    model.train()

    model.vit.compile(dynamic=True)
    model.teacher_vit.compile(dynamic=True)
    model.seg.fusion.compile(dynamic=True, fullgraph=True)
    model.seg.head.pixel_decoder.compile(dynamic=True, fullgraph=True)

    model.ssl_post = torch.compile(model.ssl_post, dynamic=True)
    model.seg_loss_sample = torch.compile(model.seg_loss_sample, dynamic=True, fullgraph=True)

    for step in range(2):
        out = model(batch)
        assert torch.isfinite(out.loss), f'step {step}: loss not finite'
        out.loss.backward()
        model.zero_grad(set_to_none=True)


def test_twin_inventory():
    """state_dict contains exactly the 2 twin prefixes; guards silent
    re-widening of the EMA scope and export-of-the-wrong-seg."""
    model = _build_model()
    twin_roots = {
        k.split('.')[0] for k in model.state_dict()
        if k.startswith(('teacher_', 'ema_'))
    }
    assert twin_roots == {'teacher_vit', 'ema_seg'}
    pair_twins = {t for _, t in model._ema_pairs()}
    assert pair_twins == {model.teacher_vit, model.ema_seg}


def test_ema_seg_never_invoked_in_training_forward():
    """ema_seg is a write-only artifact: no submodule of it may run in a
    training forward. The day it gains a reader is the rename-to-teacher_seg
    litmus event."""
    model = _build_model()
    batch = _make_batch(n_labeled=1, k=2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()
    calls: list[str] = []
    for name, mod in model.ema_seg.named_modules():
        mod.register_forward_hook(
            lambda m, i, o, _n=name: calls.append(_n or type(m).__name__)
        )
    out = model(batch)
    out.loss.backward()
    assert not calls, f'ema_seg modules invoked in training forward: {calls[:5]}'


def test_pair_asserts_fire():
    """Construction asserts: param-name misalignment raises; a persistent
    buffer on either side raises."""
    model = _build_model()
    # Name misalignment: wrap a twin so every param name gains a '0.' prefix.
    model.teacher_vit = torch.nn.Sequential(model.teacher_vit)
    with pytest.raises(AssertionError, match='misaligned'):
        model._assert_pair_invariants()

    model2 = _build_model()
    model2.ema_seg.head.register_buffer('running_stat', torch.zeros(3))
    with pytest.raises(AssertionError, match='persistent'):
        model2._assert_pair_invariants()


def test_copy_student_to_twins_resyncs_all_params():
    """The explicit re-sync hatch copies every param, frozen pair included
    (unlike ema_update, which skips frozen student params)."""
    model = _build_model()
    with torch.no_grad():
        for s, _ in model._ema_pairs():
            for p in s.parameters():
                p.add_(0.1)
    frozen_s = model.seg.head.pixel_decoder.conv_layers[2].weight
    frozen_t = model.ema_seg.head.pixel_decoder.conv_layers[2].weight
    assert not torch.equal(frozen_s, frozen_t)  # diverged by the perturbation
    model.copy_student_to_twins()
    for s, t in model._ema_pairs():
        for p_s, p_t in zip(s.parameters(), t.parameters()):
            assert torch.equal(p_s, p_t)


def test_checkpoint_slice_strict_load():
    """state_dict() carries clean teacher_vit.*/ema_seg.* keys (no _orig_mod.),
    and slicing + prefix-strip strict-loads into fresh bare ViT / SegDecoderStack.
    Replaces the old export_artifact round-trip: downstream parses the whole
    resume checkpoint, no separate export file."""
    model = _build_model()
    # Diverge twins from students so the slice provably carries EMA weights,
    # not the noisy student.
    with torch.no_grad():
        for s, t in model._ema_pairs():
            for p_s, p_t in zip(s.parameters(), t.parameters()):
                p_t.add_(0.1)
    model.teacher_vit.compile(dynamic=True)
    sd = model.state_dict()

    # Clean keys: no _orig_mod. leakage from the compile swap (substring, not
    # prefix — the compile prefix is nested as teacher_vit._orig_mod.<...>).
    assert not any('_orig_mod.' in k for k in sd)

    # Slice the EMA twins (not the student vit.*/seg.*).
    encoder_sd = {k[len('teacher_vit.'):]: v for k, v in sd.items() if k.startswith('teacher_vit.')}
    seg_decoder_sd = {k[len('ema_seg.'):]: v for k, v in sd.items() if k.startswith('ema_seg.')}
    assert encoder_sd and seg_decoder_sd, 'slice produced empty dict (prefix drift)'

    fresh_vit = _build_vit()
    fresh_vit.load_state_dict(encoder_sd, strict=True)
    fresh_seg = SegDecoderStack(embed_dim=EMBED_DIM)
    fresh_seg.load_state_dict(seg_decoder_sd, strict=True)

    # The sliced EMA weights match the twin (not the student).
    for p_twin, p_fresh in zip(model.teacher_vit.parameters(), fresh_vit.parameters()):
        assert torch.equal(p_twin, p_fresh), 'sliced encoder != teacher_vit twin'
    # The sliced seg-decoder weights match the twin (not the student).
    for p_twin, p_fresh in zip(model.ema_seg.parameters(), fresh_seg.parameters()):
        assert torch.equal(p_twin, p_fresh), 'sliced seg_decoder != ema_seg twin'


def test_aux_heads_removed():
    """SAM 3 has no aux supervision: the model carries no aux_heads, no aux_loss
    in output, and seg_decode returns (seg_logits, total_k) without hidden_states."""
    import dataclasses
    from pumit.ucpt.model import UCPTOutput
    model = _build_model()
    assert not hasattr(model, 'aux_heads'), 'aux_heads ModuleDict must be removed'
    assert not hasattr(model, 'aux_layer_indices'), 'aux_layer_indices arg must be removed'
    assert not hasattr(model, 'aux_loss_weights'), 'aux_loss_weights arg must be removed'
    out_fields = {f.name for f in dataclasses.fields(UCPTOutput)}
    assert 'aux_loss' not in out_fields, 'UCPTOutput.aux_loss must be removed'
    # batch contract: no aux_target_masks field
    from pumit.ucpt.batch import UCPTBatch
    batch_fields = {f.name for f in dataclasses.fields(UCPTBatch)}
    assert 'aux_target_masks' not in batch_fields, 'UCPTBatch.aux_target_masks must be removed'


def test_strict_resume_roundtrip():
    """state_dict -> fresh model -> load_state_dict(strict=True) -> twins
    bitwise equal. Twins are diverged from students first so the test proves
    twin values actually travel through the checkpoint."""
    model = _build_model()
    with torch.no_grad():
        for s, _ in model._ema_pairs():
            for p in s.parameters():
                p.add_(0.1)
    model.ema_update()  # twins now differ from both init and students
    sd = model.state_dict()

    fresh = _build_model()
    fresh.load_state_dict(sd, strict=True)
    for (s, t), (fs, ft) in zip(model._ema_pairs(), fresh._ema_pairs()):
        for p_t, p_ft in zip(t.parameters(), ft.parameters()):
            assert torch.equal(p_t, p_ft)
        for p_s, p_fs in zip(s.parameters(), fs.parameters()):
            assert torch.equal(p_s, p_fs)


def test_count_contract_single_process(monkeypatch):
    """All five local sums use their exact element or class counts."""
    use_cpu_attention_reference(monkeypatch)
    model = _build_model()
    batch = _make_batch(n_labeled=2, k=2, n_ssl=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()

    captured = {}
    original_reduce = model_module.start_global_count_reduce

    def spy_reduce(counts, device):
        captured['counts'] = [int(c.item()) if torch.is_tensor(c) else c for c in counts]
        return original_reduce(counts, device)

    original_sum_reduce = model_module.reduce_detached_sums

    def spy_sum_reduce(detached_sums):
        captured['sums'] = [float(s.detach()) for s in detached_sums]
        return original_sum_reduce(detached_sums)

    monkeypatch.setattr(model_module, 'start_global_count_reduce', spy_reduce)
    monkeypatch.setattr(model_module, 'reduce_detached_sums', spy_sum_reduce)
    out = model(batch)

    n_views = batch.n_views
    n_view_targets = batch.view_target_gather_idx.numel()
    expected_counts = [
        n_view_targets * batch.latents.shape[-1],
        n_views * int((~batch.sample_is_labeled).sum()) * model.vit.embed_dim,
        n_view_targets * model.vit.embed_dim,
        sum(text.shape[0] for text in batch.text_embeddings),
        sum(int(positive.sum()) for positive in batch.is_positive),
    ]
    assert captured['counts'] == expected_counts

    denominators = [max(count, 1) for count in expected_counts]
    expected_means = {
        'recon': captured['sums'][0] / denominators[0],
        'image_distill': captured['sums'][1] / denominators[1],
        'patch_distill': captured['sums'][2] / denominators[2],
        'seg_focal': captured['sums'][3] / denominators[3],
        'seg_dice': captured['sums'][4] / denominators[4],
        'seg': captured['sums'][3] / denominators[3] + captured['sums'][4] / denominators[4],
    }
    for term, expected in expected_means.items():
        got = getattr(out, f'{term}_loss').item()
        assert abs(got - expected) < 1e-4 * (abs(expected) + 1e-6), \
            f'{term}: mean {got} != sum/count {expected} (count contract violated)'


def test_nonlogging_forward_skips_detached_metric_reduce(monkeypatch):
    model = _build_model()
    batch = _make_batch(n_labeled=2, k=2, n_ssl=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dt = _dtype()
    model = model.to(device=device, dtype=dt)
    batch.to(device)
    cast_ucpt_batch_dtype(batch, dt)
    model.train()

    def unexpected_reduce(_):
        raise AssertionError('detached metric sums should not be reduced on a non-logging step')

    monkeypatch.setattr(model_module, 'reduce_detached_sums', unexpected_reduce)
    out = model(batch, reduce_metrics=False)
    out.loss.backward()
    assert torch.isfinite(out.loss)
