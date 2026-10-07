import torch
import pytest
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from pumit.ucpt.ssl.heads import (
    ReconDecoder,
    PatchDistillDecoder,
    ClsPredictor,
    ssl_post,
)


@pytest.fixture(scope='module')
def device():
    return torch.device('cuda:0')


@pytest.fixture(scope='module')
def dtype():
    return torch.bfloat16


def test_recon_decoder_predicts_masked(device, dtype):
    enc_dim, dec_dim = 1024, 384
    dec = ReconDecoder(
        encoder_dim=enc_dim, decoder_dim=dec_dim,
        depth=2, num_heads=6, latent_channels=16,
    ).to(device=device, dtype=dtype)
    total_patches, n_vis = 100, 75
    visible_emb = torch.randn(n_vis, enc_dim, device=device, dtype=dtype)
    visible_mask = torch.arange(total_patches, device=device) % 4 != 0
    visible_idx = visible_mask.nonzero(as_tuple=True)[0]
    masked_idx = (~visible_mask).nonzero(as_tuple=True)[0]
    coords = torch.randn(total_patches, 3, device=device, dtype=dtype)
    bias = BlockDiagonalMask.from_seqlens([total_patches])
    out = dec(visible_emb, visible_idx, masked_idx, coords, bias)
    assert out.shape == (total_patches - n_vis, 16)

    permuted = dec(visible_emb.flip(0), visible_idx.flip(0), masked_idx.flip(0), coords, bias)
    torch.testing.assert_close(permuted, out.flip(0))


def test_cls_predictor_shape():
    pred = ClsPredictor(embed_dim=1024, hidden_dim=4096)
    x = torch.randn(8, 1024)
    assert pred(x).shape == (8, 1024)


def _make_ssl_post_kwargs(device, dtype):
    """Build the shared ssl_post inputs for the two-view contract.

    2 unlabeled samples, V=2 views each -> 4 student blocks (sample-major/view-minor). Each block is one
    masked view (prefix + visible). Both decoders share one packing of 4 blocks (full grid each); each view's
    masked tokens are its recon/distill targets via view_target_gather_idx. Used by the loss-contract and
    cls-token tests so they stay in lockstep.

    Returns (common_kwargs, enc_dim, num_samples, n_views).
    """
    # Seed: the raw-vs-normalized allclose guard is bf16-sensitive and can flake
    # under full-suite runs where global RNG state varies by test order.
    torch.manual_seed(0)
    enc_dim = 256
    recon = ReconDecoder(
        encoder_dim=enc_dim, decoder_dim=128,
        depth=1, num_heads=4, latent_channels=16,
    ).to(device=device, dtype=dtype)
    patch_dec = PatchDistillDecoder(
        encoder_dim=enc_dim, decoder_dim=128,
        depth=1, num_heads=4, output_dim=enc_dim,
    ).to(device=device, dtype=dtype)
    cls_pred = ClsPredictor(embed_dim=enc_dim, hidden_dim=512).to(
        device=device, dtype=dtype)

    n_grid = 8       # full patch grid per sample
    n_vis = 4        # visible patches per view (masked = n_grid - n_vis = 4)
    n_prefix = 5     # 1 CLS + 4 registers
    num_samples = 2
    n_views = 2
    num_blocks = n_views * num_samples  # 4

    # Student: 4 blocks, each prefix + n_vis visible.
    student_len = num_blocks * (n_prefix + n_vis)
    student_out = torch.randn(student_len, enc_dim, device=device, dtype=dtype)
    # Teacher: 2 samples, each prefix + full grid.
    teacher_len = num_samples * (n_prefix + n_grid)
    teacher_out = torch.randn(teacher_len, enc_dim, device=device, dtype=dtype)

    s_block = n_prefix + n_vis
    spm = torch.zeros(student_len, dtype=torch.bool, device=device)
    for b in range(num_blocks):
        spm[b * s_block + n_prefix:b * s_block + s_block] = True
    t_block = n_prefix + n_grid
    tpm = torch.zeros(teacher_len, dtype=torch.bool, device=device)
    for s in range(num_samples):
        tpm[s * t_block + n_prefix:s * t_block + t_block] = True

    # Shared decoder packing: V blocks per sample over the full grid; view visible = first n_vis, masked rest.
    decoder_total = num_samples * n_views * n_grid
    view_visible_idx = torch.cat([
        torch.arange(blk * n_grid, blk * n_grid + n_vis, device=device)
        for blk in range(num_samples * n_views)
    ])
    view_masked_idx = torch.cat([
        torch.arange(blk * n_grid + n_vis, (blk + 1) * n_grid, device=device)
        for blk in range(num_samples * n_views)
    ])
    # Target gather: each view's masked tokens -> unlabeled-local patch index (per-sample [0,n_grid)).
    gather_parts = []
    for s in range(num_samples):
        for _ in range(n_views):
            gather_parts.append(torch.arange(n_vis, n_grid, device=device) + s * n_grid)
    view_target_gather_idx = torch.cat(gather_parts)

    latents = torch.randn(num_samples * n_grid, 16, device=device, dtype=dtype)
    view_decoder_coords = torch.randn(decoder_total, 3, device=device, dtype=dtype)
    view_decoder_attn_bias = BlockDiagonalMask.from_seqlens([n_grid] * (num_samples * n_views))

    with torch.no_grad():
        teacher_cls = teacher_out[~tpm][::n_prefix]           # one per sample
        teacher_patch_feats = teacher_out[tpm]                # full grid, unlabeled-local order

    common_kwargs = dict(
        recon_decoder=recon,
        patch_distill_decoder=patch_dec,
        cls_predictor=cls_pred,
        student_out=student_out,
        teacher_cls=teacher_cls,
        teacher_patch_feats=teacher_patch_feats,
        latents=latents,
        student_patch_mask=spm,
        view_visible_idx=view_visible_idx,
        view_masked_idx=view_masked_idx,
        view_target_gather_idx=view_target_gather_idx,
        view_decoder_coords=view_decoder_coords,
        view_decoder_attn_bias=view_decoder_attn_bias,
        n_prefix=n_prefix,
        n_views=n_views,
    )
    return common_kwargs, enc_dim, num_samples, n_views


def test_ssl_post_returns_three_losses(device, dtype):
    """The three SSL losses consume the supplied teacher targets over all masked views.

    teacher_patch_feats are precomputed raw backbone features. Passing targets rather than a teacher module
    prevents accidentally encoding teacher inputs with the online encoder.
    """
    common_kwargs, _, _, _ = _make_ssl_post_kwargs(device, dtype)

    out = ssl_post(**common_kwargs)
    assert set(out.keys()) == {'recon', 'image_distill', 'patch_distill'}
    assert torch.isfinite(out['image_distill'])
    assert torch.isfinite(out['recon'])
    assert torch.isfinite(out['patch_distill'])

    total = out['recon'] + out['image_distill'] + out['patch_distill']
    total.backward()

    # 8a0067e bug-class guard: perturb the passed-in teacher target NON-uniformly and confirm
    # patch_distill tracks it (proving ssl_post used the passed-in target, not a re-derived one).
    kwargs2 = dict(common_kwargs)
    with torch.no_grad():
        kwargs2['teacher_patch_feats'] = (
            kwargs2['teacher_patch_feats']
            + torch.randn_like(kwargs2['teacher_patch_feats'])
        )
        out2 = ssl_post(**kwargs2)
    assert not torch.allclose(out2['patch_distill'], out['patch_distill']), \
        'patch_distill_loss did not track the perturbed teacher_patch_feats target'

    # I-JEPA form: a uniform target shift is a LayerNorm no-op, so patch_distill must NOT change.
    with torch.no_grad():
        kwargs_shift = dict(common_kwargs)
        kwargs_shift['teacher_patch_feats'] = common_kwargs['teacher_patch_feats'] + 5.0
        out_shift = ssl_post(**kwargs_shift)
    assert torch.allclose(out_shift['patch_distill'], out['patch_distill'],
                          atol=1e-3, rtol=1e-3), \
        'uniform target shift changed patch_distill — LayerNorm(target) not applied'


def test_recon_smooth_l1_beta_one_sum_and_gradient():
    """Small residuals are quadratic; larger residuals have unit gradient."""
    latents = torch.tensor([[3.0, 0.0], [10.0, 10.0], [1.0, -2.0]])
    predictions = torch.tensor([[1.25, -2.5], [5.0, -3.0]], requires_grad=True)
    out = ssl_post(
        recon_decoder=lambda **_: predictions,
        patch_distill_decoder=lambda **_: torch.zeros(2, 2),
        cls_predictor=torch.nn.Identity(),
        student_out=torch.zeros(2, 2),
        teacher_cls=torch.zeros(1, 2),
        teacher_patch_feats=torch.zeros(3, 2),
        latents=latents,
        student_patch_mask=torch.tensor([False, True]),
        view_visible_idx=torch.tensor([1]),
        view_masked_idx=torch.tensor([0, 2]),
        view_target_gather_idx=torch.tensor([2, 0]),
        view_decoder_coords=torch.zeros(3, 3),
        view_decoder_attn_bias=None,
        n_prefix=1,
        n_views=1,
    )

    torch.testing.assert_close(out['recon'], torch.tensor(4.15625))
    out['recon'].backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[0.25, -0.5], [1.0, -1.0]]))


def test_ssl_post_cls_extraction_stride(device, dtype):
    """Focused stride check: with 2 samples x 2 views (4 student blocks) the CLS extraction must yield exactly
    4 view CLS rows in sample-major/view-minor order, each the block's CLS sentinel.
    """
    enc_dim = 8
    n_vis = 3
    n_prefix = 2  # 1 CLS + 1 register
    num_samples = 2
    n_views = 2
    num_blocks = n_views * num_samples
    block = n_prefix + n_vis

    # Distinct sentinel per block so we can tell which CLS was picked.
    student_out = torch.zeros(num_blocks * block, enc_dim, device=device, dtype=dtype)
    for b in range(num_blocks):
        for p in range(n_prefix):
            student_out[b * block + p] = float(b + 1)

    spm = torch.zeros(num_blocks * block, dtype=torch.bool, device=device)
    for b in range(num_blocks):
        spm[b * block + n_prefix:b * block + block] = True

    # Replicate ssl_post's CLS stride: one row per block.
    view_cls = student_out[~spm][::n_prefix]
    assert view_cls.shape[0] == num_blocks  # 4
    for b in range(num_blocks):
        assert torch.all(view_cls[b] == float(b + 1))
