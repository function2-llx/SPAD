import pytest
import torch

from pumit.ucpt.batch import UCPTBatch, seg_output_grid, seg_supervision_grid


def _make_seg_batch(n_labeled=2, k_per_sample=(3, 1), device='cpu'):
    """Hand-build the seg extension of UCPTBatch with synthetic tensors."""
    n_prefix = 5
    # per-sample shapes: (D, H, W) patch grids, small for tests
    shapes = [(4, 6, 6), (8, 6, 6)][:n_labeled]
    n_patches = [D * H * W for (D, H, W) in shapes]
    total_seg_len = sum(n_prefix + n for n in n_patches)

    # gather_idx: contiguous ranges (seg is unmasked, natural order)
    gather_parts = []
    src_off = 0
    for n in n_patches:
        gather_parts.append(torch.arange(src_off, src_off + n))
        src_off += n
    seg_patch_gather_idx = torch.cat(gather_parts)

    mask_parts = []
    for n in n_patches:
        mask_parts.append(torch.zeros(n_prefix, dtype=torch.bool))
        mask_parts.append(torch.ones(n, dtype=torch.bool))
    seg_patch_mask = torch.cat(mask_parts)

    seg_coords = torch.randn(total_seg_len, 3)
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    seg_attn_bias = BlockDiagonalMask.from_seqlens(
        [n_prefix + n for n in n_patches], device=torch.device('cpu'))

    seg_das = [0 for _ in range(n_labeled)]

    text_embeddings = [torch.randn(k, 1152) for k in k_per_sample]
    is_positive = [torch.tensor([True, True, False, False][:k], dtype=torch.bool) for k in k_per_sample]

    target_masks = []
    for (D, H, W), k in zip(shapes, k_per_sample):
        D_out, H_out, W_out = seg_output_grid(0, (D, H, W))
        target_masks.append(torch.zeros(k, D_out, H_out, W_out))

    return UCPTBatch(
        # SSL half (stubs — only seg extension + mover mechanics are tested here)
        patches=torch.randn(src_off, 3, 16, 16, 16),
        latents=torch.randn(src_off, 32),
        student_attn_bias=BlockDiagonalMask.from_seqlens([10]),
        teacher_attn_bias=BlockDiagonalMask.from_seqlens([10]),
        view_decoder_attn_bias=BlockDiagonalMask.from_seqlens([10]),
        student_coords=torch.randn(10, 3),
        teacher_coords=torch.randn(10, 3),
        view_decoder_coords=torch.randn(src_off, 3),
        student_patch_mask=torch.zeros(10, dtype=torch.bool),
        student_patch_gather_idx=torch.zeros(5, dtype=torch.long),
        teacher_patch_mask=torch.zeros(10, dtype=torch.bool),
        view_visible_idx=torch.zeros(0, dtype=torch.long),
        view_masked_idx=torch.arange(5),
        view_target_gather_idx=torch.zeros(5, dtype=torch.long),
        total_student_len=10,
        total_teacher_len=10,
        num_blocks=2,
        n_views=2,
        # seg extension
        seg_patch_gather_idx=seg_patch_gather_idx,
        seg_patch_mask=seg_patch_mask,
        seg_coords=seg_coords,
        seg_attn_bias=seg_attn_bias,
        seg_sample_shapes=shapes,
        seg_das=seg_das,
        text_embeddings=text_embeddings,
        is_positive=is_positive,
        target_masks=target_masks,
        total_seg_len=total_seg_len,
        n_ssl_patches=src_off,
    )


def test_seg_output_grid_da0_isotropic():
    # da=0, patch_grid (4,6,6) -> neck upsamples in-plane 4x and depth 4x (2 stages each)
    # output grid = (16, 24, 24)
    D_out, H_out, W_out = seg_output_grid(0, (4, 6, 6))
    assert (D_out, H_out, W_out) == (16, 24, 24), f'got {(D_out, H_out, W_out)}'


def test_to_moves_small_list_tensors_but_keeps_dense_targets_on_cpu():
    batch = _make_seg_batch()
    target_dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch.to(target_dev)
    for tm in batch.target_masks:
        assert tm.device.type == 'cpu'
    for te in batch.text_embeddings:
        assert te.device == target_dev


def test_to_skips_cpu_metadata_fields():
    batch = _make_seg_batch()
    target_dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch.to(target_dev)
    # Shape and DA lists stay Python-side (metadata device=cpu).
    assert isinstance(batch.seg_sample_shapes, list)
    assert isinstance(batch.seg_das, list)


def test_invariants_n_labeled_consistency():
    batch = _make_seg_batch(n_labeled=2, k_per_sample=(3, 1))
    n = len(batch.seg_sample_shapes)
    assert len(batch.seg_das) == n
    assert len(batch.text_embeddings) == n
    assert len(batch.is_positive) == n
    assert len(batch.target_masks) == n
    for i in range(n):
        k = batch.text_embeddings[i].shape[0]
        assert batch.is_positive[i].shape[0] == k
        assert batch.target_masks[i].shape[0] == k


def test_seg_output_grid_none_da_no_depth_upsample():
    # da=None (2D): no depth upsample; in-plane still 4x
    D_out, H_out, W_out = seg_output_grid(None, (4, 6, 6))
    assert (D_out, H_out, W_out) == (4, 24, 24), f'got {(D_out, H_out, W_out)}'


def test_seg_output_grid_da3_one_depth_upsample():
    # da=3: min(2, 4-3)=1 depth upsample (2x depth), 4x in-plane
    D_out, H_out, W_out = seg_output_grid(3, (4, 6, 6))
    assert (D_out, H_out, W_out) == (8, 24, 24), f'got {(D_out, H_out, W_out)}'


def test_seg_output_grid_da4_no_depth_upsample():
    # da=4: min(2, 0)=0 depth upsample, 4x in-plane only
    D_out, H_out, W_out = seg_output_grid(4, (4, 6, 6))
    assert (D_out, H_out, W_out) == (4, 24, 24), f'got {(D_out, H_out, W_out)}'


@pytest.mark.parametrize(
    ('da', 'stride', 'expected'),
    [
        (None, 1, (1, 96, 96)),
        (0, 2, (32, 48, 48)),
        (1, 2, (32, 48, 48)),
        (2, 2, (16, 48, 48)),
        (4, 1, (4, 96, 96)),
    ],
)
def test_seg_supervision_grid_is_spad_aware(da, stride, expected):
    assert seg_supervision_grid(da, (4 if da is not None else 1, 6, 6), stride=stride) == expected


def test_seg_supervision_grid_rejects_invalid_stride():
    with pytest.raises(ValueError, match='stride'):
        seg_supervision_grid(0, (4, 6, 6), stride=3)


def test_teacher_patch_gather_idx_field_exists():
    import dataclasses
    from pumit.ucpt.batch import UCPTBatch
    names = {f.name for f in dataclasses.fields(UCPTBatch)}
    assert 'teacher_patch_gather_idx' in names


def test_seg_output_grid_negative_da_raises():
    with pytest.raises(AssertionError):
        seg_output_grid(-1, (4, 6, 6))
