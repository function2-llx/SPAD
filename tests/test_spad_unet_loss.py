"""Tests for the shared Universal partial-label loss module."""

from __future__ import annotations

import torch
import pytest

from pumit.spad_unet.loss import (
    REGION_BALANCED_LOSS_NORMALIZATION,
    SAMPLE_MEAN_LOSS_NORMALIZATION,
    SOFTMAX_LABELS,
    UniversalPartialLabelBatch,
    UniversalPartialLabelLoss,
    deep_supervision_weights,
    select_sample_region_logits,
)
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


# --- select_sample_region_logits ---


def test_select_sample_region_logits_shape():
    B, C_full, D, H, W = 4, 41, 8, 8, 8
    logits = [
        torch.randn(B, C_full, D, H, W),
        torch.randn(B, C_full, D // 2, H // 2, W // 2),
    ]
    indices = torch.tensor([0, 5, 10], dtype=torch.long)
    result = select_sample_region_logits(logits, sample_index=2, region_indices=indices)
    assert len(result) == 2
    assert result[0].shape == (1, 3, D, H, W)
    assert result[1].shape == (1, 3, D // 2, H // 2, W // 2)


def test_select_sample_region_logits_values():
    B, C_full, S = 2, 10, 4
    logits = [torch.arange(B * C_full * S**3, dtype=torch.float32).reshape(B, C_full, S, S, S)]
    indices = torch.tensor([3, 7], dtype=torch.long)
    result = select_sample_region_logits(logits, sample_index=1, region_indices=indices)
    expected = logits[0][1:2].index_select(1, indices)
    assert torch.equal(result[0], expected)


def test_select_sample_region_logits_gradient_flows():
    B, C_full, S = 4, 41, 4
    logits_raw = torch.randn(B, C_full, S, S, S, requires_grad=True)
    canonical_logits = [logits_raw]
    indices = torch.tensor([2, 8, 15], dtype=torch.long)
    result = select_sample_region_logits(canonical_logits, sample_index=1, region_indices=indices)
    result[0].sum().backward()
    assert logits_raw.grad is not None
    # Only sample 1, channels [2, 8, 15] should have nonzero grad
    assert logits_raw.grad[0].abs().sum() == 0
    assert logits_raw.grad[1, 2].abs().sum() > 0
    assert logits_raw.grad[1, 8].abs().sum() > 0
    assert logits_raw.grad[1, 15].abs().sum() > 0
    assert logits_raw.grad[1, 0].abs().sum() == 0


# --- UniversalPartialLabelLoss ---


@pytest.fixture
def loss_module():
    return UniversalPartialLabelLoss(batch_dice=False)


def test_loss_matches_manual_calculation(loss_module):
    """Loss module output matches a direct manual DC_and_BCE_loss calculation."""
    torch.manual_seed(42)
    B, C_full, S = 2, 10, 8
    # Two datasets: sample 0 has channels [0,1,2], sample 1 has channels [5,6]
    indices_0 = torch.tensor([0, 1, 2], dtype=torch.long)
    indices_1 = torch.tensor([5, 6], dtype=torch.long)

    canonical_logits = [torch.randn(B, C_full, S, S, S)]
    targets = [
        torch.randint(0, 2, (1, 3, S, S, S)).float(),
        torch.randint(0, 2, (1, 2, S, S, S)).float(),
    ]
    region_indices = [indices_0, indices_1]

    result = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, region_indices),
    )

    # Manual calculation
    criterion = DC_and_BCE_loss(
        {},
        {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
        use_ignore_label=False,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    logits_0 = canonical_logits[0][0:1].index_select(1, indices_0)
    logits_1 = canonical_logits[0][1:2].index_select(1, indices_1)
    loss_0 = criterion(logits_0, targets[0])
    loss_1 = criterion(logits_1, targets[1])
    expected = (loss_0 + loss_1) / 2

    assert torch.allclose(result, expected, atol=1e-6)


def test_partial_label_batch_matches_native_trainer_target_contract(loss_module):
    logits = torch.randn(2, 5, 4, 4, 4)
    target_batch = UniversalPartialLabelBatch(
        [
            torch.randint(0, 2, (1, 2, 4, 4, 4)).float(),
            torch.randint(0, 2, (1, 1, 4, 4, 4)).float(),
        ],
        [
            torch.tensor([0, 2]),
            torch.tensor([4]),
        ],
    )

    moved = target_batch.to(torch.device('cpu'), non_blocking=True)
    result = loss_module(logits, moved)

    assert result.ndim == 0
    assert result.isfinite()


def test_loss_inactive_rows_zero_gradient(loss_module):
    """Rows not active for any sample receive zero gradient."""
    torch.manual_seed(0)
    B, C_full, S = 2, 10, 8
    indices_0 = torch.tensor([0, 1], dtype=torch.long)
    indices_1 = torch.tensor([3, 4], dtype=torch.long)
    inactive_channels = {2, 5, 6, 7, 8, 9}

    raw = torch.randn(B, C_full, S, S, S, requires_grad=True)
    canonical_logits = [raw]
    targets = [
        torch.randint(0, 2, (1, 2, S, S, S)).float(),
        torch.randint(0, 2, (1, 2, S, S, S)).float(),
    ]
    loss = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, [indices_0, indices_1]),
    )
    loss.backward()

    for ch in inactive_channels:
        assert raw.grad[:, ch].abs().sum() == 0, f'channel {ch} should have zero grad'
    # Active channels should have nonzero gradient
    assert raw.grad[0, 0].abs().sum() > 0
    assert raw.grad[1, 3].abs().sum() > 0


def test_loss_variable_channel_counts(loss_module):
    """Works with different region counts per sample in one batch."""
    torch.manual_seed(7)
    B, C_full, S = 4, 41, 4
    # 4 datasets with 2, 13, 1, 4 channels
    channel_counts = [2, 13, 1, 4]
    region_indices = []
    offset = 0
    for count in channel_counts:
        region_indices.append(torch.arange(offset, offset + count, dtype=torch.long))
        offset += count

    canonical_logits = [torch.randn(B, C_full, S, S, S)]
    targets = [torch.randint(0, 2, (1, c, S, S, S)).float() for c in channel_counts]

    result = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, region_indices),
    )
    assert result.ndim == 0
    assert result.isfinite()


def test_loss_means_regions_within_sample_then_samples_within_batch(loss_module):
    """Every dataset sample has one equal-weight unit in the batch loss."""
    torch.manual_seed(99)
    B, C_full, S = 3, 10, 4
    canonical_logits = [torch.randn(B, C_full, S, S, S)]
    region_indices = [
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([3, 4, 5, 6], dtype=torch.long),
    ]
    targets = [
        torch.randint(0, 2, (1, len(indices), S, S, S)).float()
        for indices in region_indices
    ]

    result = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, region_indices),
    )

    # Compute each sample loss individually
    criterion = DC_and_BCE_loss(
        {},
        {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
        use_ignore_label=False,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    sample_losses = []
    for i in range(B):
        logits_i = canonical_logits[0][i:i+1].index_select(1, region_indices[i])
        sample_losses.append(criterion(logits_i, targets[i]))
    expected = torch.stack(sample_losses).mean()
    assert torch.allclose(result, expected, atol=1e-6)


def test_region_balanced_loss_means_all_active_regions():
    """Every active region is one equal-weight unit in the batch loss."""
    torch.manual_seed(101)
    criterion = DC_and_BCE_loss(
        {},
        {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
        use_ignore_label=False,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
    )
    logits = torch.randn(2, 5, 4, 4, 4)
    indices = [torch.tensor([0]), torch.tensor([1, 2, 3])]
    targets = [
        torch.randint(0, 2, (1, 1, 4, 4, 4)).float(),
        torch.randint(0, 2, (1, 3, 4, 4, 4)).float(),
    ]

    result = loss_module(
        logits,
        UniversalPartialLabelBatch(targets, indices),
    )

    first = criterion(logits[0:1, 0:1], targets[0])
    second = criterion(logits[1:2, 1:4], targets[1])
    expected = (first + 3 * second) / 4
    assert torch.allclose(result, expected, atol=1e-6)


def test_fixed_region_count_is_ddp_equivalent():
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
        global_active_region_count=41,
        ddp_world_size=4,
    )

    result = loss_module.mean_sample_loss(
        [torch.tensor(2.0), torch.tensor(5.0)],
        [2, 13],
    )

    assert torch.allclose(result, torch.tensor(28 / 41))


def test_sample_mean_is_ddp_equivalent_with_unequal_rank_counts(
    loss_module,
    monkeypatch,
):
    """DDP rank averaging recovers the single-process global sample mean."""
    local_losses = [
        [torch.tensor(1.0), torch.tensor(6.0)],
        [torch.tensor(11.0)],
        [torch.tensor(10.0), torch.tensor(9.0), torch.tensor(12.0)],
        [torch.tensor(17.0)],
    ]
    global_count = sum(len(losses) for losses in local_losses)
    world_size = len(local_losses)

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: world_size)

    def replace_with_global_count(count, op):
        assert op == torch.distributed.ReduceOp.SUM
        count.fill_(global_count)

    monkeypatch.setattr(torch.distributed, 'all_reduce', replace_with_global_count)
    rank_losses = [
        loss_module.mean_sample_loss(losses)
        for losses in local_losses
    ]

    ddp_mean = torch.stack(rank_losses).mean()
    single_process_mean = torch.stack([
        loss
        for losses in local_losses
        for loss in losses
    ]).mean()
    assert torch.allclose(ddp_mean, single_process_mean)


def test_dynamic_region_count_is_ddp_equivalent_with_unequal_rank_counts(
    monkeypatch,
):
    """Without a fixed count the divisor is the summed active regions across ranks."""
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=REGION_BALANCED_LOSS_NORMALIZATION,
    )
    rank_losses = [[torch.tensor(2.0)], [torch.tensor(5.0)]]
    rank_counts = [[2], [13]]
    world_size = len(rank_losses)

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: world_size)

    def replace_with_global_count(count, op):
        assert op == torch.distributed.ReduceOp.SUM
        count.fill_(sum(sum(counts) for counts in rank_counts))

    monkeypatch.setattr(torch.distributed, 'all_reduce', replace_with_global_count)
    results = [
        loss_module.mean_sample_loss(losses, counts)
        for losses, counts in zip(rank_losses, rank_counts, strict=True)
    ]

    # DDP averages the rank losses, recovering the single-process region mean.
    assert torch.allclose(torch.stack(results).mean(), torch.tensor(7 / 15))


def test_fixed_global_sample_count_avoids_collective_inside_compiled_loss(
    monkeypatch,
):
    """The frozen replay uses a static DDP scale without a runtime collective."""
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        global_sample_count=12,
        ddp_world_size=4,
    )
    monkeypatch.setattr(
        torch.distributed,
        'all_reduce',
        lambda *_args, **_kwargs: pytest.fail('unexpected runtime collective'),
    )
    local_sums = [torch.tensor(7.0), torch.tensor(11.0)]

    result = loss_module.mean_sample_loss(local_sums)

    assert torch.allclose(result, 4 * torch.stack(local_sums).sum() / 12)


def test_loss_deep_supervision_weights():
    """Multi-level deep supervision produces correct weighted sum."""
    torch.manual_seed(3)
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
    )
    B, C_full, S = 1, 5, 16
    canonical_logits = [
        torch.randn(B, C_full, S, S, S),
        torch.randn(B, C_full, S // 2, S // 2, S // 2),
        torch.randn(B, C_full, S // 4, S // 4, S // 4),
    ]
    targets = [torch.randint(0, 2, (1, 3, S, S, S)).float()]
    region_indices = [torch.tensor([0, 1, 2], dtype=torch.long)]

    result = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, region_indices),
    )

    # Manual: weights for 3 levels = [1, 0.5, 0] normalized = [2/3, 1/3, 0]
    weights = deep_supervision_weights(3, canonical_logits[0].device)
    criterion = DC_and_BCE_loss(
        {},
        {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
        use_ignore_label=False,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    logits_0 = canonical_logits[0][0:1].index_select(1, region_indices[0])
    logits_1 = canonical_logits[1][0:1].index_select(1, region_indices[0])
    target_down = torch.nn.functional.interpolate(
        targets[0].float(), size=logits_1.shape[2:], mode='nearest-exact',
    )
    level_losses = torch.stack([
        criterion(logits_0, targets[0]),
        criterion(logits_1, target_down),
        torch.tensor(0.0),
    ])
    expected = (weights * level_losses).sum()
    assert torch.allclose(result, expected, atol=1e-6)


def test_loss_batch_size_mismatch_raises(loss_module):
    B, C, S = 2, 5, 4
    canonical_logits = [torch.randn(B, C, S, S, S)]
    targets = [torch.randn(1, 2, S, S, S)]  # only 1 target for B=2
    region_indices = [torch.tensor([0, 1], dtype=torch.long)] * B
    with pytest.raises(ValueError, match='batch_size mismatch'):
        loss_module(
            canonical_logits,
            UniversalPartialLabelBatch(targets, region_indices),
        )


def test_loss_single_level(loss_module):
    """Works with single DS level (weight = 1.0)."""
    torch.manual_seed(5)
    B, C, S = 1, 8, 4
    canonical_logits = [torch.randn(B, C, S, S, S)]
    targets = [torch.randint(0, 2, (1, 3, S, S, S)).float()]
    region_indices = [torch.tensor([0, 1, 2], dtype=torch.long)]

    result = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch(targets, region_indices),
    )
    assert result.ndim == 0
    assert result.isfinite()


# --- sample_loss and SPAD sequential pattern ---


def test_sample_loss_matches_forward(loss_module):
    """forward preserves the single-sample region mean."""
    torch.manual_seed(77)
    C_full, S = 10, 8
    canonical_logits = [torch.randn(1, C_full, S, S, S)]
    indices = torch.tensor([2, 5, 7], dtype=torch.long)
    target = torch.randint(0, 2, (1, 3, S, S, S)).float()

    # Via forward (full API)
    via_forward = loss_module(
        canonical_logits,
        UniversalPartialLabelBatch([target], [indices]),
    )

    # Via sample_loss (already-active logits)
    active_logits = select_sample_region_logits(canonical_logits, 0, indices)
    via_sample = loss_module.sample_loss(active_logits, target)

    assert torch.allclose(via_forward, via_sample, atol=1e-6)


def test_spad_sequential_pattern(loss_module):
    """SPAD pattern: sequential forwards, global sample mean, one backward."""
    torch.manual_seed(42)
    C_active_list = [2, 4, 1, 3]
    S = 8

    # Simulate 4 sequential forwards producing already-active logits
    conv = torch.nn.Conv3d(1, max(C_active_list), 1)
    sample_losses = []
    for c_active in C_active_list:
        x = torch.randn(1, 1, S, S, S)
        logits = conv(x)[:, :c_active]
        target = torch.randint(0, 2, (1, c_active, S, S, S)).float()
        sample_losses.append(loss_module.sample_loss([logits], target))

    loss = loss_module.mean_sample_loss(sample_losses)
    assert loss.ndim == 0
    assert loss.isfinite()
    loss.backward()
    assert conv.weight.grad is not None
    assert conv.weight.grad.abs().sum() > 0


def test_softmax_sample_loss_matches_native_nnunet_criterion():
    torch.manual_seed(123)
    logits = torch.randn(1, 4, 6, 6, 6)
    labels = torch.randint(0, 4, (1, 1, 6, 6, 6))
    target = torch.cat(
        [labels == value for value in range(1, 4)],
        dim=1,
    ).float()
    loss_module = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
    )

    actual = loss_module.sample_loss(
        [logits],
        target,
        SOFTMAX_LABELS,
    )
    criterion = DC_and_CE_loss(
        {
            'batch_dice': False,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': False,
        },
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    )

    assert torch.allclose(actual, criterion(logits, labels), atol=1e-6)


def test_softmax_sample_loss_supports_deep_supervision():
    labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    labels[:, :, :4] = 1
    labels[:, :, 4:, :4] = 2
    target = torch.cat(
        [labels == value for value in range(1, 3)],
        dim=1,
    ).float()
    logits = [
        torch.randn(1, 3, 8, 8, 8, requires_grad=True),
        torch.randn(1, 3, 4, 4, 4, requires_grad=True),
    ]

    loss = UniversalPartialLabelLoss(
        batch_dice=False,
        loss_normalization=SAMPLE_MEAN_LOSS_NORMALIZATION,
    ).sample_loss(
        logits,
        target,
        SOFTMAX_LABELS,
    )
    loss.backward()

    assert loss.isfinite()
    assert logits[0].grad is not None
