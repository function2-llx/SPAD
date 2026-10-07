import torch
from torch.nn import functional as F

from pumit.ucpt.seg.loss import seg_loss, focal_loss, dice_loss, dice_loss_per_class


def test_focal_loss_matches_probability_reference_value_and_gradient():
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 4, 8, 8, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = (torch.rand_like(logits) > 0.6).float()

    actual = focal_loss(logits, targets)
    prob = reference_logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(reference_logits, targets, reduction='none')
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1.0 - targets)
    expected = (alpha_t * (1.0 - pt).square() * ce).mean()
    torch.testing.assert_close(actual, expected)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_focal_dice_positive():
    logits = torch.randn(2, 1, 4, 8, 8)
    target = (torch.rand(2, 1, 4, 8, 8) > 0.5).float()
    out = seg_loss(logits, target, is_positive=True)
    assert set(out) == {'focal', 'dice_sum', 'n_pos', 'k'}
    assert out['focal'].ndim == 0 and out['dice_sum'].ndim == 0
    assert out['n_pos'] == 2
    assert out['k'] == 2


def test_negative_uses_zero_target():
    logits = torch.full((2, 1, 4, 8, 8), -10.0)
    target = (torch.rand(2, 1, 4, 8, 8) > 0.5).float()
    out = seg_loss(logits, target, is_positive=False)
    assert out['n_pos'] == 0
    assert out['dice_sum'].item() < 0.05


def test_dice_all_zero_target_drives_pred_to_zero():
    zero_logits = torch.full((1, 1, 2, 4, 4), -20.0)
    zero_target = torch.zeros(1, 1, 2, 4, 4)
    assert dice_loss(zero_logits, zero_target).item() < 1e-2


def test_seg_loss_batched_equals_per_class_loop():
    """Tensor is_positive path: K * batched(focal) + batched(dice_sum) == sum of per-class calls."""
    torch.manual_seed(0)
    k = 5
    logits = torch.randn(k, 1, 4, 8, 8)
    targets = (torch.rand(k, 1, 4, 8, 8) > 0.7).float()
    is_pos = torch.tensor([True, False, True, True, False])

    batched = seg_loss(logits, targets, is_pos)
    total_batched = k * batched['focal'] + batched['dice_sum']

    total_loop = torch.zeros([])
    for j in range(k):
        out = seg_loss(logits[j:j + 1], targets[j:j + 1], bool(is_pos[j]))
        total_loop = total_loop + out['focal'] + out['dice_sum']

    assert torch.allclose(total_batched, total_loop, atol=1e-6), \
        f'{total_batched.item():.6f} vs {total_loop.item():.6f}'


def test_dice_loss_per_class_mean_matches_scalar():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 8, 8, 8)
    targets = (torch.randn(4, 1, 8, 8, 8) > 0).float()
    assert torch.allclose(dice_loss_per_class(logits, targets).mean(), dice_loss(logits, targets), atol=1e-6)


def test_seg_loss_dice_positive_only():
    torch.manual_seed(0)
    K = 4
    logits = torch.randn(K, 1, 8, 8, 8)
    targets = (torch.randn(K, 1, 8, 8, 8) > 0).float()
    is_positive = torch.tensor([True, False, True, False])

    out = seg_loss(logits, targets, is_positive)
    assert torch.is_tensor(out['n_pos'])
    assert out['n_pos'].device == logits.device
    assert out['n_pos'] == 2
    assert out['k'] == K
    expected = dice_loss_per_class(logits[is_positive], targets[is_positive]).sum()
    assert torch.allclose(out['dice_sum'], expected, atol=1e-6)


def test_seg_loss_tensor_mask_matches_boolean_index_reference_gradient():
    torch.manual_seed(0)
    logits = torch.randn(5, 1, 4, 8, 8, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = (torch.rand_like(logits) > 0.6).float()
    is_positive = torch.tensor([True, False, True, True, False])

    actual = seg_loss(logits, targets, is_positive)['dice_sum']
    expected = dice_loss_per_class(
        reference_logits[is_positive], targets[is_positive],
    ).sum()
    torch.testing.assert_close(actual, expected)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_seg_loss_all_negative_dice_zero_but_differentiable():
    logits = torch.randn(3, 1, 8, 8, 8, requires_grad=True)
    targets = torch.zeros(3, 1, 8, 8, 8)
    out = seg_loss(logits, targets, torch.tensor([False, False, False]))
    assert torch.is_tensor(out['n_pos'])
    assert out['n_pos'] == 0
    assert float(out['dice_sum']) == 0.0
    (out['focal'] + out['dice_sum']).backward()
    assert logits.grad is not None   # graph stayed connected
