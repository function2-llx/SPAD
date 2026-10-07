"""Dense focal + dice segmentation loss, per class. Negatives = zero-mask target."""

import torch
from torch import Tensor
from torch.nn import functional as F


def focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """Compute binary focal loss.

    Args:
        logits: Binary mask logits.
        targets: Binary mask targets.
        alpha: Positive-class weighting factor.
        gamma: Focusing exponent.

    Returns:
        Mean focal loss.
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = (-ce).exp()
    alpha_t = alpha * targets + (1. - alpha) * (1. - targets)
    return (alpha_t * ((1. - pt) ** gamma) * ce).mean()


def dice_loss_per_class(logits: Tensor, targets: Tensor) -> Tensor:
    """Soft Dice per class, returns (B,) before mean."""
    probs = logits.flatten(1).sigmoid()
    targets_flat = targets.flatten(1)
    intersection = (probs * targets_flat).sum(1)
    union = probs.sum(1) + targets_flat.sum(1)
    return 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)


def dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean soft Dice loss (scalar)."""
    return dice_loss_per_class(logits, targets).mean()


def seg_loss(mask_logits: Tensor, targets: Tensor, is_positive: Tensor | bool) -> dict:
    """Focal over ALL K classes (mean); Dice over POSITIVE classes only (summed).

    Returns focal (per-class mean scalar), dice_sum (sum over positive classes),
    n_pos (positive count), k (total classes). ``n_pos`` remains a device scalar
    for tensor-valued labels so the caller can reduce it without a host sync.
    Caller builds the two global denominators. Negatives are supervised by focal
    against a zero mask.

    Operands are cast to fp32 before the sigmoid/BCE/products: autocast promotes
    BCE and reductions but NOT sigmoid, so bf16 logits would round the
    intersection/union terms before the fp32 reduction could help.
    """
    mask_logits = mask_logits.float()
    targets = targets.float()
    if isinstance(is_positive, bool):
        target = targets if is_positive else torch.zeros_like(mask_logits)
        positive_weight: Tensor | float = float(is_positive)
        n_pos: Tensor | int = mask_logits.shape[0] if is_positive else 0
    else:
        view = (-1,) + (1,) * (targets.ndim - 1)
        positive_weight = is_positive.to(dtype=mask_logits.dtype)
        target = targets * positive_weight.view(view)
        n_pos = positive_weight.sum()

    focal = focal_loss(mask_logits, target)
    # Weighting the per-class values is equivalent to boolean selection, while
    # avoiding both its data-dependent nonzero kernel and a host-synchronizing
    # branch on the number of positive classes.
    dice_sum = (dice_loss_per_class(mask_logits, targets) * positive_weight).sum()
    return {'focal': focal, 'dice_sum': dice_sum, 'n_pos': n_pos, 'k': mask_logits.shape[0]}
