"""Spacing-regression components for cached global probes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..cached_probe import CachedProbeRun, build_probe_head, train_cached_probe
from ..evaluation import MetricSelection

SPACING_SELECTION = MetricSelection(metric='mae_mean', direction='minimize')


def build_spacing_head(embed_dim: int, head_type: str) -> nn.Module:
    return build_probe_head(embed_dim, 3, head_type)


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss on valid spacing elements."""
    valid = ~torch.isnan(target)
    if not valid.any():
        raise ValueError('spacing objective received a batch without any valid targets')
    return F.l1_loss(prediction[valid], target[valid])


@torch.no_grad()
def compute_spacing_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute per-axis and mean MAE on log-spacing."""
    axis_names = ('depth', 'height', 'width')
    valid = ~torch.isnan(target)
    absolute_error = torch.where(valid, (prediction - target).abs(), torch.zeros_like(prediction))
    total_absolute_error = absolute_error.sum(dim=0)
    total_count = valid.sum(dim=0)

    metrics: dict[str, float] = {}
    axis_maes: list[float] = []
    for axis, name in enumerate(axis_names):
        if total_count[axis] == 0:
            raise ValueError(f'spacing evaluator received no valid {name} targets')
        mae = (total_absolute_error[axis] / total_count[axis]).item()
        metrics[f'mae_{name}'] = mae
        axis_maes.append(mae)
    metrics['mae_mean'] = float(np.mean(axis_maes))
    return metrics


def train_spacing_probe(
    train_features: torch.Tensor,
    train_spacing: torch.Tensor,
    val_features: torch.Tensor,
    val_spacing: torch.Tensor,
    *,
    head_type: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str | torch.device,
    seed: int,
    progress: bool = True,
) -> CachedProbeRun:
    """Train one spacing probe seed and restore the best-val-MAE head."""
    return train_cached_probe(
        train_features,
        train_spacing,
        val_features,
        val_spacing,
        head_factory=lambda input_dim: build_spacing_head(input_dim, head_type),
        objective=masked_l1_loss,
        evaluator=compute_spacing_metrics,
        selection=SPACING_SELECTION,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        seed=seed,
        progress=progress,
    )
