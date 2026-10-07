"""Shared optimization loop for probes trained on cached global features."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .evaluation import MetricSelection

Objective = Callable[[Tensor, Tensor], Tensor]
Evaluator = Callable[[Tensor, Tensor], dict[str, float]]
HeadFactory = Callable[[int], nn.Module]


@dataclass
class CachedProbeRun:
    """One seed after restoring the validation-selected head state."""

    model: nn.Module
    best_epoch: int
    best_metrics: dict[str, float]
    history: list[dict]
    best_update: int | None = None


def build_probe_head(input_dim: int, output_dim: int, head_type: str) -> nn.Module:
    """Build the linear/MLP variants used by classification and spacing probes."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError(f'head dimensions must be positive, got input={input_dim}, output={output_dim}')
    hidden_layers = {
        'linear': 0,
        'mlp': 1,
        'mlp2': 2,
        'mlp3': 3,
    }
    if head_type not in hidden_layers:
        raise ValueError(f'unknown probe head {head_type!r}; expected one of {tuple(hidden_layers)}')
    depth = hidden_layers[head_type]
    if depth == 0:
        return nn.Linear(input_dim, output_dim)
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(input_dim, input_dim), nn.GELU()))
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    if not metrics:
        raise ValueError('evaluator returned no metrics')
    normalized = {name: float(value) for name, value in metrics.items()}
    non_finite = {name: value for name, value in normalized.items() if not math.isfinite(value)}
    if non_finite:
        raise ValueError(f'evaluator metrics must be finite, got {non_finite}')
    return normalized


def _full_shuffled_batches(n_train: int, batch_size: int, device: torch.device) -> Iterator[Tensor]:
    """Join consecutive shuffled passes so every update uses a full batch."""
    permutation = torch.randperm(n_train, device=device)
    offset = 0
    while True:
        chunks = []
        remaining = batch_size
        while remaining:
            if offset == n_train:
                permutation = torch.randperm(n_train, device=device)
                offset = 0
            count = min(remaining, n_train - offset)
            chunks.append(permutation[offset:offset + count])
            offset += count
            remaining -= count
        yield torch.cat(chunks)


def train_cached_probe(
    train_features: Tensor,
    train_targets: Tensor,
    val_features: Tensor,
    val_targets: Tensor,
    *,
    head_factory: HeadFactory,
    objective: Objective,
    evaluator: Evaluator,
    selection: MetricSelection,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str | torch.device,
    seed: int,
    progress: bool = True,
    updates: int | None = None,
    eval_every: int | None = None,
) -> CachedProbeRun:
    """Train one cached probe seed and restore the validation-selected checkpoint."""
    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError('cached probe features must be 2D matrices')
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError('train and val feature dimensions must match')
    if train_features.shape[0] != train_targets.shape[0]:
        raise ValueError('train features and targets must have the same sample count')
    if val_features.shape[0] != val_targets.shape[0]:
        raise ValueError('val features and targets must have the same sample count')
    if train_features.shape[0] == 0 or val_features.shape[0] == 0:
        raise ValueError('cached probe train and val splits must be non-empty')
    if epochs <= 0 or batch_size <= 0:
        raise ValueError(f'epochs and batch_size must be positive, got {epochs} and {batch_size}')
    if updates is not None and (updates <= 0 or eval_every is None or eval_every <= 0):
        raise ValueError('updates and eval_every must be positive in fixed-update mode')

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    model = head_factory(train_features.shape[1]).to(device)
    train_x = train_features.to(device)
    train_y = train_targets.to(device)
    val_x = val_features.to(device)
    val_y_cpu = val_targets.cpu()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = train_x.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * n_batches if updates is None else updates)

    best_epoch = -1
    best_update = None
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    history: list[dict] = []
    full_batches = _full_shuffled_batches(n_train, batch_size, device) if updates is not None else None
    progress_iter = tqdm(range(epochs if updates is None else updates), desc=f'seed={seed}', ncols=80, disable=not progress)
    for step in progress_iter:
        model.train()
        if updates is None:
            permutation = torch.randperm(n_train, device=device)
            batches = (permutation[start:start + batch_size] for start in range(0, n_train, batch_size))
        else:
            batches = (next(full_batches),)
        for idx in batches:
            loss = objective(model(train_x[idx]), train_y[idx])
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f'probe objective must return one finite scalar, got shape={tuple(loss.shape)} value={loss}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if updates is not None and (step + 1) % eval_every != 0 and step + 1 != updates:
            continue
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x).detach().float().cpu()
        metrics = _validate_metrics(evaluator(val_prediction, val_y_cpu))
        history.append({'epoch': step, 'metrics': metrics} if updates is None else {'update': step + 1, 'metrics': metrics})
        if selection.is_better(metrics, best_metrics):
            if updates is None:
                best_epoch = step
            else:
                best_update = step + 1
            best_metrics = metrics
            best_state = _clone_state_dict(model)
        progress_iter.set_postfix({selection.metric: f'{selection.value(metrics):.4f}'})

    if best_metrics is None or best_state is None:
        raise RuntimeError('probe training completed without a validation-selected checkpoint')
    model.load_state_dict(best_state)
    return CachedProbeRun(
        model=model, best_epoch=best_epoch, best_metrics=best_metrics, history=history,
        best_update=best_update,
    )


def train_cached_probe_seeds(*, seeds: int, **kwargs) -> list[CachedProbeRun]:
    if seeds <= 0:
        raise ValueError(f'seeds must be positive, got {seeds}')
    return [train_cached_probe(seed=seed, **kwargs) for seed in range(seeds)]


def aggregate_metric_records(records: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float | int]]:
    """Compute population mean/std for records with one exact metric schema."""
    if not records:
        raise ValueError('cannot aggregate an empty metric record sequence')
    metric_keys = set(records[0])
    if any(set(record) != metric_keys for record in records[1:]):
        raise ValueError('all metric records must have the same metric keys')

    summary: dict[str, dict[str, float | int]] = {}
    for metric in sorted(metric_keys):
        values = np.asarray([float(record[metric]) for record in records], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'aggregate metric {metric!r} values must be finite, got {values.tolist()}')
        summary[metric] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'n': len(values),
        }
    return summary
