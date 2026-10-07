from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, roc_auc_score


def logits_to_scores(logits: Tensor) -> np.ndarray:
    """(N,K) logits -> (N,K) softmax probabilities as float64 numpy."""
    return torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float64)


def fallback_metrics(logits: Tensor, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    """Offline macro one-vs-rest AUC + accuracy (no medmnist .npz needed)."""
    scores = logits_to_scores(logits)
    preds = scores.argmax(axis=1)
    acc = float(accuracy_score(labels, preds))
    if n_classes == 2:
        auc = float(roc_auc_score(labels, scores[:, 1]))
    else:
        auc = float(roc_auc_score(labels, scores, multi_class='ovr', average='macro'))
    return {'auc': auc, 'acc': acc}


def evaluate(logits: Tensor, labels: np.ndarray, flag: str, size: int,
             split: str, *, root: str) -> dict[str, float]:
    """Evaluate with the official MedMNIST Evaluator and verify split ordering.

    Non-MedMNIST tasks in the registry (e.g. `ricord`) have no official Evaluator and fall back to
    the equivalent offline computation.
    """
    scores = logits_to_scores(logits)
    from medmnist import INFO, Evaluator

    if flag not in INFO:
        return fallback_metrics(logits, labels, logits.shape[1])

    evaluator = Evaluator(flag, split, size=size, root=root)
    expected_labels = np.asarray(evaluator.labels).astype(np.int64).reshape(-1)
    actual_labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if not np.array_equal(actual_labels, expected_labels):
        first = None
        if actual_labels.shape == expected_labels.shape:
            mismatch = np.flatnonzero(actual_labels != expected_labels)
            first = int(mismatch[0])
        raise ValueError(
            f'{flag}/{split} labels do not match the official npz ordering '
            f'(actual={actual_labels.shape}, expected={expected_labels.shape}, first_mismatch={first})'
        )
    auc, acc = evaluator.evaluate(scores)
    return {'auc': float(auc), 'acc': float(acc)}
