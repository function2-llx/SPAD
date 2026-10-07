"""Classification-specific components shared by MedMNIST probe and finetune."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from pumit.downstream.evaluation import MetricSelection

CLASSIFICATION_SELECTION = MetricSelection(metric='auc', direction='maximize')


def classification_objective(logits: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(logits, target)
