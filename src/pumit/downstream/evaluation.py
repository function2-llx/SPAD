"""Small shared evaluation semantics used by concrete downstream runners."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

SelectionDirection = Literal['maximize', 'minimize']


@dataclass(frozen=True)
class MetricSelection:
    """Select a checkpoint or candidate using one validation metric."""

    metric: str
    direction: SelectionDirection
    split: str = 'val'

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError('selection metric must be non-empty')
        if self.direction not in ('maximize', 'minimize'):
            raise ValueError(f'unknown selection direction {self.direction!r}')
        if self.split != 'val':
            raise ValueError(f'downstream model selection must use val, got {self.split!r}')

    def value(self, metrics: Mapping[str, float]) -> float:
        if self.metric not in metrics:
            raise KeyError(f'selection metric {self.metric!r} missing from {sorted(metrics)}')
        value = float(metrics[self.metric])
        if not math.isfinite(value):
            raise ValueError(f'selection metric {self.metric!r} must be finite, got {value}')
        return value

    def is_better(
        self,
        candidate: Mapping[str, float],
        incumbent: Mapping[str, float] | None,
    ) -> bool:
        if incumbent is None:
            self.value(candidate)
            return True
        candidate_value = self.value(candidate)
        incumbent_value = self.value(incumbent)
        if self.direction == 'maximize':
            return candidate_value > incumbent_value
        return candidate_value < incumbent_value

    def policy_record(self) -> dict[str, str]:
        return {
            'split': self.split,
            'metric': self.metric,
            'direction': self.direction,
        }

    def selected_record(self, epoch: int, metrics: Mapping[str, float]) -> dict:
        return {
            **self.policy_record(),
            'selected_epoch': epoch,
            'value': self.value(metrics),
            'metrics': dict(metrics),
        }
