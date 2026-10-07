"""Shared ViT feature-layer and optimizer-group helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from torch import nn


def select_evenly_spaced_layers(depth: int, count: int = 4) -> tuple[int, ...]:
    """Select 1-based Transformer depths at equal intervals, including the final block."""
    if depth <= 0 or count <= 0 or depth % count:
        raise ValueError(f'depth {depth} must be positive and divisible by count {count}')
    interval = depth // count
    return tuple(interval * index for index in range(1, count + 1))


def pinned_feature_layers(
    config: Mapping[str, object],
    expected: tuple[int, ...],
    *,
    consumer: str,
) -> tuple[int, ...] | None:
    """Read a backbone config's optional ``feature_layers``, which may only name the pinned depths.

    Plans written before the multi-depth readout carry no ``feature_layers`` and get ``None``, which the backbone
    maps to its legacy final-map path.
    """
    if 'feature_layers' not in config:
        return None
    feature_layers = tuple(int(layer) for layer in config['feature_layers'])
    if feature_layers != expected:
        raise ValueError(f'{consumer} feature_layers must be {expected}, got {feature_layers}')
    return feature_layers


def make_parameter_layers(
    embedding_parameters: Iterable[nn.Parameter],
    blocks: Sequence[nn.Module],
    top_parameters: Iterable[nn.Parameter],
) -> tuple[tuple[nn.Parameter, ...], ...]:
    """Return ViT parameters as shallow-to-deep architecture layers."""
    groups = [tuple(embedding_parameters)]
    groups.extend(tuple(block.parameters()) for block in blocks)
    if not groups or not groups[-1]:
        raise RuntimeError('parameter layers require at least one non-empty Transformer block')
    groups[-1] = (*groups[-1], *tuple(top_parameters))
    return tuple(groups)
