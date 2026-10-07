"""Raw-class pools and deterministic segmentation query sampling."""

from collections.abc import Iterable

import numpy as np

from .label_contract import LabelContract


type RawClass = tuple[str, str]
type ClassifiedClass = dict[str, str | bool | int]


def flatten_label_contract(label_contract: LabelContract) -> tuple[list[RawClass], list[RawClass]]:
    """Flatten every source namespace into volume-positive and explicit-negative class pools."""
    positive: list[RawClass] = []
    negative: list[RawClass] = []
    for source in sorted(label_contract):
        info = label_contract[source]
        positive.extend((source, name) for name in info['positive'])
        negative.extend((source, name) for name in info['negative'])
    return positive, negative


def _sample_pool(
    pool: list[ClassifiedClass],
    count: int,
    rng: np.random.Generator,
    *,
    focus: RawClass | None = None,
) -> list[ClassifiedClass]:
    if count < 0:
        raise ValueError(f'query count must be non-negative, got {count}')
    if count == 0 or not pool:
        return []
    if len(pool) <= count:
        return list(pool)

    selected: list[ClassifiedClass] = []
    remaining = pool
    if focus is not None:
        focus_item = next(
            (
                item
                for item in pool
                if (item['source'], item['name']) == focus
            ),
            None,
        )
        if focus_item is not None:
            selected.append(focus_item)
            remaining = [item for item in pool if item is not focus_item]

    indices = rng.choice(len(remaining), size=count - len(selected), replace=False)
    selected.extend(remaining[int(index)] for index in indices)
    return selected


def sample_classes(
    positive: Iterable[ClassifiedClass],
    negative: Iterable[ClassifiedClass],
    *,
    positive_queries: int,
    negative_queries: int,
    rng: np.random.Generator,
    focus: RawClass | None = None,
) -> list[ClassifiedClass]:
    """Sample final raw classes, preserving an eligible foreground-focus class."""
    positive_pool = sorted(positive, key=lambda item: (str(item['source']), str(item['name'])))
    negative_pool = sorted(negative, key=lambda item: (str(item['source']), str(item['name'])))
    selected = _sample_pool(positive_pool, positive_queries, rng, focus=focus)
    selected.extend(_sample_pool(negative_pool, negative_queries, rng))
    return selected
