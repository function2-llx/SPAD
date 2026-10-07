"""CUDA stream prefetcher for overlapping H2D transfer with GPU compute.

Wraps any DataLoader (or iterable) and copies the next batch to GPU on a
side CUDA stream while the current batch is being processed.

Requirements:
  - Batch objects must implement .to(device, non_blocking=True)
  - DataLoader should use pin_memory=True for true async overlap

Usage:
    from pumit.prefetcher import CUDAPrefetcher

    prefetcher = CUDAPrefetcher(loader, device)
    for step, batch in enumerate(prefetcher, start=start_step):
        losses = model(batch.patches, ...)
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from collections.abc import Mapping
from typing import Any

import torch

log = logging.getLogger(__name__)


def _iter_tensors(value: Any, seen: set[int] | None = None):
    """Yield each tensor in a nested batch object exactly once."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _iter_tensors(getattr(value, field.name), seen)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(key, seen)
            yield from _iter_tensors(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item, seen)


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    """Tell the CUDA allocator about consumer-stream use of a nested batch."""
    for tensor in _iter_tensors(value):
        if tensor.is_cuda:
            tensor.record_stream(stream)


class CUDAPrefetcher:
    """Prefetch batches to GPU on a side CUDA stream."""

    def __init__(self, loader, device: torch.device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.next_batch: Any = None
        self._warned_pin = False

        self._prefetch_first()

    def _prefetch_first(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        self._check_pinned(batch)
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(self.device, non_blocking=True)

    def _prefetch_next(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(self.device, non_blocking=True)

    def _check_pinned(self, batch):
        if self._warned_pin:
            return
        self._warned_pin = True
        tensor = None
        if isinstance(batch, torch.Tensor):
            tensor = batch
        elif hasattr(batch, 'patches'):
            tensor = batch.patches
        if tensor is not None and not tensor.is_pinned():
            warnings.warn(
                'CUDAPrefetcher: batch tensors are not pinned. '
                'H2D overlap requires pin_memory=True on the DataLoader.',
                stacklevel=3,
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        batch = self.next_batch
        # H2D tensors were allocated on the prefetch stream but are consumed on
        # the current stream. Without record_stream(), the caching allocator may
        # reuse their storage while consumer kernels are still in flight.
        _record_stream(batch, current_stream)
        self._prefetch_next()
        return batch
