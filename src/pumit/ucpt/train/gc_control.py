"""GC diagnostics and manual control for UCPT training.

``GCProbe`` (``UCPT_GC_PROBE``-gated, no-op unless set) logs every collection per rank to JSONL via ``gc.callbacks``.

``GCController`` freezes the compiled heap, disables automatic collection, and runs a synchronized full collection every N steps.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import orjson


class GCProbe:
    """Append one JSONL record per collection: rank, step, generation, duration, counts.

    The callback fires on the thread that triggered the collection, so ``set_step`` needs
    no locking. ``gc.get_count()`` is read at ``start`` (the pre-collection allocation
    deltas) to show how fast cyclic garbage builds between collections.
    """

    def __init__(self, path: Path, rank: int):
        self._rank = rank
        self._step = -1
        self._t0: dict[int, float] = {}
        self._count0: dict[int, tuple[int, int, int]] = {}
        self._fh = open(path, 'ab', buffering=0)  # unbuffered: a killed run keeps its events
        gc.callbacks.append(self._probe)

    def set_step(self, step: int) -> None:
        self._step = step

    def _probe(self, phase: str, info: dict) -> None:
        gen = info['generation']
        if phase == 'start':
            self._t0[gen] = time.perf_counter()
            self._count0[gen] = gc.get_count()
        else:  # 'stop'
            dur_ms = (time.perf_counter() - self._t0.pop(gen, time.perf_counter())) * 1e3
            rec = {
                'rank': self._rank,
                'step': self._step,
                'generation': gen,
                'duration_ms': round(dur_ms, 3),
                'collected': info.get('collected', -1),
                'uncollectable': info.get('uncollectable', -1),
                'count_at_start': self._count0.pop(gen, None),
            }
            self._fh.write(orjson.dumps(rec) + b'\n')

    def close(self) -> None:
        if self._probe in gc.callbacks:
            gc.callbacks.remove(self._probe)
        self._fh.close()


def maybe_install(save_dir: Path, rank: int) -> GCProbe | None:
    """Install a GCProbe iff UCPT_GC_PROBE is set; else return None (no-op)."""
    if not os.environ.get('UCPT_GC_PROBE'):
        return None
    return GCProbe(save_dir / f'gc_rank{rank}.jsonl', rank)


class GCController:
    """Manual GC control for DDP: freeze the compiled heap, disable automatic collection, and drive a synchronized full collection every ``interval`` steps.

    ``freeze()`` exempts the startup/compile heap from the scan; synchronized collections let ranks overlap their pauses.

    ``setup()`` must run AFTER compile + prewarm + DDP wrap (so the whole compiled heap is frozen) and AFTER DataLoader workers are forked (so they do not inherit disabled GC / the frozen generation).
    """

    def __init__(self, interval: int):
        if interval <= 0:
            raise ValueError(f'GCController interval must be positive, got {interval}')
        self.interval = interval

    def setup(self) -> None:
        gc.disable()   # first: close the window where an automatic collection fires mid-setup
        gc.collect()   # drain existing garbage so nothing collectable gets frozen in
        gc.freeze()    # pin the startup/compile heap into the permanent generation

    def due(self, completed_step: int) -> bool:
        """True after each completed collection interval."""
        return completed_step > 0 and completed_step % self.interval == 0

    def collect(self) -> None:
        """Synchronized full collect. Caller runs a dist.barrier() first so all ranks start
        together; the following all-reduce waits for the slowest."""
        if gc.isenabled():  # runtime check (not assert: strips under -O)
            raise RuntimeError('automatic GC was re-enabled; GCController owns collection')
        gc.collect()
