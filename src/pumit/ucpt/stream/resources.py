"""Node-local resource limits for stream generation."""

import math
import os
from pathlib import Path


_CGROUP_ROOT = Path('/sys/fs/cgroup')
_CPU_ROOT = Path('/sys/devices/system/cpu')
_WORKER_MEMORY_HEADROOM = 1.25


def _physical_core_count() -> int:
    cores = {
        (
            (_CPU_ROOT / f'cpu{cpu}/topology/physical_package_id').read_text().strip(),
            (_CPU_ROOT / f'cpu{cpu}/topology/core_id').read_text().strip(),
        )
        for cpu in os.sched_getaffinity(0)
    }
    return len(cores)


def _cpu_quota() -> int | None:
    quota_text, period_text = (_CGROUP_ROOT / 'cpu.max').read_text().split()
    if quota_text == 'max':
        return None
    quota = int(quota_text)
    period = int(period_text)
    return max(1, quota // period)


def cpu_process_capacity() -> int:
    """Return the usable full-CPU process count under affinity and cgroup quota."""
    physical_cores = _physical_core_count()
    quota = _cpu_quota()
    return physical_cores if quota is None else min(physical_cores, quota)


def process_rss_bytes() -> int:
    """Return current process RSS."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('/proc/self/status does not contain VmRSS')


def _committed_memory_bytes() -> int:
    stats = {
        key: int(value)
        for line in (_CGROUP_ROOT / 'memory.stat').read_text().splitlines()
        for key, value in [line.split()]
    }
    return stats['anon'] + stats['shmem'] + stats['slab_unreclaimable']


def memory_process_capacity(worker_peak_bytes: int) -> int | None:
    """Return the additional worker count supported by non-reclaimable cgroup memory."""
    if worker_peak_bytes < 1:
        raise ValueError(f'worker_peak_bytes must be positive, got {worker_peak_bytes}')
    limit_text = (_CGROUP_ROOT / 'memory.max').read_text().strip()
    if limit_text == 'max':
        return None

    available = int(limit_text) - _committed_memory_bytes()
    worker_budget = math.ceil(worker_peak_bytes * _WORKER_MEMORY_HEADROOM)
    return max(0, available // worker_budget)


def sample_process_capacity(worker_peak_bytes: int) -> tuple[int, int, int | None]:
    """Return selected, CPU-limited, and memory-limited sample process counts."""
    cpu_capacity = cpu_process_capacity()
    memory_capacity = memory_process_capacity(worker_peak_bytes)
    selected = cpu_capacity if memory_capacity is None else min(cpu_capacity, memory_capacity)
    if selected < 1:
        raise MemoryError(
            f'insufficient cgroup memory for one sample worker: '
            f'{worker_peak_bytes=} {memory_capacity=}'
        )
    return selected, cpu_capacity, memory_capacity
