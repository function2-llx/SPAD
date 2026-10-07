"""NUMA affinity helper for DDP training.

Pins each rank's process to exclusive physical cores on the same NUMA node as its GPU.
Requires torchrun (reads LOCAL_RANK, LOCAL_WORLD_SIZE from environment).

Usage:
    from pumit.numa import pin_to_gpu_numa

    affinity = pin_to_gpu_numa()
    if affinity:
        print(f'Pinned to NUMA {affinity.numa_node}, cores {affinity.physical_cores[0]}-{affinity.physical_cores[-1]}')
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class NumaAffinity:
    numa_node: int
    physical_cores: list[int]


def pin_to_gpu_numa(*, shared: bool = False) -> NumaAffinity | None:
    """Pin this process to physical cores on the same NUMA node as its GPU.

    Reads LOCAL_RANK and LOCAL_WORLD_SIZE from environment (set by torchrun).
    Returns None on any detection failure (non-NUMA system, container, missing nvidia-smi).

    Args:
        shared: Let every co-located rank use all physical cores on its GPU's NUMA node. The default partitions
            cores exclusively; shared mode is useful when rank-local input workloads are highly imbalanced.
    """
    local_rank_str = os.environ.get('LOCAL_RANK')
    local_world_size_str = os.environ.get('LOCAL_WORLD_SIZE')
    if local_rank_str is None or local_world_size_str is None:
        log.warning('pin_to_gpu_numa: LOCAL_RANK/LOCAL_WORLD_SIZE not set (not launched via torchrun?)')
        return None

    local_rank = int(local_rank_str)
    local_world_size = int(local_world_size_str)

    # Step 1: Detect physical cores per NUMA node
    numa_phys = _detect_numa_physical_cores()
    if not numa_phys:
        return None

    # Step 2: GPU-to-NUMA mapping
    gpu_numa = _detect_gpu_numa()
    if not gpu_numa:
        return None

    # Step 3: Resolve physical GPU index via CUDA_VISIBLE_DEVICES
    visible_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_str:
        try:
            visible_gpus = [int(x.strip()) for x in visible_str.split(',')]
        except ValueError:
            log.warning('pin_to_gpu_numa: CUDA_VISIBLE_DEVICES contains non-integer values, skipping')
            return None
    else:
        visible_gpus = sorted(gpu_numa.keys())

    if local_rank >= len(visible_gpus):
        log.warning(f'pin_to_gpu_numa: LOCAL_RANK={local_rank} >= len(visible_gpus)={len(visible_gpus)}')
        return None

    physical_gpu = visible_gpus[local_rank]
    if physical_gpu not in gpu_numa:
        log.warning(f'pin_to_gpu_numa: physical GPU {physical_gpu} not in detected GPU-NUMA mapping')
        return None

    numa_node = gpu_numa[physical_gpu]

    # Step 4: Partition cores among co-located ranks
    # Determine which ranks share this NUMA node (ordered by LOCAL_RANK)
    ranks_on_my_node = []
    for lr in range(min(local_world_size, len(visible_gpus))):
        pgpu = visible_gpus[lr]
        if pgpu in gpu_numa and gpu_numa[pgpu] == numa_node:
            ranks_on_my_node.append(lr)

    if local_rank not in ranks_on_my_node:
        log.warning(f'pin_to_gpu_numa: LOCAL_RANK={local_rank} not found in ranks_on_my_node')
        return None

    my_position = ranks_on_my_node.index(local_rank)
    n_ranks_on_node = len(ranks_on_my_node)

    available_cores = numa_phys.get(numa_node, [])
    if not available_cores:
        log.warning(f'pin_to_gpu_numa: no available physical cores on NUMA node {numa_node}')
        return None

    if shared:
        my_cores = available_cores
    else:
        n_cores = len(available_cores)
        per_rank = n_cores // n_ranks_on_node
        remainder = n_cores % n_ranks_on_node

        # Remainder cores go to the first N ranks (by LOCAL_RANK order)
        if my_position < remainder:
            start = my_position * (per_rank + 1)
            count = per_rank + 1
        else:
            start = remainder * (per_rank + 1) + (my_position - remainder) * per_rank
            count = per_rank
        my_cores = available_cores[start:start + count]

    if not my_cores:
        log.warning(f'pin_to_gpu_numa: empty core assignment for rank {local_rank}')
        return None

    # Step 5: Apply and verify
    os.sched_setaffinity(0, my_cores)

    actual = os.sched_getaffinity(0)
    expected = set(my_cores)
    if actual != expected:
        log.warning(
            f'pin_to_gpu_numa: affinity mismatch after pinning. '
            f'requested={sorted(expected)}, actual={sorted(actual)}'
        )

    return NumaAffinity(numa_node=numa_node, physical_cores=my_cores)


def _detect_numa_physical_cores() -> dict[int, list[int]]:
    """Detect physical cores per NUMA node, intersected with cgroup-allowed set."""
    numa_dir = Path('/sys/devices/system/node')
    if not numa_dir.exists():
        log.warning('pin_to_gpu_numa: /sys/devices/system/node not found (non-NUMA system?)')
        return {}

    allowed_cpus = os.sched_getaffinity(0)

    result: dict[int, list[int]] = {}
    for node_path in sorted(numa_dir.glob('node[0-9]*')):
        node_id = int(node_path.name.removeprefix('node'))
        try:
            all_cpus = _parse_cpulist((node_path / 'cpulist').read_text().strip())
        except OSError:
            continue

        phys = []
        for cpu in all_cpus:
            if cpu not in allowed_cpus:
                continue
            siblings_file = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
            try:
                siblings = _parse_cpulist(siblings_file.read_text().strip())
            except OSError:
                continue
            if cpu == siblings[0]:
                phys.append(cpu)

        if phys:
            result[node_id] = phys

    return result


def _detect_gpu_numa() -> dict[int, int]:
    """Detect GPU index -> NUMA node mapping via nvidia-smi + sysfs."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,gpu_bus_id', '--format=csv,noheader,nounits'],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning(f'pin_to_gpu_numa: nvidia-smi failed: {e}')
        return {}

    mapping: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = line.split(',', 1)
        if len(parts) != 2:
            continue
        try:
            gpu_idx = int(parts[0].strip())
        except ValueError:
            continue
        bus_id = _normalize_pci_bus_id(parts[1].strip())

        numa_file = Path(f'/sys/bus/pci/devices/{bus_id}/numa_node')
        if not numa_file.exists():
            log.warning(f'pin_to_gpu_numa: {numa_file} not found for GPU {gpu_idx}')
            return {}
        try:
            numa_node = int(numa_file.read_text().strip())
        except (OSError, ValueError):
            return {}

        if numa_node < 0:
            log.warning(f'pin_to_gpu_numa: GPU {gpu_idx} has numa_node={numa_node} (unknown)')
            return {}

        mapping[gpu_idx] = numa_node

    return mapping


def _normalize_pci_bus_id(bus_id: str) -> str:
    """Convert NVML's 8-digit PCI domain to the 4-digit sysfs form."""
    bus_id = bus_id.lower()
    domain, separator, rest = bus_id.partition(':')
    if not separator:
        return bus_id
    try:
        domain = f'{int(domain, 16):04x}'
    except ValueError:
        return bus_id
    return f'{domain}:{rest}'


def _parse_cpulist(s: str) -> list[int]:
    """Parse '0-63,128-191' into a sorted list of ints."""
    if not s.strip():
        return []

    cpus: list[int] = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(cpus)
