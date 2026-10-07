import pytest

from pumit.ucpt.stream import resources


def test_sample_process_capacity_uses_tighter_resource_limit(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_process_capacity', lambda: 64)
    monkeypatch.setattr(resources, 'memory_process_capacity', lambda worker_bytes: 48)

    assert resources.sample_process_capacity(2 * 2**30) == (48, 64, 48)


def test_sample_process_capacity_uses_cpu_when_memory_is_unlimited(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_process_capacity', lambda: 64)
    monkeypatch.setattr(resources, 'memory_process_capacity', lambda worker_bytes: None)

    assert resources.sample_process_capacity(2 * 2**30) == (64, 64, None)


def test_sample_process_capacity_rejects_insufficient_memory(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_process_capacity', lambda: 64)
    monkeypatch.setattr(resources, 'memory_process_capacity', lambda worker_bytes: 0)

    with pytest.raises(MemoryError, match='insufficient cgroup memory'):
        resources.sample_process_capacity(2 * 2**30)
