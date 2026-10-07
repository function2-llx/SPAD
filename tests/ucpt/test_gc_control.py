"""Tests for GC diagnostics/control (pumit.ucpt.train.gc_control).

GC-state tests restore gc.enable() + gc.unfreeze() in teardown so the rest of the suite
is unperturbed by the module's process-global effects.
"""
import gc

import pytest

from pumit.ucpt.train.gc_control import GCController, GCProbe


@pytest.fixture(autouse=True)
def _restore_gc():
    yield
    gc.unfreeze()
    gc.enable()


def test_controller_rejects_nonpositive_interval():
    for bad in (0, -1, -25):
        with pytest.raises(ValueError):
            GCController(bad)


def test_controller_due_schedule():
    c = GCController(25)
    assert not c.due(0)          # setup() already collected at step 0
    assert not c.due(1)
    assert not c.due(24)
    assert c.due(25)
    assert not c.due(26)
    assert c.due(50)


def test_controller_setup_freezes_and_disables():
    c = GCController(25)
    c.setup()
    assert not gc.isenabled(), 'setup must disable automatic GC'
    assert gc.get_freeze_count() > 0, 'setup must freeze the startup heap'


def test_controller_collect_guards_reenabled_gc():
    c = GCController(25)
    c.setup()
    c.collect()                  # ok while disabled
    gc.enable()
    with pytest.raises(RuntimeError):
        c.collect()              # controller must own collection


def test_probe_logs_one_record_per_collection(tmp_path):
    path = tmp_path / 'gc_rank0.jsonl'
    probe = GCProbe(path, rank=0)
    try:
        probe.set_step(42)
        gc.collect()             # fires the callback
    finally:
        probe.close()
    import orjson
    lines = [orjson.loads(l) for l in path.read_bytes().splitlines() if l]
    assert lines, 'probe wrote no records'
    r = lines[-1]
    assert r['rank'] == 0 and r['step'] == 42
    assert 'generation' in r and 'duration_ms' in r


def test_probe_close_removes_callback(tmp_path):
    n0 = len(gc.callbacks)
    probe = GCProbe(tmp_path / 'gc_rank0.jsonl', rank=0)
    assert len(gc.callbacks) == n0 + 1
    probe.close()
    assert len(gc.callbacks) == n0
