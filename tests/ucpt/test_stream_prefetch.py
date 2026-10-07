from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
import itertools
from multiprocessing import get_context
from threading import Event, Lock

import numpy as np

from pumit.ucpt.stream.prefetch import OrderedPrefetchIterator


def _draw_rng(rng):
    return rng.integers(0, 2**63, size=8)


def test_ordered_prefetch_yields_attempt_order_not_completion_order():
    release_first = Event()
    completion_order = []
    lock = Lock()

    def producer(attempt):
        if attempt == 0:
            assert release_first.wait(timeout=5)
        elif attempt == 1:
            with lock:
                completion_order.append(attempt)
            release_first.set()
            return attempt
        with lock:
            completion_order.append(attempt)
        return attempt

    with ThreadPoolExecutor(max_workers=2) as executor:
        with OrderedPrefetchIterator(producer, executor=executor, max_inflight=2) as samples:
            assert list(itertools.islice(samples, 2)) == [0, 1]

    assert completion_order[0] == 1


def test_ordered_prefetch_skips_rejected_attempts_and_stays_bounded():
    def producer(attempt):
        return attempt if attempt % 2 == 0 else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        with OrderedPrefetchIterator(producer, executor=executor, max_inflight=3) as samples:
            assert samples._next_submit - samples._next_consume == 3
            assert list(itertools.islice(samples, 3)) == [0, 2, 4]
            assert samples._next_submit - samples._next_consume == 3


def test_sample_generation_is_identical_with_and_without_prefetch(monkeypatch):
    from pumit.ucpt.stream import build

    monkeypatch.setattr(
        build,
        '_load_worker_globals',
        lambda config_path: (object(), object(), object()),
    )

    def fake_generate_sample(pipeline, pools, class_sampler, rng, labeled):
        value = int(rng.integers(0, 2**63))
        return None if value % 5 == 0 else {'value': value, 'labeled': labeled}

    monkeypatch.setattr(build, 'generate_sample', fake_generate_sample)
    sequential_rng = np.random.default_rng(123).spawn(1)[0]
    prefetched_rng = np.random.default_rng(123).spawn(1)[0]
    sequential = build._sample_iter(sequential_rng, 'config.yaml', True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        prefetched = build._sample_iter(
            prefetched_rng,
            'config.yaml',
            True,
            executor=executor,
            max_inflight=8,
        )
        with closing(sequential), closing(prefetched):
            expected = list(itertools.islice(sequential, 32))
            actual = list(itertools.islice(prefetched, 32))

    assert actual == expected


def test_one_sample_reuses_its_rng_until_generation_succeeds(monkeypatch):
    from pumit.ucpt.stream import build

    monkeypatch.setattr(
        build,
        '_load_worker_globals',
        lambda config_path: (object(), object(), object()),
    )
    seen_rngs = []

    def fake_generate_sample(pipeline, pools, class_sampler, rng, labeled):
        seen_rngs.append(rng)
        if len(seen_rngs) == 1:
            return None
        return {'labeled': labeled}

    monkeypatch.setattr(build, 'generate_sample', fake_generate_sample)
    samples = build._sample_iter(np.random.default_rng(123), 'config.yaml', True)
    with closing(samples):
        assert next(samples) == {'labeled': True}

    assert len(seen_rngs) == 2
    assert seen_rngs[0] is seen_rngs[1]


def test_spawned_sample_rngs_round_trip_through_process_pool():
    actual_rngs = np.random.default_rng(123).spawn(4)
    expected_rngs = np.random.default_rng(123).spawn(4)

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as executor:
        actual = list(executor.map(_draw_rng, actual_rngs))

    expected = [_draw_rng(rng) for rng in expected_rngs]
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
