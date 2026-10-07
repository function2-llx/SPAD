"""Bounded ordered prefetch for sample generation."""

from collections.abc import Callable, Iterator
from concurrent.futures import Executor, FIRST_COMPLETED, Future, wait
import itertools


class OrderedPrefetchIterator[T, A](Iterator[T]):
    """Run inputs concurrently while yielding successful results in input order."""

    def __init__(
        self,
        producer: Callable[[A], T | None],
        *,
        executor: Executor,
        max_inflight: int,
        inputs: Iterator[A] | None = None,
    ) -> None:
        if max_inflight < 1:
            raise ValueError(f'max_inflight must be positive, got {max_inflight}')

        self._producer = producer
        self._inputs = itertools.count() if inputs is None else inputs
        self._max_inflight = max_inflight
        self._executor: Executor | None = executor
        self._pending: dict[Future[T | None], int] = {}
        self._completed: dict[int, T | None] = {}
        self._next_submit = 0
        self._next_consume = 0
        self._fill()

    def _fill(self) -> None:
        assert self._executor is not None
        while self._next_submit - self._next_consume < self._max_inflight:
            index = self._next_submit
            future = self._executor.submit(self._producer, next(self._inputs))
            self._pending[future] = index
            self._next_submit += 1

    def __iter__(self) -> 'OrderedPrefetchIterator[T, A]':
        return self

    def __next__(self) -> T:
        if self._executor is None:
            raise StopIteration

        while True:
            while self._next_consume not in self._completed:
                done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
                for future in done:
                    attempt = self._pending.pop(future)
                    self._completed[attempt] = future.result()

            result = self._completed.pop(self._next_consume)
            self._next_consume += 1
            self._fill()
            if result is not None:
                return result

    def close(self) -> None:
        """Cancel queued attempts and finish this iterator's running attempts."""
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        for future in self._pending:
            future.cancel()
        done, _ = wait(self._pending)
        for future in done:
            if not future.cancelled():
                future.result()
        self._pending.clear()
        self._completed.clear()

    def __enter__(self) -> 'OrderedPrefetchIterator[T, A]':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
