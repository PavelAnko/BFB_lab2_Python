from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from queue import Queue

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

_BATCH_SIZE = 1 << 18

class _Latch:
    def __init__(self, count: int) -> None:
        self._count = count
        self._condition = threading.Condition()

    def count_down(self) -> None:
        with self._condition:
            if self._count == 0:
                return
            self._count -= 1
            if self._count == 0:
                self._condition.notify_all()

    def wait(self) -> None:
        with self._condition:
            while self._count > 0:
                self._condition.wait()


class _Failure:
    def __init__(self) -> None:
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def get(self) -> BaseException | None:
        with self._lock:
            return self._error

    def set_once(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def raise_if_set(self) -> None:
        error = self.get()
        if error is not None:
            raise error


class _Stop:
    pass


_STOP = _Stop()


@dataclass(frozen=True, slots=True)
class _Range:
    start: int
    end: int
    length: int
    done: _Latch


_Inbox = Queue[_Range | _Stop]


def _add_range(
    data: np.ndarray,
    command: _Range,
    worker_id: int,
    free_workers: Queue[int],
    failure: _Failure,
) -> None:
    try:
        start = command.start
        end = command.end
        length = command.length
        left = data[start:end]
        # partner of i is length - 1 - i
        right = data[length - end : length - start][::-1]
        np.add(left, right, out=left)
    except BaseException as thrown:
        failure.set_once(thrown)
    finally:
        command.done.count_down()
        free_workers.put(worker_id)


def _worker(
    worker_id: int,
    data: np.ndarray,
    inbox: _Inbox,
    free_workers: Queue[int],
    ready: _Latch,
    stopped: _Latch,
    failure: _Failure,
) -> None:
    ready.count_down()
    try:
        free_workers.put(worker_id)
        while True:
            command = inbox.get()
            if command is _STOP:
                return
            _add_range(data, command, worker_id, free_workers, failure)
    finally:
        stopped.count_down()


def _run_wave(length: int, inboxes: list[_Inbox], free_workers: Queue[int]) -> None:
    pairs = length // 2
    batches = (pairs + _BATCH_SIZE - 1) // _BATCH_SIZE
    done = _Latch(batches)
    start = 0
    for _ in range(batches):
        worker_id = free_workers.get()
        end = min(start + _BATCH_SIZE, pairs)
        inboxes[worker_id].put(_Range(start, end, length, done))
        start = end
    done.wait()


def _stop_workers(inboxes: list[_Inbox], stopped: _Latch) -> None:
    for inbox in inboxes:
        inbox.put(_STOP)
    stopped.wait()


def _controller(
    data: np.ndarray,
    inboxes: list[_Inbox],
    free_workers: Queue[int],
    stopped: _Latch,
    finished: _Latch,
    failure: _Failure,
) -> None:
    try:
        length = int(data.size)
        while length > 1:
            _run_wave(length, inboxes, free_workers)
            failure.raise_if_set()
            length -= length // 2
    except BaseException as thrown:
        failure.set_once(thrown)
    finally:
        try:
            _stop_workers(inboxes, stopped)
        finally:
            finished.count_down()


def compute(data: np.ndarray) -> int:
    if data.size == 0:
        return 0
    if data.size == 1:
        return int(data[0])

    worker_count = os.cpu_count() or 1
    ready = _Latch(worker_count)
    stopped = _Latch(worker_count)
    finished = _Latch(1)
    failure = _Failure()
    free_workers: Queue[int] = Queue(maxsize=worker_count)
    inboxes: list[_Inbox] = [Queue(maxsize=1) for _ in range(worker_count)]

    for worker_id, inbox in enumerate(inboxes):
        threading.Thread(
            target=_worker,
            name=f"worker-{worker_id}",
            args=(worker_id, data, inbox, free_workers, ready, stopped, failure),
        ).start()
    ready.wait()

    threading.Thread(
        target=_controller,
        name="controller",
        args=(data, inboxes, free_workers, stopped, finished, failure),
    ).start()
    finished.wait()
    failure.raise_if_set()
    return int(data[0])
