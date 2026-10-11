from __future__ import annotations

import sys
import time

from pairwise_sum import compute

import numpy as np

ARRAY_SIZE = 1_000_000_000
ELEMENT = 1


def main() -> None:
    try:
        array = np.full(ARRAY_SIZE, ELEMENT, dtype=np.int32)
    except MemoryError:
        print("OOM: run with -Xmx6g", file=sys.stderr)
        raise

    started = time.perf_counter_ns()
    total = compute(array)
    time_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    print(f"sum={total}, timeMs={time_ms:.2f}")


if __name__ == "__main__":
    main()
