from __future__ import annotations

import statistics
import time

import numpy as np

from apex_x import ApexXModel


def benchmark(iters: int = 50) -> tuple[float, float]:
    model = ApexXModel()
    image = np.random.RandomState(123).rand(1, 3, 128, 128).astype(np.float32)

    timings_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.forward(image)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = statistics.median(timings_ms)
    p95 = sorted(timings_ms)[int(0.95 * (len(timings_ms) - 1))]
    return p50, p95


def main() -> None:
    p50, p95 = benchmark()
    print(f"CPU baseline latency p50={p50:.3f} ms p95={p95:.3f} ms")


if __name__ == "__main__":
    main()
