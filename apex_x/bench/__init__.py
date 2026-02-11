"""Benchmark helpers for Apex-X."""

from .perf import compare_against_baseline, read_json, run_cpu_perf_suite, write_json
from .trt_engine_sweep import (
    TRTShapeSweepConfig,
    render_trt_shape_sweep_markdown,
    run_trt_engine_shape_sweep,
)


def bench_placeholder() -> None:
    return None


__all__ = [
    "bench_placeholder",
    "run_cpu_perf_suite",
    "write_json",
    "read_json",
    "compare_against_baseline",
    "TRTShapeSweepConfig",
    "run_trt_engine_shape_sweep",
    "render_trt_shape_sweep_markdown",
]
