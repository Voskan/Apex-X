"""Benchmark helpers for Apex-X."""

from .perf import compare_against_baseline, read_json, run_cpu_perf_suite, write_json


def bench_placeholder() -> None:
    return None


__all__ = [
    "bench_placeholder",
    "run_cpu_perf_suite",
    "write_json",
    "read_json",
    "compare_against_baseline",
]
