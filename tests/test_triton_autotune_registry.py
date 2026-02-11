from __future__ import annotations

import pytest
import torch

from apex_x.bench.gpu_bench import GPUBenchConfig, render_markdown_summary, run_gpu_bench
from apex_x.kernels.triton.autotune_registry import (
    build_shape_bucket,
    clear_triton_autotune_registry,
    get_cached_triton_config,
    record_triton_autotune_selection,
    resolve_triton_launch_config,
    snapshot_triton_autotune_registry,
)


class _FakeTritonConfig:
    def __init__(self) -> None:
        self.kwargs = {"BLOCK_PIX": 128}
        self.num_warps = 4
        self.num_stages = 2


class _FakeKernel:
    best_config = _FakeTritonConfig()


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    clear_triton_autotune_registry()
    yield
    clear_triton_autotune_registry()


def test_build_shape_bucket_is_stable_and_sorted() -> None:
    bucket = build_shape_bucket(kmax=32, tile_size=8, batch=1)
    assert bucket == "batch=1|kmax=32|tile_size=8"


def test_resolve_triton_launch_config_uses_kernel_best_config() -> None:
    selected, source = resolve_triton_launch_config(
        kernel=_FakeKernel(),
        fallback_config={"BLOCK_PIX": 256},
    )

    assert source == "triton_best_config"
    assert selected["BLOCK_PIX"] == 128
    assert selected["num_warps"] == 4
    assert selected["num_stages"] == 2


def test_registry_records_miss_then_hit_and_promotes_best_config() -> None:
    shape_bucket = build_shape_bucket(batch=1, channels=64, tile_size=8, kmax=16)
    record_triton_autotune_selection(
        op_name="tilepack",
        kernel_name="_tilepack_kernel",
        shape_bucket=shape_bucket,
        selected_config={"BLOCK_PIX": 256},
        selection_source="heuristic",
    )
    assert get_cached_triton_config(op_name="tilepack", shape_bucket=shape_bucket) == {
        "BLOCK_PIX": 256
    }

    record_triton_autotune_selection(
        op_name="tilepack",
        kernel_name="_tilepack_kernel",
        shape_bucket=shape_bucket,
        selected_config={"BLOCK_PIX": 128, "num_warps": 4},
        selection_source="triton_best_config",
    )

    snapshot = snapshot_triton_autotune_registry()
    summary = snapshot["summary"]
    assert summary["cache_entries"] == 1
    assert summary["launches"] == 2
    assert summary["cache_hits"] == 1
    assert summary["cache_misses"] == 1
    assert summary["cache_hit_rate"] == pytest.approx(0.5)

    entry = snapshot["entries"][0]
    assert entry["selection_source"] == "triton_best_config"
    assert entry["selected_config"]["BLOCK_PIX"] == 128
    assert entry["selected_config"]["num_warps"] == 4


def test_gpu_bench_reports_autotune_summary_on_cpu_host() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only host assertion; skip when CUDA is available")
    report = run_gpu_bench(GPUBenchConfig(warmup=0, iters=1))
    assert "triton_autotune" in report
    summary = report["triton_autotune"]["summary"]
    assert summary["cache_entries"] == 0
    assert summary["launches"] == 0
    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == 0


def test_gpu_bench_markdown_renders_autotune_section() -> None:
    report = {
        "status": "ok",
        "timestamp_utc": "2026-02-11T00:00:00+00:00",
        "environment": {
            "device": "cuda",
            "requested_dtype": "fp16",
            "effective_dtype": "fp16",
            "fp8_requested": False,
        },
        "benchmarks": {
            "tile_ops": {
                "tilepack": {
                    "backend": "triton",
                    "reference": {"p50_ms": 2.0, "p95_ms": 3.0},
                    "dispatch": {"p50_ms": 1.0, "p95_ms": 1.5},
                },
                "tileunpack": {
                    "backend": "triton",
                    "reference": {"p50_ms": 2.5, "p95_ms": 3.5},
                    "dispatch": {"p50_ms": 1.2, "p95_ms": 1.8},
                },
                "fusion_gate": {
                    "backend": "triton",
                    "reference": {"p50_ms": 1.4, "p95_ms": 2.0},
                    "dispatch": {"p50_ms": 0.8, "p95_ms": 1.2},
                },
            },
            "tilessm": {
                "backend": "triton",
                "torch_reference": {"p50_ms": 4.0, "p95_ms": 5.0, "tokens_per_s": 1000.0},
                "triton_dispatch": {"p50_ms": 2.0, "p95_ms": 2.5, "tokens_per_s": 2000.0},
                "tensorrt_plugin": {"status": "skipped", "reason": "missing_plugin_library"},
            },
            "end_to_end_infer": {
                "torch_eager": {"p50_ms": 10.0, "p95_ms": 12.0, "frames_per_s": 100.0},
                "torch_triton_fastpath": {
                    "p50_ms": 6.0,
                    "p95_ms": 7.0,
                    "frames_per_s": 170.0,
                },
                "tensorrt_engine": {"status": "skipped", "reason": "missing_trt_engine_path"},
            },
        },
        "triton_autotune": {
            "summary": {
                "cache_entries": 1,
                "launches": 3,
                "cache_hits": 2,
                "cache_misses": 1,
                "cache_hit_rate": 2.0 / 3.0,
            },
            "entries": [
                {
                    "op_name": "tilepack",
                    "kernel_name": "_tilepack_kernel",
                    "shape_bucket": "batch=1|channels=128",
                    "selected_config": {"BLOCK_PIX": 128, "num_warps": 4},
                    "selection_source": "triton_best_config",
                    "launches": 3,
                }
            ],
        },
    }

    markdown = render_markdown_summary(report)
    assert "## Triton Autotune Registry" in markdown
    assert "tilepack" in markdown
    assert '"BLOCK_PIX": 128' in markdown
