from __future__ import annotations

from scripts.benchmark_triton_fused import benchmark_triton_fused


def test_triton_bench_smoke_cpu_fallback() -> None:
    stats = benchmark_triton_fused(iters=2, warmup=0, device="cpu")
    assert stats["device"] == "cpu"
    assert stats["backend"] in {"reference", "triton"}
    assert float(stats["reference_ms"]) > 0.0
    assert float(stats["fused_ms"]) > 0.0
    assert stats["speedup_vs_reference"] is not None
