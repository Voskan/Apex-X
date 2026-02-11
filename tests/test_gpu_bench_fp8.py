from __future__ import annotations

from apex_x.bench.gpu_bench import GPUBenchConfig, render_markdown_summary, run_gpu_bench
from apex_x.runtime import runtime_reason_catalog


def test_gpu_bench_reports_fp8_request_telemetry_on_cpu_host() -> None:
    report = run_gpu_bench(GPUBenchConfig(dtype="fp8", warmup=0, iters=1))

    assert report["status"] == "skipped"
    env = report["environment"]
    assert env["requested_dtype"] == "fp8"
    assert env["fp8_requested"] is True
    assert env["fp8_enabled"] is False
    reason = env["fp8_fallback_reason"]
    if reason is not None:
        assert reason in set(runtime_reason_catalog()["fp8"])


def test_gpu_bench_markdown_includes_fp8_fallback_line() -> None:
    report = run_gpu_bench(GPUBenchConfig(dtype="fp8", warmup=0, iters=1))
    markdown = render_markdown_summary(report)

    assert "requested_dtype" in markdown
    assert "effective_dtype" in markdown
    assert "fp8: `fallback`" in markdown or "fp8: `enabled`" in markdown
