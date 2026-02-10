from __future__ import annotations

from apex_x.bench import compare_against_baseline, run_cpu_perf_suite


def test_run_cpu_perf_suite_smoke_has_expected_metrics() -> None:
    report = run_cpu_perf_suite(
        infer_warmup=0,
        infer_iters=3,
        micro_warmup=0,
        micro_iters=3,
        seed=11,
    )
    metrics = report["metrics"]
    assert report["suite"] == "apex_x_cpu_perf_regression"
    for key in (
        "infer_p50_ms",
        "infer_p95_ms",
        "tile_pack_p50_ms",
        "tile_unpack_p50_ms",
        "fusion_gate_p50_ms",
    ):
        assert key in metrics
        assert float(metrics[key]) > 0.0


def test_compare_against_baseline_reports_pass_and_fail() -> None:
    current = {
        "suite": "apex_x_cpu_perf_regression",
        "metrics": {
            "infer_p50_ms": 1.2,
            "tile_pack_p50_ms": 2.1,
        },
    }
    baseline = {
        "metrics": {
            "infer_p50_ms": {
                "value_ms": 1.0,
                "max_regression_ratio": 0.5,
                "max_regression_abs_ms": 0.0,
            },
            "tile_pack_p50_ms": {
                "value_ms": 1.0,
                "max_regression_ratio": 0.5,
                "max_regression_abs_ms": 0.0,
            },
        }
    }

    result = compare_against_baseline(current_report=current, baseline_spec=baseline)
    assert result["status"] == "fail"
    checks = {entry["metric"]: entry for entry in result["checks"]}
    assert checks["infer_p50_ms"]["status"] == "pass"
    assert checks["tile_pack_p50_ms"]["status"] == "fail"
