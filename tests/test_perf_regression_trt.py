from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "perf_regression_trt.py"
    spec = importlib.util.spec_from_file_location("perf_regression_trt", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load perf_regression_trt.py module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_script = _load_script_module()
_build_template_from_report = _script._build_template_from_report
_build_trend_payload = _script._build_trend_payload
_collect_trt_ms_metrics = _script._collect_trt_ms_metrics
_compare_against_baseline = _script._compare_against_baseline


def _sample_report() -> dict[str, object]:
    return {
        "suite": "apex_x_trt_engine_shape_sweep",
        "status": "ok",
        "summary": {
            "ok_count": 2,
            "skipped_count": 0,
            "failed_count": 0,
            "p50_ms_min": 2.0,
            "p50_ms_median": 3.0,
            "p50_ms_max": 4.0,
        },
        "cases": [
            {
                "label": "default",
                "status": "ok",
                "metrics": {"p50_ms": 2.5, "p95_ms": 3.0},
            },
            {
                "label": "case_001_1_3_128_128",
                "status": "ok",
                "metrics": {"p50_ms": 3.5, "p95_ms": 4.1},
            },
            {"label": "case_002_1_3_256_256", "status": "skipped", "reason": "engine_not_found"},
        ],
    }


def test_collect_trt_ms_metrics_flattens_summary_and_case_metrics() -> None:
    metrics = _collect_trt_ms_metrics(_sample_report())
    assert metrics["summary.p50_ms_min"] == 2.0
    assert metrics["summary.p50_ms_median"] == 3.0
    assert metrics["summary.p50_ms_max"] == 4.0
    assert metrics["case.default.p50_ms"] == 2.5
    assert metrics["case.default.p95_ms"] == 3.0
    assert metrics["case.case_001_1_3_128_128.p50_ms"] == 3.5
    assert metrics["case.case_001_1_3_128_128.p95_ms"] == 4.1


def test_build_template_from_report_emits_required_status_and_metrics() -> None:
    template = _build_template_from_report(_sample_report())
    assert template["required_status"] == "ok"
    metrics = template["metrics"]
    assert "summary.p50_ms_median" in metrics
    assert "case.default.p50_ms" in metrics
    assert metrics["case.default.p95_ms"]["max_regression_ratio"] == 0.8
    assert metrics["summary.p50_ms_max"]["max_regression_abs_ms"] == 1.0


def test_compare_against_baseline_pass_and_fail_paths() -> None:
    report = _sample_report()
    baseline = {
        "required_status": "ok",
        "metrics": {
            "summary.p50_ms_median": {
                "value_ms": 3.0,
                "max_regression_ratio": 0.2,
                "max_regression_abs_ms": 0.1,
            },
            "case.default.p95_ms": {
                "value_ms": 3.0,
                "max_regression_ratio": 0.5,
                "max_regression_abs_ms": 0.2,
            },
        },
    }
    passed = _compare_against_baseline(current_report=report, baseline_spec=baseline)
    assert passed["status"] == "pass"

    report_fail = dict(report)
    report_fail["status"] = "partial"
    fail = _compare_against_baseline(current_report=report_fail, baseline_spec=baseline)
    assert fail["status"] == "fail"
    checks = {entry["metric"]: entry for entry in fail["checks"]}
    assert checks["__status__"]["status"] == "fail"


def test_build_trend_payload_reports_failed_metrics_count() -> None:
    report = _sample_report()
    comparison = {
        "status": "fail",
        "required_status": "ok",
        "current_status": "partial",
        "checks": [
            {
                "metric": "summary.p50_ms_median",
                "status": "pass",
                "baseline_ms": 3.0,
                "allowed_max_ms": 4.0,
                "current_ms": 3.1,
                "regression_ratio": 0.0333,
            },
            {
                "metric": "__status__",
                "status": "fail",
                "expected": "ok",
                "current": "partial",
            },
        ],
    }
    trend = _build_trend_payload(current_report=report, comparison=comparison)
    assert trend["schema_version"] == 1
    assert trend["suite"] == "apex_x_trt_engine_shape_sweep"
    assert trend["overall_status"] == "fail"
    assert trend["failed_metrics"] == 1
    assert trend["total_metrics"] == 2
