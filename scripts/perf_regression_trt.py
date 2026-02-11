from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_trt_bench_module() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("apex_x.bench")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apex-X TensorRT shape-sweep perf regression suite"
    )
    parser.add_argument("--trt-engine-path", type=str, default="")
    parser.add_argument(
        "--shape-case",
        action="append",
        default=[],
        help=(
            "Input shape case, e.g. 'input=1x3x128x128' or "
            "'image=1x3x128x128;centers=1024x2;strides=1024'. Repeat for multiple cases."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("artifacts/perf_trt_current.json"))
    parser.add_argument("--baseline", type=Path, default=Path("scripts/perf_baseline_trt.json"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/perf_trt_compare.json"))
    parser.add_argument(
        "--trend-output",
        type=Path,
        default=None,
        help="Optional normalized trend artifact path for CI/history publishing.",
    )
    parser.add_argument("--compare", action="store_true", help="Compare output vs baseline")
    parser.add_argument(
        "--emit-baseline-template",
        action="store_true",
        help="Generate baseline metrics/tolerance template from current run.",
    )
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json root must be an object")
    return payload


def _collect_trt_ms_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    summary = report.get("summary")
    if isinstance(summary, dict):
        for key in ("p50_ms_min", "p50_ms_median", "p50_ms_max"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                metrics[f"summary.{key}"] = float(value)

    cases = report.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            label_raw = case.get("label")
            if not isinstance(label_raw, str) or not label_raw:
                continue
            case_metrics = case.get("metrics")
            if not isinstance(case_metrics, dict):
                continue
            for key in ("p50_ms", "p95_ms"):
                value = case_metrics.get(key)
                if isinstance(value, (int, float)):
                    metrics[f"case.{label_raw}.{key}"] = float(value)
    return metrics


def _build_template_from_report(report: dict[str, Any]) -> dict[str, Any]:
    current_metrics = _collect_trt_ms_metrics(report)
    metrics_template: dict[str, Any] = {}

    for metric_name, value in sorted(current_metrics.items()):
        ratio = 0.6
        abs_ms = 0.25
        if metric_name.endswith(".p95_ms"):
            ratio = 0.8
            abs_ms = 0.5
        if metric_name.startswith("summary."):
            ratio = 0.5
            abs_ms = 1.0
        metrics_template[metric_name] = {
            "value_ms": float(value),
            "max_regression_ratio": float(ratio),
            "max_regression_abs_ms": float(abs_ms),
        }

    return {
        "schema_version": 1,
        "suite": report.get("suite", "apex_x_trt_engine_shape_sweep"),
        "required_status": "ok",
        "baseline_notes": (
            "TensorRT shape-sweep baseline template; regenerate on the target deployment GPU "
            "runner with a production engine and shape-case set."
        ),
        "metrics": metrics_template,
    }


def _compare_against_baseline(
    *,
    current_report: dict[str, Any],
    baseline_spec: dict[str, Any],
) -> dict[str, Any]:
    required_status = str(baseline_spec.get("required_status", "ok"))
    current_status = str(current_report.get("status", "unknown"))
    status_ok = current_status == required_status

    checks: list[dict[str, Any]] = []
    failed = not status_ok
    if not status_ok:
        checks.append(
            {
                "metric": "__status__",
                "status": "fail",
                "expected": required_status,
                "current": current_status,
            }
        )

    metrics_spec = baseline_spec.get("metrics", {})
    if not isinstance(metrics_spec, dict):
        raise ValueError("baseline.metrics must be an object")

    current_metrics = _collect_trt_ms_metrics(current_report)
    for metric_name, spec in sorted(metrics_spec.items()):
        if not isinstance(spec, dict):
            raise ValueError(f"baseline spec for {metric_name!r} must be object")
        if "value_ms" not in spec:
            raise ValueError(f"baseline metric {metric_name!r} missing value_ms")

        baseline_value = float(spec["value_ms"])
        tolerance_ratio = float(spec.get("max_regression_ratio", 0.50))
        tolerance_abs_ms = float(spec.get("max_regression_abs_ms", 0.0))
        allowed_max = baseline_value * (1.0 + tolerance_ratio) + tolerance_abs_ms

        current_value = current_metrics.get(metric_name)
        if current_value is None:
            failed = True
            checks.append(
                {
                    "metric": metric_name,
                    "status": "missing",
                    "baseline_ms": baseline_value,
                    "allowed_max_ms": allowed_max,
                    "current_ms": None,
                }
            )
            continue

        check_status = "pass" if current_value <= allowed_max else "fail"
        if check_status == "fail":
            failed = True
        checks.append(
            {
                "metric": metric_name,
                "status": check_status,
                "baseline_ms": baseline_value,
                "allowed_max_ms": allowed_max,
                "current_ms": float(current_value),
                "regression_ratio": (
                    ((current_value / baseline_value) - 1.0) if baseline_value > 0.0 else None
                ),
            }
        )

    return {
        "suite": current_report.get("suite"),
        "status": "fail" if failed else "pass",
        "required_status": required_status,
        "current_status": current_status,
        "checks": checks,
    }


def _build_trend_payload(
    *,
    current_report: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    checks_raw = comparison.get("checks", [])
    checks = checks_raw if isinstance(checks_raw, list) else []

    metrics: list[dict[str, Any]] = []
    failed_metrics = 0
    for check in checks:
        if not isinstance(check, dict):
            continue
        metric_name = check.get("metric")
        if not isinstance(metric_name, str) or not metric_name:
            continue
        metric_status = str(check.get("status", "unknown"))
        baseline_ms = check.get("baseline_ms")
        allowed_max_ms = check.get("allowed_max_ms")
        current_ms = check.get("current_ms")
        regression_ratio = check.get("regression_ratio")
        if metric_status != "pass":
            failed_metrics += 1
        metrics.append(
            {
                "metric": metric_name,
                "status": metric_status,
                "baseline_ms": float(baseline_ms) if baseline_ms is not None else None,
                "allowed_max_ms": float(allowed_max_ms) if allowed_max_ms is not None else None,
                "current_ms": float(current_ms) if current_ms is not None else None,
                "regression_ratio": (
                    float(regression_ratio) if regression_ratio is not None else None
                ),
            }
        )

    metrics.sort(key=lambda entry: str(entry["metric"]))
    return {
        "schema_version": 1,
        "suite": current_report.get("suite", "apex_x_trt_engine_shape_sweep"),
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "overall_status": str(comparison.get("status", "unknown")),
        "required_status": str(comparison.get("required_status", "ok")),
        "current_status": str(comparison.get("current_status", "unknown")),
        "total_metrics": len(metrics),
        "failed_metrics": failed_metrics,
        "metrics": metrics,
    }


def main() -> int:
    bench = _load_trt_bench_module()
    parser = _build_parser()
    args = parser.parse_args()

    cfg = bench.TRTShapeSweepConfig(
        trt_engine_path=args.trt_engine_path,
        shape_cases=tuple(args.shape_case),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
        output_json="",
        output_md="",
    )
    report = bench.run_trt_engine_shape_sweep(cfg)
    output_path = _write_json(args.output, report)
    print(f"perf_trt_run status={report.get('status')} output={output_path}")

    if args.emit_baseline_template:
        baseline_template = _build_template_from_report(report)
        baseline_path = _write_json(args.baseline, baseline_template)
        print(f"perf_trt_baseline_template_written path={baseline_path}")
        return 0

    if not args.compare:
        return 0

    baseline = _read_json(args.baseline)
    comparison = _compare_against_baseline(current_report=report, baseline_spec=baseline)
    summary_path = _write_json(args.summary, comparison)
    if args.trend_output is not None:
        trend_payload = _build_trend_payload(current_report=report, comparison=comparison)
        trend_path = _write_json(args.trend_output, trend_payload)
        print(
            "perf_trt_trend "
            f"status={trend_payload['overall_status']} "
            f"failed_metrics={trend_payload['failed_metrics']} "
            f"output={trend_path}"
        )

    status = str(comparison["status"])
    print(f"perf_trt_compare status={status} summary={summary_path}")
    checks = comparison.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            print(
                "perf_trt_check "
                f"metric={check.get('metric')} status={check.get('status')} "
                f"baseline_ms={check.get('baseline_ms')} "
                f"allowed_max_ms={check.get('allowed_max_ms')} "
                f"current_ms={check.get('current_ms')}"
            )

    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
