from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gpu_bench_module() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("apex_x.bench.gpu_bench")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apex-X GPU perf regression suite")
    parser.add_argument("--output", type=Path, default=Path("artifacts/perf_gpu_current.json"))
    parser.add_argument("--baseline", type=Path, default=Path("scripts/perf_baseline_gpu.json"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/perf_gpu_compare.json"))
    parser.add_argument("--compare", action="store_true", help="Compare output vs baseline")
    parser.add_argument(
        "--emit-baseline-template",
        action="store_true",
        help="Generate baseline metrics/tolerance template from current run.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--kmax", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--budget-b1", type=float, default=16.0)
    parser.add_argument("--budget-b2", type=float, default=8.0)
    parser.add_argument("--budget-total", type=float, default=32.0)
    parser.add_argument("--trt-engine-path", type=str, default="")
    parser.add_argument("--trt-plugin-lib", type=str, default="")
    parser.add_argument(
        "--trt-input-shape",
        action="append",
        default=[],
        help="Dynamic TRT input shape override, e.g. tokens=1x256x128",
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


def _lookup_path(root: dict[str, Any], dotted: str) -> float | None:
    cursor: Any = root
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    if isinstance(cursor, (int, float)):
        return float(cursor)
    return None


def _build_template_from_report(report: dict[str, Any]) -> dict[str, Any]:
    metric_paths = [
        "benchmarks.tile_ops.tilepack.dispatch.p50_ms",
        "benchmarks.tile_ops.tilepack.dispatch.p95_ms",
        "benchmarks.tile_ops.tileunpack.dispatch.p50_ms",
        "benchmarks.tile_ops.tileunpack.dispatch.p95_ms",
        "benchmarks.tile_ops.fusion_gate.dispatch.p50_ms",
        "benchmarks.tile_ops.fusion_gate.dispatch.p95_ms",
        "benchmarks.tilessm.triton_dispatch.p50_ms",
        "benchmarks.tilessm.triton_dispatch.p95_ms",
        "benchmarks.end_to_end_infer.torch_eager.p50_ms",
        "benchmarks.end_to_end_infer.torch_eager.p95_ms",
        "benchmarks.end_to_end_infer.torch_triton_fastpath.p50_ms",
        "benchmarks.end_to_end_infer.torch_triton_fastpath.p95_ms",
    ]
    metrics_template: dict[str, Any] = {}
    for path in metric_paths:
        value = _lookup_path(report, path)
        if value is None:
            continue
        ratio = 0.6
        abs_ms = 0.25
        if path.endswith(".p95_ms"):
            ratio = 0.8
            abs_ms = 0.5
        if "end_to_end_infer" in path:
            ratio = 0.5
            abs_ms = 1.0 if path.endswith(".p50_ms") else 2.0
        metrics_template[path] = {
            "value_ms": float(value),
            "max_regression_ratio": float(ratio),
            "max_regression_abs_ms": float(abs_ms),
        }

    return {
        "schema_version": 1,
        "suite": report.get("suite", "apex_x_gpu_benchmark"),
        "required_status": "ok",
        "baseline_notes": (
            "GPU baseline template; regenerate on the target self-hosted GPU runner and "
            "adjust tolerances for stable CI."
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

    metrics = baseline_spec.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("baseline.metrics must be an object")

    for metric_path, spec in sorted(metrics.items()):
        if not isinstance(spec, dict):
            raise ValueError(f"baseline spec for {metric_path!r} must be object")
        if "value_ms" not in spec:
            raise ValueError(f"baseline metric {metric_path!r} missing value_ms")

        baseline_value = float(spec["value_ms"])
        tolerance_ratio = float(spec.get("max_regression_ratio", 0.50))
        tolerance_abs_ms = float(spec.get("max_regression_abs_ms", 0.0))
        allowed_max = baseline_value * (1.0 + tolerance_ratio) + tolerance_abs_ms

        current_value = _lookup_path(current_report, metric_path)
        if current_value is None:
            failed = True
            checks.append(
                {
                    "metric": metric_path,
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
                "metric": metric_path,
                "status": check_status,
                "baseline_ms": baseline_value,
                "allowed_max_ms": allowed_max,
                "current_ms": float(current_value),
                "regression_ratio": ((current_value / baseline_value) - 1.0)
                if baseline_value > 0.0
                else None,
            }
        )

    return {
        "suite": current_report.get("suite"),
        "status": "fail" if failed else "pass",
        "required_status": required_status,
        "current_status": current_status,
        "checks": checks,
    }


def main() -> int:
    bench = _load_gpu_bench_module()
    parser = _build_parser()
    args = parser.parse_args()

    cfg = bench.GPUBenchConfig(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        tile_size=args.tile_size,
        kmax=args.kmax,
        steps=args.steps,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        dtype=args.dtype,
        budget_b1=args.budget_b1,
        budget_b2=args.budget_b2,
        budget_total=args.budget_total,
        trt_engine_path=args.trt_engine_path,
        trt_plugin_lib=args.trt_plugin_lib,
        trt_input_shapes=tuple(args.trt_input_shape),
    )
    report = bench.run_gpu_bench(cfg)
    output_path = _write_json(args.output, report)
    print(f"perf_gpu_run status={report.get('status')} output={output_path}")

    if args.emit_baseline_template:
        baseline_template = _build_template_from_report(report)
        baseline_path = _write_json(args.baseline, baseline_template)
        print(f"perf_gpu_baseline_template_written path={baseline_path}")
        return 0

    if not args.compare:
        return 0

    baseline = _read_json(args.baseline)
    comparison = _compare_against_baseline(current_report=report, baseline_spec=baseline)
    summary_path = _write_json(args.summary, comparison)
    status = str(comparison["status"])
    print(f"perf_gpu_compare status={status} summary={summary_path}")

    checks = comparison.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            print(
                "perf_gpu_check "
                f"metric={check.get('metric')} status={check.get('status')} "
                f"baseline_ms={check.get('baseline_ms')} "
                f"allowed_max_ms={check.get('allowed_max_ms')} "
                f"current_ms={check.get('current_ms')}"
            )

    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
