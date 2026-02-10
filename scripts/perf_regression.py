from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_bench_module() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("apex_x.bench")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apex-X CPU perf regression suite")
    parser.add_argument("--output", type=Path, default=Path("artifacts/perf_current.json"))
    parser.add_argument("--baseline", type=Path, default=Path("scripts/perf_baseline_cpu.json"))
    parser.add_argument("--compare", action="store_true", help="Compare output vs baseline")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/perf_compare.json"))
    parser.add_argument("--infer-warmup", type=int, default=5)
    parser.add_argument("--infer-iters", type=int, default=30)
    parser.add_argument("--micro-warmup", type=int, default=5)
    parser.add_argument("--micro-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--emit-baseline-template",
        action="store_true",
        help="Generate baseline metrics/tolerance template from current run.",
    )
    return parser


def _make_template(current: dict[str, Any]) -> dict[str, Any]:
    metrics_raw = current.get("metrics", {})
    if not isinstance(metrics_raw, dict):
        raise ValueError("current report metrics must be object")

    metrics_template: dict[str, Any] = {}
    for name, value in sorted(metrics_raw.items()):
        ms = float(value)
        # Keep tolerant defaults for CI VM jitter while still guarding regressions.
        ratio = 0.75
        abs_ms = 2.0
        if "infer_p95" in name:
            ratio = 1.0
            abs_ms = 5.0
        metrics_template[name] = {
            "value_ms": ms,
            "max_regression_ratio": ratio,
            "max_regression_abs_ms": abs_ms,
        }

    return {
        "schema_version": 1,
        "suite": current.get("suite", "apex_x_cpu_perf_regression"),
        "baseline_notes": "CPU baseline with tolerant thresholds for CI variance.",
        "metrics": metrics_template,
    }


def main() -> int:
    bench = _load_bench_module()
    run_cpu_perf_suite = bench.run_cpu_perf_suite
    write_json = bench.write_json
    read_json = bench.read_json
    compare_against_baseline = bench.compare_against_baseline

    parser = _build_parser()
    args = parser.parse_args()

    report = run_cpu_perf_suite(
        infer_warmup=args.infer_warmup,
        infer_iters=args.infer_iters,
        micro_warmup=args.micro_warmup,
        micro_iters=args.micro_iters,
        seed=args.seed,
    )
    output_path = write_json(args.output, report)
    metrics = report["metrics"]
    print(
        "perf_run "
        f"infer_p50_ms={float(metrics['infer_p50_ms']):.4f} "
        f"infer_p95_ms={float(metrics['infer_p95_ms']):.4f} "
        f"tile_pack_p50_ms={float(metrics['tile_pack_p50_ms']):.4f} "
        f"tile_unpack_p50_ms={float(metrics['tile_unpack_p50_ms']):.4f} "
        f"fusion_gate_p50_ms={float(metrics['fusion_gate_p50_ms']):.4f} "
        f"output={output_path}"
    )

    if args.emit_baseline_template:
        baseline_template = _make_template(report)
        baseline_path = write_json(args.baseline, baseline_template)
        print(f"baseline_template_written path={baseline_path}")
        return 0

    if not args.compare:
        return 0

    baseline = read_json(args.baseline)
    comparison = compare_against_baseline(current_report=report, baseline_spec=baseline)
    summary_path = write_json(args.summary, comparison)

    status = str(comparison["status"])
    print(f"perf_compare status={status} summary={summary_path}")
    checks = comparison.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            metric = check.get("metric")
            check_status = check.get("status")
            baseline_ms = check.get("baseline_ms")
            allowed = check.get("allowed_max_ms")
            current_ms = check.get("current_ms")
            print(
                "perf_check "
                f"metric={metric} status={check_status} "
                f"baseline_ms={baseline_ms} allowed_max_ms={allowed} current_ms={current_ms}"
            )

    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
