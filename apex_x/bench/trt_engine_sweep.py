from __future__ import annotations

import argparse
import json
import platform
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from apex_x.runtime import RuntimeCaps, detect_runtime_caps

from .gpu_bench import GPUBenchConfig, _bench_tensorrt_engine


@dataclass(frozen=True, slots=True)
class TRTShapeSweepConfig:
    trt_engine_path: str
    shape_cases: tuple[str, ...]
    warmup: int = 3
    iters: int = 10
    seed: int = 123
    output_json: str = "artifacts/perf_trt_shape_sweep.json"
    output_md: str = "artifacts/perf_trt_shape_sweep.md"


def _parse_case_shapes(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if not text:
        raise ValueError("shape case must be non-empty")
    entries = [item.strip() for item in text.replace(",", ";").split(";") if item.strip()]
    if not entries:
        raise ValueError("shape case must include at least one tensor shape")

    normalized: list[str] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid shape case entry: {entry!r}")
        name, dims_text = entry.split("=", 1)
        tensor_name = name.strip()
        if not tensor_name:
            raise ValueError(f"invalid tensor name in shape case entry: {entry!r}")

        dim_tokens = dims_text.strip().replace("x", " ").split()
        if not dim_tokens:
            raise ValueError(f"invalid tensor dims in shape case entry: {entry!r}")
        dims: list[int] = []
        for token in dim_tokens:
            value = int(token)
            if value <= 0:
                raise ValueError(f"tensor dims must be > 0 in shape case entry: {entry!r}")
            dims.append(value)
        normalized.append(f"{tensor_name}={'x'.join(str(v) for v in dims)}")
    return tuple(normalized)


def _shape_case_label(case_index: int, shape_specs: tuple[str, ...]) -> str:
    if not shape_specs:
        return "default"
    primary = shape_specs[0].split("=", 1)[1] if "=" in shape_specs[0] else shape_specs[0]
    return f"case_{case_index:03d}_{primary.replace('x', '_')}"


def _summary_from_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ok_p50: list[float] = []
    ok_count = 0
    skipped_count = 0
    failed_count = 0

    for item in cases:
        status = str(item.get("status", "unknown"))
        if status == "ok":
            ok_count += 1
            metrics = item.get("metrics", {})
            if isinstance(metrics, dict):
                p50 = metrics.get("p50_ms")
                if isinstance(p50, float):
                    ok_p50.append(p50)
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

    summary: dict[str, Any] = {
        "ok_count": ok_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
    }
    if ok_p50:
        summary.update(
            {
                "p50_ms_min": float(min(ok_p50)),
                "p50_ms_max": float(max(ok_p50)),
                "p50_ms_median": float(statistics.median(ok_p50)),
            }
        )
    return summary


def run_trt_engine_shape_sweep(cfg: TRTShapeSweepConfig) -> dict[str, Any]:
    caps: RuntimeCaps = detect_runtime_caps()
    report: dict[str, Any] = {
        "schema_version": 1,
        "suite": "apex_x_trt_engine_shape_sweep",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "runtime_caps": caps.to_dict(),
        },
        "config": {
            "trt_engine_path": cfg.trt_engine_path,
            "shape_cases": list(cfg.shape_cases),
            "warmup": cfg.warmup,
            "iters": cfg.iters,
            "seed": cfg.seed,
        },
    }

    if not caps.cuda.available:
        report["status"] = "skipped"
        report["reason"] = caps.cuda.reason or "cuda_unavailable"
        report["cases"] = []
        report["summary"] = {"ok_count": 0, "skipped_count": 1, "failed_count": 0}
        return report

    if not caps.tensorrt.python_available:
        report["status"] = "skipped"
        report["reason"] = caps.tensorrt.python_reason or "tensorrt_python_unavailable"
        report["cases"] = []
        report["summary"] = {"ok_count": 0, "skipped_count": 1, "failed_count": 0}
        return report

    device = torch.device("cuda")
    parsed_cases: list[tuple[str, tuple[str, ...]]] = [("default", ())]
    for raw in cfg.shape_cases:
        parsed = _parse_case_shapes(raw)
        parsed_cases.append((_shape_case_label(len(parsed_cases), parsed), parsed))

    case_results: list[dict[str, Any]] = []
    for case_index, (label, shape_specs) in enumerate(parsed_cases):
        bench_cfg = GPUBenchConfig(
            warmup=cfg.warmup,
            iters=cfg.iters,
            seed=cfg.seed + case_index,
            trt_engine_path=cfg.trt_engine_path,
            trt_input_shapes=shape_specs,
        )
        try:
            result = _bench_tensorrt_engine(bench_cfg, device=device)
        except Exception as exc:
            case_results.append(
                {
                    "label": label,
                    "input_shapes": list(shape_specs),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        payload: dict[str, Any] = {
            "label": label,
            "input_shapes": list(shape_specs),
            "status": str(result.get("status", "unknown")),
        }
        if "reason" in result:
            payload["reason"] = result["reason"]
        if "mode" in result:
            payload["mode"] = result["mode"]
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            payload["metrics"] = metrics
        case_results.append(payload)

    summary = _summary_from_cases(case_results)
    report["cases"] = case_results
    report["summary"] = summary
    report["status"] = (
        "ok" if summary["ok_count"] > 0 and summary["failed_count"] == 0 else "partial"
    )
    return report


def render_trt_shape_sweep_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Apex-X TensorRT Engine Shape Sweep",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- timestamp_utc: `{report.get('timestamp_utc', '')}`",
        f"- trt_engine_path: `{report.get('config', {}).get('trt_engine_path', '')}`",
    ]

    if report.get("status") == "skipped":
        lines.append(f"- reason: `{report.get('reason', 'unknown')}`")
        lines.append("")
        return "\n".join(lines) + "\n"

    summary = report.get("summary", {})
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- ok_count: `{summary.get('ok_count', 0)}`",
            f"- skipped_count: `{summary.get('skipped_count', 0)}`",
            f"- failed_count: `{summary.get('failed_count', 0)}`",
        ]
    )
    if "p50_ms_min" in summary:
        lines.extend(
            [
                f"- p50_ms_min: `{summary['p50_ms_min']:.4f}`",
                f"- p50_ms_max: `{summary['p50_ms_max']:.4f}`",
                f"- p50_ms_median: `{summary['p50_ms_median']:.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| case | status | mode | p50 ms | p95 ms | fps | reason |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for case in report.get("cases", []):
        metrics = case.get("metrics", {})
        if isinstance(metrics, dict):
            p50 = metrics.get("p50_ms")
            p95 = metrics.get("p95_ms")
            fps = metrics.get("frames_per_s")
        else:
            p50 = None
            p95 = None
            fps = None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("label", "unknown")),
                    str(case.get("status", "unknown")),
                    str(case.get("mode", "n/a")),
                    "n/a" if not isinstance(p50, float) else f"{p50:.4f}",
                    "n/a" if not isinstance(p95, float) else f"{p95:.4f}",
                    "n/a" if not isinstance(fps, float) else f"{fps:.4f}",
                    str(case.get("reason", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> TRTShapeSweepConfig:
    parser = argparse.ArgumentParser(description="TensorRT engine shape sweep benchmark")
    parser.add_argument("--trt-engine-path", type=str, required=True)
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
    parser.add_argument("--output-json", type=str, default="artifacts/perf_trt_shape_sweep.json")
    parser.add_argument("--output-md", type=str, default="artifacts/perf_trt_shape_sweep.md")
    args = parser.parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("iters must be > 0")
    return TRTShapeSweepConfig(
        trt_engine_path=args.trt_engine_path,
        shape_cases=tuple(str(item) for item in args.shape_case),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
        output_json=str(args.output_json),
        output_md=str(args.output_md),
    )


def main() -> None:
    cfg = _parse_args()
    report = run_trt_engine_shape_sweep(cfg)

    json_path = Path(cfg.output_json).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = Path(cfg.output_md).expanduser().resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_trt_shape_sweep_markdown(report), encoding="utf-8")

    print(f"status={report.get('status', 'unknown')} json={json_path} md={md_path}")


if __name__ == "__main__":
    main()
