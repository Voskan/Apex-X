#!/usr/bin/env python3
"""Run deployment-readiness checks and write a unified report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex_x.runtime import detect_runtime_caps


@dataclass(slots=True)
class TaskResult:
    name: str
    status: str
    message: str
    command: str | None = None
    artifacts: list[str] | None = None


def _run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"failed_to_start:{type(exc).__name__}:{exc}"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return False, f"exit_{proc.returncode}:{tail[0]}"
    return True, "ok"


def _markdown_report(results: list[TaskResult]) -> str:
    lines = [
        "# Deployment Readiness Report",
        "",
        "| Task | Status | Message |",
        "|---|---|---|",
    ]
    for row in results:
        lines.append(f"| `{row.name}` | `{row.status}` | {row.message} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deployment readiness runner")
    parser.add_argument("--trt-engine-path", default=None)
    parser.add_argument("--github-repo", default="Voskan/Apex-X")
    parser.add_argument("--output-json", default="artifacts/deployment_readiness.json")
    parser.add_argument("--output-md", default="artifacts/deployment_readiness.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[TaskResult] = []
    artifacts: list[str] = []

    engine_path = Path(args.trt_engine_path).expanduser().resolve() if args.trt_engine_path else None
    caps = detect_runtime_caps()

    # 1) TRT shape sweep
    if engine_path is None or not engine_path.exists():
        results.append(
            TaskResult(
                name="trt_shape_sweep",
                status="blocked",
                message="missing_trt_engine_path",
            )
        )
    else:
        out_json = Path("artifacts/perf_trt_shape_sweep.json")
        out_md = Path("artifacts/perf_trt_shape_sweep.md")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        ok, msg = _run_command(
            [
                sys.executable,
                "-m",
                "apex_x.bench.trt_engine_sweep",
                "--trt-engine-path",
                str(engine_path),
                "--shape-case",
                "input=1x3x128x128",
                "--shape-case",
                "input=1x3x256x256",
                "--output-json",
                str(out_json),
                "--output-md",
                str(out_md),
            ]
        )
        results.append(
            TaskResult(
                name="trt_shape_sweep",
                status="done" if ok else "failed",
                message=msg,
                command="python -m apex_x.bench.trt_engine_sweep ...",
                artifacts=[str(out_json), str(out_md)] if ok else None,
            )
        )
        if ok:
            artifacts.extend([str(out_json), str(out_md)])

    # 2) TRT compare/trend
    if engine_path is None or not engine_path.exists():
        results.append(
            TaskResult(
                name="trt_regression_compare_trend",
                status="blocked",
                message="missing_trt_engine_path",
            )
        )
    else:
        out_current = Path("artifacts/perf_trt_current.json")
        out_compare = Path("artifacts/perf_trt_compare.json")
        out_trend = Path("artifacts/perf_trt_trend.json")
        ok, msg = _run_command(
            [
                sys.executable,
                "scripts/perf_regression_trt.py",
                "--compare",
                "--baseline",
                "scripts/perf_baseline_trt.json",
                "--output",
                str(out_current),
                "--summary",
                str(out_compare),
                "--trend-output",
                str(out_trend),
                "--trt-engine-path",
                str(engine_path),
                "--shape-case",
                "input=1x3x128x128",
                "--shape-case",
                "input=1x3x256x256",
            ]
        )
        results.append(
            TaskResult(
                name="trt_regression_compare_trend",
                status="done" if ok else "failed",
                message=msg,
                command="python scripts/perf_regression_trt.py ...",
                artifacts=[str(out_current), str(out_compare), str(out_trend)] if ok else None,
            )
        )
        if ok:
            artifacts.extend([str(out_current), str(out_compare), str(out_trend)])

    # 3) FP8 on sm90+
    cc = caps.cuda.compute_capability or (0, 0)
    if not caps.cuda.available:
        results.append(
            TaskResult(name="fp8_sm90_evidence", status="blocked", message="cuda_unavailable")
        )
    elif cc[0] < 9:
        results.append(
            TaskResult(
                name="fp8_sm90_evidence",
                status="blocked",
                message=f"compute_capability_{cc[0]}{cc[1]}_below_sm90",
            )
        )
    else:
        out_json = Path("artifacts/perf_gpu_fp8.json")
        out_md = Path("artifacts/perf_gpu_fp8.md")
        ok, msg = _run_command(
            [
                sys.executable,
                "-m",
                "apex_x.bench.gpu_bench",
                "--dtype",
                "fp8",
                "--warmup",
                "10",
                "--iters",
                "50",
                "--output-json",
                str(out_json),
                "--output-md",
                str(out_md),
            ]
        )
        results.append(
            TaskResult(
                name="fp8_sm90_evidence",
                status="done" if ok else "failed",
                message=msg,
                command="python -m apex_x.bench.gpu_bench --dtype fp8 ...",
                artifacts=[str(out_json), str(out_md)] if ok else None,
            )
        )
        if ok:
            artifacts.extend([str(out_json), str(out_md)])

    # 4) GitHub protected branch check
    ok_auth, msg_auth = _run_command(["gh", "auth", "status"])
    if not ok_auth:
        results.append(
            TaskResult(
                name="github_branch_protection",
                status="blocked",
                message=f"gh_auth_required:{msg_auth}",
            )
        )
        results.append(
            TaskResult(
                name="github_weekly_gpu_trend",
                status="blocked",
                message=f"gh_auth_required:{msg_auth}",
            )
        )
    else:
        ok, msg = _run_command(
            [
                "gh",
                "api",
                f"repos/{args.github_repo}/branches/main/protection",
                "--jq",
                ".required_status_checks.checks[].context",
            ]
        )
        results.append(
            TaskResult(
                name="github_branch_protection",
                status="done" if ok else "failed",
                message=msg,
                command=f"gh api repos/{args.github_repo}/branches/main/protection ...",
            )
        )
        results.append(
            TaskResult(
                name="github_weekly_gpu_trend",
                status="blocked",
                message="manual_workflow_dispatch_required",
            )
        )

    # 5) Go TRT bridge
    if engine_path is None or not engine_path.exists():
        results.append(
            TaskResult(name="go_trt_bridge", status="blocked", message="missing_trt_engine_path")
        )
    else:
        env = dict(os.environ)
        env["APEXX_TRT_ENGINE_PATH"] = str(engine_path)
        env["APEXX_TRT_BRIDGE_CMD"] = "python -m apex_x.runtime.service_bridge"
        env["CGO_ENABLED"] = "1"
        ok, msg = _run_command(
            ["go", "test", "-tags", "tensorrt", "./..."],
            cwd=Path("runtime/go"),
            env=env,
        )
        results.append(
            TaskResult(
                name="go_trt_bridge",
                status="done" if ok else "failed",
                message=msg,
                command=(
                    "cd runtime/go && APEXX_TRT_ENGINE_PATH=<engine> "
                    'APEXX_TRT_BRIDGE_CMD="python -m apex_x.runtime.service_bridge" '
                    "CGO_ENABLED=1 go test -tags tensorrt ./..."
                ),
            )
        )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "runtime_caps": caps.to_dict(),
        "results": [asdict(x) for x in results],
        "artifacts": sorted(set(artifacts)),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(_markdown_report(results), encoding="utf-8")

    print(
        "deployment_readiness "
        f"results={len(results)} "
        f"output_json={out_json} "
        f"output_md={out_md}"
    )


if __name__ == "__main__":
    main()
