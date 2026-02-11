from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "perf_trend_weekly.yml"


def _load_workflow() -> dict[str, Any]:
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("workflow root must be a mapping")
    return payload


def test_weekly_workflow_has_gpu_trend_job() -> None:
    wf = _load_workflow()
    jobs = wf.get("jobs")
    assert isinstance(jobs, dict)
    assert "gpu-trend" in jobs


def test_weekly_gpu_trend_runs_optional_trt_regression_wrapper() -> None:
    wf = _load_workflow()
    jobs = wf.get("jobs")
    assert isinstance(jobs, dict)

    gpu_trend = jobs.get("gpu-trend")
    assert isinstance(gpu_trend, dict)
    steps = gpu_trend.get("steps")
    assert isinstance(steps, list)

    joined_runs = "\n".join(
        str(step.get("run", "")) for step in steps if isinstance(step, dict) and "run" in step
    )
    assert "scripts/perf_regression_trt.py" in joined_runs
    assert "--trend-output artifacts/perf_trt_trend_weekly.json" in joined_runs


def test_weekly_gpu_trend_uploads_trt_artifacts() -> None:
    wf = _load_workflow()
    jobs = wf.get("jobs")
    assert isinstance(jobs, dict)

    gpu_trend = jobs.get("gpu-trend")
    assert isinstance(gpu_trend, dict)
    steps = gpu_trend.get("steps")
    assert isinstance(steps, list)

    upload_step = None
    for step in steps:
        is_upload = isinstance(step, dict) and (
            str(step.get("name", "")).strip() == "Upload GPU trend artifacts"
        )
        if is_upload:
            upload_step = step
            break
    assert isinstance(upload_step, dict)

    path_block = str(upload_step.get("with", {}).get("path", ""))
    assert "artifacts/perf_trt_current_weekly.json" in path_block
    assert "artifacts/perf_trt_compare_weekly.json" in path_block
    assert "artifacts/perf_trt_trend_weekly.json" in path_block
