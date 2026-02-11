from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "perf_gpu.yml"


def _load_workflow() -> dict[str, Any]:
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("workflow root must be a mapping")
    return payload


def test_perf_gpu_workflow_has_gpu_critical_pr_paths() -> None:
    wf = _load_workflow()
    triggers = wf.get("on", wf.get(True))
    assert isinstance(triggers, dict)

    pr = triggers.get("pull_request")
    assert isinstance(pr, dict)
    paths = pr.get("paths")
    assert isinstance(paths, list)

    expected = {
        "apex_x/kernels/**",
        "apex_x/runtime/**",
        "runtime/tensorrt/**",
        "scripts/perf_regression_trt.py",
    }
    got = {str(entry).strip("\"'") for entry in paths}
    assert expected.issubset(got)


def test_perf_gpu_workflow_blocks_untrusted_fork_prs() -> None:
    wf = _load_workflow()
    jobs = wf.get("jobs")
    assert isinstance(jobs, dict)

    blocked = jobs.get("blocked-untrusted-pr")
    assert isinstance(blocked, dict)

    condition = str(blocked.get("if", ""))
    assert "github.event_name == 'pull_request'" in condition
    assert "github.event.pull_request.head.repo.full_name != github.repository" in condition


def test_perf_gpu_workflow_runs_on_self_hosted_gpu_runner() -> None:
    wf = _load_workflow()
    jobs = wf.get("jobs")
    assert isinstance(jobs, dict)

    gpu_job = jobs.get("gpu-perf-regression")
    assert isinstance(gpu_job, dict)

    condition = str(gpu_job.get("if", ""))
    assert "github.event_name == 'pull_request'" in condition
    assert "github.event.pull_request.head.repo.full_name == github.repository" in condition

    runs_on = gpu_job.get("runs-on")
    assert isinstance(runs_on, list)
    labels = {str(label) for label in runs_on}
    assert {"self-hosted", "linux", "x64", "gpu"}.issubset(labels)

    steps = gpu_job.get("steps")
    assert isinstance(steps, list)
    joined_runs = "\n".join(
        str(step.get("run", "")) for step in steps if isinstance(step, dict) and "run" in step
    )
    assert "scripts/perf_regression_gpu.py" in joined_runs
    assert "scripts/perf_regression_trt.py" in joined_runs
    assert "--trend-output artifacts/perf_gpu_trend_ci.json" in joined_runs
