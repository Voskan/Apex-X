from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_attestation.py"


def _run_release_attestation(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT_PATH), *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_release_attestation_generates_pass_payload_for_cpu(tmp_path: Path) -> None:
    export_manifest = _write_text(tmp_path / "export" / "apex_x_manifest.json", "{}")
    onnx_graph = _write_text(tmp_path / "export" / "apex_x.onnx", "onnx")
    parity_report = _write_text(tmp_path / "parity" / "parity_report.json", '{"status":"pass"}')
    performance_report = _write_text(
        tmp_path / "perf" / "perf_report.json",
        '{"status":"pass","checks":[]}',
    )
    runtime_caps = _write_text(tmp_path / "runtime" / "caps.json", '{"cuda":false}')
    eval_report = _write_text(tmp_path / "eval" / "eval_report.json", '{"runtime":{}}')

    output_json = tmp_path / "release" / "attestation.json"
    output_md = tmp_path / "release" / "attestation.md"

    proc = _run_release_attestation(
        "--runtime-target",
        "cpu",
        "--release-tag",
        "v0.0.0-test",
        "--commit-sha",
        "deadbeef",
        "--release-owner",
        "ci-test",
        "--artifact-path",
        f"export_manifest={export_manifest}",
        "--artifact-path",
        f"onnx_graph={onnx_graph}",
        "--artifact-path",
        f"parity_report={parity_report}",
        "--artifact-path",
        f"performance_report={performance_report}",
        "--artifact-path",
        f"runtime_capability_snapshot={runtime_caps}",
        "--artifact-path",
        f"eval_report={eval_report}",
        "--gate-status",
        "lint_type_tests=PASS",
        "--gate-status",
        "cpu_perf_regression=PASS",
        "--gate-status",
        "gpu_perf_regression=N/A",
        "--gate-status",
        "runtime_backend_parity=PASS",
        "--gate-status",
        "runtime_capability_transparency=PASS",
        "--gate-status",
        "security_review=PASS",
        "--gate-status",
        "documentation_sync=PASS",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--strict",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert output_json.is_file()
    assert output_md.is_file()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["overall_status"] == "pass"
    assert int(summary["required_artifacts_failed"]) == 0
    assert int(summary["gates_failed"]) == 0
    assert int(summary["gates_pending"]) == 0

    artifact_status = {row["key"]: row["status"] for row in payload["artifacts"]}
    assert artifact_status["export_manifest"] == "PASS"
    assert artifact_status["onnx_graph"] == "PASS"
    assert artifact_status["parity_report"] == "PASS"
    assert artifact_status["performance_report"] == "PASS"
    assert artifact_status["runtime_capability_snapshot"] == "PASS"
    assert artifact_status["eval_report"] == "PASS"
    # TRT artifacts are optional for cpu target.
    assert artifact_status["trt_engine"] == "N/A"
    assert artifact_status["trt_plugin_versions"] == "N/A"

    markdown = output_md.read_text(encoding="utf-8")
    assert "# Apex-X Release Evidence" in markdown
    assert "## Artifact Evidence" in markdown
    assert "`pass`" in markdown


def test_release_attestation_strict_fails_on_missing_required_artifacts(tmp_path: Path) -> None:
    performance_report = _write_text(
        tmp_path / "perf" / "perf_report.json",
        '{"status":"pass","checks":[]}',
    )
    output_json = tmp_path / "release" / "attestation_fail.json"
    output_md = tmp_path / "release" / "attestation_fail.md"

    proc = _run_release_attestation(
        "--runtime-target",
        "cpu",
        "--artifact-path",
        f"performance_report={performance_report}",
        "--gate-status",
        "lint_type_tests=PASS",
        "--gate-status",
        "cpu_perf_regression=PASS",
        "--gate-status",
        "gpu_perf_regression=N/A",
        "--gate-status",
        "runtime_backend_parity=PENDING",
        "--gate-status",
        "runtime_capability_transparency=PENDING",
        "--gate-status",
        "security_review=PENDING",
        "--gate-status",
        "documentation_sync=PENDING",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--strict",
    )
    assert proc.returncode == 2
    assert output_json.is_file()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["overall_status"] == "fail"
    assert int(summary["required_artifacts_failed"]) > 0
