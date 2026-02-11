from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    key: str
    label: str
    default_path: str
    producer: str
    required_for: frozenset[str]


@dataclass(frozen=True, slots=True)
class GateSpec:
    key: str
    label: str


RUNTIME_TARGETS = ("cpu", "torch", "triton", "tensorrt")

ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        key="export_manifest",
        label="Export manifest",
        default_path="artifacts/export/apex_x_manifest.json",
        producer="apex-x export --config <cfg> --output-dir artifacts/export",
        required_for=frozenset(RUNTIME_TARGETS),
    ),
    ArtifactSpec(
        key="onnx_graph",
        label="ONNX graph",
        default_path="artifacts/export/apex_x.onnx",
        producer="apex-x export --config <cfg> --output-dir artifacts/export",
        required_for=frozenset(RUNTIME_TARGETS),
    ),
    ArtifactSpec(
        key="trt_engine",
        label="TRT engine (if TRT release)",
        default_path="artifacts/trt/apex_x.engine",
        producer="TensorRT build pipeline",
        required_for=frozenset({"tensorrt"}),
    ),
    ArtifactSpec(
        key="trt_plugin_versions",
        label="TRT plugin versions (if TRT release)",
        default_path="artifacts/trt/plugin_versions.json",
        producer="runtime/tensorrt build metadata export",
        required_for=frozenset({"tensorrt"}),
    ),
    ArtifactSpec(
        key="parity_report",
        label="Parity report",
        default_path="artifacts/parity/parity_report.json",
        producer="parity test/harness run",
        required_for=frozenset(RUNTIME_TARGETS),
    ),
    ArtifactSpec(
        key="performance_report",
        label="Performance report",
        default_path="artifacts/perf/perf_report.json",
        producer="scripts/perf_regression.py and/or GPU suite",
        required_for=frozenset(RUNTIME_TARGETS),
    ),
    ArtifactSpec(
        key="runtime_capability_snapshot",
        label="Runtime capability snapshot",
        default_path="artifacts/runtime/caps.json",
        producer='python -c "from apex_x.runtime import detect_runtime_caps; ..."',
        required_for=frozenset(RUNTIME_TARGETS),
    ),
    ArtifactSpec(
        key="eval_report",
        label="Eval report with runtime metadata",
        default_path="artifacts/eval/eval_report.json",
        producer="apex-x eval ... --report-json ...",
        required_for=frozenset(RUNTIME_TARGETS),
    ),
)

GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(key="lint_type_tests", label="Lint + typecheck + tests"),
    GateSpec(key="cpu_perf_regression", label="CPU perf regression"),
    GateSpec(key="gpu_perf_regression", label="GPU perf regression (if GPU scope)"),
    GateSpec(key="runtime_backend_parity", label="Runtime backend parity"),
    GateSpec(
        key="runtime_capability_transparency",
        label="Runtime capability transparency",
    ),
    GateSpec(key="security_review", label="Security review"),
    GateSpec(key="documentation_sync", label="Documentation sync"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate release attestation artifact manifest (JSON + Markdown)."
    )
    parser.add_argument("--release-tag", type=str, default="")
    parser.add_argument("--commit-sha", type=str, default="")
    parser.add_argument("--release-owner", type=str, default="")
    parser.add_argument(
        "--runtime-target",
        type=str,
        choices=RUNTIME_TARGETS,
        default="cpu",
    )
    parser.add_argument("--ci-run-url", type=str, default="")
    parser.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Override artifact path mapping: key=path",
    )
    parser.add_argument(
        "--gate-status",
        action="append",
        default=[],
        help="Override gate status: key=PASS|FAIL|PENDING|N/A",
    )
    parser.add_argument(
        "--gate-evidence",
        action="append",
        default=[],
        help="Override gate evidence text/URL: key=value",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/release/release_attestation.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/release/release_attestation.md"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 when required artifacts are missing or any gate is FAIL.",
    )
    return parser


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_mapping(items: list[str], *, valid_keys: set[str], flag: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"{flag} expects key=value, got: {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{flag} expects non-empty key=value, got: {raw!r}")
        if key not in valid_keys:
            valid = ", ".join(sorted(valid_keys))
            raise ValueError(f"unknown key {key!r} for {flag}; valid keys: {valid}")
        mapping[key] = value
    return mapping


def _default_release_tag() -> str:
    value = os.getenv("GITHUB_REF_NAME")
    if value:
        return value
    return "unversioned"


def _default_commit_sha() -> str:
    env_sha = os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _default_release_owner() -> str:
    return os.getenv("GITHUB_ACTOR", "unknown")


def _default_ci_run_url() -> str:
    server = os.getenv("GITHUB_SERVER_URL", "").strip()
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""


def _build_artifact_records(
    *,
    runtime_target: str,
    path_overrides: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in ARTIFACT_SPECS:
        path_value = path_overrides.get(spec.key, spec.default_path)
        path = Path(path_value)
        required = runtime_target in spec.required_for
        exists = path.is_file()
        sha256 = _sha256(path) if exists else None
        status = "PASS" if exists else ("N/A" if not required else "FAIL")
        records.append(
            {
                "key": spec.key,
                "label": spec.label,
                "path": str(path),
                "producer": spec.producer,
                "required": required,
                "exists": exists,
                "sha256": sha256,
                "size_bytes": int(path.stat().st_size) if exists else None,
                "status": status,
            }
        )
    return records


def _normalize_gate_status(raw: str) -> str:
    value = raw.strip().upper()
    allowed = {"PASS", "FAIL", "PENDING", "N/A"}
    if value not in allowed:
        allowed_joined = ", ".join(sorted(allowed))
        raise ValueError(f"invalid gate status {raw!r}; expected one of: {allowed_joined}")
    return value


def _build_gate_records(
    *,
    status_overrides: dict[str, str],
    evidence_overrides: dict[str, str],
    ci_run_url: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for spec in GATE_SPECS:
        status = _normalize_gate_status(status_overrides.get(spec.key, "PENDING"))
        evidence = evidence_overrides.get(spec.key, "").strip()
        if (
            not evidence
            and ci_run_url
            and spec.key
            in {
                "lint_type_tests",
                "cpu_perf_regression",
                "gpu_perf_regression",
            }
        ):
            evidence = ci_run_url
        records.append(
            {
                "key": spec.key,
                "label": spec.label,
                "evidence": evidence or "pending",
                "status": status,
            }
        )
    return records


def _build_payload(
    *,
    release_tag: str,
    commit_sha: str,
    release_owner: str,
    runtime_target: str,
    ci_run_url: str,
    artifact_records: list[dict[str, Any]],
    gate_records: list[dict[str, str]],
) -> dict[str, Any]:
    required_total = sum(1 for row in artifact_records if bool(row["required"]))
    required_passed = sum(
        1
        for row in artifact_records
        if bool(row["required"]) and str(row["status"]).upper() == "PASS"
    )
    required_failed = required_total - required_passed

    gate_total = len(gate_records)
    gate_failed = sum(1 for row in gate_records if row["status"] == "FAIL")
    gate_pending = sum(1 for row in gate_records if row["status"] == "PENDING")
    gate_passed = sum(1 for row in gate_records if row["status"] == "PASS")

    if required_failed > 0 or gate_failed > 0:
        overall_status = "fail"
    elif gate_pending > 0:
        overall_status = "pending"
    else:
        overall_status = "pass"

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "release_metadata": {
            "release_tag": release_tag,
            "commit_sha": commit_sha,
            "build_date_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "release_owner": release_owner,
            "runtime_target": runtime_target,
            "ci_run_url": ci_run_url or None,
        },
        "artifacts": artifact_records,
        "gates": gate_records,
        "summary": {
            "overall_status": overall_status,
            "required_artifacts_total": required_total,
            "required_artifacts_passed": required_passed,
            "required_artifacts_failed": required_failed,
            "gates_total": gate_total,
            "gates_passed": gate_passed,
            "gates_failed": gate_failed,
            "gates_pending": gate_pending,
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["release_metadata"]
    summary = payload["summary"]
    lines: list[str] = []
    lines.append("# Apex-X Release Evidence")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Release tag: `{meta['release_tag']}`")
    lines.append(f"- Commit SHA: `{meta['commit_sha']}`")
    lines.append(f"- Build date (UTC): `{meta['build_date_utc']}`")
    lines.append(f"- Release owner: `{meta['release_owner']}`")
    lines.append(f"- Runtime target: `{meta['runtime_target']}`")
    if meta.get("ci_run_url"):
        lines.append(f"- CI run: {meta['ci_run_url']}")
    lines.append("")
    lines.append("## Artifact Evidence")
    lines.append("")
    lines.append("| Artifact | Path | Producer | Required | Exists | SHA256 | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    artifact_row_template = (
        "| {label} | `{path}` | `{producer}` | {required} | " "{exists} | `{sha}` | `{status}` |"
    )
    for row in payload["artifacts"]:
        lines.append(
            artifact_row_template.format(
                label=row["label"],
                path=row["path"],
                producer=row["producer"],
                required="Yes" if row["required"] else "No",
                exists="Yes" if row["exists"] else "No",
                sha=row["sha256"] or "-",
                status=row["status"],
            )
        )
    lines.append("")
    lines.append("## Gate Evidence")
    lines.append("")
    lines.append("| Gate | Evidence | Status |")
    lines.append("| --- | --- | --- |")
    for row in payload["gates"]:
        lines.append(
            "| {label} | {evidence} | `{status}` |".format(
                label=row["label"],
                evidence=row["evidence"],
                status=row["status"],
            )
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Overall status: `{summary['overall_status']}`")
    lines.append(
        "- Required artifacts: `{passed}/{total}` passed (`failed={failed}`)".format(
            passed=summary["required_artifacts_passed"],
            total=summary["required_artifacts_total"],
            failed=summary["required_artifacts_failed"],
        )
    )
    lines.append(
        "- Gates: `{passed}/{total}` passed (`failed={failed}`, `pending={pending}`)".format(
            passed=summary["gates_passed"],
            total=summary["gates_total"],
            failed=summary["gates_failed"],
            pending=summary["gates_pending"],
        )
    )
    lines.append("")
    lines.append(
        "Generated by `python scripts/release_attestation.py` "
        "for `docs/release/CHECKLIST.md` evidence."
    )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    artifact_keys = {spec.key for spec in ARTIFACT_SPECS}
    gate_keys = {spec.key for spec in GATE_SPECS}

    try:
        artifact_path_overrides = _parse_mapping(
            args.artifact_path,
            valid_keys=artifact_keys,
            flag="--artifact-path",
        )
        gate_status_overrides = _parse_mapping(
            args.gate_status,
            valid_keys=gate_keys,
            flag="--gate-status",
        )
        gate_evidence_overrides = _parse_mapping(
            args.gate_evidence,
            valid_keys=gate_keys,
            flag="--gate-evidence",
        )
    except ValueError as exc:
        parser.error(str(exc))

    release_tag = args.release_tag.strip() or _default_release_tag()
    commit_sha = args.commit_sha.strip() or _default_commit_sha()
    release_owner = args.release_owner.strip() or _default_release_owner()
    ci_run_url = args.ci_run_url.strip() or _default_ci_run_url()

    artifact_records = _build_artifact_records(
        runtime_target=args.runtime_target,
        path_overrides=artifact_path_overrides,
    )
    gate_records = _build_gate_records(
        status_overrides=gate_status_overrides,
        evidence_overrides=gate_evidence_overrides,
        ci_run_url=ci_run_url,
    )
    payload = _build_payload(
        release_tag=release_tag,
        commit_sha=commit_sha,
        release_owner=release_owner,
        runtime_target=args.runtime_target,
        ci_run_url=ci_run_url,
        artifact_records=artifact_records,
        gate_records=gate_records,
    )

    json_path = _write_json(args.output_json, payload)
    markdown_path = _write_text(args.output_md, _render_markdown(payload))

    summary = payload["summary"]
    print(
        "release_attestation "
        f"status={summary['overall_status']} "
        f"required_failed={summary['required_artifacts_failed']} "
        f"gates_failed={summary['gates_failed']} "
        f"gates_pending={summary['gates_pending']} "
        f"json={json_path} md={markdown_path}"
    )

    if args.strict and (
        int(summary["required_artifacts_failed"]) > 0 or int(summary["gates_failed"]) > 0
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
