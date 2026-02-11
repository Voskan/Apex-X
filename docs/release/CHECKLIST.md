# Apex-X Release Checklist

Use this checklist for every release candidate. A release is blocked until every mandatory item is marked `PASS`.

## 1. Release Metadata

- Release tag:
- Commit SHA:
- Build date (UTC):
- Release owner:
- Runtime target (`cpu` / `torch` / `triton` / `tensorrt`):

## 2. Mandatory Artifacts

| Artifact | Required path (example) | Producer command | SHA256 recorded | Status |
| --- | --- | --- | --- | --- |
| Export manifest | `artifacts/export/apex_x_manifest.json` | `apex-x export --config <cfg> --output-dir artifacts/export` | Yes/No | `PASS/FAIL` |
| ONNX graph | `artifacts/export/apex_x.onnx` | same as above | Yes/No | `PASS/FAIL` |
| TRT engine (if TRT release) | `artifacts/trt/apex_x.engine` | TRT build pipeline | Yes/No | `PASS/FAIL` |
| TRT plugin versions (if TRT release) | `artifacts/trt/plugin_versions.json` | `runtime/tensorrt` build metadata export | Yes/No | `PASS/FAIL` |
| Parity report | `artifacts/parity/parity_report.json` | parity test/harness run | Yes/No | `PASS/FAIL` |
| Performance report | `artifacts/perf/perf_report.json` | `scripts/perf_regression.py` and/or GPU suite | Yes/No | `PASS/FAIL` |
| Runtime capability snapshot | `artifacts/runtime/caps.json` | `python -c "from apex_x.runtime import detect_runtime_caps; ..."` | Yes/No | `PASS/FAIL` |
| Eval report with runtime metadata | `artifacts/eval/eval_report.json` | `apex-x eval ... --report-json ...` | Yes/No | `PASS/FAIL` |

Notes:
- Every artifact path must be linked in release notes or attached to CI artifacts.
- For GPU/TRT releases, include both Markdown and JSON evidence where available.

Automation:
- Use `scripts/release_attestation.py` to auto-generate linked evidence bundles:
  - JSON: machine-readable evidence map with SHA256 + status
  - Markdown: checklist-friendly summary for release notes/review
- CI workflows now publish attestation drafts:
  - CPU CI: `artifacts/release/release_attestation_ci.json`, `artifacts/release/release_attestation_ci.md`
  - GPU CI: `artifacts/release/release_attestation_gpu_ci.json`, `artifacts/release/release_attestation_gpu_ci.md`
  - Weekly CPU/GPU trend workflows publish matching weekly attestation files.

## 3. Gating Checks

Mark each row as `PASS` only when evidence is attached.

| Gate | Evidence | Status |
| --- | --- | --- |
| Lint + typecheck + tests | CI run URL + commit SHA | `PASS/FAIL` |
| CPU perf regression | compare report vs `scripts/perf_baseline_cpu.json` | `PASS/FAIL` |
| GPU perf regression (if GPU scope) | compare report vs `scripts/perf_baseline_gpu.json` | `PASS/FAIL` |
| Runtime backend parity | parity report with configured tolerance profile | `PASS/FAIL` |
| Runtime capability transparency | runtime JSON includes backend selection + fallback fields | `PASS/FAIL` |
| Security review | dependency and critical CVE check result | `PASS/FAIL` |
| Documentation sync | `PRD` + `ENGINEERING_SPEC` + `DECISIONS` + `CONTEXT` updated | `PASS/FAIL` |

CLI helper example:
```bash
python scripts/release_attestation.py \
  --runtime-target cpu \
  --artifact-path performance_report=artifacts/perf_compare_ci.json \
  --artifact-path runtime_capability_snapshot=artifacts/runtime/caps_ci.json \
  --output-json artifacts/release/release_attestation_ci.json \
  --output-md artifacts/release/release_attestation_ci.md
```

## 4. Rollback Plan (Mandatory)

Fill before release:

- Previous stable tag:
- Rollback trigger conditions:
  - parity mismatch above threshold
  - p95 latency regression above threshold
  - runtime crash rate increase
- Rollback execution steps:
  1. Repoint deployment to previous stable tag/engine bundle.
  2. Restore previous export manifest and engine artifacts.
  3. Re-run smoke validation (`predict`, `eval`, service health checks).
  4. Publish incident note with root-cause owner and fix ETA.
- Rollback validation evidence:
  - post-rollback perf report:
  - post-rollback error-rate snapshot:

## 5. Sign-Off

- Engineering sign-off:
- Runtime/Infra sign-off:
- QA sign-off:
- Product sign-off:
- Final decision: `GO` / `NO-GO`
