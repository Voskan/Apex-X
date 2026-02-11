# Apex-X Migration Guide

This guide tracks backward-compatibility shims and required migrations for deprecated paths.

Deprecation policy used in this repository:
- Soft deprecation date: **February 11, 2026**
- Planned removal date: **June 30, 2026** (or first `v0.3.0` release, whichever is earlier)
- During soft deprecation, compatibility shims remain available but should not be used for new code.

## 1. Legacy Triton Fused Entrypoint

Deprecated path:
- `apex_x.runtime.triton_fused.gather_gate_scatter(...)`

Current behavior:
- API is retained for compatibility.
- Runtime behavior is reference-only and reports deprecated fallback reason:
  - `legacy_triton_entrypoint_deprecated_reference_only`

Migration target:
- Use the current FF-heavy execution path through model/runtime selectors:
  - `apex_x.model.ff_heavy_path`
  - Triton dispatch modules under `apex_x.kernels.triton.*`

Migration action:
1. Remove direct imports of `apex_x.runtime.triton_fused.gather_gate_scatter`.
2. Route execution through model inference runner or current kernel dispatch wrappers.

## 2. Eval CLI Compatibility Flag

Deprecated path:
- `apex-x eval --panoptic-pq`

Current behavior:
- Flag is accepted for compatibility but is a no-op.
- PQ is always included in eval report output.

Migration target:
- Remove `--panoptic-pq` from automation and docs.
- Use default `apex-x eval` output contracts in JSON/Markdown reports.

## 3. Export Compatibility Alias

Deprecated path:
- `apex_x.export.noop.Exporter`

Current behavior:
- Kept as compatibility alias.
- Actual implementation is backed by the real exporter pipeline.

Migration target:
- Use concrete exporter:
  - `apex_x.export.pipeline.ApexXExporter`

## 4. CLI Backend/Fallback Explicitness

Legacy usage pattern:
- Implicit backend selection without explicit fallback policy in automation scripts.

Migration target:
- Always pass explicit runtime selection flags where reproducibility matters:
  - `--backend cpu|torch|triton|tensorrt`
  - `--fallback-policy strict|permissive`

Recommended command pattern:
```bash
apex-x predict \
  --config tests/fixtures/apex_x_config.yaml \
  --backend triton \
  --fallback-policy strict \
  --report-json artifacts/predict_report.json
```

## 5. Release Upgrade Checklist

For each release candidate:
1. Scan internal scripts for deprecated paths listed above.
2. Remove deprecated usage or pin a justified temporary exception.
3. Attach evidence in:
   - `docs/release/CHECKLIST.md`
   - release attestation artifacts (`scripts/release_attestation.py`)
