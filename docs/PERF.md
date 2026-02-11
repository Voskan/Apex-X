# Apex-X Performance Regression Suite

## Scope
This document defines the CPU performance regression suite used in CI.

## Benchmarks
The suite lives in:
- `apex_x/bench/perf.py`
- `scripts/perf_regression.py`

It includes:
1. Fixed-size `infer()` benchmark (CPU baseline)
   - model: `ApexXModel`
   - input: `[1,3,128,128]`
   - metrics: `infer_p50_ms`, `infer_p95_ms`
2. Microbenchmarks (CPU)
   - `TilePackTorch`
   - `TileUnpackTorch`
   - `FusionGate`
   - metrics: `tile_pack_p50_ms`, `tile_unpack_p50_ms`, `fusion_gate_p50_ms`

## Baseline and Tolerances
Committed baseline file:
- `scripts/perf_baseline_cpu.json`

Each metric has:
- `value_ms`
- `max_regression_ratio`
- `max_regression_abs_ms`

Regression check rule:
- fail when `current_ms > value_ms * (1 + ratio) + abs_ms`

## Local Usage
Run suite only:
```bash
python scripts/perf_regression.py --output artifacts/perf_current.json
```

Run and compare with baseline:
```bash
python scripts/perf_regression.py \
  --compare \
  --baseline scripts/perf_baseline_cpu.json \
  --output artifacts/perf_current.json \
  --summary artifacts/perf_compare.json \
  --trend-output artifacts/perf_trend_cpu.json
```

Regenerate baseline template from current machine:
```bash
python scripts/perf_regression.py \
  --emit-baseline-template \
  --baseline scripts/perf_baseline_cpu.json
```

## CI
Workflow job `perf-regression` runs on `ubuntu-latest` (CPU-only):
- runs `scripts/perf_regression.py --compare`
- writes:
  - `artifacts/perf_current_ci.json`
  - `artifacts/perf_compare_ci.json`
  - `artifacts/perf_trend_cpu_ci.json`
- fails CI when status is `fail`

Weekly trend workflow:
- `.github/workflows/perf_trend_weekly.yml` (`cpu-trend` job)
- writes:
  - `artifacts/perf_current_weekly.json`
  - `artifacts/perf_compare_weekly.json`
  - `artifacts/perf_trend_cpu_weekly.json`
  - `artifacts/release/release_attestation_cpu_weekly.json`
  - `artifacts/release/release_attestation_cpu_weekly.md`

## Normalized Trend Artifact
CPU and GPU perf regression scripts emit the same trend artifact schema:
- `schema_version`
- `suite`
- `timestamp_utc`
- `overall_status`
- `total_metrics`
- `failed_metrics`
- `metrics[]` with:
  - `metric`, `status`, `baseline_ms`, `allowed_max_ms`, `current_ms`, `regression_ratio`

TensorRT shape-sweep regression wrapper follows the same compare/trend policy:
- script: `scripts/perf_regression_trt.py`
- baseline: `scripts/perf_baseline_trt.json`
- outputs:
  - `artifacts/perf_trt_current*.json`
  - `artifacts/perf_trt_compare*.json`
  - `artifacts/perf_trt_trend*.json`
- comparison rule is the same:
  - fail when `current_ms > value_ms * (1 + ratio) + abs_ms`

This provides one common reporting shape for weekly and per-PR trend tracking.

Release checklist wiring:
- CPU CI job also writes:
  - `artifacts/release/release_attestation_ci.json`
  - `artifacts/release/release_attestation_ci.md`
- These files auto-populate `docs/release/CHECKLIST.md` evidence links with SHA256/status fields.

## Notes
- This suite is intentionally CPU-only and stable-friendly.
- Tolerances are set to avoid flakiness from CI VM noise while still catching major regressions.

## Deterministic Replay Fixtures
Golden replay fixtures for deterministic selection/order contracts live in:
- `tests/fixtures/replay_golden_small.json`
- `tests/fixtures/replay_golden_medium.json`

Replay checks:
- `tests/test_replay_golden.py` validates selection/order metadata hashes against fixture expectations.
- `tests/test_repro.py` validates seed/config/artifact hashing utilities.

Replay manifest contract:
- `seed`
- `config_sha256`
- `artifact_hashes`
- `artifact_sha256`
- `manifest_sha256`

Recommended command:
```bash
python -m pytest -q tests/test_repro.py tests/test_replay_golden.py
```
