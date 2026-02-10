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
  --summary artifacts/perf_compare.json
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
- fails CI when status is `fail`

## Notes
- This suite is intentionally CPU-only and stable-friendly.
- Tolerances are set to avoid flakiness from CI VM noise while still catching major regressions.
