# Apex-X GPU Benchmark Suite

## Scope
This suite targets CUDA machines and produces a unified performance report for:
- Tile ops: `TilePack`, `TileUnpack`, `FusionGate` (torch reference vs Triton dispatch)
- TileSSM scan: torch reference vs Triton dispatch, plus TensorRT plugin path when available
- End-to-end FF inference path: torch eager vs torch+Triton fast-path, plus optional TensorRT engine run

The runner is:
- `apex_x/bench/gpu_bench.py`

Outputs:
- JSON report: latency `p50/p95`, throughput, peak memory
- Markdown summary: compact table for quick regression checks

## Fixed Baseline Profile
Default fixed profile (stable for comparisons):
- `batch=1`
- `channels=128`
- `height=128`
- `width=128`
- `tile_size=8`
- `kmax=32`
- `steps=256`
- `budget_b1=16`, `budget_b2=8`, `budget_total=32`
- `dtype=fp16`

## Local Usage
Run the full GPU benchmark suite:
```bash
python -m apex_x.bench.gpu_bench \
  --output-json artifacts/perf_gpu.json \
  --output-md artifacts/perf_gpu.md
```

Run faster smoke-like pass:
```bash
python -m apex_x.bench.gpu_bench \
  --warmup 3 \
  --iters 10 \
  --output-json artifacts/perf_gpu_smoke.json \
  --output-md artifacts/perf_gpu_smoke.md
```

## TensorRT Plugin Bench (TileSSM)
To enable TensorRT plugin microbench in the suite:
```bash
export APEXX_TRT_PLUGIN_LIB=/abs/path/to/libapexx_trt_plugins.so
python -m apex_x.bench.gpu_bench
```

If TensorRT Python or plugin library is missing, the report marks this section as `skipped`.

## TensorRT Engine Bench (Optional End-to-End)
To benchmark a serialized engine:
```bash
python -m apex_x.bench.gpu_bench \
  --trt-engine-path artifacts/trt/apex_x.engine \
  --trt-input-shape input=1x3x128x128
```

For dynamic-shape engines, provide one `--trt-input-shape name=AxBx...` per dynamic input.
If missing, unresolved dynamic dims default to `1` and may fail for some engines.

## TensorRT Engine Shape Sweep
For dynamic-shape validation across multiple deployment profiles:
```bash
python -m apex_x.bench.trt_engine_sweep \
  --trt-engine-path artifacts/trt/apex_x.engine \
  --shape-case "input=1x3x128x128" \
  --shape-case "input=1x3x256x256" \
  --output-json artifacts/perf_trt_shape_sweep.json \
  --output-md artifacts/perf_trt_shape_sweep.md
```

For multi-input engines, pass named tensors in a single case:
```bash
python -m apex_x.bench.trt_engine_sweep \
  --trt-engine-path artifacts/trt/apex_x.engine \
  --shape-case "image=1x3x128x128;centers=1024x2;strides=1024"
```

## Report Schema Notes
JSON top-level keys:
- `schema_version`
- `suite`
- `timestamp_utc`
- `environment`
- `config`
- `status`
- `benchmarks`
- `triton_autotune`

Important metrics:
- `p50_ms`, `p95_ms`
- throughput fields:
  - `tiles_per_s`
  - `tokens_per_s`
  - `elements_per_s`
  - `frames_per_s`
- `peak_memory_mb` (CUDA `max_memory_allocated`)

`triton_autotune` report block:
- `summary`:
  - `cache_entries`
  - `launches`
  - `cache_hits`
  - `cache_misses`
  - `cache_hit_rate`
- `entries[]`:
  - `op_name`
  - `kernel_name`
  - `shape_bucket`
  - `selected_config`
  - `selection_source` (`triton_best_config|heuristic|registry_cache`)
  - `launches`
  - `cache_hits`
  - `cache_misses`

## Fallback Behavior
- If CUDA is unavailable: suite returns `status=skipped`.
- If Triton is unavailable: dispatch path falls back to reference and records reason.
- If TensorRT is unavailable: TensorRT sections are reported as `skipped` with reason.

## Recommended Regression Workflow
1. Run with fixed defaults and save JSON.
2. Compare `p50/p95` and throughput against prior run.
3. Investigate regressions by section:
   - tile ops
   - TileSSM
   - end-to-end infer

## CI Regression Compare
GPU CI compares against `scripts/perf_baseline_gpu.json` with:
```bash
python scripts/perf_regression_gpu.py \
  --compare \
  --baseline scripts/perf_baseline_gpu.json \
  --output artifacts/perf_gpu_current_ci.json \
  --summary artifacts/perf_gpu_compare_ci.json \
  --trend-output artifacts/perf_gpu_trend_ci.json
```

TensorRT shape-sweep regression compare wrapper:
```bash
python scripts/perf_regression_trt.py \
  --compare \
  --baseline scripts/perf_baseline_trt.json \
  --output artifacts/perf_trt_current.json \
  --summary artifacts/perf_trt_compare.json \
  --trend-output artifacts/perf_trt_trend.json \
  --trt-engine-path artifacts/trt/apex_x.engine \
  --shape-case "input=1x3x128x128" \
  --shape-case "input=1x3x256x256"
```

TRT baseline template generation:
```bash
python scripts/perf_regression_trt.py \
  --emit-baseline-template \
  --baseline scripts/perf_baseline_trt.json \
  --trt-engine-path artifacts/trt/apex_x.engine \
  --shape-case "input=1x3x128x128"
```

Failure rule matches CPU policy:
- fail when `current_ms > value_ms * (1 + max_regression_ratio) + max_regression_abs_ms`

Trend artifact uses the same normalized schema as CPU:
- `schema_version`, `suite`, `timestamp_utc`, `overall_status`
- `total_metrics`, `failed_metrics`
- `metrics[]` with `metric/status/baseline_ms/allowed_max_ms/current_ms/regression_ratio`

Workflows:
- `.github/workflows/perf_gpu.yml` (`gpu-perf-regression`) for PR/manual/scheduled GPU gate
- `.github/workflows/perf_trend_weekly.yml` (`gpu-trend`) for weekly trend artifacts
- GPU workflows also auto-generate release evidence drafts:
  - `artifacts/release/release_attestation_gpu_ci.json`
  - `artifacts/release/release_attestation_gpu_ci.md`
  - `artifacts/release/release_attestation_gpu_weekly.json`
  - `artifacts/release/release_attestation_gpu_weekly.md`

## Replay and Hash Logging
For deterministic replay metadata (seed/config/artifact hashes), use:
- `apex_x.utils.build_replay_manifest(...)`
- golden fixtures:
  - `tests/fixtures/replay_golden_small.json`
  - `tests/fixtures/replay_golden_medium.json`

Validation command:
```bash
python -m pytest -q tests/test_repro.py tests/test_replay_golden.py
```
