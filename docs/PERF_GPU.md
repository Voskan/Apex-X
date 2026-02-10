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

## Report Schema Notes
JSON top-level keys:
- `schema_version`
- `suite`
- `timestamp_utc`
- `environment`
- `config`
- `status`
- `benchmarks`

Important metrics:
- `p50_ms`, `p95_ms`
- throughput fields:
  - `tiles_per_s`
  - `tokens_per_s`
  - `elements_per_s`
  - `frames_per_s`
- `peak_memory_mb` (CUDA `max_memory_allocated`)

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
