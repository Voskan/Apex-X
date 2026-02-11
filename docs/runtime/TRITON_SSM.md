# Triton TileSSM Scan (Baseline)

## Scope
This document defines the baseline Triton TileSSM scan implementation:
- input tokens `tokens [B,K,C]`
- linear recurrence scan over `K` for each `(B,C)` stream
- supports directions:
  - `forward`
  - `backward`
  - `bidirectional`
- output sequence `y [B,K,C]` and final state:
  - `[B,C]` for `forward`/`backward`
  - `[B,2,C]` for `bidirectional` (`forward_state`, `backward_state`)

Implementation:
- `apex_x/kernels/triton/tilessm_scan.py`

## Recurrence
Per batch `b`, step `k`, channel `c`:

- `driven = input_gain[c] * x[b,k,c] + state_bias[c]`
- `state = decay[c] * state + (1 - decay[c]) * driven`
- `y[b,k,c] = output_gain[c] * state`

Stability notes:
- token sanitization: `nan -> 0`, infinities clamped
- token clamp range default: `[-1e4, 1e4]`
- `decay` clamped into `(1e-6, 1-1e-6)`
- accumulation computed in `fp32` in both reference and Triton paths

## API
- `get_triton_tilessm_availability()`
- `tilessm_scan_reference(...)`
- `tilessm_scan_triton(...)`
- `tilessm_scan_dispatch(...)`
- `scan(tokens, direction=...) -> y` (clean API)

Direction and merge options:
- `direction`: `forward | backward | bidirectional`
- `merge_mode` (for bidirectional): `sum | avg | gated`
- `merge_gate` (optional, torch-computed): `[C]` or `[B,1,C]`, used when `merge_mode="gated"`

Dispatch semantics:
- inference-first Triton path (`prefer_triton=True`)
- reference fallback when Triton/CUDA unavailable
- reference fallback when `requires_grad` and `inference_only=True`

## Training vs Inference Integration
- Training/backward path should continue using torch scan modules:
  - `StableStateSpaceScan`
  - `StableBidirectionalStateSpaceScan`
- Inference can opt into Triton dispatch via model wiring.

Current integration:
- `apex_x/model/ff_heavy_path.py`
  - new `use_triton_inference_scan` toggle
  - when enabled and module is in `.eval()` mode, scan uses `tilessm_scan_dispatch(...)`
  - in `.train()` mode, existing torch scan path is kept

## Limitations (Baseline)
- kernel is still forward recurrence only; backward and bidirectional are built from directional composition
- no custom backward kernel (training path remains torch/reference)
- compile specialization by `K` (sequence length)
- long-sequence policy:
  - Triton forward scan uses chunked launches when `K > 4096`
  - chunk state is streamed between launches (`final_state` -> next chunk `init_state`)
  - preserves directional semantics while avoiding oversized single-launch specialization

## Tests
- `tests/test_triton_tilessm_parity_dispatch.py`
- `tests/test_triton_tilessm_parity_gpu.py`
- `tests/test_ff_heavy_path_tilessm_dispatch.py`

Coverage:
- parity vs torch stable scan on small/medium shapes
- parity for backward and bidirectional merge modes (`sum/avg/gated`)
- deterministic CPU fallback behavior
- autograd-safe fallback in inference-only mode
- long-sequence chunking contract for Triton forward helper
- CUDA parity for long-sequence chunked path (`K > 4096`)
- integration check: eval uses dispatch; train uses torch path

## Benchmark
Microbenchmark:
- `apex_x/bench/triton_tilessm_bench.py`

Run:
```bash
python -m apex_x.bench.triton_tilessm_bench \
  --batch 2 \
  --steps 256 \
  --channels 128 \
  --warmup 10 \
  --iters 50 \
  --dtype fp16
```

Reported:
- per-direction timings:
  - forward
  - backward
  - bidirectional (avg, gated)
- multi-direction overhead ratios vs forward
- forward throughput and speedup

Long-sequence evidence (`K > 4096`):
```bash
python -m pytest -q tests/test_triton_tilessm_parity_gpu.py -k long_sequence --maxfail=1
python -m apex_x.bench.triton_tilessm_bench \
  --batch 1 \
  --steps 8192 \
  --channels 64 \
  --warmup 3 \
  --iters 12 \
  --dtype fp16 \
  --output artifacts/perf_triton_tilessm_long_k.json
```

Artifacts:
- `artifacts/parity_tilessm_long_k.json`
- `artifacts/parity_tilessm_long_k.md`
- `artifacts/test_tilessm_long_k.log`
- `artifacts/perf_triton_tilessm_long_k.json`
- `artifacts/perf_triton_tilessm_long_k.md`
