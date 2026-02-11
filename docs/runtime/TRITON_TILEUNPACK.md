# Triton TileUnpack Kernel

## Scope
This document describes the Triton TileUnpack scatter kernel implemented in:
- `apex_x/kernels/triton/tileunpack.py`

Current scope:
- L0 tile scatter with overlap handling
- deterministic priority overwrite semantics
- optional blend mode with ordered composition semantics
- dedicated Triton blend-update kernel path (`_tileunpack_blend_update_kernel`)

## Tensor Contract
- Input base map: `F_base [B, C, H, W]`, contiguous `NCHW`
- Input packed map: `P_out [B, K, C, t, t]`, contiguous
- Input indices/meta:
  - either `idx [B,K]` tile ids, or
  - `meta` containing `origins [B,K,2]`
- Output merged map: `F_merged [B, C, H, W]`

Assumptions:
- `H % t == 0`, `W % t == 0`
- overlaps are allowed

Priority inputs:
- `levels [B,K]` integer levels (higher wins), or
- pre-sorted K-order (`assume_priority_sorted=True`) where later K wins when levels are absent

## Dtype Support
- Triton kernel path: `fp16`, `bf16` on CUDA
- Reference path: `fp32`, `fp16`, `bf16`

## Dispatch and Fallback
Use:
- `tileunpack_dispatch(...)`

Behavior:
- prefers Triton when available (`CUDA + Triton`)
- falls back to reference path when Triton is unavailable
- falls back to reference when autograd is requested (`requires_grad` and `inference_only=True`)
- defaults to `overlap_mode="override"` (priority overwrite)
- `overlap_mode="blend"` is supported in dispatch without forced reference-only branch
  when Triton path is selected
- `overlap_mode="blend"` in Triton path executes sorted-rank kernel updates on CUDA
  (no Python patch loop in the fast path)

## API
- `get_triton_tileunpack_availability()`
- `tileunpack_reference(...)`
- `tileunpack_triton(...)`
- `tileunpack_dispatch(...)`

## Testing
Parity tests:
- `tests/test_triton_tileunpack_parity_dispatch.py`
- `tests/test_triton_tileunpack_parity_gpu.py`

Coverage:
- CPU fallback parity with `TileUnpackTorch` and reference overlap semantics
- GPU parity with `TileUnpackTorch` on representative shapes (auto-skip without CUDA+Triton)
- blend-overlap parity contract in dispatch and GPU suites
- gradient-safe fallback behavior
- synthetic overlap fixtures with explicit overwrite expectations

## Microbenchmark
Implemented in:
- `apex_x/bench/triton_tileunpack_bench.py`

Run:
```bash
python -m apex_x.bench.triton_tileunpack_bench \
  --batch 1 \
  --channels 128 \
  --height 128 \
  --width 128 \
  --tile-size 8 \
  --kmax 32 \
  --overlap-shift 4 \
  --overlap-mode blend \
  --blend-alpha 0.25 \
  --warmup 10 \
  --iters 50 \
  --dtype fp16
```

Report includes:
- backend selected and fallback reason
- `reference_p50/p95`, `dispatch_p50/p95`
- speedup ratio (`reference / dispatch`)
