# Triton TilePack Kernel

## Scope
This document describes the Triton TilePack gather kernel implemented in:
- `apex_x/kernels/triton/tilepack.py`

The kernel gathers tiles from a dense feature map into packed layout without Python tile loops.

## Tensor Contract
- Input feature map: `F [B, C, H, W]`, contiguous `NCHW`
- Input tile ids: `idx [B, K]`, integer tile ids (expected `int32`, `int64` accepted and cast)
- Output packed tensor: `P [B, K, C, t, t]`, contiguous

Grid assumptions:
- `H % t == 0` and `W % t == 0`
- tile id domain: `[0, (H / t) * (W / t) - 1]`

Layout assumption:
- `idx` order is consumed as-is by Triton path
- no implicit ordering/reordering inside kernel

## Dtype Support
- Triton kernel path: `fp16`, `bf16`
- Reference path: `fp32`, `fp16`, `bf16`

## Dispatch and Fallback
Use:
- `tilepack_dispatch(...)`

Behavior:
- prefers Triton when available (`CUDA + Triton`)
- falls back to vectorized reference PyTorch gather when unavailable
- falls back to reference when `feature_map.requires_grad` and `inference_only=True`
  - reason: current Triton path is inference-oriented and does not register custom backward

## API
- `get_triton_tilepack_availability()`
- `tilepack_reference(...)`
- `tilepack_triton(...)`
- `tilepack_dispatch(...)`

## Testing
Parity tests:
- `tests/test_triton_tilepack_parity_dispatch.py`
- `tests/test_triton_tilepack_parity_gpu.py`

Coverage:
- CPU fallback correctness vs `TilePackTorch`
- GPU parity vs `TilePackTorch` on multiple shapes (when Triton/CUDA available)
- deterministic seed use
- gradient safety via reference fallback path

## Microbenchmark
Implemented in:
- `apex_x/bench/triton_tilepack_bench.py`

Run:
```bash
python -m apex_x.bench.triton_tilepack_bench \
  --batch 1 \
  --channels 128 \
  --height 128 \
  --width 128 \
  --tile-size 8 \
  --kmax 32 \
  --warmup 10 \
  --iters 50 \
  --dtype fp16
```

Report includes:
- availability and backend selected
- fallback reason (if any)
- `reference_p50/p95` and `dispatch_p50/p95`
- speedup ratio (`reference / dispatch`)

