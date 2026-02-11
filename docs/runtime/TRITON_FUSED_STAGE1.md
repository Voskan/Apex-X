# Triton Fused Stage-1 Tile Pipeline

## Scope
This document specifies the first practical fused Triton fast path for Apex-X:
- gather selected tiles from dense map
- apply lightweight per-tile transform (`pointwise affine + ReGLU-like gate`)
- scatter transformed tiles back to dense map

Implementation:
- `apex_x/kernels/triton/fused_pack_op_unpack.py`

## Tensor Contract
- Input feature map: `F [B, C, H, W]`, contiguous `NCHW`
- Input indices: `idx [B, K]`, integer tile ids (`int32` kernel path; `int64` accepted and cast)
- Tile size: `t`
- Output merged map: `F_out [B, C, H, W]`, contiguous

Tile id domain:
- `0 <= idx < (H / t) * (W / t)`
- `H % t == 0` and `W % t == 0`

Stage-1 deterministic overwrite assumption:
- indices are required to be unique per batch row
- this avoids write races and gives deterministic semantics

## Transform Definition
For each selected tile pixel value `x`:

- `value = value_scale * x + value_bias`
- `gate = gate_scale * x + gate_bias`
- `y = value * ReLU(gate)`

This is a minimal ReGLU-like placeholder to validate fused infrastructure.

## API
- `get_triton_fused_stage1_availability()`
- `apply_pointwise_affine_reglu(...)`
- `separate_pack_op_unpack_reference(...)`
- `fused_pack_op_unpack_reference(...)`
- `fused_pack_op_unpack_triton(...)`
- `fused_pack_op_unpack_dispatch(...)`

Dispatch behavior:
- prefers Triton on CUDA when available
- falls back to reference on unsupported environments
- falls back to reference when `requires_grad` and `inference_only=True`

FF heavy-path selector integration:
- `FFHeavyPath` now includes a compatibility-gated Stage-1 fused selector for inference.
- Selector is enabled only when conditions hold:
  - eval mode
  - `use_triton_fused_stage1=True`
  - refine block is identity (`use_refine=False`)
  - FiLM parameters are effectively global constants (within tolerance)
  - selected tile indices are unique
- When compatible, heavy-map update is executed via:
  - `fused_pack_op_unpack_dispatch(...)`
- When not compatible, path deterministically falls back to decomposed `pack -> FiLM -> unpack`.

## Parity and Correctness
Tests:
- `tests/test_triton_fused_stage1_dispatch.py`
- `tests/test_triton_fused_stage1_gpu.py`

Coverage:
- CPU fallback parity against separate reference composition (`pack -> op -> unpack`)
- deterministic duplicate-index guard
- GPU parity (fp16) when Triton/CUDA is available
- autograd-safe fallback behavior
- FF heavy-path fused selector compatibility tests:
  - `tests/test_ff_heavy_path_fused_stage1.py`

## Benchmark
Microbenchmark:
- `apex_x/bench/triton_fused_stage1_bench.py`

Command:
```bash
python -m apex_x.bench.triton_fused_stage1_bench \
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

Reported metrics:
- `reference_p50/p95`: explicit reference composition path
- `separate_dispatch_p50/p95`: split dispatch path (`TilePack -> op -> TileUnpack`)
- `fused_dispatch_p50/p95`: fused Stage-1 dispatch path
- `speedup_separate_over_fused`: primary speedup metric for Stage-1

## Limitations (Stage-1)
- transform is intentionally lightweight and local only
- no Tile-SSM fusion yet
- no overlap-priority blending semantics inside this kernel path (unique indices required)
- no custom backward kernel (inference-oriented Triton path)
