# Triton FusionGate Kernel

## Scope
This document describes the Triton FusionGate kernels implemented in:
- `apex_x/kernels/triton/fusiongate.py`

Implemented paths:
- alpha kernel: computes `alpha[B,1,H,W]`
- optional fused kernel: computes `F = F_base + alpha * (F_detail - F_base)`

## Tensor Contract
- Inputs:
  - `boundary_proxy [B,1,H,W]` (or `[B,H,W]`)
  - `uncertainty_proxy [B,1,H,W]` (or `[B,H,W]`)
- Alpha output:
  - `alpha [B,1,H,W]`
- Optional fusion inputs:
  - `base_features [B,C,H,W]`
  - `detail_features [B,C,H,W]`
- Optional fusion output:
  - `fused [B,C,H,W]`

## Formula
- Weight parameterization follows model `FusionGate`:
  - `w_b = softplus(boundary_log_weight)`
  - `w_u = softplus(uncertainty_log_weight)`
- Alpha:
  - `alpha = sigmoid(w_b * boundary + w_u * uncertainty + bias)`
- Fusion:
  - `fused = base + alpha * (detail - base)`

## Dtype Support
- Triton path: `fp16`, `bf16` on CUDA
- Reference path: `fp32`, `fp16`, `bf16`

## Dispatch and Fallback
Use:
- `fusiongate_dispatch(...)`

Behavior:
- prefers Triton when available
- falls back to reference path when Triton/CUDA unavailable
- falls back to reference path when autograd is requested and `inference_only=True`
- optional in-place fusion is supported (`inplace_fusion=True`)

## API
- `get_triton_fusiongate_availability()`
- `fusiongate_alpha_reference(...)`
- `fusiongate_fuse_reference(...)`
- `fusiongate_alpha_triton(...)`
- `fusiongate_fuse_triton(...)`
- `fusiongate_dispatch(...)`

## Testing
- `tests/test_triton_fusiongate_parity_dispatch.py`
- `tests/test_triton_fusiongate_parity_gpu.py`

Coverage:
- parity vs `apex_x.model.FusionGate` alpha behavior
- alpha range check `[0,1]`
- optional fusion parity
- GPU Triton parity (auto-skip without CUDA+Triton)

## Microbenchmark
Implemented in:
- `apex_x/bench/triton_fusiongate_bench.py`

Run:
```bash
python -m apex_x.bench.triton_fusiongate_bench \
  --batch 1 \
  --channels 128 \
  --height 128 \
  --width 128 \
  --warmup 10 \
  --iters 50 \
  --dtype fp16
```

Report includes:
- alpha path timing (reference vs dispatch)
- alpha+fusion timing (reference vs dispatch)
- backend/fallback info and speedup ratios

