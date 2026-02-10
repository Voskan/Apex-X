# Triton Fused Tile Ops

## Scope
This document describes Triton runtime paths for Apex-X tile operations.

TilePack-specific kernel notes are documented in:
- `docs/runtime/TRITON_TILEPACK.md`
TileUnpack-specific kernel notes are documented in:
- `docs/runtime/TRITON_TILEUNPACK.md`
FusionGate-specific kernel notes are documented in:
- `docs/runtime/TRITON_FUSION.md`
TileSSM scan notes are documented in:
- `docs/runtime/TRITON_SSM.md`
Stage-1 fused pack/op/unpack notes are documented in:
- `docs/runtime/TRITON_FUSED_STAGE1.md`

Current repository behavior is environment-driven:
- if CUDA + Triton are available: Triton backend can be selected
- if unavailable: reference PyTorch path is used

## APIs
Legacy runtime fused API (still reference-first) is in:
- `apex_x/runtime/triton_fused.py`

Stage-1 fused kernel API is in:
- `apex_x/kernels/triton/fused_pack_op_unpack.py`

Legacy runtime fused entrypoints:

- `get_triton_availability()`
- `gather_gate_scatter_reference(...)`
- `gather_gate_scatter(...)`

`gather_gate_scatter(...)` dispatches:
- Triton backend when available and requested
- reference backend on fallback (`allow_fallback=True`)

Result object includes:
- `backend` (`reference` or `triton`)
- `fallback_reason`
- `merged`, `priority_map`, `alpha_map`, `meta`

## Semantics
Reference path preserves the same contracts as tile ops and fusion gate modules:
- deterministic tile ordering
- overlap priority semantics
- fusion equation:
  - `fused = base + alpha * (heavy - base)`
- priority map update semantics identical to `TileUnpackTorch`

## Availability Contract
`get_triton_availability()` checks:
- Triton import availability
- CUDA availability
- CUDA device count

If any check fails, dispatch falls back with explicit reason:
- `triton_not_installed`
- `cuda_unavailable`
- `cuda_device_not_found`

## Kernel Status
Current status in this repository:
- dedicated Triton TilePack gather kernel is implemented with fallback dispatch:
  - `apex_x/kernels/triton/tilepack.py`
- dedicated Triton TileUnpack scatter kernel is implemented with deterministic overlap priority:
  - `apex_x/kernels/triton/tileunpack.py`
- dedicated Triton FusionGate alpha/fusion kernels are implemented with fallback dispatch:
  - `apex_x/kernels/triton/fusiongate.py`
- baseline Triton TileSSM scan kernel is implemented with inference-first dispatch:
  - `apex_x/kernels/triton/tilessm_scan.py`
  - supports `forward`, `backward`, and `bidirectional` directional APIs
  - bidirectional merge modes: `sum`, `avg`, `gated` (gate computed in torch)
- Stage-1 fused fast path (`gather -> affine+ReGLU -> scatter`) is implemented:
  - `apex_x/kernels/triton/fused_pack_op_unpack.py`
- legacy runtime `gather_gate_scatter(...)` Triton entrypoint remains a stub:
  - `_triton_fused_kernel_stub(...)` raises `NotImplementedError`
  - fallback keeps the reference path stable

This keeps compatibility for the old runtime API while enabling a practical fused kernel path.

## Correctness Tests
- `tests/test_triton_fused.py`
  - reference dispatch parity vs explicit reference pipeline
  - fallback behavior when Triton is unavailable
  - forced Triton path without fallback raises stub error
- `tests/test_triton_fused_stage1_dispatch.py`
  - CPU fallback parity vs separate pack/op/unpack composition
  - deterministic duplicate-index guard
- `tests/test_triton_fused_stage1_gpu.py`
  - GPU parity and dispatch checks (auto-skip when CUDA/Triton unavailable)
- `tests/test_triton_tilessm_parity_dispatch.py`
  - parity vs torch stable scan on CPU/reference path
- `tests/test_triton_tilessm_parity_gpu.py`
  - GPU parity and dispatch checks for TileSSM scan (auto-skip without CUDA/Triton)

## Microbenchmark
- `scripts/triton_fused_bench.py`
  - reports reference vs dispatched path timing
  - prints backend and fallback reason
  - on CPU/no-Triton setups, benchmark exercises fallback path

When Triton kernel is implemented, the same script should report runtime speedup.

For Stage-1 fused microbench:
- `python -m apex_x.bench.triton_fused_stage1_bench`
