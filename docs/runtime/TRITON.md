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
- reference backend in all environments (legacy API is reference-only)

Result object includes:
- `backend` (`reference`)
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
  - compatibility-gated selector is wired into `FFHeavyPath` inference path for
    decomposed-vs-fused routing where Stage-1 constraints are satisfied
- legacy runtime `gather_gate_scatter(...)` entrypoint is kept for compatibility but is
  de-facto reference-only with explicit fallback reason:
  - `legacy_triton_entrypoint_deprecated_reference_only`

This keeps compatibility for the old runtime API while enabling a practical fused kernel path.

## Autotune Registry
Triton kernel autotune telemetry is tracked by:
- `apex_x/kernels/triton/autotune_registry.py`

Registry contract:
- key: `op_name + shape_bucket`
- cached payload:
  - selected config (`BLOCK_*`, `num_warps`, `num_stages` when available)
  - selection source (`triton_best_config`, `heuristic`, or `registry_cache`)
  - launch count and cache hit/miss counters

Current instrumented kernels:
- `tilepack._tilepack_kernel`
- `tileunpack._tileunpack_priority_kernel`
- `tileunpack._tileunpack_scatter_kernel`
- `fusiongate._fusiongate_alpha_kernel`
- `fusiongate._fusiongate_fuse_kernel`
- `fused_pack_op_unpack._fused_pack_op_unpack_kernel`

GPU benchmark output (`apex_x/bench/gpu_bench.py`) exports this telemetry in:
- JSON: `triton_autotune.summary`, `triton_autotune.entries`
- Markdown: `Triton Autotune Registry` section

## Correctness Tests
- `tests/test_triton_fused.py`
  - reference dispatch parity vs explicit reference pipeline
  - fallback behavior when Triton is unavailable
  - forced Triton path stays reference-only and does not raise
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

On CUDA+Triton hosts, the same script reports runtime speedup deltas and selected backend metadata.

For Stage-1 fused microbench:
- `python -m apex_x.bench.triton_fused_stage1_bench`
