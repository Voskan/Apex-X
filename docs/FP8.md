# Apex-X FP8 Precision Policy

## Scope
This document defines how Apex-X enables FP8 for heavy compute ops and how it falls back safely.

Authoritative references:
- `docs/PRD.md` (FR-12)
- `docs/ENGINEERING_SPEC.md` (Section 13)

## Policy Summary
- Heavy ops are FP8-eligible only when support is detected.
- Router and KAN-like paths stay in FP16.
- If FP8 is requested but not supported, fallback is FP16.

## Request Rules
FP8 is requested when either is true:
- `runtime.precision_profile == "balanced"`
- `train.qat_fp8 == true`

## Support Detection
Implemented in `apex_x/runtime/precision.py`:
- device must be CUDA
- torch build must expose FP8 dtype (`torch.float8_e4m3fn`)
- CUDA capability gate is conservative (`sm90+`)

If any check fails, policy falls back to FP16.

## Effective Dtypes
Policy object (`PrecisionPolicy`) exposes:
- `heavy_ops_dtype`
- `router_dtype` (always FP16)
- `kan_dtype` (always FP16)
- `fp8_requested`
- `fp8_enabled`
- `fallback_reason`

Fallback reasons are canonical reason-codes aligned with runtime capability catalog
(for example `fp8_requires_cuda`, `compute_capability_below_sm90`).

## Trainer Integration
`ApexXTrainer` resolves precision policy at init and reports it in:
- `train_summary["precision"]`

For heavy-op execution context:
- FP16 path uses autocast where safe.
- FP8 path is marked as ready and left for specialized kernel/plugin integration.

## Fallback Behavior (Expected on CPU)
On CPU runs, balanced profile requests FP8 but falls back to FP16:
- `fp8_requested = true`
- `fp8_enabled = false`
- `fallback_reason = "fp8_requires_cuda"`

## Validation
Covered by `tests/test_precision_policy.py`:
- CPU fallback smoke
- mocked supported CUDA FP8 path
- trainer summary precision diagnostics
- fallback reason-code catalog compliance

Covered by `tests/test_gpu_bench_fp8.py`:
- GPU bench FP8 request telemetry on non-CUDA hosts
- Markdown summary visibility of FP8 fallback/enabled state

## GPU Benchmark Notes
`apex_x.bench.gpu_bench` now accepts `--dtype fp8`.

Report telemetry includes:
- `requested_dtype`
- `effective_dtype`
- `fp8_requested`
- `fp8_enabled`
- `fp8_fallback_reason`

On hosts without FP8 capability, benchmark falls back to FP16 and records fallback reason.
