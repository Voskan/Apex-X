# Runtime Parity Framework

## Scope
`apex_x/runtime/parity.py` provides a backend-agnostic parity harness for comparing:
- PyTorch reference op (CPU/GPU)
- Triton op (GPU)
- TensorRT plugin op (GPU, integration-ready)

The framework is lightweight and supports fast, small-shape checks for CI.

## Core API
- `ParityCase`
- `ParityMatrixCase`
- `ParitySweepReport`
- `run_parity_case(...)`
- `run_parity_matrix_case(...)`
- `run_parity_sweep(...)`
- `evaluate_parity_outputs(...)`
- `format_parity_report(...)`
- `format_parity_sweep_report(...)`
- `ToleranceConfig`
- `ParityToleranceProfile`
- `list_parity_tolerance_profiles()`
- `get_parity_tolerance_profile(...)`

## Determinism
`run_parity_case(...)` calls `seed_all(seed, deterministic=...)` before generating inputs.

Recommended test settings:
- fixed `seed`
- `deterministic=True`
- small static shapes for fast CI runs

## Tolerances
`ToleranceConfig` supports per-precision tolerances:
- `default` (fp32/other)
- `fp16`
- `bf16`
- `int8`
- `fp8`

Mismatch is computed with:
- `abs_err > atol + rtol * abs(reference)`

The report includes:
- `max_abs_err`
- `mean_abs_err`
- `max_rel_err`
- `mean_rel_err`
- `mismatch_count`
- `total_count`
- `mismatch_ratio`

## Precision Profiles (Contract)

Parity is evaluated with profile-specific tolerance presets.

Profiles:
- `quality`:
  - strictest profile; intended for high-fidelity deployment checks
  - `mismatch_ratio_limit = 0.0`
- `balanced`:
  - moderate tolerance for mixed-precision stacks
  - `mismatch_ratio_limit = 1e-4`
- `edge`:
  - relaxed e2e tolerance envelope for INT8-heavy paths
  - `mismatch_ratio_limit = 5e-4`

Each profile contains two tolerance sets:
- `op_tolerances`: per-op parity checks
- `e2e_tolerances`: end-to-end inference parity checks

Example:
```python
from apex_x.runtime import get_parity_tolerance_profile

profile = get_parity_tolerance_profile("balanced")
op_tol = profile.op_tolerances
e2e_tol = profile.e2e_tolerances
limit = profile.mismatch_ratio_limit
```

FP8 behavior:
- when candidate/reference dtype is FP8 (`float8_*`), parity uses `ToleranceConfig.fp8`
- when FP8 dtype is unavailable in current torch build, FP8-specific tests should skip

## Minimal Usage
```python
import torch
from apex_x.runtime import ParityCase, run_parity_case

case = ParityCase(
    name="tilepack-parity",
    input_factory=lambda: torch.randn(1, 16, 8, 8),
    reference_fn=lambda x: x + 1.0,
    candidate_fn=lambda x: x + 1.0,
    reference_backend="pytorch_ref",
    candidate_backend="triton",
)

report = run_parity_case(case, seed=123, deterministic=True)
print(report.to_dict())
```

## CI Notes
- Current CI can run CPU-safe parity unit tests only.
- GPU parity jobs (PyTorch vs Triton and later TensorRT) can reuse this API with:
  - backend-specific `candidate_fn`
  - profile-specific `ParityToleranceProfile`
  - optional `mismatch_ratio_limit` for relaxed thresholds

## Matrix and Sweep Harness
`run_parity_matrix_case(...)` compares multiple backends on one deterministic input draw.

Typical TensorRT parity matrix:
- `reference` vs `triton`
- `reference` vs `tensorrt`
- `triton` vs `tensorrt`

`run_parity_sweep(...)` runs many matrix cases under one profile and emits aggregate pass/fail.
Use this for:
- shape sweeps
- precision sweeps
- combined shape+precision regression packs

CUDA backend matrix examples in this repository:
- `tests/test_tensorrt_tilepack_parity.py`
  - validates reference vs triton vs tensorrt parity on CUDA (including triton-vs-tensorrt)
  - includes multi-shape parametrization.
- `tests/test_tensorrt_tilessm_parity.py`
  - validates reference vs triton vs tensorrt parity on CUDA for forward/backward scan.
  - includes multi-shape parametrization.

Minimal sweep example:
```python
import torch
from apex_x.runtime import ParityMatrixCase, run_parity_sweep

def make_backends(dtype: torch.dtype):
    return {
        "reference": lambda x: {"y": x * 2.0 + 0.5},
        "triton": lambda x: {"y": (x * 2.0 + 0.5).to(dtype).to(torch.float32)},
        "tensorrt": lambda x: {"y": (x * 2.0 + 0.5).to(dtype).to(torch.float32)},
    }

cases = [
    ParityMatrixCase(
        name="shape_8_fp16",
        input_factory=lambda: torch.randn(1, 3, 8, 8),
        backend_fns=make_backends(torch.float16),
        backend_pairs=(
            ("reference", "triton"),
            ("reference", "tensorrt"),
            ("triton", "tensorrt"),
        ),
    ),
]

sweep = run_parity_sweep(
    sweep_name="trt-e2e",
    cases=cases,
    profile_name="balanced",
)
print(sweep.to_dict())
```
