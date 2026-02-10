# Runtime Parity Framework

## Scope
`apex_x/runtime/parity.py` provides a backend-agnostic parity harness for comparing:
- PyTorch reference op (CPU/GPU)
- Triton op (GPU)
- TensorRT plugin op (GPU, integration-ready)

The framework is lightweight and supports fast, small-shape checks for CI.

## Core API
- `ParityCase`
- `run_parity_case(...)`
- `evaluate_parity_outputs(...)`
- `format_parity_report(...)`
- `ToleranceConfig`

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
  - profile-specific `ToleranceConfig`
  - optional `mismatch_ratio_limit` for relaxed thresholds

