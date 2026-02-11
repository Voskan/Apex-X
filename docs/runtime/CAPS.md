# Runtime Capability Detection

## Scope
`apex_x/runtime/caps.py` provides a unified capability probe for runtime decisions.

Main API:
- `detect_runtime_caps(...) -> RuntimeCaps`
- `RuntimeCaps.to_dict()`
- `runtime_reason_catalog() -> dict[str, tuple[str, ...]]`

## Backend Capability Matrix (Frozen Contract)

This matrix defines required vs optional runtime capabilities used by Apex-X selection logic.

| Backend | Required Capability | Optional Capability | Failure Behavior |
| --- | --- | --- | --- |
| `cpu` | Python runtime, torch CPU tensors | None | Never blocked by GPU checks |
| `torch_cuda` | `cuda.available = true` | None | Hard fail in strict mode; fallback to CPU in permissive mode |
| `triton` | `cuda.available = true`, `triton.available = true` | Triton version metadata | Hard fail in strict mode; fallback to `torch_cuda`/CPU in permissive mode |
| `tensorrt` | `cuda.available = true`, `tensorrt.python_available = true` | `headers_available` for local build workflows, `int8_available` | Hard fail in strict mode; fallback to Triton/torch/CPU in permissive mode |

Precision capability overlays:
- INT8 on TensorRT requires:
  - TensorRT Python module
  - CUDA availability
  - `tensorrt.BuilderFlag.INT8`
- FP8 requires:
  - FP8 dtype support in torch build
  - CUDA availability
  - compute capability `sm90+` policy gate

## What Is Detected

### CUDA
- availability (`torch.cuda.is_available()`)
- device count
- active device name
- compute capability (`major.minor`)

### Triton
- module presence (`triton`)
- version (package metadata or module `__version__`)

### TensorRT
- Python module availability (`tensorrt`)
- Python package version (if available)
- C++ headers presence (`NvInfer.h` or `NvInferRuntime.h`)
  - checks optional explicit search paths
  - checks environment hints (`TENSORRT_INCLUDE_DIR`, `TRT_INCLUDE_DIR`, `TENSORRT_ROOT`, etc.)
  - checks common include directories
- INT8 availability for TensorRT usage
  - requires TensorRT Python module
  - requires CUDA availability
  - requires `tensorrt.BuilderFlag.INT8`

### FP8
- FP8 dtype presence in torch build (`float8_e4m3fn`, `float8_e5m2`)
- CUDA + compute capability gate (`sm90+` required in this baseline policy)

## Canonical Reason Codes (Deterministic Contract)

All non-available paths return a stable reason code from the catalog below.

`cuda`:
- `cuda_unavailable`
- `cuda_device_not_found`
- `cuda_device_index_out_of_range`
- `cuda_query_failed`

`triton`:
- `triton_not_installed`

`tensorrt_python`:
- `tensorrt_python_not_installed`
- `tensorrt_python_import_failed`

`tensorrt_int8`:
- `tensorrt_python_unavailable`
- `cuda_required_for_tensorrt_int8`
- `tensorrt_int8_builder_flag_missing`

`fp8`:
- `torch_build_missing_fp8_dtype`
- `fp8_requires_cuda`
- `cuda_compute_capability_unknown`
- `compute_capability_below_sm90`

Reason code notes:
- Codes are stable and CI-assertable.
- Hardware-specific details (for example exact compute capability) are exposed via dedicated fields, not dynamic reason strings.

## Usage
```python
from apex_x.runtime import detect_runtime_caps

caps = detect_runtime_caps()
print(caps.to_dict())
```

With explicit TensorRT header probe path:
```python
caps = detect_runtime_caps(header_search_paths=["/usr/local/TensorRT/include"])
```

Reason catalog:
```python
from apex_x.runtime import runtime_reason_catalog

print(runtime_reason_catalog())
```

## Fallback Contract
- Missing optional runtimes never raise by default.
- Capability object always returns with explicit `reason` fields.
- CPU-only environments return:
  - `cuda.available = False`
  - `triton.available = False`
  - `tensorrt.int8_available = False`
  - `fp8.available = False`

## Test Strategy
- Tests in `tests/test_caps_runtime.py` and `tests/test_caps_tensorrt_fp8.py`
- Designed to pass on CPU-only machines using mocks for:
  - CUDA responses
  - Triton/TensorRT module discovery
  - FP8 dtype support checks
- Reason-code contract tests assert values from `runtime_reason_catalog()`.
