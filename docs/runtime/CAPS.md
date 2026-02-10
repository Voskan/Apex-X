# Runtime Capability Detection

## Scope
`apex_x/runtime/caps.py` provides a unified capability probe for runtime decisions.

Main API:
- `detect_runtime_caps(...) -> RuntimeCaps`
- `RuntimeCaps.to_dict()`

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

