# TensorRT INT8 Engine Build and Calibration

## Scope
This page documents Apex-X TensorRT INT8 engine building in Python:
- `apex_x/runtime/tensorrt/builder.py`
- `apex_x/runtime/tensorrt/calibrator.py`

The builder supports:
- ONNX parse path
- direct TensorRT network-definition path
- plugin registration checks for Apex-X custom plugins
- FP16 and INT8 engine builds

## Key APIs
- `TensorRTEngineBuilder`
- `TensorRTEngineBuildConfig`
- `TensorRTEntropyCalibrator`
- `CalibratorConfig`
- `build_calibration_cache_key(...)`
- `build_calibration_dataset_digest(...)`

## Plugin Registration
Builder checks plugin registry for:
- required by default:
  - `TilePack`
  - `TileSSMScan`
  - `TileUnpackFusion`
- optional:
  - `DecodeNMS`

You can disable strict checks for non-plugin tiny-network smoke builds:
- `strict_plugin_check=False`
- `expected_plugins=()`

## FP16 Build
Set:
- `enable_fp16=True`
- `enable_int8=False`

Example:
```python
from pathlib import Path
from apex_x.runtime.tensorrt import TensorRTEngineBuilder, TensorRTEngineBuildConfig

def build_net(network, trt):
    x = network.add_input("x", trt.DataType.FLOAT, (1, 8))
    y = network.add_identity(x).get_output(0)
    y.name = "y"
    network.mark_output(y)

builder = TensorRTEngineBuilder()
result = builder.build_from_network(
    network_builder=build_net,
    engine_path=Path("artifacts/trt/tiny_fp16.engine"),
    build=TensorRTEngineBuildConfig(
        enable_fp16=True,
        enable_int8=False,
        strict_plugin_check=False,
        expected_plugins=(),
    ),
)
print(result.engine_path)
```

## INT8 Build with Calibration
Set:
- `enable_int8=True`
- provide `calibration_batches` (iterable of numpy batches)
- optionally set `calibration_cache_path`

Calibration batch formats:
- single input:
  - `np.ndarray` with shape `[B,...]`
- multi-input:
  - `dict[str, np.ndarray]` keyed by TensorRT input names

Example:
```python
import numpy as np
from pathlib import Path
from apex_x.runtime.tensorrt import TensorRTEngineBuilder, TensorRTEngineBuildConfig

def build_net(network, trt):
    x = network.add_input("x", trt.DataType.FLOAT, (1, 8))
    y = network.add_identity(x).get_output(0)
    y.name = "y"
    network.mark_output(y)

calib = [
    np.random.randn(1, 8).astype(np.float32),
    np.random.randn(1, 8).astype(np.float32),
]

builder = TensorRTEngineBuilder()
result = builder.build_from_network(
    network_builder=build_net,
    engine_path=Path("artifacts/trt/tiny_int8.engine"),
    build=TensorRTEngineBuildConfig(
        enable_fp16=True,
        enable_int8=True,
        calibration_cache_path=Path("artifacts/trt/int8.cache"),
        calibration_dataset_version="calib-v2026-02-11",
        strict_plugin_check=False,
        expected_plugins=(),
    ),
    calibration_batches=calib,
)
print(result.engine_path, result.calibration_cache_path)
```

## FP16 Layers in INT8 Builds (Spec Compliance)
Per Apex-X spec, numerically sensitive routing components stay FP16 in INT8 builds:
- router / KAN-like layers remain FP16

Builder behavior:
- when `enable_int8=True`, builder applies FP16 precision constraints to layers whose names
  match `router_fp16_layer_keywords` (default: `("router", "kan")`).
- `strict_precision_constraints=True` (default) fails build when a matched layer does not expose
  TensorRT precision/output-type APIs required to enforce FP16 constraints.

This keeps router and gating logic in FP16 while heavy tensor paths can use INT8.

Per-layer enforcement evidence is returned in build result:
- `EngineBuildResult.layer_precision_status`
  - `layer_name`
  - `matched_keyword`
  - `precision_applied`
  - `output_constraints_applied`

## Calibration Cache Governance
For `enable_int8=True` with `calibration_cache_path`:
- cache reuse is guarded by a deterministic cache key.
- default key contract binds to:
  - ONNX model hash (or network-definition signature)
  - plugin registry contract metadata (name/version/namespace)
  - precision profile
  - calibration dataset version

Dataset version source:
- explicit:
  - `TensorRTEngineBuildConfig.calibration_dataset_version`
- automatic fallback:
  - `build_calibration_dataset_digest(calibration_batches)`

Calibrator cache behavior:
- when key governance is enabled:
  - calibrator stores cache as structured blob with metadata header (`cache_key`).
  - stale or mismatched keys are invalidated automatically (cache ignored).
- when key governance is disabled:
  - legacy raw cache blobs are accepted for backward compatibility.

Build result evidence:
- `EngineBuildResult.calibration_cache_key`
- `EngineBuildResult.calibration_dataset_version`
- `EngineBuildResult.calibration_cache_path`

Recommended artifact location:
- `artifacts/trt/`

## Capability Guards
Use runtime capability detection before build:
- `detect_runtime_caps().tensorrt.python_available`
- `detect_runtime_caps().tensorrt.int8_available`
- `torch.cuda.is_available()`

If unavailable, skip build/tests gracefully.
