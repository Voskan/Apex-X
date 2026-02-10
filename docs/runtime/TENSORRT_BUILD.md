# TensorRT Plugin Build and Harness

## Scope
This document describes how to build Apex-X TensorRT plugin artifacts and run the minimal runtime harness.

Directory:
- `runtime/tensorrt/`

## CMake Targets
- `apexx_trt_plugin_core` (always built)
  - static stub/core target used for metadata utilities
- `apexx_trt_plugins` (shared library, conditional)
  - built only when both TensorRT headers and CUDA compiler are found
  - linked only when TensorRT/CUDA runtime libraries are discoverable (`nvinfer`, `cudart`)
- `apexx_trt_plugin_info` (always built)
  - prints plugin build summary
- `apexx_trt_plugin_harness` (conditional)
  - built only when shared plugin library is built
  - loads the shared library dynamically and invokes minimal plugin call path
- `apexx_trt_tilepack_test` (conditional)
  - built only when real TilePack TensorRT plugin is enabled
  - runs C++ parity check for TilePack plugin enqueue output vs host reference
- `apexx_trt_tileunpackfusion_test` (conditional)
  - built only when real TileUnpackFusion TensorRT plugin is enabled
  - runs C++ priority-overlap correctness check for TileUnpackFusion enqueue path
- `apexx_trt_tilessm_test` (conditional)
  - built only when real TileSSMScan TensorRT plugin is enabled
  - runs C++ parity + benchmark harness for TileSSMScan enqueue path
- `apexx_trt_nms_decode_test` (conditional)
  - built only when real Decode+NMS TensorRT plugin is enabled
  - runs C++ parity + corner-case checks for decode/NMS enqueue path

## Environment Variables
- `TENSORRT_ROOT`
  - optional root directory containing TensorRT headers (e.g. `${TENSORRT_ROOT}/include/NvInfer.h`)
- `CUDA_HOME`
  - optional CUDA root path
- `CMAKE_PREFIX_PATH`
  - optional path list for dependency discovery
- `APEXX_TRT_PLUGIN_LIB`
  - runtime path for harness to load plugin shared library
  - if not set, harness uses platform default library name
- `APEXX_ENABLE_REAL_TILEPACK_PLUGIN`
  - CMake option to enable real TensorRT TilePack plugin build
  - default: `ON`
- `APEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN`
  - CMake option to enable real TensorRT TileUnpackFusion plugin build
  - default: `ON`
- `APEXX_ENABLE_REAL_TILESSM_PLUGIN`
  - CMake option to enable real TensorRT TileSSMScan plugin build
  - default: `ON`
- `APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN`
  - CMake option to enable real TensorRT Decode+NMS plugin build
  - default: `ON`

## Build: Default (Auto-Detect)
```bash
cd runtime/tensorrt
cmake -S . -B build
cmake --build build -j
```

Run summary:
```bash
./build/apexx_trt_plugin_info
```

If TensorRT/CUDA are unavailable:
- shared plugin target is skipped
- harness target is skipped
- core/info targets still build

## Build: Explicit TensorRT + CUDA Paths
```bash
cd runtime/tensorrt
cmake -S . -B build \
  -DAPEXX_ENABLE_REAL_TILEPACK_PLUGIN=ON \
  -DAPEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN=ON \
  -DAPEXX_ENABLE_REAL_TILESSM_PLUGIN=ON \
  -DAPEXX_ENABLE_REAL_NMS_DECODE_PLUGIN=ON \
  -DTENSORRT_INCLUDE_DIR="${TENSORRT_ROOT}/include" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
cmake --build build -j
```

If CMake cannot locate runtime libraries automatically, provide search paths via:
- `CMAKE_PREFIX_PATH`
- `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH` (platform dependent)

## Build: Force Skip Shared Plugins (Portable CI)
```bash
cd runtime/tensorrt
cmake -S . -B build \
  -DAPEXX_ENABLE_TENSORRT=OFF \
  -DAPEXX_ENABLE_CUDA=OFF \
  -DAPEXX_BUILD_PLUGIN_TEST_HARNESS=OFF
cmake --build build -j
```

## Build: Disable Real TilePack Plugin but Keep Shared Stub Library
```bash
cd runtime/tensorrt
cmake -S . -B build \
  -DAPEXX_ENABLE_REAL_TILEPACK_PLUGIN=OFF
cmake --build build -j
```

## Build: Disable Real TileUnpackFusion Plugin but Keep Shared Stub Library
```bash
cd runtime/tensorrt
cmake -S . -B build \
  -DAPEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN=OFF
cmake --build build -j
```

## Build: Disable Real TileSSMScan Plugin but Keep Shared Stub Library
```bash
cd runtime/tensorrt
cmake -S . -B build \
  -DAPEXX_ENABLE_REAL_TILESSM_PLUGIN=OFF
cmake --build build -j
```

## Build: Disable Real Decode+NMS Plugin but Keep Shared Stub Library
```bash
cd runtime/tensorrt
cmake -S . -B build \
  -DAPEXX_ENABLE_REAL_NMS_DECODE_PLUGIN=OFF
cmake --build build -j
```

## Harness Usage
If `apexx_trt_plugins` and harness are built:
```bash
cd runtime/tensorrt
./build/apexx_trt_plugin_harness ./build/libapexx_trt_plugins.so
```

Or with env var:
```bash
export APEXX_TRT_PLUGIN_LIB=./build/libapexx_trt_plugins.so
./build/apexx_trt_plugin_harness
```

macOS example library name:
- `libapexx_trt_plugins.dylib`

Windows example library name:
- `apexx_trt_plugins.dll`

## Minimal Harness Contract
Harness resolves C ABI symbols from the shared library:
- `apexx_trt_abi_version`
- `apexx_trt_build_summary_cstr`
- `apexx_trt_invoke_minimal`

Harness then:
- creates dummy float buffers
- calls minimal plugin path for:
  - `TilePack`
  - `TileSSMScan`
  - `TileUnpackFusion`
  - `DecodeNMS`

This validates loadability and basic enqueue-like plugin call flow.

## TilePack C++ Test Harness
When `apexx_trt_tilepack_test` is built:
```bash
cd runtime/tensorrt
./build/apexx_trt_tilepack_test
```

This test:
- creates TilePack plugin through creator
- serializes/deserializes plugin instance
- executes `enqueue()` with CUDA buffers
- compares output against a host reference implementation

## TileUnpackFusion C++ Test Harness
When `apexx_trt_tileunpackfusion_test` is built:
```bash
cd runtime/tensorrt
./build/apexx_trt_tileunpackfusion_test
```

This test:
- creates TileUnpackFusion plugin through creator
- executes `enqueue()` on crafted overlapping tiles with `levels[B,K]`
- validates deterministic nesting priority semantics against host reference

## TileSSMScan C++ Test + Benchmark Harness
When `apexx_trt_tilessm_test` is built:
```bash
cd runtime/tensorrt
./build/apexx_trt_tilessm_test
```

This harness:
- creates TileSSMScan plugin with direction field (`forward` and `backward` cases)
- executes `enqueue()` on small token tensors and compares to host reference recurrence
- runs a lightweight timing loop (CUDA events) and prints average latency and token throughput

## Decode+NMS C++ Test Harness
When `apexx_trt_nms_decode_test` is built:
```bash
cd runtime/tensorrt
./build/apexx_trt_nms_decode_test
```

This harness:
- compares plugin output against host decode+NMS reference
- covers corner cases:
  - no boxes after threshold
  - many candidate boxes
- prints a lightweight latency metric

## Python Parity Test (Optional)
If TensorRT Python package is available:
```bash
export APEXX_TRT_PLUGIN_LIB=/abs/path/to/libapexx_trt_plugins.so
python -m pytest -q tests/test_tensorrt_tilepack_parity.py tests/test_tensorrt_tileunpackfusion_parity.py tests/test_tensorrt_tilessm_parity.py tests/test_tensorrt_nms_decode_parity.py
```

The tests build TensorRT engines containing TilePack/TileUnpackFusion/TileSSMScan/DecodeNMS and compare output to PyTorch references.
