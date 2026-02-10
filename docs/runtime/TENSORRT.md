# TensorRT Runtime Scaffolding

## Scope
This document describes the C++ TensorRT plugin scaffolding under `runtime/tensorrt/`.

Detailed build and harness instructions:
- `docs/runtime/TENSORRT_BUILD.md`
- `docs/runtime/TENSORRT_INT8.md` (Python engine builder + INT8 calibration)

Current status:
- build system and plugin contracts are in place
- TilePack now has a real TensorRT plugin implementation path (guarded)
- TileUnpackFusion now has a real TensorRT plugin implementation path (guarded)
- TileSSMScan now has a real TensorRT plugin implementation path (guarded)
- Decode+NMS now has a real TensorRT plugin implementation path (guarded)
- remaining plugins are stubs
- code is guarded to build on machines without TensorRT/CUDA installed

## Directory Layout
- `runtime/tensorrt/CMakeLists.txt`
- `runtime/tensorrt/include/apexx_trt/`
  - `common.hpp`
  - `plugin_stub.hpp`
  - `tile_pack_plugin.hpp`
  - `tile_ssm_scan_plugin.hpp`
  - `tile_unpack_fusion_plugin.hpp`
  - `decode_nms_plugin.hpp` (optional)
- `runtime/tensorrt/src/`
  - `common.cpp`
  - `tile_pack_plugin.cpp`
  - `tile_ssm_scan_plugin.cpp`
  - `tile_unpack_fusion_plugin.cpp`
  - `decode_nms_plugin.cpp`
  - `plugin_info_main.cpp`

## Contract Mapping
Contracts are inherited from:
- `docs/runtime/PLUGIN_SPEC.md`
- `docs/ENGINEERING_SPEC.md`

Mapped plugin stubs:
- `TilePack`
  - contract: `F[B,C,Hf,Wf] + idx[B,K] -> P[B,K,C,t,t] + meta`
  - real TensorRT plugin path available under `runtime/tensorrt/plugins/tilepack.*`
- `TileSSMScan`
  - contract: `tokens + recurrent params (+optional init_state) -> y + next_state`
  - supports forward direction and backward direction flag
  - real TensorRT plugin path available under `runtime/tensorrt/plugins/tilessm_scan.*`
- `TileUnpackFusion`
  - contract: `F_base + P_out + idx + levels (+ optional alpha) -> F_merged`
  - overlap rule: deterministic priority overwrite for nesting (`L2 > L1 > L0`)
  - real TensorRT plugin path available under `runtime/tensorrt/plugins/tileunpackfusion.*`
- `DecodeNMS` (optional)
  - contract: `cls/box/quality + centers/strides -> boxes/scores/class_ids/valid_counts`
  - real TensorRT plugin path available under `runtime/tensorrt/plugins/nms_decode.*`

## Build Guard Behavior
`CMakeLists.txt` probes platform features and sets:
- `APEXX_ENABLE_TENSORRT` to `1` only if TensorRT headers are found
- `APEXX_ENABLE_CUDA` to `1` only if a CUDA compiler is available

If unavailable, it builds a stub-only static library with the same API surface.

## Build Steps
Use:
- `docs/runtime/TENSORRT_BUILD.md`

## Output
- shared library (when TRT+CUDA available): `apexx_trt_plugins`
- static core library (always): `apexx_trt_plugin_core`
- utility executable: `apexx_trt_plugin_info`
  - prints build summary and plugin availability flags
- harness executable (when shared library built): `apexx_trt_plugin_harness`
  - dynamically loads shared library and invokes minimal plugin call path

## Remaining Work
- map stubs to real TensorRT `IPluginV2DynamicExt` implementations
- add shape inference checks and serialization blobs
- add CUDA kernels for packed gather/scan/scatter fast path
- add parity tests vs CPU reference ops (pack/unpack/fusion/decode)
- `runtime/tensorrt/plugins/`
  - `tilepack.h`
  - `tilepack.cpp`
  - `tilepack.cu`
  - `tileunpackfusion.h`
  - `tileunpackfusion.cpp`
  - `tileunpackfusion.cu`
  - `tilessm_scan.h`
  - `tilessm_scan.cpp`
  - `tilessm_scan.cu`
  - `nms_decode.h`
  - `nms_decode.cpp`
  - `nms_decode.cu`
