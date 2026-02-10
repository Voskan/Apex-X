# TensorRT DET Postprocessing (Decode + NMS)

## Scope
This note describes the TensorRT-side DET postprocessing path that keeps decode and NMS inside the engine/runtime.

Implementation files:
- `runtime/tensorrt/plugins/nms_decode.h`
- `runtime/tensorrt/plugins/nms_decode.cpp`
- `runtime/tensorrt/plugins/nms_decode.cu`

Plugin name/version/namespace:
- `DecodeNMS` / `1` / `apexx`

## Tensor Contracts
Inputs:
- `cls_logits[B,N,C]` FP16
- `box_reg[B,N,4]` FP16 (`l,t,r,b` logits)
- `quality[B,N]` FP16
- `centers[N,2]` FP16 (`cx, cy`)
- `strides[N]` FP16

Outputs:
- `boxes[B,max_det,4]` FP16 (`xyxy`)
- `scores[B,max_det]` FP16
- `class_ids[B,max_det]` INT32
- `valid_counts[B]` INT32

Plugin fields:
- `max_detections` (int, default `100`)
- `pre_nms_topk` (int, default `1000`)
- `score_threshold` (float, default `0.05`)
- `iou_threshold` (float, default `0.6`)

## Decode Semantics
Per anchor:
- `dist = softplus(clamp(box_reg, -20, 20)) * stride`
- `x1 = cx - l`, `y1 = cy - t`, `x2 = cx + r`, `y2 = cy + b`

Scores:
- `score = sigmoid(clamp(cls_logit, -60, 60)) * sigmoid(clamp(quality, -60, 60))`
- candidates below `score_threshold` are dropped
- keep top `pre_nms_topk` candidates by score, tie-break by `(anchor * C + class)` ascending

NMS:
- class-wise IoU suppression using `iou_threshold`
- deterministic final ordering by score descending, tie-break by pair-id ascending
- padded outputs:
  - `class_ids=-1` for invalid rows
  - `scores=0`, `boxes=0` for invalid rows

## EfficientNMS Integration Note
If TensorRT EfficientNMS is available and matches required deterministic ordering semantics, it can be used as an alternative backend.
Current repository path implements custom `DecodeNMS` for explicit parity with project reference behavior.

## Build and Test
CMake option:
- `APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN=ON`

C++ harness target:
- `apexx_trt_nms_decode_test`

Example:
```bash
cd runtime/tensorrt
cmake -S . -B build -DAPEXX_ENABLE_REAL_NMS_DECODE_PLUGIN=ON
cmake --build build -j
./build/apexx_trt_nms_decode_test
```

Python parity tests (when TRT Python + CUDA are available):
```bash
export APEXX_TRT_PLUGIN_LIB=/abs/path/to/libapexx_trt_plugins.so
python -m pytest -q tests/test_tensorrt_nms_decode_parity.py
```

## Corner Cases Covered
- no detections after thresholding (`valid_counts=0`)
- many candidates (`N*C` large with `pre_nms_topk` cap)
- deterministic ties on equal scores
