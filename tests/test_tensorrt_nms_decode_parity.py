from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from apex_x.infer import deterministic_nms


def _maybe_import_tensorrt():
    try:
        import tensorrt as trt
    except Exception as exc:  # pragma: no cover - runtime dependent
        pytest.skip(f"TensorRT Python is not available: {type(exc).__name__}: {exc}")
    return trt


def _load_plugin_library() -> Path:
    lib_env = os.getenv("APEXX_TRT_PLUGIN_LIB", "").strip()
    if not lib_env:
        pytest.skip("APEXX_TRT_PLUGIN_LIB is not set")
    lib_path = Path(lib_env).expanduser().resolve()
    if not lib_path.exists():
        pytest.skip(f"APEXX_TRT_PLUGIN_LIB does not exist: {lib_path}")
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    return lib_path


def _get_creator(trt, *, name: str):
    registry = trt.get_plugin_registry()
    creator = None
    if hasattr(registry, "get_plugin_creator"):
        creator = registry.get_plugin_creator(name, "1", "apexx")
    if creator is not None:
        return creator
    for item in registry.plugin_creator_list:
        if item.name == name and getattr(item, "plugin_namespace", "") == "apexx":
            return item
    pytest.skip(f"{name} plugin creator was not registered")


def _build_engine(
    trt,
    creator,
    *,
    batch: int,
    anchors: int,
    classes: int,
    max_detections: int,
    pre_nms_topk: int,
    score_threshold: float,
    iou_threshold: float,
):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    cls_logits = network.add_input("cls_logits", trt.DataType.HALF, (batch, anchors, classes))
    box_reg = network.add_input("box_reg", trt.DataType.HALF, (batch, anchors, 4))
    quality = network.add_input("quality", trt.DataType.HALF, (batch, anchors))
    centers = network.add_input("centers", trt.DataType.HALF, (anchors, 2))
    strides = network.add_input("strides", trt.DataType.HALF, (anchors,))

    fields = [
        trt.PluginField(
            "max_detections",
            np.array([max_detections], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "pre_nms_topk",
            np.array([pre_nms_topk], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "score_threshold",
            np.array([score_threshold], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        trt.PluginField(
            "iou_threshold",
            np.array([iou_threshold], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
    ]
    plugin = creator.create_plugin("nms_decode_pytest", trt.PluginFieldCollection(fields))
    if plugin is None:
        pytest.skip("Failed to create DecodeNMS plugin instance")

    layer = network.add_plugin_v2([cls_logits, box_reg, quality, centers, strides], plugin)
    out_boxes = layer.get_output(0)
    out_scores = layer.get_output(1)
    out_class_ids = layer.get_output(2)
    out_valid = layer.get_output(3)
    out_boxes.name = "out_boxes"
    out_scores.name = "out_scores"
    out_class_ids.name = "out_class_ids"
    out_valid.name = "out_valid"
    network.mark_output(out_boxes)
    network.mark_output(out_scores)
    network.mark_output(out_class_ids)
    network.mark_output(out_valid)

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    else:  # pragma: no cover - TRT API version dependent
        config.max_workspace_size = 1 << 20

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(serialized)
    return builder.build_engine(network, config)  # pragma: no cover - TRT API version dependent


def _execute_engine(
    engine,
    *,
    cls_logits: torch.Tensor,
    box_reg: torch.Tensor,
    quality: torch.Tensor,
    centers: torch.Tensor,
    strides: torch.Tensor,
    out_boxes: torch.Tensor,
    out_scores: torch.Tensor,
    out_class_ids: torch.Tensor,
    out_valid: torch.Tensor,
) -> None:
    context = engine.create_execution_context()
    if hasattr(engine, "num_bindings"):
        bindings = [0] * int(engine.num_bindings)
        bindings[int(engine.get_binding_index("cls_logits"))] = int(cls_logits.data_ptr())
        bindings[int(engine.get_binding_index("box_reg"))] = int(box_reg.data_ptr())
        bindings[int(engine.get_binding_index("quality"))] = int(quality.data_ptr())
        bindings[int(engine.get_binding_index("centers"))] = int(centers.data_ptr())
        bindings[int(engine.get_binding_index("strides"))] = int(strides.data_ptr())
        bindings[int(engine.get_binding_index("out_boxes"))] = int(out_boxes.data_ptr())
        bindings[int(engine.get_binding_index("out_scores"))] = int(out_scores.data_ptr())
        bindings[int(engine.get_binding_index("out_class_ids"))] = int(out_class_ids.data_ptr())
        bindings[int(engine.get_binding_index("out_valid"))] = int(out_valid.data_ptr())
        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        torch.cuda.synchronize()
        return

    context.set_tensor_address("cls_logits", int(cls_logits.data_ptr()))  # pragma: no cover
    context.set_tensor_address("box_reg", int(box_reg.data_ptr()))  # pragma: no cover
    context.set_tensor_address("quality", int(quality.data_ptr()))  # pragma: no cover
    context.set_tensor_address("centers", int(centers.data_ptr()))  # pragma: no cover
    context.set_tensor_address("strides", int(strides.data_ptr()))  # pragma: no cover
    context.set_tensor_address("out_boxes", int(out_boxes.data_ptr()))  # pragma: no cover
    context.set_tensor_address("out_scores", int(out_scores.data_ptr()))  # pragma: no cover
    context.set_tensor_address("out_class_ids", int(out_class_ids.data_ptr()))  # pragma: no cover
    context.set_tensor_address("out_valid", int(out_valid.data_ptr()))  # pragma: no cover
    stream_ptr = int(torch.cuda.current_stream().cuda_stream)
    context.execute_async_v3(stream_ptr)  # pragma: no cover
    torch.cuda.synchronize()


def _decode_nms_reference(
    *,
    cls_logits: torch.Tensor,
    box_reg: torch.Tensor,
    quality: torch.Tensor,
    centers: torch.Tensor,
    strides: torch.Tensor,
    max_detections: int,
    pre_nms_topk: int,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, anchors, classes = cls_logits.shape
    out_boxes = torch.zeros(
        (batch, max_detections, 4), dtype=cls_logits.dtype, device=cls_logits.device
    )
    out_scores = torch.zeros(
        (batch, max_detections),
        dtype=cls_logits.dtype,
        device=cls_logits.device,
    )
    out_class_ids = torch.full(
        (batch, max_detections), -1, dtype=torch.int32, device=cls_logits.device
    )
    out_valid = torch.zeros((batch,), dtype=torch.int32, device=cls_logits.device)

    for b in range(batch):
        q = torch.sigmoid(
            torch.nan_to_num(quality[b], nan=0.0, posinf=60.0, neginf=-60.0)
        ).unsqueeze(1)
        score_mat = (
            torch.sigmoid(
                torch.nan_to_num(cls_logits[b], nan=0.0, posinf=60.0, neginf=-60.0)
            )
            * q
        )
        pairs = torch.nonzero(score_mat >= score_threshold, as_tuple=False)
        if pairs.numel() == 0:
            continue

        cand_scores = score_mat[pairs[:, 0], pairs[:, 1]]
        pair_ids = pairs[:, 0] * classes + pairs[:, 1]
        order = list(range(int(cand_scores.numel())))
        order.sort(
            key=lambda i: (-float(cand_scores[i].item()), int(pair_ids[i].item())),
        )
        order = order[:pre_nms_topk]
        if not order:
            continue
        order_t = torch.tensor(order, device=cls_logits.device, dtype=torch.int64)
        sel_pairs = pairs[order_t]
        sel_scores = cand_scores[order_t]

        anchor_ids = sel_pairs[:, 0]
        cls_ids = sel_pairs[:, 1].to(dtype=torch.int64)
        stride = strides[anchor_ids].to(dtype=cls_logits.dtype)
        center = centers[anchor_ids].to(dtype=cls_logits.dtype)
        ltrb_logits = torch.nan_to_num(
            box_reg[b, anchor_ids],
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)
        ltrb = torch.nn.functional.softplus(ltrb_logits) * stride.unsqueeze(1)
        boxes = torch.stack(
            (
                center[:, 0] - ltrb[:, 0],
                center[:, 1] - ltrb[:, 1],
                center[:, 0] + ltrb[:, 2],
                center[:, 1] + ltrb[:, 3],
            ),
            dim=1,
        )

        keep = deterministic_nms(
            boxes=boxes,
            scores=sel_scores,
            class_ids=cls_ids,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        valid = min(int(keep.numel()), max_detections)
        out_valid[b] = valid
        if valid == 0:
            continue
        sel = keep[:valid]
        out_boxes[b, :valid] = boxes[sel]
        out_scores[b, :valid] = sel_scores[sel]
        out_class_ids[b, :valid] = cls_ids[sel].to(dtype=torch.int32)

    return out_boxes, out_scores, out_class_ids, out_valid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("score_threshold", "iou_threshold"),
    [
        (0.2, 0.5),
        (0.9999, 0.5),  # no boxes corner case
    ],
)
def test_tensorrt_decode_nms_parity(
    score_threshold: float,
    iou_threshold: float,
) -> None:
    trt = _maybe_import_tensorrt()
    _load_plugin_library()
    creator = _get_creator(trt, name="DecodeNMS")

    batch, anchors, classes = 1, 48, 4
    max_detections, pre_nms_topk = 20, 120
    torch.manual_seed(123)
    cls_logits = torch.randn((batch, anchors, classes), device="cuda", dtype=torch.float16)
    box_reg = torch.randn((batch, anchors, 4), device="cuda", dtype=torch.float16) + 1.5
    quality = torch.randn((batch, anchors), device="cuda", dtype=torch.float16) * 0.5 + 0.3
    centers = torch.zeros((anchors, 2), device="cuda", dtype=torch.float16)
    centers[:, 0] = torch.arange(anchors, device="cuda", dtype=torch.float16) * 2.0 + 4.0
    centers[:, 1] = torch.arange(anchors, device="cuda", dtype=torch.float16) * 1.0 + 6.0
    strides = torch.full((anchors,), 8.0, device="cuda", dtype=torch.float16)

    engine = _build_engine(
        trt,
        creator,
        batch=batch,
        anchors=anchors,
        classes=classes,
        max_detections=max_detections,
        pre_nms_topk=pre_nms_topk,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    if engine is None:
        pytest.skip("TensorRT engine build returned None")

    out_boxes = torch.empty((batch, max_detections, 4), device="cuda", dtype=torch.float16)
    out_scores = torch.empty((batch, max_detections), device="cuda", dtype=torch.float16)
    out_class_ids = torch.empty((batch, max_detections), device="cuda", dtype=torch.int32)
    out_valid = torch.empty((batch,), device="cuda", dtype=torch.int32)
    _execute_engine(
        engine,
        cls_logits=cls_logits,
        box_reg=box_reg,
        quality=quality,
        centers=centers,
        strides=strides,
        out_boxes=out_boxes,
        out_scores=out_scores,
        out_class_ids=out_class_ids,
        out_valid=out_valid,
    )

    ref_boxes, ref_scores, ref_class_ids, ref_valid = _decode_nms_reference(
        cls_logits=cls_logits,
        box_reg=box_reg,
        quality=quality,
        centers=centers,
        strides=strides,
        max_detections=max_detections,
        pre_nms_topk=pre_nms_topk,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    assert torch.equal(out_valid, ref_valid)
    assert torch.equal(out_class_ids, ref_class_ids)
    assert torch.allclose(out_scores, ref_scores, atol=6e-2, rtol=6e-2)
    assert torch.allclose(out_boxes, ref_boxes, atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensorrt_decode_nms_many_boxes_corner_case() -> None:
    trt = _maybe_import_tensorrt()
    _load_plugin_library()
    creator = _get_creator(trt, name="DecodeNMS")

    batch, anchors, classes = 1, 128, 6
    max_detections, pre_nms_topk = 32, 512
    score_threshold, iou_threshold = 0.01, 0.6
    torch.manual_seed(999)
    cls_logits = torch.randn((batch, anchors, classes), device="cuda", dtype=torch.float16) + 1.0
    box_reg = torch.randn((batch, anchors, 4), device="cuda", dtype=torch.float16) + 2.0
    quality = torch.randn((batch, anchors), device="cuda", dtype=torch.float16) + 0.5
    centers = torch.zeros((anchors, 2), device="cuda", dtype=torch.float16)
    centers[:, 0] = torch.arange(anchors, device="cuda", dtype=torch.float16) * 1.2
    centers[:, 1] = torch.arange(anchors, device="cuda", dtype=torch.float16) * 0.9
    strides = torch.full((anchors,), 8.0, device="cuda", dtype=torch.float16)

    engine = _build_engine(
        trt,
        creator,
        batch=batch,
        anchors=anchors,
        classes=classes,
        max_detections=max_detections,
        pre_nms_topk=pre_nms_topk,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    if engine is None:
        pytest.skip("TensorRT engine build returned None")

    out_boxes = torch.empty((batch, max_detections, 4), device="cuda", dtype=torch.float16)
    out_scores = torch.empty((batch, max_detections), device="cuda", dtype=torch.float16)
    out_class_ids = torch.empty((batch, max_detections), device="cuda", dtype=torch.int32)
    out_valid = torch.empty((batch,), device="cuda", dtype=torch.int32)
    _execute_engine(
        engine,
        cls_logits=cls_logits,
        box_reg=box_reg,
        quality=quality,
        centers=centers,
        strides=strides,
        out_boxes=out_boxes,
        out_scores=out_scores,
        out_class_ids=out_class_ids,
        out_valid=out_valid,
    )
    assert int(out_valid[0].item()) <= max_detections
    assert torch.all(out_class_ids[0, int(out_valid[0].item()) :] == -1)
    assert torch.all(out_scores[0, int(out_valid[0].item()) :] == 0)
