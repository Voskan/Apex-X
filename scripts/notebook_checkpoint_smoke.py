#!/usr/bin/env python3
"""Reproducible smoke runner for checkpoint-image inference notebook paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from apex_x.config import ApexXConfig
from apex_x.model import (
    DetHead,
    DualPathFPN,
    PVModule,
    TeacherModel,
    TeacherModelV3,
    TimmBackboneAdapter,
    post_process_detections_per_class,
)
from apex_x.train.checkpoint import extract_model_state_dict, safe_torch_load


def _parse_devices(raw: str) -> list[str]:
    allowed = {"cpu", "cuda"}
    devices: list[str] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"Unsupported device '{value}'. Expected one of: cpu,cuda")
        if value not in devices:
            devices.append(value)
    if not devices:
        raise ValueError("No devices provided. Use at least one of: cpu,cuda")
    return devices


def _infer_model_family(state_dict: dict[str, torch.Tensor], hint: str = "auto") -> str:
    forced = hint.strip().lower()
    if forced in {"teacher", "teacher_v3"}:
        return forced
    markers = ("backbone.", "neck.", "mask_head.", "quality_head.", "rpn_objectness")
    if any(any(k.startswith(prefix) for prefix in markers) for k in state_dict):
        return "teacher_v3"
    return "teacher"


def _infer_num_classes(state_dict: dict[str, torch.Tensor], family: str) -> int:
    if family == "teacher":
        weight = state_dict.get("det_head.cls_pred.weight")
    else:
        weight = state_dict.get("det_head.stages.0.cls_head.4.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])
    return 3


def _infer_teacher_backbone_type(state_dict: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("pv_module.backbone.blocks.") for k in state_dict):
        return "timm"
    return "pv"


def _build_teacher_model_for_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    num_classes: int,
) -> TeacherModel:
    cfg = ApexXConfig()
    backbone_type = _infer_teacher_backbone_type(state_dict)
    if backbone_type == "timm":
        pv_module = TimmBackboneAdapter(
            model_name="efficientnet_b0",
            pretrained=False,
            out_indices=(2, 3, 4),
        )
        p3_ch = pv_module.p3_channels
        p4_ch = pv_module.p4_channels
        p5_ch = pv_module.p5_channels
        ff_channels = p3_ch
    else:
        pv_module = PVModule(
            in_channels=3,
            p3_channels=16,
            p4_channels=24,
            p5_channels=32,
            coarse_level="P4",
        )
        p3_ch, p4_ch, p5_ch = 16, 24, 32
        ff_channels = 16

    fpn = DualPathFPN(
        pv_p3_channels=p3_ch,
        pv_p4_channels=p4_ch,
        pv_p5_channels=p5_ch,
        ff_channels=ff_channels,
        out_channels=16,
    )
    det_head = DetHead(in_channels=16, num_classes=num_classes, hidden_channels=16, depth=1)
    return TeacherModel(
        num_classes=num_classes,
        config=cfg,
        pv_module=pv_module,
        fpn=fpn,
        det_head=det_head,
        feature_layers=("P3", "P4"),
        use_ema=True,
        ema_decay=0.99,
        use_ema_for_forward=False,
    )


def _load_state_dict_non_strict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool,
) -> tuple[torch.nn.modules.module._IncompatibleKeys, list[str]]:
    if strict:
        incompatible = model.load_state_dict(state_dict, strict=True)
        return incompatible, []

    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        expected = model_state.get(key)
        if expected is None:
            continue
        if tuple(expected.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered[key] = value
    incompatible = model.load_state_dict(filtered, strict=False)
    return incompatible, skipped


def _prepare_image_tensor(
    image: Image.Image,
    *,
    target_size: int,
    keep_aspect: bool,
    align_to: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    width, height = image.size
    target_size = int(max(64, target_size))
    if keep_aspect:
        scale = float(target_size) / float(max(height, width))
        align = max(1, int(align_to))
        out_h = max(align, int(round((height * scale) / float(align)) * align))
        out_w = max(align, int(round((width * scale) / float(align)) * align))
    else:
        out_h = target_size
        out_w = target_size
    resized = image.resize((out_w, out_h))
    image_np = np.array(resized)
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return image_np, tensor.to(device)


def _clip_boxes_to_image(boxes: torch.Tensor, *, image_h: int, image_w: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, max(0, image_w - 1))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, max(0, image_h - 1))
    return boxes


def _postprocess_teacher_v3_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    conf_threshold: float,
    nms_iou: float,
    max_dets: int,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    image_h, image_w = image_hw
    boxes = outputs.get("boxes")
    score_matrix = outputs.get("scores")
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("TeacherModelV3 output has invalid boxes tensor")
    if not isinstance(score_matrix, torch.Tensor) or score_matrix.ndim != 2:
        raise ValueError("TeacherModelV3 output has invalid scores tensor")
    if score_matrix.shape[0] != boxes.shape[0]:
        n = min(score_matrix.shape[0], boxes.shape[0])
        boxes = boxes[:n]
        score_matrix = score_matrix[:n]
    if float(score_matrix.min().item()) < 0.0 or float(score_matrix.max().item()) > 1.0:
        score_matrix = torch.sigmoid(score_matrix)
    scores, classes = score_matrix.max(dim=1)
    boxes = _clip_boxes_to_image(boxes, image_h=image_h, image_w=image_w)
    keep = scores >= float(conf_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    if boxes.numel() > 0:
        keep_idx = torchvision.ops.batched_nms(
            boxes,
            scores,
            classes,
            iou_threshold=float(nms_iou),
        )
        keep_idx = keep_idx[: int(max_dets)]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        classes = classes[keep_idx]
    else:
        keep_idx = torch.zeros((0,), dtype=torch.int64, device=boxes.device)

    masks = outputs.get("masks")
    masks_up: torch.Tensor | None = None
    if isinstance(masks, torch.Tensor) and masks.numel() > 0 and keep_idx.numel() > 0:
        masks_sel = masks[keep][keep_idx]
        if masks_sel.ndim == 4 and masks_sel.shape[1] == 1:
            masks_sel = masks_sel[:, 0]
        if masks_sel.ndim == 3:
            masks_up = F.interpolate(
                masks_sel.unsqueeze(1),
                size=(image_h, image_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
    return boxes, scores, classes, masks_up


def _load_image(image_path: str | None) -> tuple[Image.Image, str]:
    if image_path:
        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB"), str(path)
    synthetic = np.full((1024, 1024, 3), fill_value=128, dtype=np.uint8)
    return Image.fromarray(synthetic), "synthetic_1024_gray"


def _run_smoke_for_device(
    *,
    device_name: str,
    family: str,
    num_classes: int,
    state_dict: dict[str, torch.Tensor],
    image: Image.Image,
    image_name: str,
    strict_load: bool,
    inference_size: int,
    keep_aspect: bool,
    conf_threshold: float,
    nms_iou: float,
    max_dets: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "device": device_name,
        "status": "fail",
        "retried_square_resize": False,
    }
    if device_name == "cuda" and not torch.cuda.is_available():
        result["status"] = "skip"
        result["reason"] = "cuda_unavailable"
        return result
    device = torch.device(device_name)
    try:
        if family == "teacher_v3":
            model = TeacherModelV3(num_classes=num_classes)
            align_to = 14
        else:
            model = _build_teacher_model_for_state_dict(state_dict, num_classes=num_classes)
            align_to = 32

        incompatible, shape_skipped = _load_state_dict_non_strict(
            model,
            state_dict,
            strict=strict_load,
        )
        model = model.to(device).eval()
        image_np, image_tensor = _prepare_image_tensor(
            image,
            target_size=inference_size,
            keep_aspect=keep_aspect,
            align_to=align_to,
            device=device,
        )

        with torch.no_grad():
            try:
                if family == "teacher_v3":
                    outputs = model(image_tensor)
                    boxes, scores, classes, masks_up = _postprocess_teacher_v3_outputs(
                        outputs,
                        conf_threshold=conf_threshold,
                        nms_iou=nms_iou,
                        max_dets=max_dets,
                        image_hw=(int(image_np.shape[0]), int(image_np.shape[1])),
                    )
                else:
                    outputs = model(image_tensor, use_ema=False)
                    det = post_process_detections_per_class(
                        outputs.logits_by_level,
                        outputs.boxes_by_level,
                        outputs.quality_by_level,
                        conf_threshold=conf_threshold,
                        nms_threshold=nms_iou,
                        max_detections=max_dets,
                    )[0]
                    boxes = det["boxes"]
                    scores = det["scores"]
                    classes = det["classes"]
                    masks_up = None
                    if isinstance(outputs.masks, torch.Tensor) and outputs.masks.numel() > 0:
                        masks = outputs.masks[0]
                        if masks.ndim == 4 and masks.shape[1] == 1:
                            masks = masks[:, 0]
                        if masks.ndim == 3:
                            masks_up = F.interpolate(
                                masks.unsqueeze(1),
                                size=(image_np.shape[0], image_np.shape[1]),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(1)
            except ValueError as exc:
                if family != "teacher_v3" or "Cannot reshape" not in str(exc):
                    raise
                result["retried_square_resize"] = True
                image_np, image_tensor = _prepare_image_tensor(
                    image,
                    target_size=inference_size,
                    keep_aspect=False,
                    align_to=14,
                    device=device,
                )
                outputs = model(image_tensor)
                boxes, scores, classes, masks_up = _postprocess_teacher_v3_outputs(
                    outputs,
                    conf_threshold=conf_threshold,
                    nms_iou=nms_iou,
                    max_dets=max_dets,
                    image_hw=(int(image_np.shape[0]), int(image_np.shape[1])),
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.update(
            {
                "status": "ok",
                "image": image_name,
                "image_hw": [int(image_np.shape[0]), int(image_np.shape[1])],
                "align_to": int(align_to),
                "detection_count": int(boxes.shape[0]),
                "max_score": float(scores.max().item()) if scores.numel() > 0 else 0.0,
                "mask_count": int(masks_up.shape[0]) if masks_up is not None else 0,
                "missing_keys": int(len(getattr(incompatible, "missing_keys", []))),
                "unexpected_keys": int(len(getattr(incompatible, "unexpected_keys", []))),
                "shape_skipped": int(len(shape_skipped)),
                "elapsed_ms": float(elapsed_ms),
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - smoke path keeps diagnostics payload.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_ms"] = float(elapsed_ms)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible checkpoint-image smoke for notebook inference paths.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--image", default="", help="Optional path to image file")
    parser.add_argument("--devices", default="cpu,cuda", help="Comma list: cpu,cuda")
    parser.add_argument("--model-hint", default="auto", choices=["auto", "teacher", "teacher_v3"])
    parser.add_argument("--strict-load", action="store_true", help="Use strict state_dict load")
    parser.add_argument("--inference-size", type=int, default=1024)
    parser.add_argument("--keep-aspect", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument(
        "--output-json",
        default="artifacts/notebook_smoke/report.json",
        help="Smoke artifact path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    devices = _parse_devices(args.devices)
    payload = safe_torch_load(args.checkpoint, map_location="cpu")
    state_dict, checkpoint_format = extract_model_state_dict(payload)
    family = _infer_model_family(state_dict, args.model_hint)
    classes = _infer_num_classes(state_dict, family)
    image, image_name = _load_image(args.image.strip())

    runs = [
        _run_smoke_for_device(
            device_name=device_name,
            family=family,
            num_classes=classes,
            state_dict=state_dict,
            image=image,
            image_name=image_name,
            strict_load=bool(args.strict_load),
            inference_size=int(args.inference_size),
            keep_aspect=bool(args.keep_aspect),
            conf_threshold=float(args.conf_threshold),
            nms_iou=float(args.nms_iou),
            max_dets=int(args.max_dets),
        )
        for device_name in devices
    ]

    output = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "checkpoint_format": checkpoint_format,
        "model_family": family,
        "num_classes": int(classes),
        "settings": {
            "devices": devices,
            "strict_load": bool(args.strict_load),
            "inference_size": int(args.inference_size),
            "keep_aspect": bool(args.keep_aspect),
            "conf_threshold": float(args.conf_threshold),
            "nms_iou": float(args.nms_iou),
            "max_dets": int(args.max_dets),
        },
        "runs": runs,
    }
    out_path = Path(args.output_json).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fail_count = sum(1 for run in runs if run.get("status") == "fail")
    ok_count = sum(1 for run in runs if run.get("status") == "ok")
    print(f"notebook_smoke output={out_path} ok={ok_count} fail={fail_count}")
    return 1 if (fail_count > 0 or ok_count == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
