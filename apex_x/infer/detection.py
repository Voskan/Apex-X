from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as f

from apex_x.model import DetHeadOutput

_DEFAULT_LEVEL_STRIDES: Final[dict[str, int]] = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}
_EPS: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class DetectionCandidates:
    """Decoded per-image detection candidates before NMS."""

    boxes: list[Tensor]  # each: [Ni, 4] in xyxy image coordinates
    scores: list[Tensor]  # each: [Ni]
    class_ids: list[Tensor]  # each: [Ni] int64


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Batched detection outputs after deterministic NMS."""

    boxes: Tensor  # [B, N, 4] xyxy
    scores: Tensor  # [B, N]
    class_ids: Tensor  # [B, N] int64, -1 for padded entries
    valid_counts: Tensor  # [B] int64


def _level_sort_key(level: str) -> tuple[int, int | str]:
    if level.startswith("P") and level[1:].isdigit():
        return (0, int(level[1:]))
    return (1, level)


def _resolve_level_strides(
    levels: list[str],
    strides: dict[str, int] | None,
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    source = _DEFAULT_LEVEL_STRIDES if strides is None else strides
    for level in levels:
        if level not in source:
            raise ValueError(f"missing stride for level {level}")
        stride = int(source[level])
        if stride <= 0:
            raise ValueError(f"stride for level {level} must be > 0")
        resolved[level] = stride
    return resolved


def _validate_level_tensors(
    level: str,
    cls_logits: Tensor,
    box_reg: Tensor,
    quality: Tensor,
    *,
    batch_size: int | None,
) -> int:
    if cls_logits.ndim != 4:
        raise ValueError(f"{level}.cls_logits must be [B,C,H,W]")
    if box_reg.ndim != 4 or box_reg.shape[1] != 4:
        raise ValueError(f"{level}.box_reg must be [B,4,H,W]")
    if quality.ndim != 4 or quality.shape[1] != 1:
        raise ValueError(f"{level}.quality must be [B,1,H,W]")
    if cls_logits.shape[0] != box_reg.shape[0] or cls_logits.shape[0] != quality.shape[0]:
        raise ValueError(f"{level} tensors must share batch size")
    if cls_logits.shape[2:] != box_reg.shape[2:] or cls_logits.shape[2:] != quality.shape[2:]:
        raise ValueError(f"{level} tensors must share spatial shape")
    if batch_size is not None and cls_logits.shape[0] != batch_size:
        raise ValueError("all levels must share the same batch size")

    if not torch.isfinite(cls_logits).all():
        raise ValueError(f"{level}.cls_logits must be finite")
    if not torch.isfinite(box_reg).all():
        raise ValueError(f"{level}.box_reg must be finite")
    if not torch.isfinite(quality).all():
        raise ValueError(f"{level}.quality must be finite")

    return int(cls_logits.shape[0])


def _decode_level_boxes(
    box_reg: Tensor,
    *,
    stride: int,
    image_size: tuple[int, int] | None,
) -> Tensor:
    # Box parameterization: l/t/r/b distances from anchor center.
    clean = torch.nan_to_num(box_reg, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    distances = f.softplus(clean) * float(stride)

    batch_size, _, h, w = distances.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=distances.device, dtype=distances.dtype),
        torch.arange(w, device=distances.device, dtype=distances.dtype),
        indexing="ij",
    )
    cx = (xx + 0.5) * float(stride)
    cy = (yy + 0.5) * float(stride)
    cx = cx.unsqueeze(0)
    cy = cy.unsqueeze(0)

    x1 = cx - distances[:, 0]
    y1 = cy - distances[:, 1]
    x2 = cx + distances[:, 2]
    y2 = cy + distances[:, 3]
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)  # [B,H,W,4]

    if image_size is not None:
        image_h, image_w = image_size
        if image_h <= 0 or image_w <= 0:
            raise ValueError("image_size must be (H,W) with positive dimensions")
        boxes[..., 0] = boxes[..., 0].clamp(0.0, float(image_w))
        boxes[..., 2] = boxes[..., 2].clamp(0.0, float(image_w))
        boxes[..., 1] = boxes[..., 1].clamp(0.0, float(image_h))
        boxes[..., 3] = boxes[..., 3].clamp(0.0, float(image_h))

    return boxes.reshape(batch_size, h * w, 4)


def _decode_level_scores(cls_logits: Tensor, quality_logits: Tensor) -> Tensor:
    cls_prob = torch.sigmoid(torch.nan_to_num(cls_logits, nan=0.0, posinf=60.0, neginf=-60.0))
    qual_prob = torch.sigmoid(torch.nan_to_num(quality_logits, nan=0.0, posinf=60.0, neginf=-60.0))
    combined = cls_prob * qual_prob
    return combined.permute(0, 2, 3, 1).reshape(cls_logits.shape[0], -1, cls_logits.shape[1])


def decode_anchor_free_candidates(
    det_output: DetHeadOutput,
    *,
    image_size: tuple[int, int] | None = None,
    strides: dict[str, int] | None = None,
    score_threshold: float = 0.05,
    pre_nms_topk: int = 1000,
) -> DetectionCandidates:
    """Decode DET logits/regression maps into per-image candidate tensors."""
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be in [0,1]")
    if pre_nms_topk <= 0:
        raise ValueError("pre_nms_topk must be > 0")

    cls_levels = set(det_output.cls_logits.keys())
    box_levels = set(det_output.box_reg.keys())
    qual_levels = set(det_output.quality.keys())
    if cls_levels != box_levels or cls_levels != qual_levels:
        raise ValueError("DetHeadOutput levels must match across cls_logits/box_reg/quality")
    if not cls_levels:
        raise ValueError("DetHeadOutput must contain at least one level")

    ordered_levels = sorted(cls_levels, key=_level_sort_key)
    resolved_strides = _resolve_level_strides(ordered_levels, strides)

    batch_size: int | None = None
    per_batch_boxes: list[list[Tensor]] = []
    per_batch_scores: list[list[Tensor]] = []
    per_batch_classes: list[list[Tensor]] = []

    for level in ordered_levels:
        cls_logits = det_output.cls_logits[level]
        box_reg = det_output.box_reg[level]
        quality_logits = det_output.quality[level]
        batch_size = _validate_level_tensors(
            level,
            cls_logits,
            box_reg,
            quality_logits,
            batch_size=batch_size,
        )
        if not per_batch_boxes:
            per_batch_boxes = [[] for _ in range(batch_size)]
            per_batch_scores = [[] for _ in range(batch_size)]
            per_batch_classes = [[] for _ in range(batch_size)]

        level_boxes = _decode_level_boxes(
            box_reg,
            stride=resolved_strides[level],
            image_size=image_size,
        )
        level_scores = _decode_level_scores(cls_logits, quality_logits)

        for batch_idx in range(batch_size):
            score_matrix = level_scores[batch_idx]  # [A,C]
            candidate_pairs = torch.nonzero(score_matrix >= score_threshold, as_tuple=False)
            if candidate_pairs.numel() == 0:
                continue

            candidate_scores = score_matrix[candidate_pairs[:, 0], candidate_pairs[:, 1]]
            score_order = torch.argsort(candidate_scores, descending=True, stable=True)
            if score_order.numel() > pre_nms_topk:
                score_order = score_order[:pre_nms_topk]

            selected_pairs = candidate_pairs[score_order]
            selected_scores = candidate_scores[score_order]
            selected_anchor_ids = selected_pairs[:, 0]
            selected_class_ids = selected_pairs[:, 1].to(dtype=torch.int64)
            selected_boxes = level_boxes[batch_idx, selected_anchor_ids]

            per_batch_boxes[batch_idx].append(selected_boxes)
            per_batch_scores[batch_idx].append(selected_scores)
            per_batch_classes[batch_idx].append(selected_class_ids)

    assert batch_size is not None
    out_boxes: list[Tensor] = []
    out_scores: list[Tensor] = []
    out_classes: list[Tensor] = []

    for batch_idx in range(batch_size):
        if not per_batch_boxes[batch_idx]:
            device = det_output.cls_logits[ordered_levels[0]].device
            dtype = det_output.cls_logits[ordered_levels[0]].dtype
            out_boxes.append(torch.zeros((0, 4), device=device, dtype=dtype))
            out_scores.append(torch.zeros((0,), device=device, dtype=dtype))
            out_classes.append(torch.zeros((0,), device=device, dtype=torch.int64))
            continue
        out_boxes.append(torch.cat(per_batch_boxes[batch_idx], dim=0))
        out_scores.append(torch.cat(per_batch_scores[batch_idx], dim=0))
        out_classes.append(torch.cat(per_batch_classes[batch_idx], dim=0))

    return DetectionCandidates(boxes=out_boxes, scores=out_scores, class_ids=out_classes)


def _canonicalize_xyxy(boxes: Tensor) -> Tensor:
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    return torch.stack((x1, y1, x2, y2), dim=1)


def _iou_with_one(box: Tensor, others: Tensor) -> Tensor:
    if others.ndim != 2 or others.shape[1] != 4:
        raise ValueError("others must be [N,4]")
    if others.numel() == 0:
        return others.new_zeros((0,), dtype=box.dtype)

    inter_x1 = torch.maximum(box[0], others[:, 0])
    inter_y1 = torch.maximum(box[1], others[:, 1])
    inter_x2 = torch.minimum(box[2], others[:, 2])
    inter_y2 = torch.minimum(box[3], others[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp(min=0.0) * (box[3] - box[1]).clamp(min=0.0)
    area_others = (others[:, 2] - others[:, 0]).clamp(min=0.0) * (
        others[:, 3] - others[:, 1]
    ).clamp(min=0.0)
    union = (area_box + area_others - inter).clamp(min=_EPS)
    return inter / union


def deterministic_nms(
    boxes: Tensor,
    scores: Tensor,
    class_ids: Tensor,
    *,
    iou_threshold: float = 0.6,
    max_detections: int = 100,
) -> Tensor:
    """Deterministic class-wise NMS with stable score tie-breaking."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be [N,4]")
    if scores.ndim != 1 or class_ids.ndim != 1:
        raise ValueError("scores and class_ids must be [N]")
    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != class_ids.shape[0]:
        raise ValueError("boxes, scores, and class_ids must share N")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0,1]")
    if max_detections < 0:
        raise ValueError("max_detections must be >= 0")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must be finite")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    n = boxes.shape[0]
    if n == 0 or max_detections == 0:
        return torch.zeros((0,), dtype=torch.int64, device=boxes.device)

    boxes_xyxy = _canonicalize_xyxy(boxes)
    class_ids_i64 = class_ids.to(dtype=torch.int64)
    unique_classes = torch.unique(class_ids_i64, sorted=True)

    kept: list[int] = []
    for cls in unique_classes.tolist():
        cls_mask = class_ids_i64 == int(cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()
        if cls_indices.numel() == 0:
            continue
        cls_scores = scores[cls_indices]
        cls_order = torch.argsort(cls_scores, descending=True, stable=True)
        active = cls_indices[cls_order]

        while active.numel() > 0:
            keep_idx = int(active[0].item())
            kept.append(keep_idx)
            if active.numel() == 1:
                break
            rest = active[1:]
            iou = _iou_with_one(boxes_xyxy[keep_idx], boxes_xyxy[rest])
            active = rest[iou <= iou_threshold]

    kept_sorted = sorted(kept, key=lambda idx: (-float(scores[idx].item()), idx))
    if len(kept_sorted) > max_detections:
        kept_sorted = kept_sorted[:max_detections]
    return torch.tensor(kept_sorted, device=boxes.device, dtype=torch.int64)


def batched_deterministic_nms(
    candidates: DetectionCandidates,
    *,
    iou_threshold: float = 0.6,
    max_detections: int = 100,
) -> DetectionBatch:
    """Apply deterministic NMS to a batch of decoded candidates and pad outputs."""
    if max_detections < 0:
        raise ValueError("max_detections must be >= 0")
    if len(candidates.boxes) != len(candidates.scores) or len(candidates.boxes) != len(
        candidates.class_ids
    ):
        raise ValueError("candidates lists must have equal batch length")

    batch_size = len(candidates.boxes)
    if batch_size == 0:
        empty_i64 = torch.zeros((0,), dtype=torch.int64)
        return DetectionBatch(
            boxes=torch.zeros((0, max_detections, 4), dtype=torch.float32),
            scores=torch.zeros((0, max_detections), dtype=torch.float32),
            class_ids=torch.zeros((0, max_detections), dtype=torch.int64),
            valid_counts=empty_i64,
        )

    first_nonempty = next((tensor for tensor in candidates.boxes if tensor.numel() > 0), None)
    if first_nonempty is None:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = first_nonempty.device
        dtype = first_nonempty.dtype

    out_boxes = torch.zeros((batch_size, max_detections, 4), device=device, dtype=dtype)
    out_scores = torch.zeros((batch_size, max_detections), device=device, dtype=dtype)
    out_class_ids = torch.full(
        (batch_size, max_detections),
        -1,
        device=device,
        dtype=torch.int64,
    )
    valid_counts = torch.zeros((batch_size,), device=device, dtype=torch.int64)

    for batch_idx in range(batch_size):
        boxes = candidates.boxes[batch_idx]
        scores = candidates.scores[batch_idx]
        class_ids = candidates.class_ids[batch_idx]
        keep = deterministic_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        valid = min(int(keep.numel()), max_detections)
        valid_counts[batch_idx] = valid
        if valid == 0:
            continue
        selected = keep[:valid]
        out_boxes[batch_idx, :valid] = boxes[selected]
        out_scores[batch_idx, :valid] = scores[selected]
        out_class_ids[batch_idx, :valid] = class_ids[selected].to(dtype=torch.int64)

    return DetectionBatch(
        boxes=out_boxes,
        scores=out_scores,
        class_ids=out_class_ids,
        valid_counts=valid_counts,
    )


def decode_and_nms(
    det_output: DetHeadOutput,
    *,
    image_size: tuple[int, int] | None = None,
    strides: dict[str, int] | None = None,
    score_threshold: float = 0.05,
    pre_nms_topk: int = 1000,
    iou_threshold: float = 0.6,
    max_detections: int = 100,
) -> DetectionBatch:
    """Decode anchor-free outputs and run deterministic class-wise NMS."""
    candidates = decode_anchor_free_candidates(
        det_output,
        image_size=image_size,
        strides=strides,
        score_threshold=score_threshold,
        pre_nms_topk=pre_nms_topk,
    )
    return batched_deterministic_nms(
        candidates,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )


__all__ = [
    "DetectionCandidates",
    "DetectionBatch",
    "decode_anchor_free_candidates",
    "deterministic_nms",
    "batched_deterministic_nms",
    "decode_and_nms",
]
