from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .pq_eval import PQMetrics, evaluate_panoptic_quality

_DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(0.5 + 0.05 * i for i in range(10))


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    image_id: int
    category_id: int
    score: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class MaskRecord:
    image_id: int
    category_id: int
    score: float
    mask: np.ndarray  # [H,W] bool


@dataclass(frozen=True, slots=True)
class EvalSummary:
    det_map: float
    det_ap50: float
    det_ap75: float
    mask_map: float
    mask_ap50: float
    mask_ap75: float
    semantic_miou: float
    panoptic_pq: float
    panoptic_source: str
    per_class_semantic_iou: dict[int, float]
    num_det_gt: int
    num_det_pred: int
    num_mask_gt: int
    num_mask_pred: int

    def to_dict(self) -> dict[str, Any]:
        per_class_iou = {
            str(key): value for key, value in sorted(self.per_class_semantic_iou.items())
        }
        return {
            "det": {
                "map": self.det_map,
                "ap50": self.det_ap50,
                "ap75": self.det_ap75,
                "num_gt": self.num_det_gt,
                "num_pred": self.num_det_pred,
            },
            "inst_seg": {
                "map": self.mask_map,
                "ap50": self.mask_ap50,
                "ap75": self.mask_ap75,
                "num_gt": self.num_mask_gt,
                "num_pred": self.num_mask_pred,
            },
            "semantic": {
                "miou": self.semantic_miou,
                "per_class_iou": per_class_iou,
            },
            "panoptic": {
                "pq": self.panoptic_pq,
                "source": self.panoptic_source,
            },
        }


def _require_keys(record: dict[str, object], required: set[str], name: str) -> None:
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")


def _as_int(value: object, *, field_name: str, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    out = int(value)
    if min_value is not None and out < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return out


def _as_float(value: object, *, field_name: str, min_value: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    if min_value is not None and out < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return out


def _bbox_xywh_to_xyxy(xywh: list[object], *, field_name: str) -> tuple[float, float, float, float]:
    if len(xywh) != 4:
        raise ValueError(f"{field_name} must be a 4-item list")
    x = _as_float(xywh[0], field_name=f"{field_name}[0]")
    y = _as_float(xywh[1], field_name=f"{field_name}[1]")
    w = _as_float(xywh[2], field_name=f"{field_name}[2]", min_value=0.0)
    h = _as_float(xywh[3], field_name=f"{field_name}[3]", min_value=0.0)
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"{field_name} width/height must be > 0")
    return (x, y, x + w, y + h)


def _parse_det_records(
    raw_records: list[object],
    *,
    with_score: bool,
    prefix: str,
) -> list[DetectionRecord]:
    out: list[DetectionRecord] = []
    required = {"image_id", "category_id", "bbox_xywh"}
    if with_score:
        required = required | {"score"}
    for idx, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"{prefix}[{idx}] must be an object")
        record = cast(dict[str, object], raw)
        _require_keys(record, required, f"{prefix}[{idx}]")
        bbox_raw = record["bbox_xywh"]
        if not isinstance(bbox_raw, list):
            raise ValueError(f"{prefix}[{idx}].bbox_xywh must be a list")
        score = _as_float(
            record.get("score", 1.0),
            field_name=f"{prefix}[{idx}].score",
            min_value=0.0,
        )
        out.append(
            DetectionRecord(
                image_id=_as_int(
                    record["image_id"],
                    field_name=f"{prefix}[{idx}].image_id",
                    min_value=0,
                ),
                category_id=_as_int(
                    record["category_id"],
                    field_name=f"{prefix}[{idx}].category_id",
                    min_value=0,
                ),
                score=score,
                bbox_xyxy=_bbox_xywh_to_xyxy(bbox_raw, field_name=f"{prefix}[{idx}].bbox_xywh"),
            ),
        )
    return out


def _parse_mask(mask_raw: object, *, field_name: str) -> np.ndarray:
    arr = np.asarray(mask_raw)
    if arr.ndim != 2:
        raise ValueError(f"{field_name} must be a 2D array-like")
    if arr.size == 0:
        raise ValueError(f"{field_name} must be non-empty")
    return arr.astype(bool, copy=False)


def _parse_mask_records(
    raw_records: list[object],
    *,
    with_score: bool,
    prefix: str,
) -> list[MaskRecord]:
    out: list[MaskRecord] = []
    required = {"image_id", "category_id", "mask"}
    if with_score:
        required = required | {"score"}
    for idx, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"{prefix}[{idx}] must be an object")
        record = cast(dict[str, object], raw)
        _require_keys(record, required, f"{prefix}[{idx}]")
        score = _as_float(
            record.get("score", 1.0),
            field_name=f"{prefix}[{idx}].score",
            min_value=0.0,
        )
        out.append(
            MaskRecord(
                image_id=_as_int(
                    record["image_id"],
                    field_name=f"{prefix}[{idx}].image_id",
                    min_value=0,
                ),
                category_id=_as_int(
                    record["category_id"],
                    field_name=f"{prefix}[{idx}].category_id",
                    min_value=0,
                ),
                score=score,
                mask=_parse_mask(record["mask"], field_name=f"{prefix}[{idx}].mask"),
            ),
        )
    return out


def _box_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("mask shapes must match for IoU")
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _ap_from_pr(tp: np.ndarray, fp: np.ndarray, *, num_gt: int) -> float:
    if num_gt <= 0:
        return 0.0
    cum_tp = np.cumsum(tp, dtype=np.float64)
    cum_fp = np.cumsum(fp, dtype=np.float64)
    recall = cum_tp / float(num_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    mrec = np.concatenate([np.asarray([0.0]), recall, np.asarray([1.0])], axis=0)
    mpre = np.concatenate([np.asarray([0.0]), precision, np.asarray([0.0])], axis=0)
    for idx in range(mpre.shape[0] - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    changed = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _average_precision_detection(
    pred: list[DetectionRecord],
    gt: list[DetectionRecord],
    *,
    iou_threshold: float,
    class_id: int,
) -> float:
    gt_filtered = [rec for rec in gt if rec.category_id == class_id]
    pred_filtered = [rec for rec in pred if rec.category_id == class_id]
    if not gt_filtered:
        return 0.0
    gt_by_image: dict[int, list[DetectionRecord]] = {}
    for record in gt_filtered:
        gt_by_image.setdefault(record.image_id, []).append(record)
    gt_matched_by_image: dict[int, list[bool]] = {
        image_id: [False] * len(records) for image_id, records in gt_by_image.items()
    }

    pred_sorted = sorted(
        pred_filtered,
        key=lambda record: (-record.score, record.image_id),
    )
    tp = np.zeros((len(pred_sorted),), dtype=np.float64)
    fp = np.zeros((len(pred_sorted),), dtype=np.float64)
    for idx, pred_record in enumerate(pred_sorted):
        gt_records = gt_by_image.get(pred_record.image_id, [])
        gt_matched = gt_matched_by_image.get(pred_record.image_id, [])
        best_iou = -1.0
        best_gt = -1
        for gt_idx, gt_record in enumerate(gt_records):
            if gt_matched[gt_idx]:
                continue
            iou = _box_iou_xyxy(pred_record.bbox_xyxy, gt_record.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            gt_matched[best_gt] = True
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0
    return _ap_from_pr(tp, fp, num_gt=len(gt_filtered))


def _average_precision_mask(
    pred: list[MaskRecord],
    gt: list[MaskRecord],
    *,
    iou_threshold: float,
    class_id: int,
) -> float:
    gt_filtered = [rec for rec in gt if rec.category_id == class_id]
    pred_filtered = [rec for rec in pred if rec.category_id == class_id]
    if not gt_filtered:
        return 0.0
    gt_by_image: dict[int, list[MaskRecord]] = {}
    for record in gt_filtered:
        gt_by_image.setdefault(record.image_id, []).append(record)
    gt_matched_by_image: dict[int, list[bool]] = {
        image_id: [False] * len(records) for image_id, records in gt_by_image.items()
    }

    pred_sorted = sorted(
        pred_filtered,
        key=lambda record: (-record.score, record.image_id),
    )
    tp = np.zeros((len(pred_sorted),), dtype=np.float64)
    fp = np.zeros((len(pred_sorted),), dtype=np.float64)
    for idx, pred_record in enumerate(pred_sorted):
        gt_records = gt_by_image.get(pred_record.image_id, [])
        gt_matched = gt_matched_by_image.get(pred_record.image_id, [])
        best_iou = -1.0
        best_gt = -1
        for gt_idx, gt_record in enumerate(gt_records):
            if gt_matched[gt_idx]:
                continue
            iou = _mask_iou(pred_record.mask, gt_record.mask)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            gt_matched[best_gt] = True
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0
    return _ap_from_pr(tp, fp, num_gt=len(gt_filtered))


def _coco_map_det(
    pred: list[DetectionRecord],
    gt: list[DetectionRecord],
    *,
    iou_thresholds: tuple[float, ...] = _DEFAULT_IOU_THRESHOLDS,
) -> tuple[float, float, float]:
    classes = sorted({rec.category_id for rec in gt})
    if not classes:
        return 0.0, 0.0, 0.0
    aps_by_threshold: dict[float, float] = {}
    for threshold in iou_thresholds:
        per_class = [
            _average_precision_detection(pred, gt, iou_threshold=threshold, class_id=class_id)
            for class_id in classes
        ]
        aps_by_threshold[threshold] = float(np.mean(np.asarray(per_class, dtype=np.float64)))
    map_value = float(np.mean(np.asarray(list(aps_by_threshold.values()), dtype=np.float64)))
    return (
        map_value,
        aps_by_threshold.get(0.5, 0.0),
        aps_by_threshold.get(0.75, 0.0),
    )


def _coco_map_mask(
    pred: list[MaskRecord],
    gt: list[MaskRecord],
    *,
    iou_thresholds: tuple[float, ...] = _DEFAULT_IOU_THRESHOLDS,
) -> tuple[float, float, float]:
    classes = sorted({rec.category_id for rec in gt})
    if not classes:
        return 0.0, 0.0, 0.0
    aps_by_threshold: dict[float, float] = {}
    for threshold in iou_thresholds:
        per_class = [
            _average_precision_mask(pred, gt, iou_threshold=threshold, class_id=class_id)
            for class_id in classes
        ]
        aps_by_threshold[threshold] = float(np.mean(np.asarray(per_class, dtype=np.float64)))
    map_value = float(np.mean(np.asarray(list(aps_by_threshold.values()), dtype=np.float64)))
    return (
        map_value,
        aps_by_threshold.get(0.5, 0.0),
        aps_by_threshold.get(0.75, 0.0),
    )


def _semantic_miou(
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None,
) -> tuple[float, dict[int, float]]:
    if pred_semantic.shape != gt_semantic.shape:
        raise ValueError("semantic pred/gt shape mismatch")
    if pred_semantic.ndim == 2:
        pred_arr = pred_semantic[None, ...]
        gt_arr = gt_semantic[None, ...]
    elif pred_semantic.ndim == 3:
        pred_arr = pred_semantic
        gt_arr = gt_semantic
    else:
        raise ValueError("semantic pred/gt must be [H,W] or [B,H,W]")

    valid_mask = np.ones_like(gt_arr, dtype=bool)
    if ignore_index is not None:
        valid_mask &= gt_arr != int(ignore_index)

    per_class: dict[int, float] = {}
    class_ious: list[float] = []
    for class_id in range(num_classes):
        gt_c = gt_arr == class_id
        pred_c = pred_arr == class_id
        inter = int(np.logical_and(np.logical_and(gt_c, pred_c), valid_mask).sum())
        union = int(np.logical_and(np.logical_or(gt_c, pred_c), valid_mask).sum())
        if union > 0:
            iou = float(inter / union)
            per_class[class_id] = iou
            class_ious.append(iou)
        else:
            per_class[class_id] = 0.0
    miou = float(np.mean(np.asarray(class_ious, dtype=np.float64))) if class_ious else 0.0
    return miou, per_class


def evaluate_fixture_payload(payload: dict[str, object]) -> EvalSummary:
    if not isinstance(payload, dict):
        raise ValueError("fixture payload must be an object")
    for key in ("det", "inst_seg", "semantic", "panoptic"):
        if key not in payload:
            raise ValueError(f"fixture payload missing key: {key}")

    det_obj = payload["det"]
    seg_obj = payload["inst_seg"]
    sem_obj = payload["semantic"]
    pano_obj = payload["panoptic"]
    if not isinstance(det_obj, dict):
        raise ValueError("det section must be an object")
    if not isinstance(seg_obj, dict):
        raise ValueError("inst_seg section must be an object")
    if not isinstance(sem_obj, dict):
        raise ValueError("semantic section must be an object")
    if not isinstance(pano_obj, dict):
        raise ValueError("panoptic section must be an object")

    det_pred_raw = det_obj.get("pred")
    det_gt_raw = det_obj.get("gt")
    if not isinstance(det_pred_raw, list) or not isinstance(det_gt_raw, list):
        raise ValueError("det.pred and det.gt must be lists")
    det_pred = _parse_det_records(det_pred_raw, with_score=True, prefix="det.pred")
    det_gt = _parse_det_records(det_gt_raw, with_score=False, prefix="det.gt")
    det_map, det_ap50, det_ap75 = _coco_map_det(det_pred, det_gt)

    mask_pred_raw = seg_obj.get("pred")
    mask_gt_raw = seg_obj.get("gt")
    if not isinstance(mask_pred_raw, list) or not isinstance(mask_gt_raw, list):
        raise ValueError("inst_seg.pred and inst_seg.gt must be lists")
    mask_pred = _parse_mask_records(mask_pred_raw, with_score=True, prefix="inst_seg.pred")
    mask_gt = _parse_mask_records(mask_gt_raw, with_score=False, prefix="inst_seg.gt")
    mask_map, mask_ap50, mask_ap75 = _coco_map_mask(mask_pred, mask_gt)

    sem_pred_raw = sem_obj.get("pred")
    sem_gt_raw = sem_obj.get("gt")
    num_classes_raw = sem_obj.get("num_classes")
    ignore_index_raw = sem_obj.get("ignore_index")
    if num_classes_raw is None:
        raise ValueError("semantic.num_classes is required")
    num_classes = _as_int(num_classes_raw, field_name="semantic.num_classes", min_value=1)
    semantic_pred = np.asarray(sem_pred_raw)
    semantic_gt = np.asarray(sem_gt_raw)
    ignore_index = (
        None
        if ignore_index_raw is None
        else _as_int(ignore_index_raw, field_name="semantic.ignore_index")
    )
    semantic_miou, per_class_iou = _semantic_miou(
        semantic_pred,
        semantic_gt,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    pred_map_raw = pano_obj.get("pred_map")
    gt_map_raw = pano_obj.get("gt_map")
    pred_segments_raw = pano_obj.get("pred_segments")
    gt_segments_raw = pano_obj.get("gt_segments")
    thing_ids_raw = pano_obj.get("thing_class_ids", [])
    if pred_map_raw is None or gt_map_raw is None:
        raise ValueError("panoptic pred_map/gt_map required")
    if not isinstance(pred_segments_raw, list) or not isinstance(gt_segments_raw, list):
        raise ValueError("panoptic pred_segments/gt_segments must be lists")
    if not isinstance(thing_ids_raw, list):
        raise ValueError("panoptic.thing_class_ids must be a list")

    pred_map_tensor = torch.tensor(np.asarray(pred_map_raw), dtype=torch.int64)
    gt_map_tensor = torch.tensor(np.asarray(gt_map_raw), dtype=torch.int64)
    if pred_map_tensor.ndim == 2:
        pred_map_tensor = pred_map_tensor.unsqueeze(0)
    if gt_map_tensor.ndim == 2:
        gt_map_tensor = gt_map_tensor.unsqueeze(0)

    pred_segments = [cast(list[dict[str, object]], pred_segments_raw)]
    gt_segments = [cast(list[dict[str, object]], gt_segments_raw)]
    thing_ids = {int(v) for v in thing_ids_raw}
    pq_metrics: PQMetrics = evaluate_panoptic_quality(
        pred_panoptic_map=pred_map_tensor,
        pred_segments_info=pred_segments,
        gt_panoptic_map=gt_map_tensor,
        gt_segments_info=gt_segments,
        thing_class_ids=thing_ids,
    )

    return EvalSummary(
        det_map=det_map,
        det_ap50=det_ap50,
        det_ap75=det_ap75,
        mask_map=mask_map,
        mask_ap50=mask_ap50,
        mask_ap75=mask_ap75,
        semantic_miou=semantic_miou,
        panoptic_pq=float(pq_metrics.all_pq),
        panoptic_source=pq_metrics.source,
        per_class_semantic_iou=per_class_iou,
        num_det_gt=len(det_gt),
        num_det_pred=len(det_pred),
        num_mask_gt=len(mask_gt),
        num_mask_pred=len(mask_pred),
    )


def evaluate_fixture_file(path: str | Path) -> EvalSummary:
    fixture_path = Path(path)
    payload = cast(dict[str, object], json.loads(fixture_path.read_text(encoding="utf-8")))
    return evaluate_fixture_payload(payload)


def _markdown_report(summary: EvalSummary, *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| COCO mAP (det) | {summary.det_map:.6f} |",
        f"| AP50 (det) | {summary.det_ap50:.6f} |",
        f"| AP75 (det) | {summary.det_ap75:.6f} |",
        f"| COCO mAP (inst-seg) | {summary.mask_map:.6f} |",
        f"| AP50 (inst-seg) | {summary.mask_ap50:.6f} |",
        f"| AP75 (inst-seg) | {summary.mask_ap75:.6f} |",
        f"| mIoU (semantic) | {summary.semantic_miou:.6f} |",
        f"| PQ (panoptic) | {summary.panoptic_pq:.6f} |",
        "",
        f"Panoptic source: `{summary.panoptic_source}`",
    ]
    return "\n".join(lines) + "\n"


def write_eval_reports(
    summary: EvalSummary,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    title: str = "Apex-X Evaluation Report",
) -> tuple[Path, Path]:
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "summary": summary.to_dict(),
    }
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(_markdown_report(summary, title=title), encoding="utf-8")
    return json_out, md_out


def tiny_eval_fixture_payload() -> dict[str, object]:
    """Deterministic tiny payload used as eval fallback and smoke data."""
    return {
        "det": {
            "pred": [
                {"image_id": 1, "category_id": 1, "score": 0.95, "bbox_xywh": [1, 1, 2, 2]},
                {"image_id": 1, "category_id": 2, "score": 0.85, "bbox_xywh": [0, 0, 2, 2]},
            ],
            "gt": [
                {"image_id": 1, "category_id": 1, "bbox_xywh": [1, 1, 2, 2]},
                {"image_id": 1, "category_id": 2, "bbox_xywh": [0, 0, 2, 2]},
            ],
        },
        "inst_seg": {
            "pred": [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "score": 0.93,
                    "mask": [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                },
                {
                    "image_id": 1,
                    "category_id": 2,
                    "score": 0.90,
                    "mask": [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                },
            ],
            "gt": [
                {"image_id": 1, "category_id": 1, "mask": [[0, 0, 0], [0, 1, 1], [0, 1, 1]]},
                {"image_id": 1, "category_id": 2, "mask": [[1, 1, 0], [1, 0, 0], [0, 0, 0]]},
            ],
        },
        "semantic": {
            "pred": [[0, 1, 1], [0, 1, 2], [2, 2, 2]],
            "gt": [[0, 1, 1], [0, 1, 2], [2, 2, 2]],
            "num_classes": 3,
            "ignore_index": None,
        },
        "panoptic": {
            "pred_map": [[0, 1, 1], [0, 1, 2], [2, 2, 2]],
            "gt_map": [[0, 1, 1], [0, 1, 2], [2, 2, 2]],
            "pred_segments": [
                {"id": 1, "category_id": 1, "isthing": True},
                {"id": 2, "category_id": 2, "isthing": False},
            ],
            "gt_segments": [
                {"id": 1, "category_id": 1, "isthing": True},
                {"id": 2, "category_id": 2, "isthing": False},
            ],
            "thing_class_ids": [1],
        },
    }


__all__ = [
    "DetectionRecord",
    "MaskRecord",
    "EvalSummary",
    "evaluate_fixture_payload",
    "evaluate_fixture_file",
    "write_eval_reports",
    "tiny_eval_fixture_payload",
]
