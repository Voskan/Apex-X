from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .panoptic import PanopticSegmentInfo


@dataclass(frozen=True, slots=True)
class PQClassMetrics:
    """Per-class PQ statistics."""

    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    iou_sum: float


@dataclass(frozen=True, slots=True)
class PQMetrics:
    """Panoptic Quality aggregate metrics."""

    all_pq: float
    all_sq: float
    all_rq: float
    things_pq: float
    things_sq: float
    things_rq: float
    stuff_pq: float
    stuff_sq: float
    stuff_rq: float
    per_class: dict[int, PQClassMetrics]
    used_official_api: bool
    source: str


@dataclass(frozen=True, slots=True)
class OfficialPQPaths:
    """Filesystem inputs for official panopticapi evaluation."""

    gt_json: Path
    pred_json: Path
    gt_folder: Path | None = None
    pred_folder: Path | None = None


@dataclass(slots=True)
class _ClassAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0


def _segment_to_dict(segment: PanopticSegmentInfo | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(segment, PanopticSegmentInfo):
        return {
            "id": int(segment.id),
            "category_id": int(segment.category_id),
            "isthing": bool(segment.isthing),
            "area": int(segment.area),
            "score": segment.score,
            "instance_index": segment.instance_index,
        }
    if not isinstance(segment, Mapping):
        raise ValueError("segment entries must be PanopticSegmentInfo or mapping")
    return dict(segment)


def _normalize_segments(
    segments: Sequence[PanopticSegmentInfo | Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    normalized: dict[int, dict[str, Any]] = {}
    for segment in segments:
        item = _segment_to_dict(segment)
        if "id" not in item or "category_id" not in item:
            raise ValueError("segment entries must include id and category_id")
        seg_id = int(item["id"])
        if seg_id <= 0:
            raise ValueError("segment id must be > 0")
        if seg_id in normalized:
            raise ValueError("duplicate segment id in segments_info")
        category_id = int(item["category_id"])
        isthing = bool(item.get("isthing", False))
        normalized[seg_id] = {
            "id": seg_id,
            "category_id": category_id,
            "isthing": isthing,
        }
    return normalized


def _ensure_panoptic_shapes(
    pred_panoptic_map: Tensor,
    gt_panoptic_map: Tensor,
    pred_segments_info: Sequence[Sequence[PanopticSegmentInfo | Mapping[str, Any]]],
    gt_segments_info: Sequence[Sequence[PanopticSegmentInfo | Mapping[str, Any]]],
) -> None:
    if pred_panoptic_map.ndim != 3 or gt_panoptic_map.ndim != 3:
        raise ValueError("pred_panoptic_map and gt_panoptic_map must be [B,H,W]")
    if pred_panoptic_map.shape != gt_panoptic_map.shape:
        raise ValueError("pred_panoptic_map and gt_panoptic_map must have identical shape")
    if pred_panoptic_map.shape[0] != len(pred_segments_info):
        raise ValueError("pred_segments_info batch length must match pred_panoptic_map batch")
    if gt_panoptic_map.shape[0] != len(gt_segments_info):
        raise ValueError("gt_segments_info batch length must match gt_panoptic_map batch")
    if pred_panoptic_map.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("pred_panoptic_map must be integer typed")
    if gt_panoptic_map.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("gt_panoptic_map must be integer typed")


def _collect_present_ids(map_2d: Tensor, *, void_id: int) -> set[int]:
    values = torch.unique(map_2d).tolist()
    return {int(v) for v in values if int(v) != void_id}


def _compute_iou(pred_mask: Tensor, gt_mask: Tensor) -> float:
    inter = int((pred_mask & gt_mask).sum().item())
    if inter == 0:
        return 0.0
    pred_area = int(pred_mask.sum().item())
    gt_area = int(gt_mask.sum().item())
    union = pred_area + gt_area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _acc_to_metrics(acc: _ClassAccumulator) -> PQClassMetrics:
    denom = float(acc.tp) + 0.5 * float(acc.fp) + 0.5 * float(acc.fn)
    pq = float(acc.iou_sum / denom) if denom > 0.0 else 0.0
    sq = float(acc.iou_sum / float(acc.tp)) if acc.tp > 0 else 0.0
    rq = float(float(acc.tp) / denom) if denom > 0.0 else 0.0
    return PQClassMetrics(
        pq=pq,
        sq=sq,
        rq=rq,
        tp=acc.tp,
        fp=acc.fp,
        fn=acc.fn,
        iou_sum=float(acc.iou_sum),
    )


def _mean_metric(metrics: Iterable[PQClassMetrics], field: str) -> float:
    vals = [float(getattr(metric, field)) for metric in metrics]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _compute_fallback_pq(
    pred_panoptic_map: Tensor,
    pred_segments_info: Sequence[Sequence[PanopticSegmentInfo | Mapping[str, Any]]],
    gt_panoptic_map: Tensor,
    gt_segments_info: Sequence[Sequence[PanopticSegmentInfo | Mapping[str, Any]]],
    *,
    iou_threshold: float,
    void_id: int,
    thing_class_ids: set[int] | None,
) -> PQMetrics:
    class_acc: dict[int, _ClassAccumulator] = {}
    class_isthing: dict[int, bool] = {}
    batch_size = pred_panoptic_map.shape[0]

    for b in range(batch_size):
        pred_map = pred_panoptic_map[b]
        gt_map = gt_panoptic_map[b]
        pred_info = _normalize_segments(pred_segments_info[b])
        gt_info = _normalize_segments(gt_segments_info[b])

        present_pred_ids = _collect_present_ids(pred_map, void_id=void_id)
        present_gt_ids = _collect_present_ids(gt_map, void_id=void_id)

        for pred_id in present_pred_ids:
            if pred_id not in pred_info:
                raise ValueError(f"pred_segments_info missing id {pred_id}")
        for gt_id in present_gt_ids:
            if gt_id not in gt_info:
                raise ValueError(f"gt_segments_info missing id {gt_id}")

        matched_pred: set[int] = set()
        matched_gt: set[int] = set()

        for pred_id in sorted(present_pred_ids):
            pred_seg = pred_info[pred_id]
            pred_class = int(pred_seg["category_id"])
            pred_mask = pred_map == pred_id
            candidate_gt = sorted(
                {int(v) for v in torch.unique(gt_map[pred_mask]).tolist() if int(v) != void_id}
            )

            best_gt: int | None = None
            best_iou = -1.0
            for gt_id in candidate_gt:
                if gt_id in matched_gt:
                    continue
                if gt_id not in present_gt_ids:
                    continue
                gt_seg = gt_info[gt_id]
                gt_class = int(gt_seg["category_id"])
                if gt_class != pred_class:
                    continue
                gt_mask = gt_map == gt_id
                iou = _compute_iou(pred_mask, gt_mask)
                if iou > best_iou or (iou == best_iou and best_gt is not None and gt_id < best_gt):
                    best_iou = iou
                    best_gt = gt_id

            if best_gt is not None and best_iou > iou_threshold:
                matched_pred.add(pred_id)
                matched_gt.add(best_gt)
                acc = class_acc.setdefault(pred_class, _ClassAccumulator())
                acc.tp += 1
                acc.iou_sum += best_iou
                if thing_class_ids is None:
                    class_isthing[pred_class] = bool(pred_seg["isthing"])
                else:
                    class_isthing[pred_class] = pred_class in thing_class_ids

        for pred_id in sorted(present_pred_ids):
            if pred_id in matched_pred:
                continue
            pred_class = int(pred_info[pred_id]["category_id"])
            acc = class_acc.setdefault(pred_class, _ClassAccumulator())
            acc.fp += 1
            if thing_class_ids is None:
                class_isthing[pred_class] = bool(pred_info[pred_id]["isthing"])
            else:
                class_isthing[pred_class] = pred_class in thing_class_ids

        for gt_id in sorted(present_gt_ids):
            if gt_id in matched_gt:
                continue
            gt_class = int(gt_info[gt_id]["category_id"])
            acc = class_acc.setdefault(gt_class, _ClassAccumulator())
            acc.fn += 1
            if thing_class_ids is None:
                class_isthing[gt_class] = bool(gt_info[gt_id]["isthing"])
            else:
                class_isthing[gt_class] = gt_class in thing_class_ids

    per_class = {class_id: _acc_to_metrics(acc) for class_id, acc in sorted(class_acc.items())}
    all_metrics = list(per_class.values())
    thing_metrics = [
        metric for class_id, metric in per_class.items() if class_isthing.get(class_id, False)
    ]
    stuff_metrics = [
        metric for class_id, metric in per_class.items() if not class_isthing.get(class_id, False)
    ]

    return PQMetrics(
        all_pq=_mean_metric(all_metrics, "pq"),
        all_sq=_mean_metric(all_metrics, "sq"),
        all_rq=_mean_metric(all_metrics, "rq"),
        things_pq=_mean_metric(thing_metrics, "pq"),
        things_sq=_mean_metric(thing_metrics, "sq"),
        things_rq=_mean_metric(thing_metrics, "rq"),
        stuff_pq=_mean_metric(stuff_metrics, "pq"),
        stuff_sq=_mean_metric(stuff_metrics, "sq"),
        stuff_rq=_mean_metric(stuff_metrics, "rq"),
        per_class=per_class,
        used_official_api=False,
        source="fallback",
    )


def _try_official_api(paths: OfficialPQPaths) -> PQMetrics | None:
    try:
        from panopticapi import evaluation as panoptic_eval  # type: ignore[import-not-found]
    except Exception:
        return None

    pq_compute = getattr(panoptic_eval, "pq_compute", None)
    if not callable(pq_compute):
        return None

    try:
        if paths.gt_folder is not None and paths.pred_folder is not None:
            result = pq_compute(
                str(paths.gt_json),
                str(paths.pred_json),
                str(paths.gt_folder),
                str(paths.pred_folder),
            )
        else:
            result = pq_compute(str(paths.gt_json), str(paths.pred_json))
    except TypeError:
        return None
    except Exception:
        return None

    if not isinstance(result, Mapping):
        return None

    all_block = result.get("All", {})
    things_block = result.get("Things", {})
    stuff_block = result.get("Stuff", {})
    raw_per_class = result.get("per_class", {})

    per_class: dict[int, PQClassMetrics] = {}
    if isinstance(raw_per_class, Mapping):
        for key, value in raw_per_class.items():
            if not isinstance(value, Mapping):
                continue
            class_id = int(key)
            per_class[class_id] = PQClassMetrics(
                pq=float(value.get("pq", 0.0)),
                sq=float(value.get("sq", 0.0)),
                rq=float(value.get("rq", 0.0)),
                tp=int(value.get("tp", 0)),
                fp=int(value.get("fp", 0)),
                fn=int(value.get("fn", 0)),
                iou_sum=float(value.get("iou", 0.0)),
            )

    return PQMetrics(
        all_pq=float(all_block.get("pq", 0.0)),
        all_sq=float(all_block.get("sq", 0.0)),
        all_rq=float(all_block.get("rq", 0.0)),
        things_pq=float(things_block.get("pq", 0.0)),
        things_sq=float(things_block.get("sq", 0.0)),
        things_rq=float(things_block.get("rq", 0.0)),
        stuff_pq=float(stuff_block.get("pq", 0.0)),
        stuff_sq=float(stuff_block.get("sq", 0.0)),
        stuff_rq=float(stuff_block.get("rq", 0.0)),
        per_class=per_class,
        used_official_api=True,
        source="official",
    )


def evaluate_panoptic_quality(
    pred_panoptic_map: Tensor,
    pred_segments_info: Sequence[Sequence[PanopticSegmentInfo | Mapping[str, Any]]],
    gt_panoptic_map: Tensor,
    gt_segments_info: Sequence[Sequence[PanopticSegmentInfo | Mapping[str, Any]]],
    *,
    thing_class_ids: Iterable[int] | None = None,
    iou_threshold: float = 0.5,
    void_id: int = 0,
    official_paths: OfficialPQPaths | None = None,
    use_official_if_available: bool = True,
) -> PQMetrics:
    """Evaluate panoptic quality using official API when available, else fallback."""
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in (0,1]")
    _ensure_panoptic_shapes(
        pred_panoptic_map=pred_panoptic_map,
        gt_panoptic_map=gt_panoptic_map,
        pred_segments_info=pred_segments_info,
        gt_segments_info=gt_segments_info,
    )

    if use_official_if_available and official_paths is not None:
        official = _try_official_api(official_paths)
        if official is not None:
            return official

    normalized_thing = None if thing_class_ids is None else {int(v) for v in thing_class_ids}
    return _compute_fallback_pq(
        pred_panoptic_map=pred_panoptic_map.to(dtype=torch.int64),
        pred_segments_info=pred_segments_info,
        gt_panoptic_map=gt_panoptic_map.to(dtype=torch.int64),
        gt_segments_info=gt_segments_info,
        iou_threshold=float(iou_threshold),
        void_id=int(void_id),
        thing_class_ids=normalized_thing,
    )


__all__ = [
    "PQClassMetrics",
    "PQMetrics",
    "OfficialPQPaths",
    "evaluate_panoptic_quality",
]
