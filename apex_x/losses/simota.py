from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as f

ClassificationCostType = Literal["focal", "bce"]


@dataclass(frozen=True)
class SimOTACostOutput:
    """Cost breakdown for SimOTA/OTA-style assignment."""

    cls_cost: Tensor  # [M, N]
    iou_cost: Tensor  # [M, N]
    center_prior_cost: Tensor  # [M, N]
    total_cost: Tensor  # [M, N]
    candidate_indices: Tensor  # [M, K]
    candidate_mask: Tensor  # [M, N] bool


@dataclass(frozen=True)
class DynamicKMatchingOutput:
    """Dynamic-K matching output for SimOTA/OTA assignment."""

    dynamic_ks: Tensor  # [M] int64
    matching_matrix: Tensor  # [M, N] bool
    foreground_mask: Tensor  # [N] bool
    matched_gt_indices: Tensor  # [N] int64, -1 for background
    assigned_cost: Tensor  # [N] float, inf for background
    num_foreground: int


def _validate_xyxy(boxes: Tensor, *, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must be [N,4] in xyxy format")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must contain finite values")
    if torch.any(boxes[:, 2] < boxes[:, 0]) or torch.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError(f"{name} must satisfy x2>=x1 and y2>=y1")


def _box_centers(boxes_xyxy: Tensor) -> Tensor:
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    return torch.stack((cx, cy), dim=1)


def _pairwise_iou_xyxy(pred_boxes: Tensor, gt_boxes: Tensor, *, eps: float = 1e-9) -> Tensor:
    """Pairwise IoU matrix [M, N] for GT vs predicted boxes."""
    _validate_xyxy(pred_boxes, name="pred_boxes")
    _validate_xyxy(gt_boxes, name="gt_boxes")

    m = gt_boxes.shape[0]
    n = pred_boxes.shape[0]
    if m == 0 or n == 0:
        return pred_boxes.new_zeros((m, n))

    gt = gt_boxes[:, None, :]  # [M,1,4]
    pr = pred_boxes[None, :, :]  # [1,N,4]

    inter_x1 = torch.maximum(gt[..., 0], pr[..., 0])
    inter_y1 = torch.maximum(gt[..., 1], pr[..., 1])
    inter_x2 = torch.minimum(gt[..., 2], pr[..., 2])
    inter_y2 = torch.minimum(gt[..., 3], pr[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_gt = (gt[..., 2] - gt[..., 0]).clamp(min=0.0) * (gt[..., 3] - gt[..., 1]).clamp(min=0.0)
    area_pr = (pr[..., 2] - pr[..., 0]).clamp(min=0.0) * (pr[..., 3] - pr[..., 1]).clamp(min=0.0)
    union = (area_gt + area_pr - inter).clamp(min=eps)
    return inter / union


def classification_cost(
    pred_cls_logits: Tensor,
    gt_classes: Tensor,
    *,
    mode: ClassificationCostType = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    eps: float = 1e-9,
) -> Tensor:
    """Classification cost matrix [M, N] for GT classes against anchor logits."""
    if pred_cls_logits.ndim != 2:
        raise ValueError("pred_cls_logits must be [N,C]")
    if gt_classes.ndim != 1:
        raise ValueError("gt_classes must be [M]")
    if pred_cls_logits.shape[0] == 0 or gt_classes.shape[0] == 0:
        return pred_cls_logits.new_zeros((gt_classes.shape[0], pred_cls_logits.shape[0]))
    if gt_classes.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("gt_classes must be integer tensor")
    if not torch.isfinite(pred_cls_logits).all():
        raise ValueError("pred_cls_logits must contain finite values")

    n_anchors, n_classes = pred_cls_logits.shape
    gt_classes_i64 = gt_classes.to(dtype=torch.int64, device=pred_cls_logits.device)
    if torch.any(gt_classes_i64 < 0) or torch.any(gt_classes_i64 >= n_classes):
        raise ValueError("gt_classes must be in [0, num_classes)")

    # [N, M] logits for each anchor against each GT class, then transpose -> [M, N].
    logits_nm = pred_cls_logits[:, gt_classes_i64]
    logits_mn = logits_nm.transpose(0, 1)

    if mode == "bce":
        targets = torch.ones_like(logits_mn)
        return f.binary_cross_entropy_with_logits(logits_mn, targets, reduction="none")

    if mode != "focal":
        raise ValueError("mode must be one of {'focal', 'bce'}")
    if focal_alpha < 0.0 or focal_alpha > 1.0:
        raise ValueError("focal_alpha must be in [0,1]")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be >= 0")

    prob_pos = torch.sigmoid(logits_mn).clamp(min=eps, max=1.0 - eps)
    cost = -focal_alpha * torch.pow(1.0 - prob_pos, focal_gamma) * torch.log(prob_pos)
    return torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=0.0)


def iou_cost(pred_boxes_xyxy: Tensor, gt_boxes_xyxy: Tensor) -> Tensor:
    """IoU cost matrix [M, N] = 1 - IoU."""
    iou = _pairwise_iou_xyxy(pred_boxes_xyxy, gt_boxes_xyxy)
    return 1.0 - iou


def center_prior_cost(
    anchor_centers_xy: Tensor,
    gt_boxes_xyxy: Tensor,
    *,
    normalize_by_gt_size: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    """Center prior matrix [M, N] using GT-center to anchor-center distance."""
    if anchor_centers_xy.ndim != 2 or anchor_centers_xy.shape[1] != 2:
        raise ValueError("anchor_centers_xy must be [N,2]")
    if not torch.isfinite(anchor_centers_xy).all():
        raise ValueError("anchor_centers_xy must contain finite values")
    _validate_xyxy(gt_boxes_xyxy, name="gt_boxes_xyxy")

    m = gt_boxes_xyxy.shape[0]
    n = anchor_centers_xy.shape[0]
    if m == 0 or n == 0:
        return anchor_centers_xy.new_zeros((m, n))

    gt_centers = _box_centers(gt_boxes_xyxy)  # [M,2]
    delta = anchor_centers_xy[None, :, :] - gt_centers[:, None, :]  # [M,N,2]

    if not normalize_by_gt_size:
        return torch.sqrt(torch.sum(delta * delta, dim=-1) + eps)

    gt_wh = torch.stack(
        (
            (gt_boxes_xyxy[:, 2] - gt_boxes_xyxy[:, 0]).clamp(min=eps),
            (gt_boxes_xyxy[:, 3] - gt_boxes_xyxy[:, 1]).clamp(min=eps),
        ),
        dim=1,
    )  # [M,2]
    norm = gt_wh[:, None, :]  # [M,1,2]
    delta_norm = delta / norm
    # Stability: clamp normalized delta to prevent huge costs for extreme distant boxes
    delta_norm = delta_norm.clamp(min=-100.0, max=100.0)
    return torch.sqrt(torch.sum(delta_norm * delta_norm, dim=-1) + eps)


def topk_center_candidates(
    anchor_centers_xy: Tensor,
    gt_boxes_xyxy: Tensor,
    *,
    topk: int,
) -> Tensor:
    """Per-GT top-k candidate anchors by center distance. Returns [M, K]."""
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if anchor_centers_xy.ndim != 2 or anchor_centers_xy.shape[1] != 2:
        raise ValueError("anchor_centers_xy must be [N,2]")
    _validate_xyxy(gt_boxes_xyxy, name="gt_boxes_xyxy")

    m = gt_boxes_xyxy.shape[0]
    n = anchor_centers_xy.shape[0]
    if m == 0 or n == 0:
        return torch.empty((m, 0), dtype=torch.int64, device=anchor_centers_xy.device)

    gt_centers = _box_centers(gt_boxes_xyxy)
    delta = anchor_centers_xy[None, :, :] - gt_centers[:, None, :]  # [M,N,2]
    dist2 = torch.sum(delta * delta, dim=-1)  # [M,N]
    k = min(int(topk), n)
    # stable=True gives deterministic tie handling.
    return torch.argsort(dist2, dim=1, descending=False, stable=True)[:, :k].to(dtype=torch.int64)


def candidate_mask_from_indices(candidate_indices: Tensor, num_anchors: int) -> Tensor:
    """Build boolean mask [M,N] from candidate indices [M,K]."""
    if candidate_indices.ndim != 2:
        raise ValueError("candidate_indices must be [M,K]")
    if num_anchors < 0:
        raise ValueError("num_anchors must be >= 0")

    m, k = candidate_indices.shape
    mask = torch.zeros((m, num_anchors), dtype=torch.bool, device=candidate_indices.device)
    if num_anchors == 0 or k == 0:
        return mask

    idx = candidate_indices.to(dtype=torch.int64)
    if torch.any(idx < 0) or torch.any(idx >= num_anchors):
        raise ValueError("candidate_indices contain out-of-bounds anchor ids")
    row_ids = torch.arange(m, device=idx.device).unsqueeze(1).expand(m, k)
    mask[row_ids, idx] = True
    return mask


def dynamic_k_from_top_ious(
    iou_matrix: Tensor,
    *,
    candidate_mask: Tensor | None = None,
    topk: int = 10,
    min_k: int = 1,
) -> Tensor:
    """Compute per-GT dynamic_k from top IoU values."""
    if iou_matrix.ndim != 2:
        raise ValueError("iou_matrix must be [M,N]")
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if min_k <= 0:
        raise ValueError("min_k must be > 0")
    if not torch.isfinite(iou_matrix).all():
        raise ValueError("iou_matrix must contain finite values")

    m, n = iou_matrix.shape
    if candidate_mask is not None:
        if candidate_mask.shape != (m, n):
            raise ValueError("candidate_mask must be [M,N] matching iou_matrix")
        if candidate_mask.dtype != torch.bool:
            candidate_mask = candidate_mask.to(dtype=torch.bool)

    dynamic_ks = torch.zeros((m,), dtype=torch.int64, device=iou_matrix.device)
    iou_nonneg = iou_matrix.clamp(min=0.0)

    for gt_idx in range(m):
        if n == 0:
            dynamic_ks[gt_idx] = 0
            continue

        if candidate_mask is None:
            candidate_ious = iou_nonneg[gt_idx]
        else:
            row_mask = candidate_mask[gt_idx]
            if not torch.any(row_mask):
                dynamic_ks[gt_idx] = 0
                continue
            candidate_ious = iou_nonneg[gt_idx, row_mask]

        num_candidates = int(candidate_ious.numel())
        k_for_sum = min(int(topk), num_candidates)
        top_values = torch.topk(candidate_ious, k=k_for_sum, largest=True, sorted=True).values
        raw_k = int(torch.floor(top_values.sum()).item())
        clamped = max(min_k, raw_k)
        clamped = min(clamped, num_candidates)
        dynamic_ks[gt_idx] = int(clamped)

    return dynamic_ks


def dynamic_k_matching(
    total_cost: Tensor,
    iou_matrix: Tensor,
    *,
    candidate_mask: Tensor | None = None,
    dynamic_ks: Tensor | None = None,
    dynamic_topk: int = 10,
    min_k: int = 1,
) -> DynamicKMatchingOutput:
    """Assign positives via dynamic-k minimal-cost matching and conflict resolution."""
    if total_cost.ndim != 2 or iou_matrix.ndim != 2:
        raise ValueError("total_cost and iou_matrix must be [M,N]")
    if total_cost.shape != iou_matrix.shape:
        raise ValueError("total_cost and iou_matrix must have identical shape")
    if not torch.isfinite(total_cost).all():
        raise ValueError("total_cost must contain finite values")

    m, n = total_cost.shape
    if candidate_mask is not None:
        if candidate_mask.shape != (m, n):
            raise ValueError("candidate_mask must be [M,N] matching total_cost")
        if candidate_mask.dtype != torch.bool:
            candidate_mask = candidate_mask.to(dtype=torch.bool)

    if dynamic_ks is None:
        dynamic_ks = dynamic_k_from_top_ious(
            iou_matrix,
            candidate_mask=candidate_mask,
            topk=dynamic_topk,
            min_k=min_k,
        )
    else:
        if dynamic_ks.ndim != 1 or dynamic_ks.shape[0] != m:
            raise ValueError("dynamic_ks must be [M]")
        dynamic_ks = dynamic_ks.to(dtype=torch.int64, device=total_cost.device)

    matching = torch.zeros((m, n), dtype=torch.bool, device=total_cost.device)

    for gt_idx in range(m):
        k = int(dynamic_ks[gt_idx].item())
        if k <= 0 or n == 0:
            continue

        if candidate_mask is None:
            candidate_ids = torch.arange(n, device=total_cost.device, dtype=torch.int64)
        else:
            row_mask = candidate_mask[gt_idx]
            if not torch.any(row_mask):
                continue
            candidate_ids = torch.nonzero(row_mask, as_tuple=False).flatten().to(dtype=torch.int64)

        candidate_cost = total_cost[gt_idx, candidate_ids]
        order = torch.argsort(candidate_cost, dim=0, descending=False, stable=True)
        selected = candidate_ids[order[: min(k, int(candidate_ids.numel()))]]
        matching[gt_idx, selected] = True

    # Resolve conflicts: an anchor can match at most one GT, choose minimal cost GT.
    conflicts = torch.nonzero(matching.sum(dim=0) > 1, as_tuple=False).flatten()
    for anchor_idx in conflicts.tolist():
        gt_ids = torch.nonzero(matching[:, anchor_idx], as_tuple=False).flatten()
        gt_costs = total_cost[gt_ids, anchor_idx]
        best_local = int(torch.argmin(gt_costs).item())
        best_gt = int(gt_ids[best_local].item())
        matching[gt_ids, anchor_idx] = False
        matching[best_gt, anchor_idx] = True

    foreground = torch.any(matching, dim=0)
    matched_gt_indices = torch.full((n,), -1, dtype=torch.int64, device=total_cost.device)
    if n > 0 and torch.any(foreground):
        fg_idx = torch.nonzero(foreground, as_tuple=False).flatten()
        matched_gt_indices[fg_idx] = torch.argmax(matching[:, fg_idx].to(dtype=torch.int64), dim=0)

    assigned_cost = torch.full_like(matched_gt_indices, float("inf"), dtype=total_cost.dtype)
    if n > 0 and torch.any(foreground):
        fg_idx = torch.nonzero(foreground, as_tuple=False).flatten()
        fg_gt = matched_gt_indices[fg_idx]
        assigned_cost[fg_idx] = total_cost[fg_gt, fg_idx]

    return DynamicKMatchingOutput(
        dynamic_ks=dynamic_ks,
        matching_matrix=matching,
        foreground_mask=foreground,
        matched_gt_indices=matched_gt_indices,
        assigned_cost=assigned_cost,
        num_foreground=int(foreground.sum().item()),
    )


def compute_simota_cost(
    *,
    pred_cls_logits: Tensor,
    pred_boxes_xyxy: Tensor,
    anchor_centers_xy: Tensor,
    gt_boxes_xyxy: Tensor,
    gt_classes: Tensor,
    topk_center: int = 10,
    classification_mode: ClassificationCostType = "focal",
    cls_weight: float = 1.0,
    iou_weight: float = 3.0,
    center_weight: float = 1.0,
    non_candidate_penalty: float = 1e6,
) -> SimOTACostOutput:
    """Compute SimOTA/OTA cost components and combined matrix [M, N]."""
    if pred_boxes_xyxy.ndim != 2 or pred_boxes_xyxy.shape[1] != 4:
        raise ValueError("pred_boxes_xyxy must be [N,4]")
    if anchor_centers_xy.ndim != 2 or anchor_centers_xy.shape[1] != 2:
        raise ValueError("anchor_centers_xy must be [N,2]")
    if pred_boxes_xyxy.shape[0] != anchor_centers_xy.shape[0]:
        raise ValueError("pred_boxes_xyxy and anchor_centers_xy must have same N")
    if pred_cls_logits.ndim != 2 or pred_cls_logits.shape[0] != pred_boxes_xyxy.shape[0]:
        raise ValueError("pred_cls_logits must be [N,C] and align with anchor count")
    if gt_boxes_xyxy.ndim != 2 or gt_boxes_xyxy.shape[1] != 4:
        raise ValueError("gt_boxes_xyxy must be [M,4]")
    if gt_classes.ndim != 1 or gt_classes.shape[0] != gt_boxes_xyxy.shape[0]:
        raise ValueError("gt_classes must be [M] and align with GT box count")
    if non_candidate_penalty < 0.0:
        raise ValueError("non_candidate_penalty must be >= 0")
    for name, weight in (
        ("cls_weight", cls_weight),
        ("iou_weight", iou_weight),
        ("center_weight", center_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be >= 0")

    cls = classification_cost(pred_cls_logits, gt_classes, mode=classification_mode)
    iou_c = iou_cost(pred_boxes_xyxy, gt_boxes_xyxy)
    center_c = center_prior_cost(anchor_centers_xy, gt_boxes_xyxy, normalize_by_gt_size=True)
    total = cls_weight * cls + iou_weight * iou_c + center_weight * center_c

    candidates = topk_center_candidates(anchor_centers_xy, gt_boxes_xyxy, topk=topk_center)
    cand_mask = candidate_mask_from_indices(candidates, num_anchors=pred_boxes_xyxy.shape[0])
    if cand_mask.numel() > 0:
        total = total.masked_fill(~cand_mask, float(non_candidate_penalty))

    return SimOTACostOutput(
        cls_cost=cls,
        iou_cost=iou_c,
        center_prior_cost=center_c,
        total_cost=total,
        candidate_indices=candidates,
        candidate_mask=cand_mask,
    )
