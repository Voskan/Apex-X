from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as f

from .simota import (
    ClassificationCostType,
    DynamicKMatchingOutput,
    compute_simota_cost,
    dynamic_k_matching,
)

ClsLossType = Literal["bce", "focal"]
QualityLossType = Literal["bce", "qfl"]


@dataclass(frozen=True)
class SimOTATargets:
    """Anchor targets produced by SimOTA dynamic-k assignment."""

    foreground_mask: Tensor  # [N] bool
    matched_gt_indices: Tensor  # [N] int64, -1 for background
    cls_target: Tensor  # [N, C]
    box_target: Tensor  # [N, 4] xyxy
    quality_target: Tensor  # [N]
    positive_weights: Tensor  # [N], >0 for positives and 0 for background
    matching: DynamicKMatchingOutput
    num_foreground: int


@dataclass(frozen=True)
class DetLossOutput:
    """DET loss bundle with assignment targets and component losses."""

    total_loss: Tensor
    cls_loss: Tensor
    box_loss: Tensor
    quality_loss: Tensor
    targets: SimOTATargets


def _sanitize_logits(logits: Tensor, *, logit_clip: float = 30.0) -> Tensor:
    if logit_clip <= 0.0:
        raise ValueError("logit_clip must be > 0")
    return torch.nan_to_num(logits, nan=0.0, posinf=logit_clip, neginf=-logit_clip).clamp(
        -logit_clip,
        logit_clip,
    )


def _validate_pred_shapes(
    pred_cls_logits: Tensor,
    pred_boxes_xyxy: Tensor,
    pred_quality_logits: Tensor,
    anchor_centers_xy: Tensor,
) -> None:
    if pred_cls_logits.ndim != 2:
        raise ValueError("pred_cls_logits must be [N,C]")
    if pred_boxes_xyxy.ndim != 2 or pred_boxes_xyxy.shape[1] != 4:
        raise ValueError("pred_boxes_xyxy must be [N,4]")
    if anchor_centers_xy.ndim != 2 or anchor_centers_xy.shape[1] != 2:
        raise ValueError("anchor_centers_xy must be [N,2]")

    n = pred_cls_logits.shape[0]
    if pred_boxes_xyxy.shape[0] != n or anchor_centers_xy.shape[0] != n:
        raise ValueError("prediction tensors must share anchor dimension N")

    if pred_quality_logits.ndim == 2:
        if pred_quality_logits.shape != (n, 1):
            raise ValueError("pred_quality_logits must be [N] or [N,1]")
    elif pred_quality_logits.ndim == 1:
        if pred_quality_logits.shape[0] != n:
            raise ValueError("pred_quality_logits must be [N] or [N,1]")
    else:
        raise ValueError("pred_quality_logits must be [N] or [N,1]")

    if not torch.isfinite(pred_cls_logits).all():
        raise ValueError("pred_cls_logits must contain finite values")
    if not torch.isfinite(pred_boxes_xyxy).all():
        raise ValueError("pred_boxes_xyxy must contain finite values")
    if not torch.isfinite(pred_quality_logits).all():
        raise ValueError("pred_quality_logits must contain finite values")
    if not torch.isfinite(anchor_centers_xy).all():
        raise ValueError("anchor_centers_xy must contain finite values")


def _canonicalize_xyxy(boxes_xyxy: Tensor) -> Tensor:
    x1 = torch.minimum(boxes_xyxy[:, 0], boxes_xyxy[:, 2])
    y1 = torch.minimum(boxes_xyxy[:, 1], boxes_xyxy[:, 3])
    x2 = torch.maximum(boxes_xyxy[:, 0], boxes_xyxy[:, 2])
    y2 = torch.maximum(boxes_xyxy[:, 1], boxes_xyxy[:, 3])
    return torch.stack((x1, y1, x2, y2), dim=1)


def _pairwise_iou(
    boxes_a_xyxy: Tensor,
    boxes_b_xyxy: Tensor,
    *,
    eps: float = 1e-9,
) -> Tensor:
    # boxes_a: [M,4], boxes_b: [N,4] -> [M,N]
    m = boxes_a_xyxy.shape[0]
    n = boxes_b_xyxy.shape[0]
    if m == 0 or n == 0:
        return boxes_a_xyxy.new_zeros((m, n))

    a = boxes_a_xyxy[:, None, :]
    b = boxes_b_xyxy[None, :, :]
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]).clamp(min=0.0) * (a[..., 3] - a[..., 1]).clamp(min=0.0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0.0) * (b[..., 3] - b[..., 1]).clamp(min=0.0)
    union = (area_a + area_b - inter).clamp(min=eps)
    return inter / union


def _box_iou_diag(pred_xyxy: Tensor, target_xyxy: Tensor, *, eps: float = 1e-9) -> Tensor:
    if pred_xyxy.shape != target_xyxy.shape:
        raise ValueError("pred_xyxy and target_xyxy must have identical shape")
    if pred_xyxy.ndim != 2 or pred_xyxy.shape[1] != 4:
        raise ValueError("pred_xyxy and target_xyxy must be [N,4]")
    if pred_xyxy.shape[0] == 0:
        return pred_xyxy.new_zeros((0,))

    inter_x1 = torch.maximum(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.maximum(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.minimum(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.minimum(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    ).clamp(min=0.0)
    area_t = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0.0) * (
        target_xyxy[:, 3] - target_xyxy[:, 1]
    ).clamp(min=0.0)
    union = (area_p + area_t - inter).clamp(min=eps)
    return inter / union


def build_simota_targets_for_anchors(
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
    dynamic_topk: int = 10,
    min_dynamic_k: int = 1,
    small_object_boost: float = 2.0,
    boost_min: float = 1.0,
    boost_max: float = 4.0,
) -> SimOTATargets:
    """Build SimOTA assignment targets for anchors.

    Includes small-object stability weighting by inverse sqrt(area) with clipping.
    """
    _validate_pred_shapes(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes_xyxy,
        pred_quality_logits=pred_cls_logits.new_zeros((pred_cls_logits.shape[0],)),
        anchor_centers_xy=anchor_centers_xy,
    )
    if gt_boxes_xyxy.ndim != 2 or gt_boxes_xyxy.shape[1] != 4:
        raise ValueError("gt_boxes_xyxy must be [M,4]")
    if gt_classes.ndim != 1 or gt_classes.shape[0] != gt_boxes_xyxy.shape[0]:
        raise ValueError("gt_classes must be [M]")
    if gt_classes.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("gt_classes must be integer tensor")
    if small_object_boost < 0.0:
        raise ValueError("small_object_boost must be >= 0")
    if boost_min <= 0.0 or boost_max < boost_min:
        raise ValueError("boost_min/boost_max must satisfy 0 < boost_min <= boost_max")

    n, num_classes = pred_cls_logits.shape
    m = gt_boxes_xyxy.shape[0]
    device = pred_cls_logits.device
    dtype = pred_cls_logits.dtype

    pred_boxes = _canonicalize_xyxy(pred_boxes_xyxy)
    gt_boxes = _canonicalize_xyxy(gt_boxes_xyxy.to(device=device, dtype=dtype))
    gt_classes_i64 = gt_classes.to(dtype=torch.int64, device=device)

    foreground_mask = torch.zeros((n,), dtype=torch.bool, device=device)
    matched_gt = torch.full((n,), -1, dtype=torch.int64, device=device)
    cls_target = torch.zeros((n, num_classes), dtype=dtype, device=device)
    box_target = torch.zeros((n, 4), dtype=dtype, device=device)
    quality_target = torch.zeros((n,), dtype=dtype, device=device)
    positive_weights = torch.zeros((n,), dtype=dtype, device=device)

    if n == 0 or m == 0:
        empty_matching = DynamicKMatchingOutput(
            dynamic_ks=torch.zeros((m,), dtype=torch.int64, device=device),
            matching_matrix=torch.zeros((m, n), dtype=torch.bool, device=device),
            foreground_mask=foreground_mask,
            matched_gt_indices=matched_gt,
            assigned_cost=torch.full((n,), float("inf"), dtype=dtype, device=device),
            num_foreground=0,
        )
        return SimOTATargets(
            foreground_mask=foreground_mask,
            matched_gt_indices=matched_gt,
            cls_target=cls_target,
            box_target=box_target,
            quality_target=quality_target,
            positive_weights=positive_weights,
            matching=empty_matching,
            num_foreground=0,
        )

    simota_cost = compute_simota_cost(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes,
        anchor_centers_xy=anchor_centers_xy,
        gt_boxes_xyxy=gt_boxes,
        gt_classes=gt_classes_i64,
        topk_center=topk_center,
        classification_mode=classification_mode,
        cls_weight=cls_weight,
        iou_weight=iou_weight,
        center_weight=center_weight,
        non_candidate_penalty=non_candidate_penalty,
    )
    iou_matrix = (1.0 - simota_cost.iou_cost).clamp(min=0.0, max=1.0)
    matching = dynamic_k_matching(
        total_cost=simota_cost.total_cost,
        iou_matrix=iou_matrix,
        candidate_mask=simota_cost.candidate_mask,
        dynamic_topk=dynamic_topk,
        min_k=min_dynamic_k,
    )

    foreground_mask = matching.foreground_mask
    matched_gt = matching.matched_gt_indices
    num_foreground = int(foreground_mask.sum().item())
    if num_foreground == 0:
        return SimOTATargets(
            foreground_mask=foreground_mask,
            matched_gt_indices=matched_gt,
            cls_target=cls_target,
            box_target=box_target,
            quality_target=quality_target,
            positive_weights=positive_weights,
            matching=matching,
            num_foreground=0,
        )

    fg_idx = torch.nonzero(foreground_mask, as_tuple=False).flatten()
    fg_gt = matched_gt[fg_idx]
    fg_classes = gt_classes_i64[fg_gt]

    cls_target[fg_idx, fg_classes] = 1.0
    box_target[fg_idx] = gt_boxes[fg_gt]
    quality_target[fg_idx] = iou_matrix[fg_gt, fg_idx].detach().clamp(min=0.0, max=1.0)

    # Small-object stability: inverse sqrt(area), clipped to avoid exploding gradients.
    gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
    gt_area = gt_w * gt_h
    area_ref = torch.median(gt_area).clamp(min=1e-6)
    area_ratio = torch.sqrt(area_ref / gt_area)
    gt_pos_weight = (1.0 + small_object_boost * area_ratio).clamp(min=boost_min, max=boost_max)
    positive_weights[fg_idx] = gt_pos_weight[fg_gt]

    return SimOTATargets(
        foreground_mask=foreground_mask,
        matched_gt_indices=matched_gt,
        cls_target=cls_target,
        box_target=box_target,
        quality_target=quality_target,
        positive_weights=positive_weights,
        matching=matching,
        num_foreground=num_foreground,
    )


def det_loss_with_simota(
    *,
    pred_cls_logits: Tensor,
    pred_boxes_xyxy: Tensor,
    pred_quality_logits: Tensor,
    anchor_centers_xy: Tensor,
    gt_boxes_xyxy: Tensor,
    gt_classes: Tensor,
    topk_center: int = 10,
    classification_mode: ClassificationCostType = "focal",
    assign_on_detached_preds: bool = True,
    cls_cost_weight: float = 1.0,
    iou_cost_weight: float = 3.0,
    center_cost_weight: float = 1.0,
    non_candidate_penalty: float = 1e6,
    dynamic_topk: int = 10,
    min_dynamic_k: int = 1,
    small_object_boost: float = 2.0,
    cls_loss_type: ClsLossType = "bce",
    quality_loss_type: QualityLossType = "bce",
    cls_loss_weight: float = 1.0,
    box_loss_weight: float = 2.0,
    quality_loss_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    quality_focal_beta: float = 2.0,
    logit_clip: float = 30.0,
    box_loss_type: str = "mpdiou",
) -> DetLossOutput:
    """Compute DET loss with SimOTA dynamic-k target assignment."""
    _validate_pred_shapes(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes_xyxy,
        pred_quality_logits=pred_quality_logits,
        anchor_centers_xy=anchor_centers_xy,
    )
    if cls_loss_type not in {"bce", "focal"}:
        raise ValueError("cls_loss_type must be one of {'bce', 'focal'}")
    if quality_loss_type not in {"bce", "qfl"}:
        raise ValueError("quality_loss_type must be one of {'bce', 'qfl'}")
    if quality_focal_beta < 0.0:
        raise ValueError("quality_focal_beta must be >= 0")

    pred_cls = _sanitize_logits(pred_cls_logits, logit_clip=logit_clip)
    pred_quality = _sanitize_logits(pred_quality_logits.reshape(-1), logit_clip=logit_clip)
    pred_boxes = _canonicalize_xyxy(pred_boxes_xyxy)

    if assign_on_detached_preds:
        assign_cls = pred_cls.detach()
        assign_boxes = pred_boxes.detach()
    else:
        assign_cls = pred_cls
        assign_boxes = pred_boxes

    targets = build_simota_targets_for_anchors(
        pred_cls_logits=assign_cls,
        pred_boxes_xyxy=assign_boxes,
        anchor_centers_xy=(
            anchor_centers_xy.detach() if assign_on_detached_preds else anchor_centers_xy
        ),
        gt_boxes_xyxy=gt_boxes_xyxy,
        gt_classes=gt_classes,
        topk_center=topk_center,
        classification_mode=classification_mode,
        cls_weight=cls_cost_weight,
        iou_weight=iou_cost_weight,
        center_weight=center_cost_weight,
        non_candidate_penalty=non_candidate_penalty,
        dynamic_topk=dynamic_topk,
        min_dynamic_k=min_dynamic_k,
        small_object_boost=small_object_boost,
    )

    fg = targets.foreground_mask
    anchor_weights = torch.ones_like(pred_quality)
    anchor_weights[fg] = targets.positive_weights[fg]

    # Classification loss (binary one-vs-rest over classes).
    if cls_loss_type == "bce":
        cls_loss_raw = f.binary_cross_entropy_with_logits(
            pred_cls,
            targets.cls_target,
            reduction="none",
        )
    else:
        p = torch.sigmoid(pred_cls)
        p_t = p * targets.cls_target + (1.0 - p) * (1.0 - targets.cls_target)
        alpha_t = targets.cls_target * focal_alpha + (1.0 - targets.cls_target) * (
            1.0 - focal_alpha
        )
        cls_loss_raw = (
            f.binary_cross_entropy_with_logits(
                pred_cls,
                targets.cls_target,
                reduction="none",
            )
            * alpha_t
            * torch.pow((1.0 - p_t).clamp(min=0.0), focal_gamma)
        )

    cls_denom = anchor_weights.sum().clamp(min=1.0)
    cls_loss = (cls_loss_raw * anchor_weights[:, None]).sum() / cls_denom

    # Box IoU loss on positives.
    if targets.num_foreground > 0:
        fg_pred_boxes = pred_boxes[fg]
        fg_target_boxes = targets.box_target[fg]
        
        from apex_x.losses.iou_loss import bbox_iou
        
        # Determine IoU loss type flags
        giou = box_loss_type == "giou"
        diou = box_loss_type == "diou"
        ciou = box_loss_type == "ciou"
        mpdiou = box_loss_type == "mpdiou"
        
        iou = bbox_iou(
            fg_pred_boxes, 
            fg_target_boxes, 
            xywh=False, 
            GIoU=giou, 
            DIoU=diou, 
            CIoU=ciou, 
            MPDIoU=mpdiou
        ).clamp(min=-1.0, max=1.0)
        
        fg_w = targets.positive_weights[fg]
        box_loss = ((1.0 - iou) * fg_w).sum() / fg_w.sum().clamp(min=1.0)
    else:
        box_loss = pred_boxes.new_zeros(())

    # Quality target predicts assigned IoU for positives and 0 for background.
    quality_target = targets.quality_target
    if quality_loss_type == "bce":
        quality_loss_raw = f.binary_cross_entropy_with_logits(
            pred_quality,
            quality_target,
            reduction="none",
        )
    else:
        q_prob = torch.sigmoid(pred_quality)
        quality_loss_raw = f.binary_cross_entropy_with_logits(
            pred_quality,
            quality_target,
            reduction="none",
        ) * torch.pow(torch.abs(quality_target - q_prob), quality_focal_beta)
    quality_denom = anchor_weights.sum().clamp(min=1.0)
    quality_loss = (quality_loss_raw * anchor_weights).sum() / quality_denom

    total = (
        cls_loss_weight * cls_loss + box_loss_weight * box_loss + quality_loss_weight * quality_loss
    )
    return DetLossOutput(
        total_loss=total,
        cls_loss=cls_loss,
        box_loss=box_loss,
        quality_loss=quality_loss,
        targets=targets,
    )
