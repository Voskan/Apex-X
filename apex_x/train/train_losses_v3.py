"""Enhanced training losses for TeacherModelV3."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from apex_x.losses.iou_loss import bbox_iou
from apex_x.losses.lovasz_loss import lovasz_instance_loss
from apex_x.losses.seg_loss import (
    boundary_iou_loss,
    mask_bce_loss,
    mask_dice_loss,
    multi_scale_instance_segmentation_losses,
)
from apex_x.model.mask_quality_head import mask_iou_loss
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

_LOGIT_CLIP = 30.0
_BOX_CLIP = 1e6


def _resolve_device(outputs: dict[str, Any], model: Any) -> torch.device:
    for value in outputs.values():
        if isinstance(value, Tensor):
            return value.device
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Tensor):
                    return item.device
                if isinstance(item, list):
                    for nested in item:
                        if isinstance(nested, Tensor):
                            return nested.device
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _to_instance_masks(mask: Tensor | None) -> Tensor | None:
    """Normalize masks to [N, H, W] format."""
    if mask is None:
        return None
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask.squeeze(1)
    if mask.ndim == 4 and mask.shape[0] == 1:
        return mask.squeeze(0)
    return mask


def _focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    logits = torch.nan_to_num(
        logits,
        nan=0.0,
        posinf=_LOGIT_CLIP,
        neginf=-_LOGIT_CLIP,
    ).clamp(min=-_LOGIT_CLIP, max=_LOGIT_CLIP)
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce
    return focal.mean()


def _sanitize_boxes(boxes: Tensor) -> Tensor:
    clean = torch.nan_to_num(boxes, nan=0.0, posinf=_BOX_CLIP, neginf=-_BOX_CLIP)
    if clean.ndim != 2 or clean.shape[-1] != 4:
        return clean
    x1 = torch.minimum(clean[:, 0], clean[:, 2])
    y1 = torch.minimum(clean[:, 1], clean[:, 3])
    x2 = torch.maximum(clean[:, 0], clean[:, 2])
    y2 = torch.maximum(clean[:, 1], clean[:, 3])
    return torch.stack((x1, y1, x2, y2), dim=1)


def _sanitize_mask_logits(mask_logits: Tensor) -> Tensor:
    return torch.nan_to_num(
        mask_logits,
        nan=0.0,
        posinf=_LOGIT_CLIP,
        neginf=-_LOGIT_CLIP,
    ).clamp(min=-_LOGIT_CLIP, max=_LOGIT_CLIP)


def _sanitize_binary_mask(mask: Tensor) -> Tensor:
    return torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)


def _finite_or_zero(loss_value: Tensor, *, key: str, device: torch.device) -> Tensor:
    if loss_value.ndim != 0:
        loss_value = loss_value.mean()
    if torch.isfinite(loss_value).all():
        return loss_value
    LOGGER.warning("Non-finite loss component %s detected; forcing to 0.", key)
    return torch.zeros((), device=device, dtype=loss_value.dtype)


def _resolve_box_loss_type(config: Any) -> str:
    default = "mpdiou"
    if hasattr(config, "train") and hasattr(config.train, "box_loss_type"):
        value = str(config.train.box_loss_type).lower()
        if value in {"iou", "giou", "diou", "ciou", "mpdiou"}:
            return value
    return default


def _loss_weight(config: Any, key: str, default: float) -> float:
    if not hasattr(config, "loss"):
        return default
    attr = f"{key}_weight"
    if hasattr(config.loss, attr):
        try:
            return float(getattr(config.loss, attr))
        except (TypeError, ValueError):
            return default
    return default


def compute_v3_training_losses(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    model: Any,
    config: Any,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute v3 training losses with finite-safe sanitization."""
    device = _resolve_device(outputs, model)
    loss_dict: dict[str, Tensor] = {}

    # 1) Classification focal loss
    if "scores" in outputs and "labels" in targets:
        logits_raw = outputs["scores"]
        labels_raw = targets["labels"]
        if isinstance(logits_raw, Tensor) and isinstance(labels_raw, Tensor):
            if logits_raw.ndim == 2 and labels_raw.ndim == 1:
                if logits_raw.numel() > 0 and labels_raw.numel() > 0 and logits_raw.shape[1] > 0:
                    n = min(logits_raw.shape[0], labels_raw.shape[0])
                    logits = _sanitize_mask_logits(logits_raw[:n])
                    labels = labels_raw[:n].to(device=device, dtype=torch.long)
                    labels = labels.clamp(min=0, max=logits.shape[1] - 1)
                    loss_dict["cls"] = _focal_loss(logits, labels)

    # 2) Box regression loss (IoU family)
    box_loss_type = _resolve_box_loss_type(config)
    if "boxes" in outputs and "boxes" in targets:
        pred_b_raw = outputs["boxes"]
        gt_b_raw = targets["boxes"]
        if isinstance(pred_b_raw, Tensor) and isinstance(gt_b_raw, Tensor):
            pred_b = _sanitize_boxes(pred_b_raw)
            gt_b = _sanitize_boxes(gt_b_raw.to(device))
            if pred_b.numel() > 0 and gt_b.numel() > 0:
                n = min(pred_b.shape[0], gt_b.shape[0])
                iou = bbox_iou(
                    pred_b[:n],
                    gt_b[:n],
                    xywh=False,
                    GIoU=box_loss_type == "giou",
                    DIoU=box_loss_type == "diou",
                    CIoU=box_loss_type == "ciou",
                    MPDIoU=box_loss_type == "mpdiou",
                )
                loss_dict[f"box_{box_loss_type}"] = (1.0 - iou).mean()

    # 3) Segmentation losses (BCE + Dice + Lovasz)
    if "masks" in outputs and outputs["masks"] is not None and "masks" in targets:
        pred_masks_raw = outputs["masks"]
        gt_masks_raw = targets["masks"]
        if isinstance(pred_masks_raw, Tensor) and isinstance(gt_masks_raw, Tensor):
            mask_pred = _to_instance_masks(_sanitize_mask_logits(pred_masks_raw))
            mask_gt = _to_instance_masks(gt_masks_raw)
            if mask_pred is not None and mask_gt is not None:
                if mask_pred.ndim == 2:
                    mask_pred = mask_pred.unsqueeze(0)
                if mask_gt.ndim == 2:
                    mask_gt = mask_gt.unsqueeze(0)
                if mask_pred.ndim == 3 and mask_gt.ndim == 3:
                    # [N,H,W] -> [1,N,H,W]
                    mask_pred_4d = mask_pred.unsqueeze(0)
                    mask_gt_4d = _sanitize_binary_mask(mask_gt.to(device).unsqueeze(0))
                    if mask_pred_4d.shape[-2:] != mask_gt_4d.shape[-2:]:
                        mask_gt_4d = F.interpolate(
                            mask_gt_4d.float(),
                            size=mask_pred_4d.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    n = min(mask_pred_4d.shape[1], mask_gt_4d.shape[1])
                    if n > 0:
                        mask_pred_4d = mask_pred_4d[:, :n]
                        mask_gt_4d = mask_gt_4d[:, :n]
                        loss_dict["mask_bce"] = mask_bce_loss(mask_pred_4d, mask_gt_4d)
                        loss_dict["mask_dice"] = mask_dice_loss(mask_pred_4d, mask_gt_4d)
                        loss_dict["lovasz"] = lovasz_instance_loss(mask_pred_4d, mask_gt_4d)

    # 4) Boundary IoU
    mask_logits = outputs.get("masks")
    target_masks = targets.get("masks")
    if isinstance(mask_logits, Tensor) and isinstance(target_masks, Tensor):
        mask_logits = _to_instance_masks(_sanitize_mask_logits(mask_logits))
        mask_gt_b = _to_instance_masks(target_masks)
        if mask_logits is not None and mask_gt_b is not None:
            if mask_logits.ndim == 2:
                mask_logits = mask_logits.unsqueeze(0)
            if mask_gt_b.ndim == 2:
                mask_gt_b = mask_gt_b.unsqueeze(0)
            if mask_logits.ndim == 3 and mask_gt_b.ndim == 3:
                mask_logits_4d = mask_logits.unsqueeze(0)
                mask_gt_b_4d = _sanitize_binary_mask(mask_gt_b.to(device).unsqueeze(0))
                if mask_logits_4d.shape[-2:] != mask_gt_b_4d.shape[-2:]:
                    mask_gt_b_4d = F.interpolate(
                        mask_gt_b_4d.float(),
                        size=mask_logits_4d.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                n = min(mask_logits_4d.shape[1], mask_gt_b_4d.shape[1])
                if n > 0:
                    loss_dict["boundary_iou"] = boundary_iou_loss(
                        mask_logits_4d[:, :n],
                        mask_gt_b_4d[:, :n],
                        boundary_width=3,
                        reduction="mean",
                    )

    # 5) Mask quality loss
    pred_quality = outputs.get("predicted_quality")
    if isinstance(pred_quality, Tensor) and isinstance(mask_logits, Tensor) and isinstance(target_masks, Tensor):
        mp = _to_instance_masks(_sanitize_mask_logits(mask_logits))
        mg = _to_instance_masks(target_masks)
        if mp is not None and mg is not None:
            if mp.ndim == 4 and mp.shape[1] == 1:
                mp = mp.squeeze(1)
            if mp.ndim == 2:
                mp = mp.unsqueeze(0)
            if mg.ndim == 4 and mg.shape[1] == 1:
                mg = mg.squeeze(1)
            if mg.ndim == 2:
                mg = mg.unsqueeze(0)
            if mp.ndim == 3 and mg.ndim == 3:
                mg = _sanitize_binary_mask(mg.to(device))
                if mp.shape[-2:] != mg.shape[-2:]:
                    mg = F.interpolate(
                        mg.unsqueeze(0).float(),
                        size=mp.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                n = min(pred_quality.shape[0], mp.shape[0], mg.shape[0])
                if n > 0:
                    pq = torch.nan_to_num(
                        pred_quality[:n],
                        nan=0.5,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(min=0.0, max=1.0)
                    loss_dict["mask_quality"] = mask_iou_loss(pq, mp[:n], mg[:n])

    # 6) Optional multi-scale supervision
    use_ms = hasattr(config, "loss") and bool(getattr(config.loss, "multi_scale_supervision", False))
    if use_ms and isinstance(mask_logits, Tensor) and isinstance(target_masks, Tensor):
        ms_pred = _sanitize_mask_logits(mask_logits)
        ms_gt = _to_instance_masks(target_masks)
        if ms_gt is not None:
            ms_gt = _sanitize_binary_mask(ms_gt.to(device))
            if ms_pred.ndim == 4 and ms_pred.shape[1] == 1:
                ms_pred = ms_pred.squeeze(1)
            if ms_gt.ndim == 4 and ms_gt.shape[1] == 1:
                ms_gt = ms_gt.squeeze(1)
            if ms_pred.ndim == 3 and ms_gt.ndim == 3 and ms_pred.numel() > 0 and ms_gt.numel() > 0:
                ms_pred_4d = ms_pred.unsqueeze(0)
                ms_gt_4d = ms_gt.unsqueeze(0)
                n = min(ms_pred_4d.shape[1], ms_gt_4d.shape[1])
                if n > 0:
                    ms_out = multi_scale_instance_segmentation_losses(ms_pred_4d[:, :n], ms_gt_4d[:, :n])
                    loss_dict["multi_scale"] = ms_out.total_loss

    # 7) Optional cascade stage losses
    all_boxes = outputs.get("all_boxes")
    gt_boxes = targets.get("boxes")
    if isinstance(all_boxes, list) and isinstance(gt_boxes, Tensor):
        gt_b = _sanitize_boxes(gt_boxes.to(device))
        cascade_w = 0.3
        for stage_idx, stage_boxes_list in enumerate(all_boxes[1:]):
            if isinstance(stage_boxes_list, list):
                non_empty = [b for b in stage_boxes_list if isinstance(b, Tensor) and b.numel() > 0]
                stage_boxes = (
                    torch.cat(non_empty, dim=0)
                    if non_empty
                    else torch.zeros((0, 4), device=device)
                )
            elif isinstance(stage_boxes_list, Tensor):
                stage_boxes = stage_boxes_list
            else:
                continue
            stage_boxes = _sanitize_boxes(stage_boxes)
            if stage_boxes.numel() > 0 and gt_b.numel() > 0:
                n = min(stage_boxes.shape[0], gt_b.shape[0])
                iou_stage = bbox_iou(
                    stage_boxes[:n],
                    gt_b[:n],
                    xywh=False,
                    GIoU=box_loss_type == "giou",
                    DIoU=box_loss_type == "diou",
                    CIoU=box_loss_type == "ciou",
                    MPDIoU=box_loss_type == "mpdiou",
                )
                loss_dict[f"cascade_s{stage_idx}_{box_loss_type}"] = cascade_w * (1.0 - iou_stage).mean()

    default_weights = {
        "cls": 1.0,
        "mask_bce": 1.0,
        "mask_dice": 1.0,
        "lovasz": _loss_weight(config, "lovasz", 0.5),
        "boundary_iou": _loss_weight(config, "boundary", 0.5),
        "mask_quality": _loss_weight(config, "quality", 1.0),
        "multi_scale": 1.0,
    }
    box_weight = _loss_weight(config, "box", 2.0)

    safe_loss_dict: dict[str, Tensor] = {}
    total = torch.zeros((), device=device)
    for key, val in loss_dict.items():
        safe_val = _finite_or_zero(val, key=key, device=device)
        safe_loss_dict[key] = safe_val
        if key.startswith("box_"):
            weight = box_weight
        else:
            weight = default_weights.get(key, 1.0)
        total = total + float(weight) * safe_val

    if not safe_loss_dict:
        anchor = next((v for v in outputs.values() if isinstance(v, Tensor)), None)
        if anchor is not None:
            total = anchor.sum() * 0.0
        else:
            total = torch.zeros((), device=device)

    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)
    return total, safe_loss_dict


__all__ = ["compute_v3_training_losses"]
