"""Enhanced training losses for TeacherModelV3.

Integrates all v2.0 loss functions:
- Classification loss (cross-entropy / focal)
- Box regression (GIoU)
- Mask BCE + Dice
- Boundary IoU loss   (+0.5-1% AP)
- Mask quality loss   (+1-2% AP)
- Multi-scale mask supervision (+0.5-1% AP)
- Cascade stage-wise supervision
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from apex_x.losses.seg_loss import (
    mask_bce_loss,
    mask_dice_loss,
    boundary_iou_loss,
    multi_scale_instance_segmentation_losses,
)
from apex_x.model.mask_quality_head import mask_iou_loss


# --------------------------------------------------------------------------- #
# GIoU box loss (no external dep required)
# --------------------------------------------------------------------------- #

def _giou_loss(pred_boxes: Tensor, gt_boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """Generalised IoU loss for bounding-box regression.

    Both inputs are ``[N, 4]`` in ``(x1, y1, x2, y2)`` format.
    Returns scalar loss (1 − GIoU), averaged over N.
    """
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * \
              (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)

    union = area_pred + area_gt - inter + eps
    iou = inter / union

    # enclosing box
    ex1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    ey1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    ex2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    ey2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enclose_area = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + eps

    giou = iou - (enclose_area - union) / enclose_area
    return (1.0 - giou).mean()


# --------------------------------------------------------------------------- #
# Main loss function
# --------------------------------------------------------------------------- #

def compute_v3_training_losses(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    model: Any,
    config: Any,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute all losses for TeacherModelV3.

    Args:
        outputs: Model predictions dict.
        targets: Ground truth dict (``labels``, ``boxes``, ``masks``).
        model: The model (used only to discover device).
        config: Training config with optional ``.loss`` namespace.

    Returns:
        ``(total_loss, loss_dict)``
    """
    loss_dict: dict[str, Tensor] = {}
    device = next(model.parameters()).device

    # helper
    def _w(name: str, default: float) -> float:
        if hasattr(config, "loss"):
            return float(getattr(config.loss, f"{name}_weight", default))
        return default

    # ----- 1. classification loss ------------------------------------------
    if "scores" in outputs and "labels" in targets:
        scores = outputs["scores"]
        labels = targets["labels"].long().to(device)
        if scores.numel() > 0 and labels.numel() > 0:
            # match N dimension
            if scores.shape[0] != labels.shape[0]:
                n = min(scores.shape[0], labels.shape[0])
                scores = scores[:n]
                labels = labels[:n]
            # clamp labels to valid range
            num_cls = scores.shape[-1]
            labels = labels.clamp(0, num_cls - 1)
            loss_dict["cls"] = F.cross_entropy(scores, labels)

    # ----- 2. box regression loss (GIoU) -----------------------------------
    if "boxes" in outputs and "boxes" in targets:
        pred_b = outputs["boxes"]
        gt_b = targets["boxes"].to(device)
        if pred_b.numel() > 0 and gt_b.numel() > 0:
            n = min(pred_b.shape[0], gt_b.shape[0])
            loss_dict["box"] = _giou_loss(pred_b[:n], gt_b[:n])

    # ----- 3. segmentation losses (BCE + Dice) -----------------------------
    if "masks" in outputs and outputs["masks"] is not None and "masks" in targets:
        mask_pred = outputs["masks"]
        mask_gt = targets["masks"].to(device)
        if mask_pred.numel() > 0 and mask_gt.numel() > 0:
            # align spatial dimensions
            if mask_pred.shape != mask_gt.shape:
                mask_gt = F.interpolate(
                    mask_gt.float(),
                    size=mask_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss_dict["mask_bce"] = mask_bce_loss(mask_pred, mask_gt)
            loss_dict["mask_dice"] = mask_dice_loss(mask_pred, mask_gt)

    # ----- 4. boundary IoU loss (+0.5-1% AP) -------------------------------
    mask_logits = outputs.get("masks")
    if mask_logits is not None and "masks" in targets:
        mask_gt_b = targets["masks"].to(device)
        if mask_logits.shape != mask_gt_b.shape:
            mask_gt_b = F.interpolate(
                mask_gt_b.float(),
                size=mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss_dict["boundary_iou"] = boundary_iou_loss(
            mask_logits, mask_gt_b, boundary_width=3, reduction="mean",
        )

    # ----- 5. mask quality loss (+1-2% AP) ---------------------------------
    if (
        "predicted_quality" in outputs
        and "masks" in outputs
        and outputs["masks"] is not None
        and "masks" in targets
    ):
        mask_pred_q = outputs["masks"]
        mask_gt_q = targets["masks"].to(device)
        if mask_pred_q.shape != mask_gt_q.shape:
            mask_gt_q = F.interpolate(
                mask_gt_q.float(),
                size=mask_pred_q.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        # mask_iou_loss expects [N, H, W] — squeeze channel dim if present
        mp = mask_pred_q.squeeze(1) if mask_pred_q.dim() == 4 else mask_pred_q
        mg = mask_gt_q.squeeze(1) if mask_gt_q.dim() == 4 else mask_gt_q
        n = min(outputs["predicted_quality"].shape[0], mp.shape[0])
        loss_dict["mask_quality"] = mask_iou_loss(
            outputs["predicted_quality"][:n], mp[:n], mg[:n],
        )

    # ----- 6. multi-scale supervision (+0.5-1% AP) -------------------------
    use_ms = hasattr(config, "loss") and getattr(config.loss, "multi_scale_supervision", False)
    if use_ms and mask_logits is not None and "masks" in targets:
        ms_gt = targets["masks"].to(device)
        if mask_logits.shape != ms_gt.shape:
            ms_gt = F.interpolate(
                ms_gt.float(), size=mask_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        ms_out = multi_scale_instance_segmentation_losses(mask_logits, ms_gt)
        loss_dict["multi_scale"] = ms_out.total_loss

    # ----- 7. cascade stage losses -----------------------------------------
    if "all_boxes" in outputs and "boxes" in targets:
        gt_b = targets["boxes"].to(device)
        cascade_w = 0.3
        # outputs["all_boxes"] is list[list[Tensor]] -> [Stage][Batch]
        for stage_idx, stage_boxes_list in enumerate(outputs["all_boxes"][1:]):
            if isinstance(stage_boxes_list, list):
                # Flatten the list of tensors per batch image into one tensor for matching gt_b
                stage_boxes = torch.cat([b for b in stage_boxes_list if b.numel() > 0]) if any(b.numel() > 0 for b in stage_boxes_list) else torch.zeros((0, 4), device=device)
            else:
                stage_boxes = stage_boxes_list

            if stage_boxes.numel() > 0 and gt_b.numel() > 0:
                n = min(stage_boxes.shape[0], gt_b.shape[0])
                loss_dict[f"cascade_s{stage_idx}"] = cascade_w * _giou_loss(
                    stage_boxes[:n], gt_b[:n],
                )

    # ----- weighted total --------------------------------------------------
    default_weights = {
        "cls": 1.0,
        "box": 2.0,
        "mask_bce": 1.0,
        "mask_dice": 1.0,
        "boundary_iou": _w("boundary", 0.5),
        "mask_quality": _w("quality", 1.0),
        "multi_scale": 1.0,
    }

    total = torch.tensor(0.0, device=device)
    for key, val in loss_dict.items():
        w = default_weights.get(key, 1.0)
        total = total + w * val

    return total, loss_dict


__all__ = ["compute_v3_training_losses"]
