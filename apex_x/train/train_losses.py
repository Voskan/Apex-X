"""Training loss computation utilities for Apex-X.

This module provides helper functions to compute training losses from model outputs
and ground truth annotations.
"""

from __future__ import annotations


import torch
from torch import Tensor

from apex_x.data import TransformSample
from apex_x.model import TeacherDistillOutput, compute_anchor_centers
from apex_x.losses.det_loss import (
    build_simota_targets_for_anchors,
    det_loss_with_simota,
)
from apex_x.losses.seg_loss import instance_segmentation_losses


def compute_teacher_training_loss(
    output: TeacherDistillOutput,
    samples: list[TransformSample] | None = None,
    *,
    det_weight: float = 1.0,
    boundary_weight: float = 0.05,
    quality_weight: float = 1.0,
    cls_weight: float = 1.0,
    box_weight: float = 2.0,
    epoch: int | None = None,
    total_epochs: int | None = None,
    adaptive_boundary: bool = True,
    box_loss_type: str = "mpdiou",
) -> tuple[Tensor, dict[str, float]]:
    """Compute training loss for teacher model using SimOTA.
    
    Args:
        output: Teacher model forward pass output
        samples: List of training samples with GT annotations
        det_weight: Overall weight for detection loss
        boundary_weight: Base weight for boundary distillation loss
        quality_weight: Weight for quality prediction loss
        cls_weight: Weight for classification loss component
        box_weight: Weight for box regression loss component
        epoch: Current epoch (for progressive loss balancing)
        total_epochs: Total training epochs (for progressive loss balancing)
        adaptive_boundary: Use adaptive boundary weight scheduling
        box_loss_type: Type of box loss (iou, giou, diou, ciou, mpdiou)
    
    Returns:
        tuple of (total_loss, loss_dict) with individual components
    """
    device = output.logits.device
    loss_dict: dict[str, float] = {}
    
    # Adaptive boundary weight: 0.05 → 0.5 during training
    # Early: focus on detection, Late: refine boundaries
    if adaptive_boundary and epoch is not None and total_epochs is not None and total_epochs > 10:
        warmup = 10  # First 10 epochs keep low boundary weight
        progress = max(0.0, min(1.0, (epoch - warmup) / (total_epochs - warmup)))
        # Linear increase from base (0.05) to 10x base (0.5)
        boundary_weight = boundary_weight * (1.0 + 9.0 * progress)
        loss_dict['boundary_weight_adaptive'] = boundary_weight
    
    if samples is None or len(samples) == 0:
        # Fallback: dummy loss for synthetic data
        det_loss = output.logits.pow(2).mean()
        boundary_loss = output.boundaries.mean()
        total_loss = det_loss + boundary_weight * boundary_loss
        
        loss_dict["det_loss"] = float(det_loss.item())
        loss_dict["boundary_loss"] = float(boundary_loss.item())
        loss_dict["total_loss"] = float(total_loss.item())
        
        return total_loss, loss_dict
    
    # Progressive loss balancing (YOLO26-style)
    if epoch is not None and total_epochs is not None and total_epochs > 5:
        warmup = 5
        progress = max(0.0, min(1.0, (epoch - warmup) / (total_epochs - warmup)))
        # Early: focus classification, Late: refine boxes
        cls_weight = cls_weight * 1.0
        box_weight = box_weight * (0.5 + 1.5 * progress)  # 0.5 → 2.0
    
    # Extract GT from samples
    gt_boxes_list = []
    gt_classes_list = []
    
    for sample in samples:
        if sample.boxes_xyxy.shape[0] > 0:
            gt_boxes_list.append(torch.from_numpy(sample.boxes_xyxy).to(device))
        else:
            gt_boxes_list.append(torch.zeros((0, 4), device=device))
        
        if sample.class_ids.shape[0] > 0:
            gt_classes_list.append(torch.from_numpy(sample.class_ids).to(device))
        else:
            gt_classes_list.append(torch.zeros((0,), dtype=torch.int64, device=device))
    
    # Multi-level detection loss using SimOTA
    levels = [
        ('P3', 8),
        ('P4', 16),
        ('P5', 32),
        ('P6', 64),
        ('P7', 128),
    ]
    
    total_cls_loss = torch.tensor(0.0, device=device)
    total_box_loss = torch.tensor(0.0, device=device)
    total_quality_loss = torch.tensor(0.0, device=device)
    num_levels_used = 0
    
    for level_name, stride in levels:
        if level_name not in output.logits_by_level:
            continue
        
        # Extract predictions for this level
        # Shape: [B, C, H, W]
        pred_cls_logits = output.logits_by_level[level_name]
        pred_box_reg = output.boxes_by_level[level_name]
        pred_quality = output.quality_by_level[level_name]
        
        B, C, H, W = pred_cls_logits.shape
        
        # Get anchor centers for this level
        anchor_centers = compute_anchor_centers((H, W), stride, device)  # [H*W, 2]
        
        # Process each image in batch
        for batch_idx in range(B):
            # Reshape predictions: [C, H, W] → [H*W, C]
            cls_logits = pred_cls_logits[batch_idx].permute(1, 2, 0).reshape(-1, C)
            box_reg = pred_box_reg[batch_idx].permute(1, 2, 0).reshape(-1, 4)
            quality = pred_quality[batch_idx].permute(1, 2, 0).reshape(-1)
            
            # Convert box regression to xyxy (from distance format)
            # box_reg: [left, top, right, bottom] distances
            distances = box_reg * stride
            x1 = anchor_centers[:, 0] - distances[:, 0]
            y1 = anchor_centers[:, 1] - distances[:, 1]
            x2 = anchor_centers[:, 0] + distances[:, 2]
            y2 = anchor_centers[:, 1] + distances[:, 3]
            pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            
            # Get GT for this image
            gt_boxes = gt_boxes_list[batch_idx]
            gt_classes = gt_classes_list[batch_idx]
            
            # Skip if no GT
            if len(gt_boxes) == 0:
                continue
            
            # Build SimOTA targets
            targets = build_simota_targets_for_anchors(
                pred_cls_logits=cls_logits,
                pred_boxes_xyxy=pred_boxes_xyxy,
                anchor_centers_xy=anchor_centers,
                gt_boxes_xyxy=gt_boxes,
                gt_classes=gt_classes,
                topk_center=10,
                classification_mode="focal",
                cls_weight=1.0,
                iou_weight=3.0,
                center_weight=1.0,
                dynamic_topk=10,
                min_dynamic_k=1,
                small_object_boost=2.0,  # STAL-inspired boosting
            )
            
            # Compute loss for this level/image
            loss_out = det_loss_with_simota(
                pred_cls_logits=cls_logits,
                pred_boxes_xyxy=pred_boxes_xyxy,
                pred_quality_logits=quality,
                targets=targets,
                cls_loss_type="focal",
                quality_loss_type="qfl",
                cls_weight=cls_weight,
                box_weight=box_weight,
                quality_weight=quality_weight,
                box_loss_type=box_loss_type,
            )
            
            total_cls_loss += loss_out.cls_loss
            total_box_loss += loss_out.box_loss
            total_quality_loss += loss_out.quality_loss
            num_levels_used += 1
    
    # Average losses across levels and batch
    if num_levels_used > 0:
        total_cls_loss = total_cls_loss / num_levels_used
        total_box_loss = total_box_loss / num_levels_used
        total_quality_loss = total_quality_loss / num_levels_used
    
    # Detection loss
    det_loss = total_cls_loss + total_box_loss + total_quality_loss
    
    # Boundary distillation loss
    boundary_loss = output.boundaries.abs().mean()
    boundary_loss = torch.nan_to_num(boundary_loss, nan=0.0)

    # Instance segmentation loss when masks are available from both prediction and GT.
    seg_loss = torch.tensor(0.0, device=device)
    seg_batches = 0
    if output.masks is not None and output.masks.ndim == 4:
        pred_masks = output.masks
        for batch_idx, sample in enumerate(samples[: pred_masks.shape[0]]):
            if sample.masks is None or sample.masks.shape[0] <= 0:
                continue
            pred_batch = pred_masks[batch_idx]
            if pred_batch.ndim != 3 or pred_batch.shape[0] <= 0:
                continue
            gt_masks = torch.from_numpy(sample.masks).to(device=device, dtype=pred_batch.dtype)
            if gt_masks.ndim == 2:
                gt_masks = gt_masks.unsqueeze(0)
            num_instances = min(int(pred_batch.shape[0]), int(gt_masks.shape[0]))
            if num_instances <= 0:
                continue

            pred_used = pred_batch[:num_instances]
            gt_used = gt_masks[:num_instances]
            if gt_used.shape[-2:] != pred_used.shape[-2:]:
                gt_used = torch.nn.functional.interpolate(
                    gt_used.unsqueeze(1),
                    size=pred_used.shape[-2:],
                    mode="nearest",
                ).squeeze(1)

            pred_logits = torch.logit(pred_used.clamp(min=1e-4, max=1.0 - 1e-4))
            seg_out = instance_segmentation_losses(
                mask_logits=pred_logits.unsqueeze(0),
                target_masks=gt_used.unsqueeze(0),
                bce_weight=1.0,
                dice_weight=1.0,
                boundary_weight=1.0,
            )
            seg_loss += seg_out.total_loss
            seg_batches += 1
    if seg_batches > 0:
        seg_loss = seg_loss / seg_batches
    
    # Combine
    total_loss = det_weight * det_loss + boundary_weight * boundary_loss + seg_loss
    
    # Record components
    loss_dict["cls_loss"] = float(total_cls_loss.item())
    loss_dict["box_loss"] = float(total_box_loss.item())
    loss_dict["quality_loss"] = float(total_quality_loss.item())
    loss_dict["det_loss"] = float(det_loss.item())
    loss_dict["boundary_loss"] = float(boundary_loss.item())
    loss_dict["seg_loss"] = float(seg_loss.item())
    loss_dict["total_loss"] = float(total_loss.item())
    loss_dict["num_levels"] = num_levels_used
    
    return total_loss, loss_dict


__all__ = [
    "compute_teacher_training_loss",
]
