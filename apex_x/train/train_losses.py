"""Training loss computation utilities for Apex-X.

This module provides helper functions to compute training losses from model outputs
and ground truth annotations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from apex_x.data import TransformSample
from apex_x.losses.det_loss import (
    det_loss_with_simota,
)
from apex_x.losses.seg_loss import instance_segmentation_losses
from apex_x.model import TeacherDistillOutput, compute_anchor_centers


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
    boundary_warmup_epochs: int = 10,
    boundary_max_scale: float = 10.0,
    box_warmup_epochs: int = 5,
    box_scale_start: float = 0.5,
    box_scale_end: float = 2.0,
    max_det_component: float = 1e4,
    max_boundary_component: float = 1e3,
    max_seg_component: float = 1e3,
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
        boundary_warmup_epochs: Epoch count before boundary ramp starts
        boundary_max_scale: Maximum boundary weight multiplier at end of ramp
        box_warmup_epochs: Epoch count before box-weight ramp starts
        box_scale_start: Initial scale multiplier for box regression term
        box_scale_end: Final scale multiplier for box regression term
        max_det_component: Clamp for detection loss component (stability guard)
        max_boundary_component: Clamp for boundary loss component (stability guard)
        max_seg_component: Clamp for segmentation loss component (stability guard)
    
    Returns:
        tuple of (total_loss, loss_dict) with individual components
    """
    device = output.logits.device
    loss_dict: dict[str, float] = {}
    
    # Adaptive boundary weight: 0.05 → 0.5 during training
    # Early: focus on detection, Late: refine boundaries
    if (
        adaptive_boundary
        and epoch is not None
        and total_epochs is not None
        and total_epochs > boundary_warmup_epochs
    ):
        warmup = max(0, int(boundary_warmup_epochs))
        progress = max(0.0, min(1.0, (epoch - warmup) / max(total_epochs - warmup, 1)))
        max_scale = max(1.0, float(boundary_max_scale))
        boundary_scale = 1.0 + (max_scale - 1.0) * progress
        boundary_weight = boundary_weight * boundary_scale
        loss_dict['boundary_scale_adaptive'] = boundary_scale
        loss_dict['boundary_weight_adaptive'] = boundary_weight
    
    if samples is None or len(samples) == 0:
        # Fallback: dummy loss for synthetic data
        det_loss = output.logits.pow(2).mean()
        det_loss = torch.nan_to_num(
            det_loss,
            nan=0.0,
            posinf=max_det_component,
            neginf=0.0,
        ).clamp(min=0.0, max=max_det_component)
        boundary_loss = output.boundaries.mean()
        boundary_loss = torch.nan_to_num(
            boundary_loss,
            nan=0.0,
            posinf=max_boundary_component,
            neginf=0.0,
        ).clamp(min=0.0, max=max_boundary_component)
        total_loss = det_loss + boundary_weight * boundary_loss
        total_loss = torch.nan_to_num(
            total_loss,
            nan=0.0,
            posinf=max_det_component,
            neginf=0.0,
        )
        
        loss_dict["det_loss"] = float(det_loss.item())
        loss_dict["boundary_loss"] = float(boundary_loss.item())
        loss_dict["total_loss"] = float(total_loss.item())
        loss_dict["num_assignment_samples"] = 0.0
        loss_dict["assignment_num_anchors"] = 0.0
        loss_dict["assignment_num_foreground"] = 0.0
        loss_dict["assignment_foreground_ratio"] = 0.0
        
        return total_loss, loss_dict
    
    # Progressive loss balancing (YOLO26-style)
    if epoch is not None and total_epochs is not None and total_epochs > box_warmup_epochs:
        warmup = max(0, int(box_warmup_epochs))
        progress = max(0.0, min(1.0, (epoch - warmup) / max(total_epochs - warmup, 1)))
        box_start = float(max(box_scale_start, 1e-6))
        box_end = float(max(box_scale_end, box_start))
        # Early: focus classification, Late: refine boxes
        cls_weight = cls_weight * 1.0
        box_weight = box_weight * (box_start + (box_end - box_start) * progress)
        loss_dict["box_weight_adaptive"] = box_weight
    
    # Extract GT from samples
    gt_boxes_list = []
    gt_classes_list = []
    gt_masks_list = []
    
    if isinstance(samples, dict):
        # Handle unified batch dict
        batch_size = output.logits.shape[0]
        boxes = samples.get("boxes")
        labels = samples.get("labels")
        masks = samples.get("masks")
        batch_idx_tensor = samples.get("batch_idx")
        
        for i in range(batch_size):
            if batch_idx_tensor is not None and boxes is not None:
                mask_i = (batch_idx_tensor == i)
                gt_boxes_list.append(boxes[mask_i].to(device))
                gt_classes_list.append(labels[mask_i].to(device))
                if masks is not None:
                    gt_masks_list.append(masks[mask_i].to(device))
            else:
                gt_boxes_list.append(torch.zeros((0, 4), device=device))
                gt_classes_list.append(torch.zeros((0,), dtype=torch.int64, device=device))
    elif samples is not None:
        # Handle list of TransformSample
        for sample in samples:
            if sample.boxes_xyxy.shape[0] > 0:
                gt_boxes_list.append(torch.from_numpy(sample.boxes_xyxy).to(device))
            else:
                gt_boxes_list.append(torch.zeros((0, 4), device=device))
            
            if sample.class_ids.shape[0] > 0:
                gt_classes_list.append(torch.from_numpy(sample.class_ids).to(device))
            else:
                gt_classes_list.append(torch.zeros((0,), dtype=torch.int64, device=device))
            
            if sample.masks is not None and sample.masks.shape[0] > 0:
                gt_masks_list.append(torch.from_numpy(sample.masks).to(device))
            else:
                gt_masks_list.append(None)
    
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
    assignment_num_anchors = 0.0
    assignment_num_foreground = 0.0
    assignment_samples = 0
    
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
            
            # Compute loss for this level/image (includes target assignment)
            loss_out = det_loss_with_simota(
                pred_cls_logits=cls_logits,
                pred_boxes_xyxy=pred_boxes_xyxy,
                pred_quality_logits=quality,
                anchor_centers_xy=anchor_centers,
                gt_boxes_xyxy=gt_boxes,
                gt_classes=gt_classes,
                topk_center=10,
                classification_mode="focal",
                dynamic_topk=10,
                min_dynamic_k=1,
                small_object_boost=2.0,  # STAL-inspired boosting
                cls_loss_type="focal",
                quality_loss_type="qfl",
                cls_loss_weight=cls_weight,
                box_loss_weight=box_weight,
                quality_loss_weight=quality_weight,
                box_loss_type=box_loss_type,
            )
            
            total_cls_loss += loss_out.cls_loss
            total_box_loss += loss_out.box_loss
            total_quality_loss += loss_out.quality_loss
            num_levels_used += 1
            assignment_num_anchors += float(loss_out.assignment_stats.get("num_anchors", 0.0))
            assignment_num_foreground += float(loss_out.assignment_stats.get("num_foreground", 0.0))
            assignment_samples += 1
    
    # Average losses across levels and batch
    if num_levels_used > 0:
        total_cls_loss = total_cls_loss / num_levels_used
        total_box_loss = total_box_loss / num_levels_used
        total_quality_loss = total_quality_loss / num_levels_used
    
    # Detection loss
    det_loss = total_cls_loss + total_box_loss + total_quality_loss
    det_loss = torch.nan_to_num(det_loss, nan=0.0, posinf=max_det_component, neginf=0.0).clamp(
        min=0.0,
        max=max_det_component,
    )
    
    # Boundary distillation loss
    boundary_loss = output.boundaries.abs().mean()
    boundary_loss = torch.nan_to_num(
        boundary_loss,
        nan=0.0,
        posinf=max_boundary_component,
        neginf=0.0,
    ).clamp(min=0.0, max=max_boundary_component)

    # Instance segmentation loss when masks are available from both prediction and GT.
    seg_loss = torch.tensor(0.0, device=device)
    seg_batches = 0
    if output.masks is not None and output.masks.ndim == 4:
        pred_masks = output.masks
        for batch_idx in range(min(pred_masks.shape[0], len(gt_masks_list))):
            gt_masks = gt_masks_list[batch_idx]
            if gt_masks is None or gt_masks.shape[0] <= 0:
                continue
            pred_batch = pred_masks[batch_idx]
            if pred_batch.ndim != 3 or pred_batch.shape[0] <= 0:
                continue
            # (gt_masks is already on device and correct dtype from extraction)
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
    seg_loss = torch.nan_to_num(seg_loss, nan=0.0, posinf=max_seg_component, neginf=0.0).clamp(
        min=0.0,
        max=max_seg_component,
    )
    
    # Combine
    total_loss = det_weight * det_loss + boundary_weight * boundary_loss + seg_loss
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=max_det_component, neginf=0.0)
    
    # Record components
    loss_dict["cls_loss"] = float(total_cls_loss.item())
    loss_dict["box_loss"] = float(total_box_loss.item())
    loss_dict["quality_loss"] = float(total_quality_loss.item())
    loss_dict["det_loss"] = float(det_loss.item())
    loss_dict["boundary_loss"] = float(boundary_loss.item())
    loss_dict["seg_loss"] = float(seg_loss.item())
    loss_dict["total_loss"] = float(total_loss.item())
    loss_dict["num_levels"] = num_levels_used
    loss_dict["num_assignment_samples"] = float(assignment_samples)
    loss_dict["assignment_num_anchors"] = float(assignment_num_anchors)
    loss_dict["assignment_num_foreground"] = float(assignment_num_foreground)
    if assignment_num_anchors > 0.0:
        loss_dict["assignment_foreground_ratio"] = float(
            assignment_num_foreground / assignment_num_anchors
        )
    else:
        loss_dict["assignment_foreground_ratio"] = 0.0

    return total_loss, loss_dict


__all__ = [
    "compute_teacher_training_loss",
]
