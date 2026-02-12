"""Detection post-processing utilities for Apex-X.

Converts raw model outputs (logits, box regressions, quality scores) to final
detections using NMS and confidence thresholding.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torchvision.ops

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


def compute_anchor_centers(
    feature_size: tuple[int, int],
    stride: int,
    device: torch.device,
) -> Tensor:
    """Generate anchor center coordinates for a feature map level.
    
    Args:
        feature_size: (height, width) of feature map
        stride: Stride of this pyramid level (8, 16, 32, 64, 128 for P3-P7)
        device: Device to create tensors on
    
    Returns:
        Tensor of shape [H*W, 2] with (x, y) center coordinates in image space
    
    Example:
        >>> centers = compute_anchor_centers((80, 80), stride=8, device='cuda')
        >>> centers.shape
        torch.Size([6400, 2])
        >>> centers[0]  # First anchor at (4, 4) - center of first 8x8 cell
        tensor([4., 4.])
    """
    h, w = feature_size
    
    # Create coordinate grids
    shift_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=device)
    shift_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=device)
    
    # Create meshgrid and stack to [H, W, 2]
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    
    # Add half-stride to get cell centers
    centers = torch.stack(
        [shift_x + stride // 2, shift_y + stride // 2],
        dim=-1
    )
    
    # Reshape to [H*W, 2]
    centers = centers.reshape(-1, 2)
    
    return centers


def decode_boxes_distance(
    box_reg: Tensor,
    anchor_centers: Tensor,
    stride: int,
) -> Tensor:
    """Decode bounding boxes from distance-based regression.
    
    Box regression format: [left, top, right, bottom] distances from anchor center
    
    Args:
        box_reg: [N, 4] regression outputs (distances)
        anchor_centers: [N, 2] anchor (x, y) centers
        stride: Feature stride
    
    Returns:
        [N, 4] boxes in xyxy format
    """
    # box_reg: [left, top, right, bottom] distances
    # Scale by stride
    distances = box_reg * stride
    
    x1 = anchor_centers[:, 0] - distances[:, 0]
    y1 = anchor_centers[:, 1] - distances[:, 1]
    x2 = anchor_centers[:, 0] + distances[:, 2]
    y2 = anchor_centers[:, 1] + distances[:, 3]
    
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes_xyxy


def decode_boxes_direct(
    box_reg: Tensor,
    anchor_centers: Tensor,
    stride: int,
) -> Tensor:
    """Decode bounding boxes from direct xyxy regression.
    
    Box regression format: [x1, y1, x2, y2] normalized to feature space
    
    Args:
        box_reg: [N, 4] regression outputs (normalized xyxy)
        anchor_centers: [N, 2] anchor (x, y) centers (not used for direct)
        stride: Feature stride for scaling
    
    Returns:
        [N, 4] boxes in xyxy format (image space)
    """
    # Simply scale to image coordinates
    boxes_xyxy = box_reg * stride
    return boxes_xyxy


def post_process_detections(
    cls_logits_by_level: dict[str, Tensor],  # [B, C, H, W]
    box_reg_by_level: dict[str, Tensor],      # [B, 4, H, W]
    quality_by_level: dict[str, Tensor],      # [B, 1, H, W]
    *,
    conf_threshold: float = 0.001,
    nms_threshold: float = 0.65,
    max_detections: int = 300,
    box_format: str = "distance",  # "distance" or "direct"
) -> list[dict[str, Tensor]]:
    """Convert model outputs to final detections with NMS.
    
    Args:
        cls_logits_by_level: Classification logits per pyramid level
        box_reg_by_level: Box regression outputs per level
        quality_by_level: Quality/objectness scores per level
        conf_threshold: Minimum confidence score to keep detection
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections to return per image
        box_format: How boxes are encoded - "distance" or "direct"
    
    Returns:
        List of dicts (one per image in batch) with:
            - boxes: [N, 4] tensor in xyxy format
            - scores: [N] confidence scores
            - classes: [N] predicted class indices
    """
    # Level definitions (name → stride)
    levels = [
        ('P3', 8),
        ('P4', 16),
        ('P5', 32),
        ('P6', 64),
        ('P7', 128),
    ]
    
    # Get batch size and device
    first_level = next(iter(cls_logits_by_level.values()))
    batch_size = first_level.shape[0]
    device = first_level.device
    
    all_detections = []
    
    for batch_idx in range(batch_size):
        boxes_list = []
        scores_list = []
        class_ids_list = []
        
        for level_name, stride in levels:
            if level_name not in cls_logits_by_level:
                continue
            
            # Extract batch item: [C, H, W], [4, H, W], [1, H, W]
            cls_logits = cls_logits_by_level[level_name][batch_idx]
            box_reg = box_reg_by_level[level_name][batch_idx]
            quality = quality_by_level[level_name][batch_idx]
            
            C, H, W = cls_logits.shape
            
            # Reshape to [H*W, C], [H*W, 4], [H*W]
            cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
            box_reg = box_reg.permute(1, 2, 0).reshape(-1, 4)        # [H*W, 4]
            quality = quality.permute(1, 2, 0).reshape(-1)           # [H*W]
            
            # Compute anchor centers
            anchor_centers = compute_anchor_centers((H, W), stride, device)
            
            # Decode boxes
            if box_format == "distance":
                boxes_xyxy = decode_boxes_distance(box_reg, anchor_centers, stride)
            else:  # "direct"
                boxes_xyxy = decode_boxes_direct(box_reg, anchor_centers, stride)
            
            # Compute scores: cls_prob * quality (task-aligned scoring)
            cls_probs = torch.sigmoid(cls_logits)  # [H*W, C]
            quality_scores = torch.sigmoid(quality).unsqueeze(1)  # [H*W, 1]
            scores = cls_probs * quality_scores  # [H*W, C]
            
            # Get max score and corresponding class per anchor
            max_scores, class_ids = scores.max(dim=1)  # [H*W], [H*W]
            
            # Threshold by confidence
            keep = max_scores > conf_threshold
            
            if keep.sum() > 0:
                boxes_list.append(boxes_xyxy[keep])
                scores_list.append(max_scores[keep])
                class_ids_list.append(class_ids[keep])
        
        # Concatenate all levels
        if len(boxes_list) == 0:
            # No detections for this image
            all_detections.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'classes': torch.zeros((0,), dtype=torch.int64, device=device),
            })
            continue
        
        boxes = torch.cat(boxes_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        class_ids = torch.cat(class_ids_list, dim=0)
        
        # Apply NMS (class-agnostic)
        keep_indices = torchvision.ops.nms(
            boxes,
            scores,
            iou_threshold=nms_threshold,
        )
        
        # Keep top-K detections
        keep_indices = keep_indices[:max_detections]
        
        all_detections.append({
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'classes': class_ids[keep_indices],
        })
    
    return all_detections


def post_process_detections_per_class(
    cls_logits_by_level: dict[str, Tensor],
    box_reg_by_level: dict[str, Tensor],
    quality_by_level: dict[str, Tensor],
    *,
    conf_threshold: float = 0.001,
    nms_threshold: float = 0.65,
    max_detections: int = 300,
    box_format: str = "distance",
) -> list[dict[str, Tensor]]:
    """Post-process with per-class NMS (slower but more accurate).
    
    Same as post_process_detections but applies NMS separately per class.
    """
    # Similar implementation to above but with per-class NMS
    # Using torchvision.ops.batched_nms with class indices
    
    levels = [
        ('P3', 8),
        ('P4', 16),
        ('P5', 32),
        ('P6', 64),
        ('P7', 128),
    ]
    
    first_level = next(iter(cls_logits_by_level.values()))
    batch_size = first_level.shape[0]
    device = first_level.device
    
    all_detections = []
    
    for batch_idx in range(batch_size):
        boxes_list = []
        scores_list = []
        class_ids_list = []
        
        for level_name, stride in levels:
            if level_name not in cls_logits_by_level:
                continue
            
            cls_logits = cls_logits_by_level[level_name][batch_idx]
            box_reg = box_reg_by_level[level_name][batch_idx]
            quality = quality_by_level[level_name][batch_idx]
            
            C, H, W = cls_logits.shape
            cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, C)
            box_reg = box_reg.permute(1, 2, 0).reshape(-1, 4)
            quality = quality.permute(1, 2, 0).reshape(-1)
            
            anchor_centers = compute_anchor_centers((H, W), stride, device)
            
            if box_format == "distance":
                boxes_xyxy = decode_boxes_distance(box_reg, anchor_centers, stride)
            else:
                boxes_xyxy = decode_boxes_direct(box_reg, anchor_centers, stride)
            
            cls_probs = torch.sigmoid(cls_logits)
            quality_scores = torch.sigmoid(quality).unsqueeze(1)
            scores = cls_probs * quality_scores
            
            max_scores, class_ids = scores.max(dim=1)
            keep = max_scores > conf_threshold
            
            if keep.sum() > 0:
                boxes_list.append(boxes_xyxy[keep])
                scores_list.append(max_scores[keep])
                class_ids_list.append(class_ids[keep])
        
        if len(boxes_list) == 0:
            all_detections.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'classes': torch.zeros((0,), dtype=torch.int64, device=device),
            })
            continue
        
        boxes = torch.cat(boxes_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        class_ids = torch.cat(class_ids_list, dim=0)
        
        # Per-class NMS using batched_nms
        keep_indices = torchvision.ops.batched_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold=nms_threshold,
        )
        
        keep_indices = keep_indices[:max_detections]
        
        all_detections.append({
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'classes': class_ids[keep_indices],
        })
    
    return all_detections


__all__ = [
    "compute_anchor_centers",
    "decode_boxes_distance",
    "decode_boxes_direct",
    "post_process_detections",
    "post_process_detections_per_class",
]
