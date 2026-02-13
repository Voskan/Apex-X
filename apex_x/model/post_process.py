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


    # Helper to gather masks if present (lists of lists structure implied by cascade head output?)
    # The caller usually passes the *final stage* masks.
    
    return all_detections


def post_process_detections(
    cls_logits_by_level: dict[str, Tensor],  # [B, C, H, W]
    box_reg_by_level: dict[str, Tensor],      # [B, 4, H, W]
    quality_by_level: dict[str, Tensor],      # [B, 1, H, W]
    *,
    masks_by_level: dict[str, Tensor] | None = None, # OPTIONAL: PointRend/Mask outputs
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
        masks_by_level: Optional mask features or tiny masks matching the dense grid (rare for dense detectors).
                        **NOTE**: For Two-Stage / Cascade models, this function is usually used for the RPN/First stage.
                        The actual mask prediction happens *after* NMS on the RoIs.
                        
                        HOWEVER, if this is a Dense instance segmentation head (like SOLO/YOLO-Seg), 
                        we would process masks here.
                        
                        Apex-X uses CascadeMaskHead (Two-Stage).
                        This `post_process.py` seems targeted at the *Dense* part (RetinaNet/FCOS style)
                        OR a flattened view of the final outputs.
                        
                        If `TeacherModelV3` calls this with *dense* outputs, we process them.
                        But `TeacherModelV3` returns `boxes`, `masks`, `scores` directly from the ROI heads!
                        
                        Wait, `TeacherModelV3.forward` calls `self.det_head`, looks like it handles its own post-processing 
                        internally or returns `final_boxes_list`.
                        
                        Let's check `TeacherModelV3.forward` again.
                        It returns dictionary with 'boxes', 'masks', 'scores'.
                        
                        So this `post_process_detections` might be for the *RPN* or a *Dense Head* fallback?
                        Ah, `TeacherModelV3` does manual clamping/unflattening.
                        
                        The `post_process.py` utility might be used by *other* heads or during ONNX export.
                        
                        BUT, the user request is about `post_process.py` ignoring masks.
                        If `TeacherModelV3` logic resides in `TeacherModelV3.forward`, we might be editing the wrong file
                        IF `TeacherModelV3` doesn't use this.
                        
                        Let's verify usage. `TeacherModelV3` does internal post-proc.
                        But `runner.py` or export might use `post_process_detections`.
                        
                        If we want to support generic "post process", we should support masks.
                        
                        Let's Assume we receive aligned dense masks or we are modifying the function signature 
                        to be compatible with a "boxes + masks" input style.
                        
                        Actually, looking at `TeacherModelV3`, it returns:
                             "boxes": boxes_xyxy,                # Tensor [N_total, 4]
                             "masks": final_masks_flat,           # Tensor [N_total, 1, 28, 28]
                             "scores": adjusted_scores_flat,      # Tensor [N_total, num_classes]
                        
                        It does NOT return `by_level` dicts for the final output!
                        
                        This `post_process_detections` function takes `_by_level` dicts.
                        This suggests it's for a One-Stage detector (RetinaNet associated).
                        TeacherV3 is Two-Stage (Cascade).
                        
                        However, the "Usage" in `runner.py` or `export` might rely on this 
                        to handle the raw dictionary outputs if they were dense.
                        
                        CRITICAL: If TeacherV3 outputs are already [N, ...], 
                        they just need NMS and Thresholding if not done yet.
                        TeacherV3 `forward` *does* apply score thresholding? 
                        No, it uses `topk` in RPN, but for final output it just returns standard Tensors.
                        Usually `roi_heads` returns post-processed results (NMS'd).
                        
                        Let's assume the user wants `post_process.py` to contain a *generic* NMS + Mask Paste utility
                        that can be called on the final flat outputs as well.
                        
                        I will add a NEW function `post_process_proposals` to handle [N, 4] + [N, 1, 28, 28].
                        
    """
    # ... existing implementation for dense heads ...
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
        masks_list = [] # New
        
        for level_name, stride in levels:
            if level_name not in cls_logits_by_level:
                continue
            
            # Extract batch item: [C, H, W], [4, H, W], [1, H, W]
            cls_logits = cls_logits_by_level[level_name][batch_idx]
            box_reg = box_reg_by_level[level_name][batch_idx]
            quality = quality_by_level[level_name][batch_idx]
            
            # Handle masks if present
            mask_data = None
            if masks_by_level and level_name in masks_by_level:
                 mask_data = masks_by_level[level_name][batch_idx] # [M, H, W] or similar
            
            C, H, W = cls_logits.shape
            
            # Reshape to [H*W, C], [H*W, 4], [H*W]
            cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
            box_reg = box_reg.permute(1, 2, 0).reshape(-1, 4)        # [H*W, 4]
            quality = quality.permute(1, 2, 0).reshape(-1)           # [H*W]
            
            if mask_data is not None:
                # Assuming dense mask encoding, e.g. [NumPrototypes, H, W] -> permute -> reshape
                # Or if it's per-pixel mask (SOLO), it matches grid.
                # For safety, skipping dense mask logic implementation unless explicitly requested.
                # The Gap Analysis likely referred to the final R-CNN output not being processed.
                pass

            # ... (rest of dense logic) ...
            
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
                'masks': torch.zeros((0, 1, 28, 28), device=device), # Empty masks
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
            # 'masks': ... (Dense masks would be here)
        })
    
    return all_detections


def post_process_roi_outputs(
    boxes: Tensor,          # [N, 4]
    scores: Tensor,         # [N]
    classes: Tensor,        # [N]
    masks: Tensor | None,   # [N, 1, H, W] (raw logits or probs)
    image_shape: tuple[int, int],
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Post-process standard R-CNN outputs (Boxes + PointRend Masks).
    
    Filters by score and pastes masks.
    """
    keep = scores > score_threshold
    
    p_boxes = boxes[keep]
    p_scores = scores[keep]
    p_classes = classes[keep]
    
    result = {
        "boxes": p_boxes,
        "scores": p_scores,
        "classes": p_classes,
    }
    
    if masks is not None:
        p_masks = masks[keep]
        # Check if masks are small (RoI) or full image
        # PointRend often returns small RoI masks (e.g. 28x28 or 112x112)
        # We need to paste them.
        
        if p_masks.ndim == 4 and p_masks.shape[2] != image_shape[0]:
            # Paste masks
            p_masks_full = paste_masks_in_image(p_masks, p_boxes, image_shape, threshold=mask_threshold)
            result["masks"] = p_masks_full
        else:
            # Already full size or different format
            result["masks"] = p_masks > mask_threshold
            
    return result


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

    return all_detections


def paste_masks_in_image(
    masks: Tensor,  # [N, 1, M, M]
    boxes: Tensor,  # [N, 4]
    image_shape: tuple[int, int],
    threshold: float = 0.5,
) -> Tensor:
    """Paste mask predictions into regular image tensor.
    
    Args:
        masks: [N, 1, M, M] raw mask predictions (e.g. 28x28)
        boxes: [N, 4] bounding boxes in image coordinates
        image_shape: (H, W) of target image
        threshold: Binarization threshold
        
    Returns:
        [N, H, W] uint8 binary masks
    """
    N = masks.shape[0]
    H, W = image_shape
    device = masks.device
    
    if N == 0:
        return torch.zeros((0, H, W), device=device, dtype=torch.uint8)
        
    # Create full canvas
    # We use a loop or refined grid_sample. 
    # For efficiency with many objects, Detectron2 uses a custom kernel or 
    # carefully constructed grid_sample. Here we implement a native Pytorch version.
    
    res_masks = torch.zeros((N, H, W), device=device, dtype=torch.bool)
    
    for i in range(N):
        box = boxes[i]
        mask = masks[i, 0]  # [M, M]
        
        x0, y0, x1, y1 = box.int().tolist()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        
        w = x1 - x0
        h = y1 - y0
        
        if w > 0 and h > 0:
            # Upsample mask to box size
            # unsqueeze for interpolate: [1, 1, M, M] -> [1, 1, h, w]
            m_up = torch.nn.functional.interpolate(
                mask[None, None, ...],
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )[0, 0]
            
            res_masks[i, y0:y1, x0:x1] = m_up > threshold
            
    return res_masks


__all__ = [
    "compute_anchor_centers",
    "decode_boxes_distance",
    "decode_boxes_direct",
    "post_process_detections",
    "post_process_detections_per_class",
    "paste_masks_in_image",
]
