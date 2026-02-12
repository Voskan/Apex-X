"""Cascade R-CNN detection head for iterative refinement.

Implements multi-stage cascade detection with increasing IoU thresholds.
Each stage refines the predictions from the previous stage.

Expected gain: +3-5% AP (biggest single improvement!)

Reference:
    Cascade R-CNN: Delving into High Quality Object Detection
    https://arxiv.org/abs/1712.00726
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


class CascadeStage(nn.Module):
    """Single stage of cascade detection.
    
    Args:
        in_channels: Input feature channels
        num_classes: Number of object classes
        iou_threshold: IoU threshold for this stage
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        iou_threshold: float = 0.5,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.hidden_dim = hidden_dim
        
        # Box regression head (refines previous boxes)
        self.box_head = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),  # (dx, dy, dw, dh)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_features: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through cascade stage.
        
        Args:
            roi_features: RoI-pooled features [N, C, 7, 7]
            
        Returns:
            Tuple of (box_deltas, class_logits)
                - box_deltas: [N, 4] box refinements
                - class_logits: [N, num_classes] classification scores
        """
        # Flatten features
        x = roi_features.flatten(1)  # [N, C*7*7]
        
        # Box regression
        box_deltas = self.box_head(x)  # [N, 4]
        
        # Classification
        class_logits = self.cls_head(x)  # [N, num_classes]
        
        return box_deltas, class_logits


class CascadeDetHead(nn.Module):
    """Cascade R-CNN detection head with 3 stages.
    
    Stage 1: IoU = 0.5 (easy positives)
    Stage 2: IoU = 0.6 (medium quality)
    Stage 3: IoU = 0.7 (high quality)
    
    Each stage refines the boxes from previous stage.
    
    Expected AP gain: +3-5% over single-stage detector
    
    Args:
        in_channels: Input feature channels
        num_classes: Number of object classes
        num_stages: Number of cascade stages (default: 3)
        iou_thresholds: IoU thresholds for each stage
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_stages: int = 3,
        iou_thresholds: list[float] | None = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = num_stages
        # Clamp (Faster R-CNN convention): log(1000 / 16) ~= 4.135.
        self._bbox_scale_clip = float(math.log(1000.0 / 16.0))
        self._center_shift_clip = 10.0
        self._min_box_size = 1e-6
        self._max_box_size = 1e6
        
        # Default IoU thresholds: [0.5, 0.6, 0.7]
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.6, 0.7]
        
        if len(iou_thresholds) != num_stages:
            raise ValueError(f"iou_thresholds length must equal num_stages ({num_stages})")
        
        self.iou_thresholds = iou_thresholds
        
        # Create cascade stages
        self.stages = nn.ModuleList([
            CascadeStage(
                in_channels=in_channels,
                num_classes=num_classes,
                iou_threshold=iou_thresh,
            )
            for iou_thresh in iou_thresholds
        ])
    
    def forward(
        self,
        features: Tensor,
        initial_boxes: list[Tensor] | Tensor,
    ) -> dict[str, list[list[Tensor]] | list[Tensor]]:
        """Forward pass through cascade stages.
        
        Args:
            features: Feature maps [B, C, H, W]
            initial_boxes: List of boxes per batch element [B, N_i, 4]
            
        Returns:
            Dict with cascade outputs:
                - 'boxes': list[list[Tensor]] Refined boxes for each stage for each batch element
                - 'scores': list[Tensor] Class logits [B*N, num_classes] for each stage
                - 'box_deltas': list[Tensor] Box deltas [B*N, 4] for each stage
        """
        if isinstance(initial_boxes, Tensor):
            if initial_boxes.ndim != 2 or initial_boxes.shape[1] != 4:
                raise ValueError("initial_boxes tensor must be [N, 4]")
            current_boxes: list[Tensor] = [initial_boxes]
        else:
            current_boxes = list(initial_boxes)
        all_stage_boxes: list[list[Tensor]] = [current_boxes]
        all_scores: list[Tensor] = []
        all_deltas: list[Tensor] = []
        
        for _stage_idx, stage in enumerate(self.stages):
            # 1. Flatten boxes for efficient batch ROI Align
            flat_boxes, box_counts = self.flatten_boxes_for_roi(current_boxes, features.device)
            
            # 2. RoI Align
            roi_features = self._roi_align(
                features,
                flat_boxes,
                output_size=(7, 7),
            )
            
            # 3. Forward through stage
            box_deltas, class_logits = stage(roi_features)
            
            # 4. Apply box deltas and unflatten
            refined_flat_boxes = self._apply_deltas(flat_boxes[:, 1:], box_deltas)
            
            refined_boxes_list = []
            start = 0
            for count in box_counts:
                refined_boxes_list.append(refined_flat_boxes[start : start + count])
                start += count
            
            # Store outputs
            all_stage_boxes.append(refined_boxes_list)
            all_scores.append(class_logits)
            all_deltas.append(box_deltas)
            
            # Use refined boxes for next stage
            current_boxes = refined_boxes_list
        
        return {
            'boxes': all_stage_boxes,
            'scores': all_scores,
            'box_deltas': all_deltas,
        }

    def flatten_boxes_for_roi(
        self,
        boxes_list: list[Tensor],
        device: torch.device,
    ) -> tuple[Tensor, list[int]]:
        """Helpers to flatten list of boxes into [N_total, 5] (batch_idx, x1, y1, x2, y2)."""
        flat_boxes = []
        counts = []
        for i, boxes in enumerate(boxes_list):
            if boxes.numel() == 0:
                counts.append(0)
                continue
            batch_idx = torch.full((boxes.shape[0], 1), i, dtype=boxes.dtype, device=device)
            flat_boxes.append(torch.cat([batch_idx, boxes], dim=1))
            counts.append(boxes.shape[0])
        
        if not flat_boxes:
            # Empty fallback
            return torch.zeros((0, 5), dtype=torch.float32, device=device), counts
            
        return torch.cat(flat_boxes, dim=0), counts

    def _roi_align(
        self,
        features: Tensor,
        boxes_with_batch: Tensor,
        output_size: tuple[int, int] = (7, 7),
    ) -> Tensor:
        """RoI Align pooling.
        
        Args:
            features: Feature maps [B, C, H, W]
            boxes_with_batch: Boxes [N, 5] (batch_idx, x1, y1, x2, y2)
            output_size: Output spatial size
            
        Returns:
            RoI features [N, C, H_out, W_out]
        """
        try:
            from torchvision.ops import roi_align
            return roi_align(
                features,
                boxes_with_batch,
                output_size=output_size,
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
        except ImportError:
            # Fallback (simplified)
            LOGGER.warning("torchvision.ops.roi_align not found, using zero fallback")
            B, C, _, _ = features.shape
            N = boxes_with_batch.shape[0]
            return torch.zeros((N, C, *output_size), device=features.device, dtype=features.dtype)
    
    def _roi_align_fallback(
        self,
        features: Tensor,
        boxes: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        """Fallback RoI pooling using grid_sample."""
        # Simplified fallback - not as accurate as RoI Align
        b, c, h_feat, w_feat = features.shape
        h_out, w_out = output_size
        
        roi_features = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Normalize to [-1, 1] for grid_sample
            grid_x = torch.linspace(-1, 1, w_out, device=features.device)
            grid_y = torch.linspace(-1, 1, h_out, device=features.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
            
            # Sample features
            sampled = F.grid_sample(
                features,
                grid,
                mode='bilinear',
                align_corners=False,
            )  # [B, C, H_out, W_out]
            
            roi_features.append(sampled[0])  # [C, H_out, W_out]
        
        roi_features = torch.stack(roi_features, dim=0)  # [N, C, H_out, W_out]
        return roi_features
    
    def _apply_deltas(
        self,
        boxes: Tensor,
        deltas: Tensor,
    ) -> Tensor:
        """Apply box deltas to boxes.
        
        Args:
            boxes: Input boxes [N, 4] in (x1, y1, x2, y2) format
            deltas: Box deltas [N, 4] in (dx, dy, dw, dh) format
            
        Returns:
            Refined boxes [N, 4]
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))

        # Protect downstream ops from non-finite values.
        boxes = torch.nan_to_num(
            boxes,
            nan=0.0,
            posinf=self._max_box_size,
            neginf=-self._max_box_size,
        )
        deltas = torch.nan_to_num(
            deltas,
            nan=0.0,
            posinf=self._center_shift_clip,
            neginf=-self._center_shift_clip,
        )

        # Convert to center format
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=self._min_box_size)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=self._min_box_size)
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        # Apply deltas
        dx, dy, dw, dh = deltas.unbind(dim=1)
        dx = dx.clamp(min=-self._center_shift_clip, max=self._center_shift_clip)
        dy = dy.clamp(min=-self._center_shift_clip, max=self._center_shift_clip)
        dw = dw.clamp(min=-self._bbox_scale_clip, max=self._bbox_scale_clip)
        dh = dh.clamp(min=-self._bbox_scale_clip, max=self._bbox_scale_clip)
        
        # Predict new center and size
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = (widths * torch.exp(dw)).clamp(min=self._min_box_size, max=self._max_box_size)
        pred_h = (heights * torch.exp(dh)).clamp(min=self._min_box_size, max=self._max_box_size)
        
        # Convert back to (x1, y1, x2, y2)
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ], dim=1)
        pred_boxes = torch.nan_to_num(
            pred_boxes,
            nan=0.0,
            posinf=self._max_box_size,
            neginf=-self._max_box_size,
        )

        # Guarantee valid coordinate ordering.
        x1 = torch.minimum(pred_boxes[:, 0], pred_boxes[:, 2])
        y1 = torch.minimum(pred_boxes[:, 1], pred_boxes[:, 3])
        x2 = torch.maximum(pred_boxes[:, 0], pred_boxes[:, 2])
        y2 = torch.maximum(pred_boxes[:, 1], pred_boxes[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=1)


__all__ = ['CascadeDetHead', 'CascadeStage']
