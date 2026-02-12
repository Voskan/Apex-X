"""Cascade R-CNN detection head for iterative refinement.

Implements multi-stage cascade detection with increasing IoU thresholds.
Each stage refines the predictions from the previous stage.

Expected gain: +3-5% AP (biggest single improvement!)

Reference:
    Cascade R-CNN: Delving into High Quality Object Detection
    https://arxiv.org/abs/1712.00726
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


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
        initial_boxes: Tensor,
    ) -> dict[str, list[Tensor]]:
        """Forward pass through cascade stages.
        
        Args:
            features: Feature maps [B, C, H, W]
            initial_boxes: Initial box proposals [N, 4] in (x1, y1, x2, y2) format
            
        Returns:
            Dict with cascade outputs:
                - 'boxes': List of refined boxes from each stage
                - 'scores': List of class scores from each stage
                - 'box_deltas': List of box refinements from each stage
        """
        all_boxes = [initial_boxes]
        all_scores = []
        all_deltas = []
        
        current_boxes = initial_boxes
        
        for stage_idx, stage in enumerate(self.stages):
            # RoI Align on current boxes
            roi_features = self._roi_align(
                features,
                current_boxes,
                output_size=(7, 7),
            )
            
            # Forward through stage
            box_deltas, class_logits = stage(roi_features)
            
            # Apply box deltas to refine boxes
            refined_boxes = self._apply_deltas(current_boxes, box_deltas)
            
            # Store outputs
            all_boxes.append(refined_boxes)
            all_scores.append(class_logits)
            all_deltas.append(box_deltas)
            
            # Use refined boxes for next stage
            current_boxes = refined_boxes
        
        return {
            'boxes': all_boxes,
            'scores': all_scores,
            'box_deltas': all_deltas,
        }
    
    def _roi_align(
        self,
        features: Tensor,
        boxes: Tensor,
        output_size: tuple[int, int] = (7, 7),
    ) -> Tensor:
        """RoI Align pooling.
        
        Args:
            features: Feature maps [B, C, H, W]
            boxes: Boxes [N, 4] in (x1, y1, x2, y2) format
            output_size: Output spatial size
            
        Returns:
            RoI features [N, C, H_out, W_out]
        """
        try:
            from torchvision.ops import roi_align
            
            # Add batch index (assume all boxes from batch 0)
            batch_indices = torch.zeros(
                (boxes.shape[0], 1),
                dtype=boxes.dtype,
                device=boxes.device,
            )
            boxes_with_batch = torch.cat([batch_indices, boxes], dim=1)
            
            # RoI Align
            roi_features = roi_align(
                features,
                boxes_with_batch,
                output_size=output_size,
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
            
            return roi_features
            
        except ImportError:
            # Fallback to grid_sample
            return self._roi_align_fallback(features, boxes, output_size)
    
    def _roi_align_fallback(
        self,
        features: Tensor,
        boxes: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        """Fallback RoI pooling using grid_sample."""
        # Simplified fallback - not as accurate as RoI Align
        b, c, h_feat, w_feat = features.shape
        n_boxes = boxes.shape[0]
        h_out, w_out = output_size
        
        roi_features = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            
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
        # Convert to center format
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        # Apply deltas
        dx, dy, dw, dh = deltas.unbind(dim=1)
        
        # Predict new center and size
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)
        
        # Convert back to (x1, y1, x2, y2)
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ], dim=1)
        
        return pred_boxes


__all__ = ['CascadeDetHead', 'CascadeStage']
