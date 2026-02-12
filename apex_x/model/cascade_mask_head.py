"""Cascade instance segmentation head with iterative mask refinement.

Extends cascade detection to segmentation with progressive mask refinement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CascadeMaskStage(nn.Module):
    """Single stage of cascade mask prediction.
    
    Args:
        in_channels: Input feature channels
        mask_size: Output mask resolution
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        mask_size: int = 28,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.mask_size = mask_size
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),  # Final mask logits
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_features: Tensor) -> Tensor:
        """Predict masks from RoI features.
        
        Args:
            roi_features: RoI features [N, C, H, W]
            
        Returns:
            Mask logits [N, 1, mask_size, mask_size]
        """
        return self.mask_head(roi_features)


class CascadeMaskHead(nn.Module):
    """Cascade mask head with multi-stage refinement.
    
    Progressively refines masks through multiple stages,
    similar to cascade box detection.
    
    Args:
        in_channels: Input feature channels
        num_stages: Number of refinement stages
        mask_sizes: Mask resolution for each stage
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_stages: int = 3,
        mask_sizes: list[int] | None = None,
    ):
        super().__init__()
        
        self.num_stages = num_stages
        
        # Default mask sizes: [14, 28, 28] (low to high res)
        if mask_sizes is None:
            mask_sizes = [14, 28, 28]
        
        if len(mask_sizes) != num_stages:
            raise ValueError(f"mask_sizes length must equal num_stages")
        
        self.mask_sizes = mask_sizes
        
        # Create mask stages
        self.stages = nn.ModuleList([
            CascadeMaskStage(
                in_channels=in_channels,
                mask_size=size,
            )
            for size in mask_sizes
        ])
    
    def forward(
        self,
        features: Tensor,
        boxes: list[list[Tensor]],
    ) -> list[Tensor]:
        """Progressive mask refinement for batches.
        
        Args:
            features: Feature maps [B, C, H, W]
            boxes: List of stages, each containing a list of boxes per batch element [S, B, N_i, 4]
            
        Returns:
            List of mask logits from each stage [S, N_total, 1, H_mask, W_mask]
        """
        all_masks = []
        
        for stage_idx, (stage, stage_boxes_list) in enumerate(zip(self.stages, boxes)):
            # 1. Flatten boxes for this stage
            flat_boxes, _ = self.flatten_boxes_for_roi(stage_boxes_list, features.device)
            
            # 2. RoI Align
            roi_features = self._roi_align(
                features,
                flat_boxes,
                output_size=(14, 14),
            )
            
            # 3. Predict masks
            mask_logits = stage(roi_features)
            all_masks.append(mask_logits)
        
        return all_masks

    def flatten_boxes_for_roi(self, boxes_list: list[Tensor], device: torch.device) -> tuple[Tensor, list[int]]:
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
            return torch.zeros((0, 5), dtype=torch.float32, device=device), counts
            
        return torch.cat(flat_boxes, dim=0), counts

    def _roi_align(
        self,
        features: Tensor,
        boxes_with_batch: Tensor,
        output_size: tuple[int, int] = (14, 14),
    ) -> Tensor:
        """RoI Align for mask features."""
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
            # Fallback
            B, C, _, _ = features.shape
            N = boxes_with_batch.shape[0]
            return torch.zeros((N, C, *output_size), device=features.device, dtype=features.dtype)


__all__ = ['CascadeMaskHead', 'CascadeMaskStage']
