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
        boxes: list[Tensor],
    ) -> list[Tensor]:
        """Progressive mask refinement.
        
        Args:
            features: Feature maps [B, C, H, W]
            boxes: Refined boxes from each cascade detection stage
            
        Returns:
            List of mask logits from each stage
        """
        all_masks = []
        
        for stage_idx, (stage, stage_boxes) in enumerate(zip(self.stages, boxes)):
            # RoI Align on current boxes
            roi_features = self._roi_align(
                features,
                stage_boxes,
                output_size=(14, 14),  # Standard RoI size
            )
            
            # Predict masks
            mask_logits = stage(roi_features)
            all_masks.append(mask_logits)
        
        return all_masks
    
    def _roi_align(
        self,
        features: Tensor,
        boxes: Tensor,
        output_size: tuple[int, int] = (14, 14),
    ) -> Tensor:
        """RoI Align for mask features."""
        try:
            from torchvision.ops import roi_align
            
            # Add batch index
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
            # Fallback
            b, c, h, w = features.shape
            n = boxes.shape[0]
            return torch.zeros(
                (n, c, *output_size),
                dtype=features.dtype,
                device=features.device,
            )


__all__ = ['CascadeMaskHead', 'CascadeMaskStage']
