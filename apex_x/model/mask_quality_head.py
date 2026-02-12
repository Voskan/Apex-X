"""Mask Quality Prediction Head for IoU-aware confidence.

Predicts the IoU between predicted and ground truth masks.
This improves confidence calibration and NMS ranking.

Expected gain: +1-2% mask AP
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MaskQualityHead(nn.Module):
    """Mask Quality (IoU) Prediction Head.
    
    Predicts the IoU score for each predicted mask to improve
    confidence calibration. Can be used for:
    - NMS re-ranking (use predicted IoU instead of classification score)
    - Quality-aware loss weighting
    - Better AP metrics
    
    Architecture:
        mask_features [N, C, H, W]
        → Global Average Pool → [N, C]
        → FC layers → [N, 1]
        → Sigmoid → predicted IoU [N]
    
    Args:
        in_channels: Number of input feature channels
        hidden_dim: Hidden layer dimension (default: 256)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Quality prediction network
        self.quality_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output: predicted IoU in [0, 1]
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
    
    def forward(self, mask_features: Tensor) -> Tensor:
        """Predict mask quality (IoU).
        
        Args:
            mask_features: Mask feature maps [N, C, H, W]
            
        Returns:
            Predicted IoU scores [N]
        """
        # Global average pooling
        # [N, C, H, W] → [N, C]
        pooled = F.adaptive_avg_pool2d(mask_features, 1)
        pooled = pooled.flatten(1)  # [N, C]
        
        # Predict IoU
        quality = self.quality_net(pooled)  # [N, 1]
        quality = quality.squeeze(-1)  # [N]
        
        return quality


def mask_iou_loss(
    predicted_iou: Tensor,
    mask_logits: Tensor,
    target_masks: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Mask IoU prediction loss.
    
    Supervises the predicted IoU to match the actual IoU between
    predicted and ground truth masks.
    
    Args:
        predicted_iou: Predicted IoU scores [N]
        mask_logits: Predicted mask logits [N, H, W]
        target_masks: Ground truth masks [N, H, W]
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Mask IoU loss (MSE between predicted and actual IoU)
    """
    # Compute actual IoU
    mask_probs = torch.sigmoid(mask_logits)
    mask_binary = (mask_probs > 0.5).float()
    
    intersection = (mask_binary * target_masks).sum(dim=[-2, -1])
    union = (mask_binary + target_masks).clamp(0, 1).sum(dim=[-2, -1])
    
    # Actual IoU
    actual_iou = (intersection + 1e-7) / (union + 1e-7)
    
    # MSE loss between predicted and actual IoU
    loss = F.mse_loss(predicted_iou, actual_iou, reduction=reduction)
    
    return loss


def mask_iou_loss_smooth_l1(
    predicted_iou: Tensor,
    mask_logits: Tensor,
    target_masks: Tensor,
    *,
    beta: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """Mask IoU prediction loss with Smooth L1 (more robust).
    
    Args:
        predicted_iou: Predicted IoU scores [N]
        mask_logits: Predicted mask logits [N, H, W]
        target_masks: Ground truth masks [N, H, W]
        beta: Smooth L1 threshold (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Smooth L1 loss
    """
    # Compute actual IoU
    mask_probs = torch.sigmoid(mask_logits)
    mask_binary = (mask_probs > 0.5).float()
    
    intersection = (mask_binary * target_masks).sum(dim=[-2, -1])
    union = (mask_binary + target_masks).clamp(0, 1).sum(dim=[-2, -1])
    
    actual_iou = (intersection + 1e-7) / (union + 1e-7)
    
    # Smooth L1 loss
    loss = F.smooth_l1_loss(predicted_iou, actual_iou, beta=beta, reduction=reduction)
    
    return loss


__all__ = [
    'MaskQualityHead',
    'mask_iou_loss',
    'mask_iou_loss_smooth_l1',
]
