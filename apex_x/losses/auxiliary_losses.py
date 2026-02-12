"""Auxiliary decoder losses for improved segmentation training.

Adds intermediate supervision at multiple decoder stages to improve
gradient flow and feature quality.

Expected impact: +2-3% mask AP
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def auxiliary_mask_loss(
    *,
    aux_mask_outputs: list[Tensor],
    target_masks: Tensor,
    weights: list[float] | None = None,
    loss_type: str = 'bce',
) -> Tensor:
    """Compute auxiliary losses from intermediate decoder outputs.
    
    Args:
        aux_mask_outputs: List of intermediate mask predictions [N_aux, B, H, W]
        target_masks: Ground truth masks [B, H, W]
        weights: Loss weights for each auxiliary output (default: decreasing)
        loss_type: 'bce', 'dice', or 'focal' (default: 'bce')
    
    Returns:
        Weighted auxiliary loss scalar
    """
    if len(aux_mask_outputs) == 0:
        return torch.tensor(0.0, device=target_masks.device)
    
    # Default weights: decreasing from early to late layers
    if weights is None:
        n_aux = len(aux_mask_outputs)
        weights = [0.5 ** (n_aux - i) for i in range(n_aux)]
    
    total_loss = torch.tensor(0.0, device=target_masks.device)
    
    for aux_output, weight in zip(aux_mask_outputs, weights):
        # Resize auxiliary output to match target size
        if aux_output.shape[-2:] != target_masks.shape[-2:]:
            aux_output = F.interpolate(
                aux_output.unsqueeze(1),
                size=target_masks.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        
        # Compute loss based on type
        if loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(
                aux_output,
                target_masks,
                reduction='mean',
            )
        elif loss_type == 'dice':
            loss = dice_loss(aux_output, target_masks)
        elif loss_type == 'focal':
            loss = focal_loss(aux_output, target_masks, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        total_loss = total_loss + weight * loss
    
    return total_loss


def dice_loss(
    pred_logits: Tensor,
    target: Tensor,
    smooth: float = 1.0,
) -> Tensor:
    """Differentiable Dice loss for segmentation.
    
    Args:
        pred_logits: Predicted logits [B, H, W]
        target: Target masks [B, H, W]
        smooth: Smoothing factor (default: 1.0)
    
    Returns:
        Dice loss scalar
    """
    pred = torch.sigmoid(pred_logits)
    
    # Flatten
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    # Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice


def focal_loss(
    pred_logits: Tensor,
    target: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Focal loss for handling class imbalance in masks.
    
    Args:
        pred_logits: Predicted logits [B, H, W]
        target: Target masks [B, H, W]
        gamma: Focusing parameter (default: 2.0)
        alpha: Weighting factor (default: 0.25)
    
    Returns:
        Focal loss scalar
    """
    pred = torch.sigmoid(pred_logits)
    
    # Binary cross entropy
    bce = F.binary_cross_entropy_with_logits(
        pred_logits,
        target,
        reduction='none',
    )
    
    # Focal weight
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    
    # Alpha weighting
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    loss = alpha_t * focal_weight * bce
    
    return loss.mean()


__all__ = [
    'auxiliary_mask_loss',
    'dice_loss',
    'focal_loss',
]
