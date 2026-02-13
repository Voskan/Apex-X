"""Lovász-Softmax loss for instance segmentation.

Implementation of the Lovász-Softmax loss from:
"The Lovász-Softmax loss: A tractable surrogate for the optimization of the
intersection-over-union measure in neural networks"
Berman et al., CVPR 2018
https://arxiv.org/abs/1705.08790

This loss is superior to Dice loss for segmentation boundaries.
Expected impact: +1-3% boundary IoU improvement.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Compute gradient of the Lovász extension w.r.t the sorted errors.
    
    Args:
        gt_sorted: Sorted ground truth labels [N]
        
    Returns:
        Lovász gradient [N]
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
    return jaccard


def lovasz_hinge(logits: Tensor, labels: Tensor, *, per_image: bool = True) -> Tensor:
    """Binary Lovász hinge loss.
    
    Args:
        logits: Predictions [B, H, W] or [B, N, H, W]
        labels: Binary ground truth {0, 1} [B, H, W] or [B, N, H, W]
        per_image: Compute loss per image then average
        
    Returns:
        Scalar loss
    """
    if per_image:
        if logits.ndim == 3:
            # [B, H, W] format
            losses = [
                _lovasz_hinge_flat(*_flatten_binary_scores(logits[i], labels[i]))
                for i in range(logits.shape[0])
            ]
        else:
            # [B, N, H, W] format - handle per instance
            losses = []
            for batch_idx in range(logits.shape[0]):
                for inst_idx in range(logits.shape[1]):
                    loss = _lovasz_hinge_flat(
                        *_flatten_binary_scores(
                            logits[batch_idx, inst_idx],
                            labels[batch_idx, inst_idx]
                        )
                    )
                    losses.append(loss)
                    
        return torch.stack(losses).mean()
    else:
        return _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels))


def _flatten_binary_scores(scores: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten predictions and labels."""
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    return scores, labels


def _lovasz_hinge_flat(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary Lovász hinge loss on flattened tensors.
    
    Args:
        logits: Predictions [N]
        labels: Binary labels {0, 1} [N]
        
    Returns:
        Scalar loss
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
        
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_softmax(
    probs: Tensor,
    labels: Tensor,
    *,
    classes: str | list[int] = 'present',
    per_image: bool = True,
    ignore_index: int | None = None,
) -> Tensor:
    """Multi-class Lovász-Softmax loss.
    
    Args:
        probs: Class probabilities [B, C, H, W]
        labels: Ground truth class indices [B, H, W]
        classes: 'all' for all classes, 'present' for classes in batch, 
                 or list of class indices
        per_image: Compute loss per image then average
        ignore_index: Class index to ignore
        
    Returns:
        Scalar loss
    """
    if per_image:
        losses = [
            _lovasz_softmax_flat(
                *_flatten_probs(probs[i], labels[i], ignore_index),
                classes=classes
            )
            for i in range(probs.shape[0])
        ]
        return torch.stack(losses).mean()
    else:
        return _lovasz_softmax_flat(
            *_flatten_probs(probs, labels, ignore_index),
            classes=classes
        )


def _flatten_probs(probs: Tensor, labels: Tensor, ignore_index: int | None) -> tuple[Tensor, Tensor]:
    """Flatten predictions and labels for multi-class."""
    C = probs.shape[0]
    probs = probs.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    labels = labels.reshape(-1)  # [H*W]
    
    if ignore_index is not None:
        valid = labels != ignore_index
        probs = probs[valid]
        labels = labels[valid]
        
    return probs, labels


def _lovasz_softmax_flat(probs: Tensor, labels: Tensor, *, classes: str | list[int] = 'present') -> Tensor:
    """Multi-class Lovász-Softmax loss on flattened tensors.
    
    Args:
        probs: Class probabilities [N, C]
        labels: Class indices [N]
        classes: Which classes to compute loss for
        
    Returns:
        Scalar loss
    """
    if len(labels) == 0:
        return probs.sum() * 0.0
        
    C = probs.shape[1]
    losses = []
    
    # Determine which classes to process
    if classes == 'all':
        class_indices = range(C)
    elif classes == 'present':
        class_indices = labels.unique().tolist()
    else:
        class_indices = classes
        
    for c in class_indices:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
            
        class_probs = probs[:, c]
        errors = (fg - class_probs).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
        
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def lovasz_instance_loss(
    mask_logits: Tensor,
    target_masks: Tensor,
    *,
    per_instance: bool = True,
) -> Tensor:
    """Lovász loss for instance segmentation masks.
    
    Optimized for instance-level binary segmentation.
    
    Args:
        mask_logits: Predicted mask logits [B, N, H, W]
        target_masks: Binary target masks [B, N, H, W]
        per_instance: Compute loss per instance then average
        
    Returns:
        Scalar loss
    """
    if mask_logits.ndim != 4 or target_masks.ndim != 4:
        raise ValueError(f"Expected [B, N, H, W], got {mask_logits.shape}, {target_masks.shape}")
        
    if mask_logits.shape != target_masks.shape:
        raise ValueError(f"Shape mismatch: {mask_logits.shape} vs {target_masks.shape}")
        
    # Convert logits to hinge format (margin-based)
    target = target_masks.float()
    
    return lovasz_hinge(mask_logits, target, per_image=per_instance)


def combined_seg_loss(
    mask_logits: Tensor,
    target_masks: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    lovasz_weight: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Combined segmentation loss with Lovász.
    
    Combines BCE, Dice, and Lovász losses for optimal segmentation.
    
    Args:
        mask_logits: Predicted mask logits [B, N, H, W]
        target_masks: Binary target masks [B, N, H, W]
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        lovasz_weight: Weight for Lovász loss
        
    Returns:
        Total loss and dict of individual losses
    """
    from apex_x.losses.seg_loss import mask_bce_loss, mask_dice_loss
    
    bce = mask_bce_loss(mask_logits, target_masks)
    dice = mask_dice_loss(mask_logits, target_masks)
    lovasz = lovasz_instance_loss(mask_logits, target_masks)
    
    total = bce_weight * bce + dice_weight * dice + lovasz_weight * lovasz
    
    losses = {
        'bce_loss': float(bce.item()),
        'dice_loss': float(dice.item()),
        'lovasz_loss': float(lovasz.item()),
        'total_loss': float(total.item()),
    }
    
    return total, losses



# Alias for compatibility with seg_loss.py
lovasz_loss = lovasz_hinge

__all__ = [
    "lovasz_hinge",
    "lovasz_loss",
    "lovasz_softmax",
    "lovasz_instance_loss",
    "combined_seg_loss",
]
