"""Pseudo-labeling (Silver Label) Generator.

Generates high-confidence labels from a teacher model for semi-supervised 
training on unlabeled datasets.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

class PseudoLabeler:
    """Generates and filters high-quality pseudo-labels."""
    
    def __init__(self, conf_threshold: float = 0.5, quality_threshold: float = 0.6):
        self.conf_threshold = conf_threshold
        self.quality_threshold = quality_threshold

    def generate_silver_labels(
        self, 
        model_output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Filter model outputs to produce reliable pseudo-labels.
        
        Args:
            model_output: Raw detection results.
            
        Returns:
            Filtered targets for semi-supervised training.
        """
        scores = model_output.get("scores")   # [N, C]
        masks = model_output.get("masks")     # [N, 1, 28, 28]
        boxes = model_output.get("boxes")     # [N, 4]
        quality = model_output.get("predicted_quality") # [N]
        
        if scores is None or masks is None or boxes is None or quality is None:
            return {}

        # 1. Get max scores and classes
        max_scores, classes = torch.max(scores, dim=-1)
        
        # 2. Filter by confidence AND predicted quality
        # This is a critical v2.0 optimization: trust the QualityHead
        keep_mask = (max_scores > self.conf_threshold) & (quality > self.quality_threshold)
        
        silver_indices = torch.where(keep_mask)[0]
        
        if silver_indices.numel() == 0:
            return {}
            
        return {
            "boxes": boxes[silver_indices],
            "labels": classes[silver_indices],
            "masks": masks[silver_indices],
            "scores": max_scores[silver_indices],
            "quality": quality[silver_indices]
        }

__all__ = ["PseudoLabeler"]
