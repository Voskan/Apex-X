"""Active Learning Data Miner.

Identifies 'hard' examples from unlabeled satellite imagery by analyzing:
- Predictive Entropy: High entropy = model is confused.
- Margin Sampling: Small gap between top-2 classes.
- Mask Quality: Low predicted quality = geometric uncertainty.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, List

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

def compute_entropy(logits: Tensor) -> Tensor:
    """Compute Shannon entropy from logits.
    
    Args:
        logits: Tensor of shape [N, C]
        
    Returns:
        Entropy tensor of shape [N]
    """
    probs = torch.softmax(logits, dim=-1)
    # Clamp to avoid log(0)
    probs = probs.clamp(min=1e-12)
    entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
    return entropy

class DataMiner:
    """Miner for identifying hard examples in new datasets."""
    
    def __init__(self, entropy_threshold: float = 0.5, quality_threshold: float = 0.4):
        self.entropy_threshold = entropy_threshold
        self.quality_threshold = quality_threshold

    def find_hard_tiles(
        self, 
        model_output: Dict[str, Tensor],
        num_tiles: int = 10
    ) -> List[int]:
        """Find indices of tiles that require attention.
        
        Args:
            model_output: Output dictionary from TeacherModelV3.
            num_tiles: Number of hard tiles to return.
            
        Returns:
            List of tile indices sorted by 'hardness'.
        """
        scores = model_output.get("scores") # [N, C]
        quality = model_output.get("predicted_quality") # [N]
        
        if scores is None:
            return []
            
        # 1. Compute entropy of scores
        entropy = compute_entropy(scores) # [N]
        
        # 2. Heuristic: Hardness = Entropy / (Quality + eps)
        # We want high entropy (confusion) and low quality (geometric uncertainty)
        eps = 1e-6
        if quality is not None:
            hardness = entropy / (quality + eps)
        else:
            hardness = entropy
            
        # Get top-k indices
        values, indices = torch.topk(hardness, min(num_tiles, hardness.size(0)))
        
        return indices.tolist()

__all__ = ["DataMiner", "compute_entropy"]
