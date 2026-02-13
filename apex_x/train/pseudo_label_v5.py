import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SelfMaskTeacher:
    """World-Class Self-Distillation Engine.
    
    Implements the 'Ascension' feedback loop:
    1. Coarse Prediction -> Diffusion Refinement -> High-Q Pseudo-Mask.
    2. Pseudo-Mask supervises Coarse Prediction with a Consistency Loss.
    
    This allows the model to learn its own refinement mistakes and 
    converge to 'Infinite Precision'.
    """
    
    def __init__(self, temperature: float = 0.5, consistency_weight: float = 1.0):
        self.temperature = temperature
        self.consistency_weight = consistency_weight

    def distillation_loss(self, coarse_logits: Tensor, refined_probs: Tensor) -> Tensor:
        """
        Args:
            coarse_logits: [B, 1, H, W] initial coarse output
            refined_probs: [B, 1, H, W] output from Diffusion Refiner (Stop Gradient)
        """
        # 1. Stop gradient on teacher (refined_probs)
        teacher_mask = refined_probs.detach() > self.temperature
        teacher_mask = teacher_mask.float()
        
        # 2. Kullback-Leibler or BCE consistency
        # We want coarse_logits to 'look ahead' to the refined version
        loss = F.binary_cross_entropy_with_logits(coarse_logits, teacher_mask)
        
        return loss * self.consistency_weight

def apply_self_supervision(model_output: dict[str, Tensor], loss_dict: dict[str, Tensor]):
    """Helper to integrate SelfMaskTeacher into the loss pipeline."""
    coarse = model_output.get("coarse_mask")
    diffused = model_output.get("diffused_mask")
    
    if coarse is not None and diffused is not None:
        teacher = SelfMaskTeacher()
        # diffused_mask is already sigmoid-probs from previous implementation
        loss_dict["self_teacher_consistency"] = teacher.distillation_loss(coarse, diffused)
