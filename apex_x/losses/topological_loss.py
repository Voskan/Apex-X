import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TopologicalPersistenceLoss(nn.Module):
    """World-Class Differentiable Topological Loss.
    
    Instead of full persistent homology (which is slow), we use a 
    Differentiable Euler Characteristic (EC) approximation. 
    EC = V - E + F (Vertices - Edges + Faces).
    
    For a binary mask, EC correlates with the number of objects (Betti-0) 
    minus the number of holes (Betti-1). 
    We regularize the predicted EC to match the GT EC.
    """
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, pred_probs: Tensor, gt_masks: Tensor) -> Tensor:
        """
        Args:
            pred_probs: [B, 1, H, W] sigmoid probabilities
            gt_masks: [B, 1, H, W] binary ground truth
        """
        # 1. Soft Euler Characteristic Calculation
        # V: Soft vertices (sum of probabilities)
        v = pred_probs.sum(dim=(2, 3))
        
        # E: Soft edges (horizontal and vertical adjacencies)
        # horizontal: p(i, j) * p(i, j+1)
        e_h = (pred_probs[:, :, :, :-1] * pred_probs[:, :, :, 1:]).sum(dim=(2, 3))
        # vertical: p(i, j) * p(i+1, j)
        e_v = (pred_probs[:, :, :-1, :] * pred_probs[:, :, 1:, :]).sum(dim=(2, 3))
        e = e_h + e_v
        
        # F: Soft faces (2x2 squares)
        # p(i, j) * p(i+1, j) * p(i, j+1) * p(i+1, j+1)
        f = (pred_probs[:, :, :-1, :-1] * 
             pred_probs[:, :, 1:, :-1] * 
             pred_probs[:, :, :-1, 1:] * 
             pred_probs[:, :, 1:, 1:]).sum(dim=(2, 3))
        
        soft_ec = v - e + f
        
        # 2. GT EC (Analytical)
        # We calculate it once for GT
        with torch.no_grad():
            v_gt = gt_masks.sum(dim=(2, 3))
            e_h_gt = (gt_masks[:, :, :, :-1] * gt_masks[:, :, :, 1:]).sum(dim=(2, 3))
            e_v_gt = (gt_masks[:, :, :-1, :] * gt_masks[:, :, 1:, :]).sum(dim=(2, 3))
            f_gt = (gt_masks[:, :, :-1, :-1] * 
                    gt_masks[:, :, 1:, :-1] * 
                    gt_masks[:, :, :-1, 1:] * 
                    gt_masks[:, :, 1:, 1:]).sum(dim=(2, 3))
            gt_ec = v_gt - (e_h_gt + e_v_gt) + f_gt
            
        # 3. Persistence Loss: Match EC and penalize "dust" (high frequency small components)
        ec_loss = F.mse_loss(soft_ec, gt_ec)
        
        # Total variation as a proxy for boundary smoothness (Topological Regularization)
        tv_loss = (torch.abs(pred_probs[:, :, :, 1:] - pred_probs[:, :, :, :-1]).mean() + 
                   torch.abs(pred_probs[:, :, 1:, :] - pred_probs[:, :, :-1, :]).mean())
                   
        return (ec_loss * self.scale) + (tv_loss * 0.1)
