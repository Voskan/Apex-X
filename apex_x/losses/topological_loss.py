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
        B, C, H, W = pred_probs.shape
        num_pixels = H * W
        eps = 1e-6
        
        # Clamp for numerical stability in products
        p = pred_probs.clamp(min=eps, max=1.0 - eps)
        
        # 1. Soft Euler Characteristic Calculation (Area-Normalized)
        # V: Soft vertices (sum of probabilities)
        v = p.sum(dim=(2, 3))
        
        # E: Soft edges (horizontal and vertical adjacencies)
        e_h = (p[:, :, :, :-1] * p[:, :, :, 1:]).sum(dim=(2, 3))
        e_v = (p[:, :, :-1, :] * p[:, :, 1:, :]).sum(dim=(2, 3))
        e = e_h + e_v
        
        # F: Soft faces (2x2 squares)
        f = (p[:, :, :-1, :-1] * 
             p[:, :, 1:, :-1] * 
             p[:, :, :-1, 1:] * 
             p[:, :, 1:, 1:]).sum(dim=(2, 3))
        
        soft_ec = (v - e + f) / num_pixels
        
        # 2. GT EC (Analytical, Area-Normalized)
        with torch.no_grad():
            gt = gt_masks.float()
            v_gt = gt.sum(dim=(2, 3))
            e_h_gt = (gt[:, :, :, :-1] * gt[:, :, :, 1:]).sum(dim=(2, 3))
            e_v_gt = (gt[:, :, :-1, :] * gt[:, :, 1:, :]).sum(dim=(2, 3))
            f_gt = (gt[:, :, :-1, :-1] * 
                    gt[:, :, 1:, :-1] * 
                    gt[:, :, :-1, 1:] * 
                    gt[:, :, 1:, 1:]).sum(dim=(2, 3))
            gt_ec = (v_gt - (e_h_gt + e_v_gt) + f_gt) / num_pixels
            
        # 3. Persistence Loss: Match EC
        ec_loss = F.mse_loss(soft_ec, gt_ec)
        
        # Total variation as a proxy for boundary smoothness
        tv_loss = (torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1]).mean() + 
                   torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :]).mean())
                   
        return (ec_loss * self.scale) + (tv_loss * 0.1)
