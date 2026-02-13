import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FlowSymmetryLoss(nn.Module):
    """World-Class Physics-Informed Symmetry Loss.
    
    Instead of just matching vectors, we enforce that the predicted 
    Boundary Force Field (BFF) is a Conservative Field (Irrotational).
    
    A field V is conservative if Curl(V) = 0.
    In 2D, this means dFy/dx - dFx/dy = 0.
    
    We also add a Flow-Invariance term: the potential should be 
    symmetric relative to the boundary midline.
    """
    
    def __init__(self, curl_weight: float = 1.0, alignment_weight: float = 1.0):
        super().__init__()
        self.curl_weight = curl_weight
        self.alignment_weight = alignment_weight
        
    def forward(self, pred_bff: Tensor, gt_bff: Tensor) -> Tensor:
        """
        Args:
            pred_bff: [B, 2, H, W] (Fx, Fy)
            gt_bff: [B, 2, H, W] Ground truth force field
        """
        # 1. Curl Regularization (Hamiltonian Constraint)
        # dFy/dx
        dfy_dx = pred_bff[:, 1, :, 1:] - pred_bff[:, 1, :, :-1]
        # dFx/dy
        dfx_dy = pred_bff[:, 0, 1:, :] - pred_bff[:, 0, :-1, :]
        
        # Pad to match resolution
        dfy_dx = F.pad(dfy_dx, (0, 1, 0, 0))
        dfx_dy = F.pad(dfx_dy, (0, 0, 0, 1))
        
        curl = dfy_dx - dfx_dy
        curl_loss = torch.mean(curl**2)
        
        # 2. Potential Alignment (MSE with GT)
        potential_loss = F.mse_loss(pred_bff, gt_bff)
        
        # 3. Flow Symmetry (Cosine Similarity)
        # We want the directions to match perfectly even if magnitude varies
        cos_sim = F.cosine_similarity(pred_bff, gt_bff, dim=1)
        symmetry_loss = 1.0 - cos_sim.mean()
        
        return (self.curl_weight * curl_loss) + \
               (self.alignment_weight * potential_loss) + \
               (0.5 * symmetry_loss)
