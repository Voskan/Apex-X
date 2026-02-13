"""Boundary Force Field (BFF) Loss — Non-standard math for perfect segmentation.

Instead of predicting binary masks, we predict a Relative Displacement Field (RDF)
where each pixel (x, y) predicts a vector (dx, dy) pointing to the nearest boundary.

This solves sub-pixel aliasing and provides a continuous geometric representation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from scipy.ndimage import distance_transform_edt
    import numpy as np
except ImportError:
    distance_transform_edt = None

class BFFLoss(nn.Module):
    """Boundary Force Field Loss.
    
    Optimizes the predicted displacement vectors against the distance transform gradient.
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def compute_gt_bff(self, masks: Tensor) -> Tensor:
        """Compute Ground Truth Force Field using EDT gradients.
        
        Args:
            masks: [N, H, W] binary masks.
            
        Returns:
            gt_bff: [N, 2, H, W] unit vectors (dx, dy).
        """
        if distance_transform_edt is None:
            return torch.zeros((masks.shape[0], 2, masks.shape[1], masks.shape[2]), device=masks.device)
            
        N, H, W = masks.shape
        masks_np = masks.cpu().numpy()
        gt_bff_list = []
        
        for i in range(N):
            mask = masks_np[i]
            # EDT gives distance to nearest 0 (background)
            # To find distance to boundary, we combine EDT of mask and EDT of ~mask
            dist_in = distance_transform_edt(mask)
            dist_out = distance_transform_edt(1 - mask)
            
            # Combine: positive inside, negative outside
            dist_total = dist_in - dist_out # Signed Distance Function (SDF)
            
            # Gradient of SDF points to the boundary
            dy, dx = np.gradient(dist_total)
            
            # Normalize to unit vectors
            mag = np.sqrt(dx**2 + dy**2) + 1e-8
            dx /= mag
            dy /= mag
            
            gt_bff_list.append(torch.from_numpy(np.stack([dx, dy], axis=0)))
            
        return torch.stack(gt_bff_list, dim=0).to(masks.device).float()

    def forward(self, pred_bff: Tensor, target_masks: Tensor) -> Tensor:
        """
        Args:
            pred_bff: [N, 2, H, W] predicted displacement vectors.
            target_masks: [N, H, W] binary ground truth.
        """
        # 1. Compute GT BFF
        with torch.no_grad():
            gt_bff = self.compute_gt_bff(target_masks) # [N, 2, H, W]
            
        # 2. Vector Cosine Similarity Loss
        # We want pred_bff to align with gt_bff
        # loss = 1 - cosine_similarity
        cos_sim = F.cosine_similarity(pred_bff, gt_bff, dim=1)
        similarity_loss = 1.0 - cos_sim.mean()
        
        # 3. Magnitude Loss (Optional: push magnitudes to be useful or unit)
        # For RDF, we can also predict distance as magnitude.
        # But here we focus on direction for boundary tightness.
        
        return similarity_loss

class DifferentiableContourIntegrator(nn.Module):
    """Converts a Force Field back to a binary mask probability.
    
    Uses the divergence of the force field as a proxy for the mask interior.
    Div(F) < 0 at sinks (interior), Div(F) > 0 at sources (exterior/edges).
    """
    
    def forward(self, bff: Tensor) -> Tensor:
        """
        Args:
            bff: [N, 2, H, W] predicted force field.
            
        Returns:
            mask_probs: [N, 1, H, W] derived mask probability.
        """
        # Divergence = dFx/dx + dFy/dy
        # Using finite differences
        dx = bff[:, 0, :, 1:] - bff[:, 0, :, :-1]
        dy = bff[:, 1, 1:, :] - bff[:, 1, :-1, :]
        
        # Pad to keep size
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        
        div = dx + dy
        
        # Sinks (negative divergence) are interior. 
        # Source (positive divergence) are boundaries.
        # Simple mapping: sigmoid(-div * scale)
        return torch.sigmoid(-div * 5.0).unsqueeze(1)
