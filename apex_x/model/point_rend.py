"""PointRend: Image Segmentation as Rendering.

Implementation of PointRend for efficient high-resolution mask refinement.
Paper: https://arxiv.org/abs/1912.08193

This module provides:
1. Point sampling based on uncertainty.
2. Point-wise feature extraction.
3. MLP for point-wise prediction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sampling_points(
    mask_logits: Tensor,
    num_points: int,
    oversample_ratio: int = 3,
    importance_sample_ratio: float = 0.75,
) -> Tensor:
    """Sample points from mask logits based on uncertainty.
    
    Args:
        mask_logits: [B, N, H, W]
        num_points: Number of points to sample
        oversample_ratio: Multiplier for initial random sampling
        importance_sample_ratio: Ratio of points sampled by uncertainty vs random
        
    Returns:
        point_coords: [B, N, num_points, 2] in [0, 1]
    """
    assert mask_logits.dim() == 4
    B, N, H, W = mask_logits.shape
    device = mask_logits.device
    
    # 1. Uniform sampling
    num_oversampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(B, N, num_oversampled, 2, device=device)
    
    # 2. Compute uncertainty at sampled points
    point_logits = point_sample(mask_logits, point_coords)
    
    # Handle dimension squeezing
    if point_logits.dim() == 4 and point_logits.shape[2] == point_coords.shape[1]:
        if point_logits.shape[1] == 1:
             point_logits = point_logits.squeeze(1)

    if point_logits.dim() == 4 and point_logits.shape[2] == 1:
         point_logits = point_logits.squeeze(2)
         
    # Uncertainty = -p*log(p) - (1-p)*log(1-p) approx |p - 0.5| for binary
    point_probs = point_logits.sigmoid()
    point_uncertainties = -torch.abs(point_probs - 0.5)
    
    # 3. Select points
    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain
    
    actual_uncertain = min(num_uncertain, num_oversampled)
    
    if actual_uncertain > 0:
        _, idx = point_uncertainties.topk(actual_uncertain, dim=2)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 2) 
        sampled_uncertain = torch.gather(point_coords, 2, idx_expanded)
    else:
        sampled_uncertain = torch.zeros((B, N, 0, 2), device=device)
    
    if num_random > 0:
        sampled_random = torch.rand(B, N, num_random, 2, device=device)
        sampled_points = torch.cat([sampled_uncertain, sampled_random], dim=2)
    else:
        sampled_points = sampled_uncertain
        
    # Ensure correct number of points if we fell short or went over
    if sampled_points.shape[2] != num_points:
        if sampled_points.shape[2] > num_points:
            sampled_points = sampled_points[:, :, :num_points, :]
        else:
            # Pad with random
             pad = torch.rand(B, N, num_points - sampled_points.shape[2], 2, device=device)
             sampled_points = torch.cat([sampled_points, pad], dim=2)

    return sampled_points


def point_sample(input: Tensor, point_coords: Tensor, **kwargs) -> Tensor:
    """Sample features at point coordinates.
    
    Args:
        input: [B, C, H, W] features
        point_coords: [B, N, P, 2] coordinates in [0, 1] range (x, y)
        
    Returns:
        [B, N, C, P] sampled features
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(1) # [B, 1, P, 2]
        
    # Convert [0, 1] to [-1, 1]
    grid = 2.0 * point_coords - 1.0
    
    # [B, C, 1, P]
    # align_corners=False is standard for segmentation
    output = F.grid_sample(input, grid, align_corners=False, **kwargs)
    
    if add_dim:
        output = output.squeeze(2)
        
    return output


class PointRendModule(nn.Module):
    """PointRend refinement module for crisp boundary segmentation.
    
    Attributes:
        mlp: Multi-layer perceptron for point-wise prediction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_points: int = 8096,
        num_fc: int = 3,
        fc_dim: int = 256,
        subdivision_steps: int = 5,
        subdivision_num_points: int = 8192,
    ):
        super().__init__()
        self.num_points = num_points
        self.subdivision_steps = subdivision_steps
        self.subdivision_num_points = subdivision_num_points
        
        self.mlp = nn.Sequential()
        for i in range(num_fc):
            # Input dim is fine_features (in_channels) + coarse_prediction (out_channels)
            last_dim = in_channels + out_channels if i == 0 else fc_dim
            self.mlp.add_module(f"fc{i+1}", nn.Conv1d(last_dim, fc_dim, 1))
            self.mlp.add_module(f"relu{i+1}", nn.ReLU(inplace=True))
            
        self.mlp.add_module("final", nn.Conv1d(fc_dim, out_channels, 1))
        
    def forward(self, coarse_logits: Tensor, fine_features: Tensor, point_coords: Tensor) -> Tensor:
        """
        Predict masks at specific points (Training Mode).
        
        Args:
            coarse_logits: [N, C_out, H, W] Coarse prediction (e.g. from Mask Head)
            fine_features: [N, C_in, H, W] Fine features (e.g. P2 or stride-4 backbone)
            point_coords: [N, P, 2] Point coordinates in [0, 1]
            
        Returns:
            point_logits: [N, C_out, P] Logits at sampled points
        """
        # 1. Sample coarse prediction at points
        coarse_sampled = point_sample(coarse_logits, point_coords) 
        if coarse_sampled.dim() == 4 and coarse_sampled.shape[2] == 1:
             coarse_sampled = coarse_sampled.squeeze(2) # [N, C_out, P]

        # 2. Sample fine features at points
        fine_sampled = point_sample(fine_features, point_coords) 
        if fine_sampled.dim() == 4 and fine_sampled.shape[2] == 1:
             fine_sampled = fine_sampled.squeeze(2) # [N, C_in, P]

        # 3. Concatenate: [N, C_in + C_out, P]
        features = torch.cat([fine_sampled, coarse_sampled], dim=1) 
        
        # 4. Predict
        return self.mlp(features)

    @torch.no_grad()
    def inference(self, coarse_logits: Tensor, fine_features: Tensor) -> Tensor:
        """
        Refine masks using iterative subdivision (Inference Mode).
        
        Args:
            coarse_logits: [N, C_out, H_mask, W_mask] Coarse mask prediction
            fine_features: [N, C_in, H_feat, W_feat] Fine features
            
        Returns:
            refined_logits: [N, C_out, H_feat, W_feat] High-resolution mask logits
        """
        N, C, H_mask, W_mask = coarse_logits.shape
        _, _, H_feat, W_feat = fine_features.shape
        
        # 1. Initial Upsample to Fine Resolution
        refined_logits = F.interpolate(
            coarse_logits, 
            size=(H_feat, W_feat), 
            mode="bilinear", 
            align_corners=False
        )
        
        # 2. Iterative Subdivision
        for _ in range(self.subdivision_steps):
            # Select uncertain points from current refined logits
            # point_coords: [N, subdivision_num_points, 2]
            point_coords = sampling_points(
                refined_logits, 
                self.subdivision_num_points, 
                oversample_ratio=3, 
                importance_sample_ratio=1.0 # Focus purely on uncertain regions
            )
            
            # Predict new values at these points
            # point_logits: [N, C_out, P]
            point_logits = self.forward(coarse_logits, fine_features, point_coords)
            
            # Scatter predictions back to the high-res map
            # We map [0,1] coords to [0, W_feat-1], [0, H_feat-1] indices
            # Note: point_coords are (x, y) in [0, 1] relative to the box/mask
            
            point_indices_x = (point_coords[..., 0] * W_feat).long().clamp(0, W_feat - 1)
            point_indices_y = (point_coords[..., 1] * H_feat).long().clamp(0, H_feat - 1)
            
            # For each instance in batch
            for i in range(N):
                # refined_logits[i]: [C, H, W]
                # point_logits[i]: [C, P]
                # indices: [P]
                
                # Careful with shape: point_logits is [C, P]
                # We want to assign to [C, y, x]
                
                # Flatten spatial indices for advanced indexing?
                # Or just loop. Loop over P is slow. Loop over C is fast (C=1).
                
                # Vectorized scatter:
                # refined_logits[i, c, y[i], x[i]] = point_logits[i, c]
                
                # Since N is batch of ROIs, we do a loop over N is acceptable (usually < 100)
                # But here we can use fancy indexing
                
                px = point_indices_x[i] # [P]
                py = point_indices_y[i] # [P]
                val = point_logits[i]   # [C, P]
                
                refined_logits[i, :, py, px] = val
                
        return refined_logits
