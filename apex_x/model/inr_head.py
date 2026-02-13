import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ImplicitNeuralHead(nn.Module):
    """World-Class Implicit Neural Representation (INR) Head.
    
    Instead of using standard convolution upsampling, this head treats 
    segmentation as a continuous function f(x, y, context).
    It allows querying for mask values at any sub-pixel coordinate.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        # Fourier Feature Encoding for Coordinates (PE)
        # Allows the MLP to learn high-frequency boundary details
        self.register_buffer("freqs", torch.randn(2, hidden_dim // 2) * 10.0)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels)
        )
        
    def _encode_coords(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: [B, N, 2] in [-1, 1]
        Returns:
            pe: [B, N, hidden_dim]
        """
        # [B, N, 2] @ [2, D/2] -> [B, N, D/2]
        proj = torch.matmul(coords, self.freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [B, C, H, W] context features
            coords: [B, N, 2] normalized coordinates [-1, 1]
        Returns:
            logits: [B, N, Out]
        """
        B, C, H, W = features.shape
        N = coords.shape[1]
        
        # 1. Sample features at coordinates
        # grid: [B, 1, N, 2]
        grid = coords.unsqueeze(1)
        # sampled: [B, C, 1, N] -> [B, N, C]
        sampled_feats = F.grid_sample(features, grid, align_corners=True).squeeze(2).permute(0, 2, 1)
        
        # 2. Encode Coordinates with PE
        pe = self._encode_coords(coords)
        
        # 3. Concatenate and MLP
        x = torch.cat([sampled_feats, pe], dim=-1)
        logits = self.mlp(x)
        
        return logits
