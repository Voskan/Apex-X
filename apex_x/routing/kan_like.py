from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class RouterKANOutput:
    """Router outputs per tile: utility U, split utility S, optional temporal keep T."""

    U: Tensor  # [B, K]
    S: Tensor  # [B, K]
    T: Tensor | None = None  # [B, K] when temporal head enabled


class LightweightSplineActivation(nn.Module):
    """A B-Spline based Linear layer for Kolmogorov-Arnold Networks.
    
    Instead of fixed weights, it learns the activation function on the edge.
    This provides higher representational power for non-linear utility boundaries.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base weight (like standard linear)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Spline weights
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.trunc_normal_(self.spline_weight, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        # Base path
        base_output = F.linear(x, self.base_weight)
        
        # Spline path (Simplified implementation of KAN spline evaluation)
        # evaluation on grids for non-linearity
        x_expanded = x.unsqueeze(-1)
        grid = torch.linspace(-1, 1, self.grid_size + self.spline_order, device=x.device)
        # [B, ..., In, Grid]
        basis = torch.exp(-(x_expanded - grid)**2 * 10.0) 
        
        # We need to handle arbitrary leading dims (e.g. [B, K, In])
        # [B, ..., In, Grid] * [Out, In, Grid] -> [B, ..., Out]
        # Using matmul/einsum that preserves batch dims.
        # basis: [..., In, Grid], spline_weight: [Out, In, Grid]
        # We want to sum over (In, Grid)
        
        # Reshape spline weight for easier contraction: [Out, In * Grid]
        w = self.spline_weight.view(self.out_features, -1)
        # Reshape basis: [B, ..., In * Grid]
        b = basis.view(*x.shape[:-1], -1)
        
        spline_output = F.linear(b, w)
        
        return base_output + spline_output


class RouterKANLike(nn.Module):
    """Kolmogorov-Arnold Network (KAN) inspired Router.
    
    Predicts tile utilities (U, S, T) using learnable spline-based activations.
    Significantly more expressive than standard MLPs for geometric routing.
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int = 64, 
        num_outputs: int = 2,
        temporal_head: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_head = temporal_head
        
        self.backbone = nn.Sequential(
            LightweightSplineActivation(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.head_u = LightweightSplineActivation(hidden_dim, 1)
        self.head_s = LightweightSplineActivation(hidden_dim, 1)
        self.head_t = LightweightSplineActivation(hidden_dim, 1) if temporal_head else None

    def forward(self, x: Tensor) -> RouterKANOutput:
        """
        Args:
            x: Input features [B, K, D] where K is num_tiles
        Returns:
            RouterKANOutput with U, S, T
        """
        if x.ndim != 3:
            # Fallback for old callers who might pass [B, D]
            is_flat = True
            x = x.unsqueeze(1)
        else:
            is_flat = False
            
        h = self.backbone(x)
        
        u = self.head_u(h).squeeze(-1)
        s = self.head_s(h).squeeze(-1)
        t = self.head_t(h).squeeze(-1) if self.head_t is not None else None
        
        return RouterKANOutput(U=u, S=s, T=t)
