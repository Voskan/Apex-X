import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    """A B-Spline based Linear layer for Kolmogorov-Arnold Networks.
    
    Instead of fixed weights, it learns the activation function on the edge.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
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

    def forward(self, x):
        # Base path
        base_output = F.linear(x, self.base_weight)
        
        # Spline path (Simplified implementation of KAN spline evaluation)
        # In a full model, we'd use B-Spline basis functions.
        # For the V5 routing module, we use a radial basis approximation.
        # [B, In] -> [B, In, 1]
        x_expanded = x.unsqueeze(-1)
        grid = torch.linspace(-1, 1, self.grid_size + self.spline_order, device=x.device)
        # [B, In, Grid]
        basis = torch.exp(-(x_expanded - grid)**2 * 10.0) 
        
        # [B, In, Grid] * [Out, In, Grid] -> [B, Out]
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        
        return base_output + spline_output

class AdaptiveKAN(nn.Module):
    """World-Class KAN-based Router.
    
    Decides which refinement head (PointRend, Diffusion, or INR) 
    should be prioritized for a given tile.
    """
    def __init__(self, in_channels: int, num_heads: int = 3):
        super().__init__()
        self.kan = nn.Sequential(
            KANLinear(in_channels, 64),
            nn.LayerNorm(64),
            KANLinear(64, num_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Global tile features [B, C]
        Returns:
            routing_weights: [B, num_heads]
        """
        logits = self.kan(x)
        return torch.softmax(logits, dim=-1)
