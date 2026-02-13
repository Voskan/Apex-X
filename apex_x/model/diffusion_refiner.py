"""Iterative Mask Diffusion (IMD) Refiner — God-Tier Sharpness.

Inspired by SegRefiner (NeurIPS 2024), this head treats mask refinement
as a generative denoising task.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskDiffusionBlock(nn.Module):
    """A lightweight residual block for mask denoising."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn = nn.GroupNorm(8, channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.gn(x)
        x = self.silu(x)
        x = self.conv2(x)
        return identity + x

class MaskDiffusionRefiner(nn.Module):
    """Iterative Mask Diffusion Head.
    
    Refines a coarse mask (logits) by 'denoising' it using high-res features.
    """
    
    def __init__(
        self, 
        feature_channels: int = 256, 
        hidden_channels: int = 64,
        num_iterations: int = 2
    ):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Initial projection: Coarse mask (1) + Features (C) -> Hidden
        self.proj = nn.Conv2d(feature_channels + 1, hidden_channels, 1)
        
        # Iterative blocks
        self.blocks = nn.ModuleList([
            MaskDiffusionBlock(hidden_channels) for _ in range(num_iterations)
        ])
        
        # Final Readout: Sharp Mask Logits
        self.readout = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, coarse_logits: Tensor, features: Tensor) -> Tensor:
        """
        Args:
            coarse_logits: [B, 1, H, W] initial coarse mask.
            features: [B, C, H, W] backbone features (usually P2 or P3).
            
        Returns:
            refined_logits: [B, 1, H, W] denoised sharp mask.
        """
        # Interpolate logits to match feature resolution if needed
        if coarse_logits.shape[-2:] != features.shape[-2:]:
            coarse_logits = F.interpolate(
                coarse_logits, 
                size=features.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
        # Concatenate and Project
        x = torch.cat([coarse_logits, features], dim=1)
        x = self.proj(x)
        
        # Iterative 'Denoising'
        # In a real diffusion model, we'd add noise and step through timesteps.
        # Here we perform 'Direct Denoising' as a learned iterative update.
        for block in self.blocks:
            x = block(x)
            
        # Final Sharpening
        delta = self.readout(x)
        
        # Residual update to coarse logits
        return coarse_logits + delta
