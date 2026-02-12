"""Learnable image enhancement for low-quality satellite imagery.

This module provides trainable preprocessing to improve Google Maps imagery
before feeding to the backbone. Handles:
- JPEG compression artifacts
- Varying brightness/contrast
- Shadow/illumination issues
- Atmospheric haze

Expected impact: +3-5% mAP on satellite images
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LearnableImageEnhancer(nn.Module):
    """Trainable image enhancement for satellite imagery.
    
    Applies learned denoising, color correction, and sharpening to improve
    low-quality satellite images from Google Maps.
    
    Architecture:
    - Denoising branch: Residual conv network
    - Color correction: Per-channel learned curves
    - Sharpening: High-pass filter with learnable strength
    
    Args:
        enhance_denoise: Enable denoising branch
        enhance_color: Enable color correction
        enhance_sharpen: Enable sharpening
        denoise_channels: Hidden channels for denoising
    """
    
    def __init__(
        self,
        *,
        enhance_denoise: bool = True,
        enhance_color: bool = True,
        enhance_sharpen: bool = True,
        denoise_channels: int = 32,
    ) -> None:
        super().__init__()
        
        self.enhance_denoise = enhance_denoise
        self.enhance_color = enhance_color
        self.enhance_sharpen = enhance_sharpen
        
        # Denoising branch (residual)
        if self.enhance_denoise:
            self.denoise_conv1 = nn.Conv2d(3, denoise_channels, 3, padding=1, bias=False)
            self.denoise_bn1 = nn.BatchNorm2d(denoise_channels)
            self.denoise_conv2 = nn.Conv2d(denoise_channels, denoise_channels, 3, padding=1, bias=False)
            self.denoise_bn2 = nn.BatchNorm2d(denoise_channels)
            self.denoise_conv3 = nn.Conv2d(denoise_channels, 3, 3, padding=1, bias=True)
            
        # Color correction (learned curves)
        if self.enhance_color:
            # 3 MLP for RGB channel correction
            self.color_mlp = nn.Sequential(
                nn.Conv2d(3, 16, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 1, bias=True),
                nn.Sigmoid(),  # Scale factor in [0, 2]
            )
            # Learnable bias for brightness adjustment
            self.color_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
            
        # Sharpening (high-pass filter with learned strength)
        if self.enhance_sharpen:
            # Learnable sharpening strength
            self.sharpen_strength = nn.Parameter(torch.tensor(0.5))
            
    def forward(self, x: Tensor) -> Tensor:
        """Apply learned enhancements.
        
        Args:
            x: Input image tensor [B, 3, H, W] in [0, 1]
            
        Returns:
            Enhanced image [B, 3, H, W] in [0, 1]
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {x.shape}")
            
        enhanced = x
        
        # 1. Denoising (additive residual)
        if self.enhance_denoise:
            residual = F.relu(self.denoise_bn1(self.denoise_conv1(enhanced)))
            residual = F.relu(self.denoise_bn2(self.denoise_conv2(residual)))
            residual = self.denoise_conv3(residual)
            enhanced = enhanced + residual * 0.1  # Scale down residual
            
        # 2. Color correction (multiplicative + additive)
        if self.enhance_color:
            # Multiplicative (learned curves)
            color_scale = self.color_mlp(enhanced) * 2.0  # [0, 2] range
            enhanced = enhanced * color_scale
            
            # Additive (brightness)
            enhanced = enhanced + self.color_bias
            
        # 3. Sharpening (Laplacian-based)
        if self.enhance_sharpen:
            # Compute Laplacian (high-pass filter)
            kernel = enhanced.new_tensor([
                [[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]]
            ]).view(1, 1, 3, 3) / 4.0
            
            # Apply to each channel
            laplacian = F.conv2d(
                enhanced.reshape(-1, 1, *enhanced.shape[2:]),
                kernel,
                padding=1
            ).reshape(enhanced.shape)
            
            # Add weighted high-frequency component
            strength = torch.sigmoid(self.sharpen_strength)  # [0, 1]
            enhanced = enhanced + laplacian * strength * 0.2
            
        # Clamp to valid range
        enhanced = enhanced.clamp(0.0, 1.0)
        
        return enhanced
    
    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """Freeze all parameters (for inference)."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self) -> None:
        """Unfreeze all parameters (for training)."""
        for param in self.parameters():
            param.requires_grad = True


class SatelliteImagePreprocessor(nn.Module):
    """Combined preprocessing pipeline for satellite images.
    
    Combines the learnable enhancer with normalization.
    
    Args:
        use_enhancement: Enable learnable enhancement
        mean: ImageNet normalization mean
        std: ImageNet normalization std
    """
    
    def __init__(
        self,
        *,
        use_enhancement: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        
        self.use_enhancement = use_enhancement
        
        if self.use_enhancement:
            self.enhancer = LearnableImageEnhancer()
        
        # Normalization
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        """Preprocess satellite image.
        
        Args:
            x: Input [B, 3, H, W] in [0, 1]
            
        Returns:
            Preprocessed [B, 3, H, W] (normalized)
        """
        # 1. Enhancement (if enabled)
        if self.use_enhancement:
            x = self.enhancer(x)
            
        # 2. Normalization
        x = (x - self.mean) / self.std
        
        return x
    
    def denormalize(self, x: Tensor) -> Tensor:
        """Reverse normalization for visualization.
        
        Args:
            x: Normalized tensor
            
        Returns:
            Denormalized tensor in [0, 1]
        """
        x = x * self.std + self.mean
        return x.clamp(0.0, 1.0)


__all__ = [
    "LearnableImageEnhancer",
    "SatelliteImagePreprocessor",
]
