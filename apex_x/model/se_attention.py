"""Squeeze-and-Excitation (SE) attention module for FPN enhancement.

Adds lightweight channel attention to improve feature discrimination.
Expected impact: +1-2% mAP with minimal compute overhead.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    Applies global pooling + FC layers to recalibrate channel responses.
    Widely used in EfficientNet, ResNet-SE, and modern detectors.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
        activation: Activation function (default: 'relu')
    
    Reference: https://arxiv.org/abs/1709.01507
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        
        reduced_channels = max(channels // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply SE attention.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Squeeze: global average pooling
        squeezed = self.squeeze(x).view(B, C)
        
        # Excitation: channel-wise gating
        attention = self.excitation(squeezed).view(B, C, 1, 1)
        
        # Scale input by attention
        return x * attention


class SEDualPathFPN(nn.Module):
    """DualPathFPN enhanced with SE attention blocks.
    
    Adds SE blocks after each FPN level to improve feature quality.
    This is a drop-in replacement for standard DualPathFPN.
    
    Args:
        Same as DualPathFPN, plus:
        se_reduction: SE channel reduction ratio (default: 16)
        se_layers: Which layers to add SE ('all', 'output', or list of levels)
    
    Expected impact: +1-2% mAP
    """
    
    def __init__(
        self,
        base_fpn: nn.Module,
        se_reduction: int = 16,
        se_layers: str | list[str] = 'output',
    ) -> None:
        super().__init__()
        
        self.base_fpn = base_fpn
        self.se_layers = se_layers
        
        # Add SE blocks for specified layers
        self.se_blocks = nn.ModuleDict()
        
        if se_layers == 'all':
            # Add SE to all FPN outputs
            layer_names = ['P3', 'P4', 'P5', 'P6', 'P7']
        elif se_layers == 'output':
            # Only output layers (P3, P4, P5)
            layer_names = ['P3', 'P4', 'P5']
        else:
            # Custom list
            layer_names = se_layers
        
        # Assume standard FPN channel dims
        channel_dims = {
            'P3': 256,
            'P4': 256,
            'P5': 256,
            'P6': 256,
            'P7': 256,
        }
        
        for layer in layer_names:
            if layer in channel_dims:
                self.se_blocks[layer] = SEBlock(
                    channels=channel_dims[layer],
                    reduction=se_reduction,
                )
    
    def forward(self, *args, **kwargs):
        """Forward pass with SE attention.
        
        Returns same output format as base FPN, with SE-enhanced features.
        """
        # Forward through base FPN
        outputs = self.base_fpn(*args, **kwargs)
        
        # Apply SE blocks to specified layers
        if isinstance(outputs, dict):
            # Dictionary output (level -> features)
            for level, se_block in self.se_blocks.items():
                if level in outputs:
                    outputs[level] = se_block(outputs[level])
        else:
            # Assume tuple/list output, apply to first few
            outputs_list = list(outputs) if isinstance(outputs, tuple) else outputs
            for i, level in enumerate(['P3', 'P4', 'P5', 'P6', 'P7']):
                if level in self.se_blocks and i < len(outputs_list):
                    outputs_list[i] = self.se_blocks[level](outputs_list[i])
            outputs = tuple(outputs_list) if isinstance(outputs, tuple) else outputs_list
        
        return outputs


__all__ = ['SEBlock', 'SEDualPathFPN']
