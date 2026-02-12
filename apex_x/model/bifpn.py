"""BiFPN - Bi-directional Feature Pyramid Network.

Efficient bi-directional cross-scale connections and weighted feature fusion.
Better than standard FPN for multi-scale feature aggregation.

Expected gain: +1-2% AP over standard FPN

Reference:
    EfficientDet: Scalable and Efficient Object Detection
    https://arxiv.org/abs/1911.09070
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FastNormalizedFusion(nn.Module):
    """Fast normalized fusion with learnable weights.
    
    Learns to weight different feature levels during fusion.
    Uses ReLU + normalization for efficient weighted fusion.
    
    Args:
        num_inputs: Number of input feature maps to fuse
        eps: Small constant for numerical stability
    """
    
    def __init__(self, num_inputs: int = 2, eps: float = 1e-4):
        super().__init__()
        self.num_inputs = num_inputs
        self.eps = eps
        
        # Learnable fusion weights (initialized to 1.0)
        self.weights = nn.Parameter(torch.ones(num_inputs))
    
    def forward(self, *inputs: Tensor) -> Tensor:
        """Fuse multiple input features with learned weights.
        
        Args:
            *inputs: Variable number of input tensors [B, C, H, W]
            
        Returns:
            Fused features [B, C, H, W]
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Apply ReLU to weights (ensure non-negative)
        weights = F.relu(self.weights)
        
        # Normalize weights to sum to 1
        weights = weights / (weights.sum() + self.eps)
        
        # Weighted fusion
        fused = sum(w * x for w, x in zip(weights, inputs))
        
        return fused


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with top-down and bottom-up paths.
    
    Implements one layer of bi-directional feature pyramid:
    1. Top-down pathway (high-resolution features)
    2. Bottom-up pathway (semantic features)
    3. Weighted feature fusion at each level
    
    Args:
        channels: Number of channels for all feature levels
        num_levels: Number of pyramid levels (default: 5)
        use_separable_conv: Use separable convolutions for efficiency
    """
    
    def __init__(
        self,
        channels: int = 256,
        num_levels: int = 5,
        use_separable_conv: bool = True,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_levels = num_levels
        
        # Convolution type
        if use_separable_conv:
            conv_fn = self._make_separable_conv
        else:
            conv_fn = self._make_conv
        
        # Top-down pathway convolutions
        self.td_convs = nn.ModuleList([
            conv_fn(channels) for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway convolutions
        self.bu_convs = nn.ModuleList([
            conv_fn(channels) for _ in range(num_levels - 1)
        ])
        
        # Fast normalized fusion weights
        # For intermediate levels (need to fuse 2 inputs)
        self.td_fusions = nn.ModuleList([
            FastNormalizedFusion(num_inputs=2)
            for _ in range(num_levels - 1)
        ])
        
        # For bottom-up path (need to fuse 3 inputs: input, td, bu)
        self.bu_fusions = nn.ModuleList([
            FastNormalizedFusion(num_inputs=3)
            for _ in range(num_levels - 2)
        ])
    
    def _make_conv(self, channels: int) -> nn.Module:
        """Standard 3x3 convolution."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    
    def _make_separable_conv(self, channels: int) -> nn.Module:
        """Depthwise separable convolution (more efficient)."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            # Pointwise
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Forward pass through BiFPN layer.
        
        Args:
            features: List of feature maps [P3, P4, P5, P6, P7]
                     Each is [B, C, H, W] with decreasing spatial size
        
        Returns:
            List of fused feature maps
        """
        if len(features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature levels")
        
        # Top-down pathway
        td_features = [features[-1]]  # Start from coarsest level (P7)
        
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample higher level
            upsampled = F.interpolate(
                td_features[0],
                size=features[i].shape[-2:],
                mode='nearest',
            )
            
            # Fuse with current level
            fused = self.td_fusions[i](upsampled, features[i])
            
            # Apply convolution
            fused = self.td_convs[i](fused)
            
            td_features.insert(0, fused)
        
        # Bottom-up pathway
        bu_features = [td_features[0]]  # Start from finest level (P3)
        
        for i in range(self.num_levels - 1):
            # Downsample lower level
            if i == self.num_levels - 2:
                # Last level - only fuse 2 inputs
                downsampled = F.interpolate(
                    bu_features[-1],
                    size=features[i + 1].shape[-2:],
                    mode='nearest',
                )
                fused = self.td_fusions[i](downsampled, td_features[i + 1])
            else:
                # Intermediate levels - fuse 3 inputs
                downsampled = F.interpolate(
                    bu_features[-1],
                    size=features[i + 1].shape[-2:],
                    mode='nearest',
                )
                fused = self.bu_fusions[i](downsampled, features[i + 1], td_features[i + 1])
            
            # Apply convolution
            fused = self.bu_convs[i](fused)
            
            bu_features.append(fused)
        
        return bu_features


class BiFPN(nn.Module):
    """BiFPN - Bi-directional Feature Pyramid Network.
    
    Stacks multiple BiFPN layers for iterative multi-scale feature fusion.
    
    Expected AP gain: +1-2% over standard FPN
    
    Args:
        in_channels_list: List of input channels for each level
        out_channels: Unified output channels for all levels
        num_layers: Number of BiFPN layers to stack (default: 3)
        num_levels: Number of pyramid levels (default: 5)
    """
    
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        num_layers: int = 3,
        num_levels: int = 5,
    ):
        super().__init__()
        
        if len(in_channels_list) != num_levels:
            raise ValueError(f"in_channels_list length must equal num_levels ({num_levels})")
        
        self.num_levels = num_levels
        self.num_layers = num_layers
        
        # Input projection to unify channels
        self.input_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for in_ch in in_channels_list
        ])
        
        # Stack BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(
                channels=out_channels,
                num_levels=num_levels,
                use_separable_conv=True,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Forward pass through BiFPN.
        
        Args:
            features: List of feature maps from backbone
                     [C3, C4, C5, C6, C7] with different channels
        
        Returns:
            List of fused feature maps [P3, P4, P5, P6, P7]
                     All with unified channels
        """
        if len(features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} input features")
        
        # Project all inputs to same channel dimension
        features = [conv(feat) for conv, feat in zip(self.input_convs, features)]
        
        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
        
        return features


__all__ = ['BiFPN', 'BiFPNLayer', 'FastNormalizedFusion']
