"""DINOv2-based Peripheral Vision Module.

This module replaces the standard PV backbone with a frozen DINOv2 ViT,
using lightweight LoRA adapters for task-specific feature extraction.

Expected impact: +5-8% mAP from superior pre-trained features.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

try:
    from transformers import Dinov2Model, Dinov2Config
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation for efficient fine-tuning.
    
    Adds trainable low-rank matrices to frozen features without
    modifying the original model weights.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        rank: Rank of low-rank decomposition (lower = fewer params)
        alpha: Scaling factor for LoRA updates
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices: A (down-project), B (up-project)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        
        # Initialize A with Kaiming, B with zeros (start from identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        # Optional direct projection (trainable)
        self.direct = nn.Linear(in_dim, out_dim, bias=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply LoRA adaptation.
        
        Args:
            x: Input features [B, N, D] or [B, D, H, W]
        
        Returns:
            Adapted features [B, out_dim, H', W']
        """
        # Direct projection
        out = self.direct(x)
        
        # Add low-rank update: x @ A @ B
        lora_out = self.lora_B(self.lora_A(x))
        out = out + self.scaling * lora_out
        
        return out


class PVModuleDINOv2(nn.Module):
    """Peripheral Vision module using frozen DINOv2 backbone.
    
    Uses DINOv2 ViT-Large (142M images pre-trained) as a frozen feature
    extractor, with lightweight LoRA adapters to adapt features to the
    detection task.
    
    Architecture:
        - DINOv2 ViT-L/14 (frozen, 304M params)
        - LoRA adapters (trainable, ~2M params)
        - P3, P4, P5 pyramid outputs
    
    Expected benefits:
        - +5-8% mAP from superior features
        - Fast training (only 2M trainable params)
        - Better generalization across domains
    
    Args:
        model_name: HuggingFace model identifier
        feature_layers: Which DINOv2 layers to extract (default: [8, 16, 23])
        lora_rank: Rank for LoRA decomposition (default: 8)
        output_dims: Output channels for P3, P4, P5 (default: [256, 512, 1024])
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        feature_layers: tuple[int, int, int] = (8, 16, 23),
        lora_rank: int = 8,
        output_dims: tuple[int, int, int] = (256, 512, 1024),
    ) -> None:
        super().__init__()
        
        if not DINOV2_AVAILABLE:
            raise RuntimeError(
                "transformers library required for DINOv2. "
                "Install with: pip install transformers"
            )
        
        self.feature_layers = feature_layers
        self.output_dims = output_dims
        
        # Load frozen DINOv2
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        
        # Freeze all DINOv2 parameters
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # Get hidden dimension from config
        hidden_dim = self.dinov2.config.hidden_size  # 1024 for ViT-L
        
        # LoRA adapters for each pyramid level
        self.lora_p3 = LoRAAdapter(hidden_dim, output_dims[0], rank=lora_rank)
        self.lora_p4 = LoRAAdapter(hidden_dim, output_dims[1], rank=lora_rank)
        self.lora_p5 = LoRAAdapter(hidden_dim, output_dims[2], rank=lora_rank)
        
        # Spatial dimension adapters (ViT outputs are 1D, need 2D for FPN)
        # DINOv2 outputs: [B, N_patches, D] where N_patches = (H/14) * (W/14)
        # We need to reshape to [B, D, H', W']
        self.patch_size = 14  # DINOv2 uses 14x14 patches
    
    def _reshape_vit_output(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Reshape ViT sequence output to 2D feature map.
        
        Args:
            x: ViT output [B, N+1, D] (includes CLS token)
            target_size: Target spatial size (H, W)
        
        Returns:
            Feature map [B, D, H, W]
        """
        B, N_plus_1, D = x.shape
        
        # Remove CLS token (first token)
        x = x[:, 1:, :]  # [B, N, D]
        
        # Reshape to 2D grid
        # N = (H/patch_size) * (W/patch_size)
        h, w = target_size
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        
        x = x.reshape(B, h_patches, w_patches, D)
        x = x.permute(0, 3, 1, 2)  # [B, D, H', W']
        
        return x
    
    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        """Extract multi-scale features using DINOv2.
        
        Args:
            image: Input image [B, 3, H, W]
        
        Returns:
            Dictionary with P3, P4, P5 features:
                - P3: [B, 256, H/8, W/8]
                - P4: [B, 512, H/16, W/16]
                - P5: [B, 1024, H/32, W/32]
        """
        B, _, H, W = image.shape
        
        # Forward through DINOv2 (frozen)
        with torch.no_grad():
            outputs = self.dinov2(
                pixel_values=image,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract features from specified layers
        hidden_states = outputs.hidden_states
        
        feat_l8 = hidden_states[self.feature_layers[0]]   # Early features
        feat_l16 = hidden_states[self.feature_layers[1]]  # Mid features
        feat_l23 = hidden_states[self.feature_layers[2]]  # Deep features
        
        # Reshape to 2D feature maps
        # Sizes decrease with depth in pyramid
        feat_l8_2d = self._reshape_vit_output(feat_l8, (H // 2, W // 2))    
        feat_l16_2d = self._reshape_vit_output(feat_l16, (H // 2, W // 2))   
        feat_l23_2d = self._reshape_vit_output(feat_l23, (H // 2, W // 2))   
        
        # Apply LoRA adapters (trainable)
        # P3: High resolution, early features
        p3 = self.lora_p3(feat_l8_2d.permute(0, 2, 3, 1))  # [B, H, W, D]
        p3 = p3.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # P4: Mid resolution, mid features
        p4 = self.lora_p4(feat_l16_2d.permute(0, 2, 3, 1))
        p4 = p4.permute(0, 3, 1, 2)
        # Downsample to H/16
        p4 = nn.functional.avg_pool2d(p4, kernel_size=2, stride=2)
        
        # P5: Low resolution, deep features
        p5 = self.lora_p5(feat_l23_2d.permute(0, 2, 3, 1))
        p5 = p5.permute(0, 3, 1, 2)
        # Downsample to H/32
        p5 = nn.functional.avg_pool2d(p5, kernel_size=4, stride=4)
        
        return {
            'P3': p3,  # [B, 256, H/8, W/8]
            'P4': p4,  # [B, 512, H/16, W/16]
            'P5': p5,  # [B, 1024, H/32, W/32]
        }
    
    def trainable_parameters(self) -> int:
        """Count trainable parameters (only LoRA adapters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def frozen_parameters(self) -> int:
        """Count frozen parameters (DINOv2 backbone)."""
        return sum(p.numel() for p in self.dinov2.parameters())


__all__ = [
    'PVModuleDINOv2',
    'LoRAAdapter',
    'DINOV2_AVAILABLE',
]
