"""Gradient checkpointing utilities for memory-efficient training.

Wraps model components with gradient checkpointing to reduce memory usage
during backpropagation. Essential for training on 1024x1024 images.

Expected benefit: 50% memory reduction with ~15% slowdown.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class CheckpointedSequential(nn.Sequential):
    """Sequential module with gradient checkpointing.
    
    Wraps nn.Sequential to use gradient checkpointing during forward pass.
    
    Args:
        *args: Modules to wrap
        checkpoint_segments: Number of checkpoint segments (more = less memory)
    """
    
    def __init__(self, *args: nn.Module, checkpoint_segments: int = 1) -> None:
        super().__init__(*args)
        self.checkpoint_segments = checkpoint_segments
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward with gradient checkpointing.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if not self.training or self.checkpoint_segments == 0:
            # No checkpointing during inference
            return super().forward(x)
            
        # Split modules into segments
        num_modules = len(self)
        segment_size = max(1, num_modules // self.checkpoint_segments)
        
        for i in range(0, num_modules, segment_size):
            segment_modules = list(self)[i:i+segment_size]
            
            if len(segment_modules) == 0:
                continue
                
            # Create forward function for this segment
            def segment_forward(input_tensor: Tensor) -> Tensor:
                for module in segment_modules:
                    input_tensor = module(input_tensor)
                return input_tensor
                
            # Apply checkpointing
            x = checkpoint(segment_forward, x, use_reentrant=False)
            
        return x


def checkpoint_wrapper(
    module: nn.Module,
    *,
    enabled: bool = True,
) -> nn.Module:
    """Wrap a module with gradient checkpointing.
    
    Args:
        module: Module to wrap
        enabled: Whether to enable checkpointing
        
    Returns:
        Wrapped module (or original if not enabled)
    """
    if not enabled:
        return module
        
    class CheckpointedModule(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner
            
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if not self.training:
                return self.inner(*args, **kwargs)
                
            # Wrap forward call with checkpoint
            def forward_fn(*inputs: Any) -> Any:
                return self.inner(*inputs, **kwargs)
                
            return checkpoint(forward_fn, *args, use_reentrant=False)
            
    return CheckpointedModule(module)


def enable_gradient_checkpointing(
    model: nn.Module,
    module_types: tuple[type, ...] | None = None,
) -> None:
    """Enable gradient checkpointing for specific module types in model.
    
    Recursively finds and wraps modules of specified types.
    
    Args:
        model: Model to modify
        module_types: Types of modules to checkpoint (default: Conv2d, Linear)
    """
    if module_types is None:
        module_types = (nn.Conv2d, nn.Linear)
        
    def recursive_wrap(module: nn.Module, name: str = '') -> None:
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, module_types):
                # Wrap this module
                wrapped = checkpoint_wrapper(child, enabled=True)
                setattr(module, child_name, wrapped)
            else:
                # Recurse into children
                recursive_wrap(child, full_name)
                
    recursive_wrap(model)


class GradientCheckpointConfig:
    """Configuration for gradient checkpointing strategy.
    
    Attributes:
        enabled: Global enable/disable
        checkpoint_backbone: Checkpoint backbone conv layers
        checkpoint_fpn: Checkpoint FPN layers
        checkpoint_heads: Checkpoint detection/segmentation heads
        segment_size: Number of layers per checkpoint segment
    """
    
    def __init__(
        self,
        *,
        enabled: bool = True,
        checkpoint_backbone: bool = True,
        checkpoint_fpn: bool = True,
        checkpoint_heads: bool = False,  # Heads are usually small
        segment_size: int = 4,
    ) -> None:
        self.enabled = enabled
        self.checkpoint_backbone = checkpoint_backbone
        self.checkpoint_fpn = checkpoint_fpn
        self.checkpoint_heads = checkpoint_heads
        self.segment_size = segment_size
        
    def apply_to_model(self, model: nn.Module) -> None:
        """Apply checkpointing configuration to model.
        
        Args:
            model: Model to configure
        """
        if not self.enabled:
            return
            
        # Checkpoint backbone if enabled
        if self.checkpoint_backbone and hasattr(model, 'pv_module'):
            if hasattr(model.pv_module, 'backbone'):
                enable_gradient_checkpointing(
                    model.pv_module.backbone,
                    module_types=(nn.Conv2d,)
                )
                
        # Checkpoint FPN if enabled
        if self.checkpoint_fpn and hasattr(model, 'fpn'):
            enable_gradient_checkpointing(
                model.fpn,
                module_types=(nn.Conv2d,)
            )
            
        # Checkpoint heads if enabled
        if self.checkpoint_heads:
            if hasattr(model, 'det_head'):
                enable_gradient_checkpointing(
                    model.det_head,
                    module_types=(nn.Conv2d, nn.Linear)
                )
            if hasattr(model, 'seg_head'):
                enable_gradient_checkpointing(
                    model.seg_head,
                    module_types=(nn.Conv2d, nn.Linear)
                )


__all__ = [
    "CheckpointedSequential",
    "checkpoint_wrapper",
    "enable_gradient_checkpointing",
    "GradientCheckpointConfig",
]
