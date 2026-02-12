"""CPU training support utilities.

Provides fallback logic for training on CPU when CUDA is unavailable.
"""

from __future__ import annotations

import torch
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Get training device with automatic fallback.
    
    Args:
        device_str: 'auto', 'cuda', 'cpu', or 'cuda:N'
        
    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            LOGGER.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            LOGGER.warning("CUDA not available, falling back to CPU training")
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device_str)
            LOGGER.info(f"Using {device_str}")
        else:
            LOGGER.error(f"CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
   else:
        device = torch.device(device_str)
        LOGGER.info(f"Using device: {device_str}")
    
    return device


def should_use_amp(device: torch.device, force_amp: bool = False) -> bool:
    """Determine if AMP should be used.
    
    AMP (Automatic Mixed Precision) only works on CUDA devices.
    
    Args:
        device: Training device
        force_amp: Force AMP even on CPU (will be ignored)
        
    Returns:
        True if AMP should be enabled
    """
    if device.type == "cpu":
        if force_amp:
            LOGGER.warning("AMP requested but not supported on CPU, disabling")
        return False
    
    # CUDA device
    return True


__all__ = ['get_device', 'should_use_amp']
