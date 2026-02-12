"""Focal Frequency Loss for improved geometric regularization.

Focuses learning on high-frequency components (edges, textures) which
are often critical for accurate object localization and segmentation.

Reference: https://arxiv.org/abs/2012.12821
Expected impact: +1-2% AP, especially on complex scenes
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def focal_frequency_loss(
    pred: Tensor,
    target: Tensor,
    *,
    alpha: float = 1.0,
    patch_factor: int = 1,
    ave_spectrum: bool = False,
    log_matrix: bool = False,
    reduction: str = 'mean',
) -> Tensor:
    """Focal Frequency Loss comparing frequency spectra.
    
    Computes loss in frequency domain using 2D FFT. This emphasizes
    high-frequency components (edges, fine details) which are important
    for accurate localization.
    
    Args:
        pred: Predicted features or images [B, C, H, W]
        target: Target features or images [B, C, H, W]
        alpha: Spectrum weight scaling factor (default: 1.0)
        patch_factor: Patch size factor for patch-based spectrum (default: 1)
        ave_spectrum: Average spectrum across batch (default: False)
        log_matrix: Apply log to spectrum magnitudes (default: False)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    
    Returns:
        Frequency domain loss scalar
    """
    # Ensure same spatial size
    if pred.shape != target.shape:
        target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    
    # Convert to frequency domain via 2D FFT
    pred_freq = torch.fft.fft2(pred, norm='ortho')
    target_freq = torch.fft.fft2(target, norm='ortho')
    
    # Compute magnitude spectra
    pred_mag = torch.abs(pred_freq)
    target_mag = torch.abs(target_freq)
    
    # Optional: apply log to reduce dynamic range
    if log_matrix:
        pred_mag = torch.log(pred_mag + 1.0)
        target_mag = torch.log(target_mag + 1.0)
    
    # Optionally average spectrum across batch
    if ave_spectrum:
        pred_mag = pred_mag.mean(dim=0, keepdim=True)
        target_mag = target_mag.mean(dim=0, keepdim=True)
    
    # Compute frequency weight matrix (emphasize high frequencies)
    freq_weight = get_frequency_weight_matrix(
        pred_mag.shape,
        alpha=alpha,
        device=pred.device,
    )
    
    # Weighted L1 loss in frequency domain
    freq_loss = torch.abs(pred_mag - target_mag) * freq_weight
    
    # Apply patch factor if requested
    if patch_factor > 1:
        freq_loss = avg_pool_spectrum(freq_loss, patch_factor)
    
    # Reduction
    if reduction == 'mean':
        return freq_loss.mean()
    elif reduction == 'sum':
        return freq_loss.sum()
    else:
        return freq_loss


def get_frequency_weight_matrix(
    shape: tuple[int, ...],
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> Tensor:
    """Generate frequency weight matrix emphasizing high frequencies.
    
    Creates a radial weight matrix where higher frequencies (farther from DC)
    receive higher weights.
    
    Args:
        shape: Shape of frequency tensor [B, C, H, W]
        alpha: Scaling factor for frequency emphasis (default: 1.0)
        device: Device to create tensor on
    
    Returns:
        Frequency weight matrix [B, C, H, W]
    """
    _, _, h, w = shape
    
    # Create coordinate grids
    y_coords = torch.arange(h, device=device, dtype=torch.float32) - h // 2
    x_coords = torch.arange(w, device=device, dtype=torch.float32) - w // 2
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Radial distance from DC component (center)
    radius = torch.sqrt(x_grid ** 2 + y_grid ** 2)
    
    # Shift to match FFT layout (DC at corners)
    radius = torch.fft.ifftshift(radius)
    
    # Normalize to [0, 1]
    radius = radius / radius.max()
    
    # Apply exponential weighting: higher frequencies get more weight
    weight = torch.exp(alpha * radius)
    
    # Expand to match input shape
    weight = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    weight = weight.expand(shape)
    
    return weight


def avg_pool_spectrum(
    spectrum: Tensor,
    patch_factor: int,
) -> Tensor:
    """Average pool frequency spectrum for patch-based loss.
    
    Args:
        spectrum: Frequency spectrum [B, C, H, W]
        patch_factor: Pooling factor
    
    Returns:
        Pooled spectrum
    """
    return F.avg_pool2d(
        spectrum,
        kernel_size=patch_factor,
        stride=patch_factor,
    )


class FocalFrequencyLoss(torch.nn.Module):
    """Focal Frequency Loss module.
    
    Can be used standalone or combined with spatial losses.
    
    Args:
        alpha: Frequency emphasis factor (default: 1.0)
        patch_factor: Patch pooling factor (default: 1)
        ave_spectrum: Average spectrum across batch (default: False)
        log_matrix: Apply log to magnitudes (default: True)
        loss_weight: Overall loss weight (default: 1.0)
    
    Example:
        >>> ffl = FocalFrequencyLoss(alpha=1.0, log_matrix=True, loss_weight=0.1)
        >>> pred = model(x)
        >>> loss = ffl(pred, target)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = True,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.loss_weight = loss_weight
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute focal frequency loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
        
        Returns:
            Weighted frequency loss
        """
        loss = focal_frequency_loss(
            pred,
            target,
            alpha=self.alpha,
            patch_factor=self.patch_factor,
            ave_spectrum=self.ave_spectrum,
            log_matrix=self.log_matrix,
            reduction='mean',
        )
        
        return self.loss_weight * loss


__all__ = [
    'focal_frequency_loss',
    'get_frequency_weight_matrix',
    'FocalFrequencyLoss',
]
