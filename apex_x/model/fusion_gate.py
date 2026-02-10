from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionGate(nn.Module):
    """Fuse base/heavy features using proxy-conditioned spatial alpha."""

    def __init__(
        self,
        init_boundary_weight: float = 1.0,
        init_uncertainty_weight: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.boundary_log_weight = nn.Parameter(torch.tensor(float(init_boundary_weight)))
        self.uncertainty_log_weight = nn.Parameter(torch.tensor(float(init_uncertainty_weight)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def _prepare_proxy(self, proxy: Tensor, *, name: str, like: Tensor) -> Tensor:
        if proxy.ndim == 3:
            proxy = proxy.unsqueeze(1)
        if proxy.ndim != 4:
            raise ValueError(f"{name} must be [B,1,H,W] or [B,H,W]")
        if proxy.shape[1] != 1:
            raise ValueError(f"{name} channel dimension must be 1")
        if proxy.shape[0] != like.shape[0] or proxy.shape[2:] != like.shape[2:]:
            raise ValueError(f"{name} must match base/heavy batch and spatial shape")

        return torch.nan_to_num(
            proxy.to(dtype=like.dtype, device=like.device),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def compute_alpha(
        self,
        boundary_proxy: Tensor,
        uncertainty_proxy: Tensor,
        like: Tensor,
    ) -> Tensor:
        boundary = self._prepare_proxy(boundary_proxy, name="boundary_proxy", like=like)
        uncertainty = self._prepare_proxy(uncertainty_proxy, name="uncertainty_proxy", like=like)
        boundary_w = torch.nn.functional.softplus(self.boundary_log_weight)
        uncertainty_w = torch.nn.functional.softplus(self.uncertainty_log_weight)
        logits = boundary_w * boundary + uncertainty_w * uncertainty + self.bias
        return torch.sigmoid(logits)

    def forward(
        self,
        base_features: Tensor,
        heavy_features: Tensor,
        boundary_proxy: Tensor,
        uncertainty_proxy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_features.ndim != 4 or heavy_features.ndim != 4:
            raise ValueError("base_features and heavy_features must be [B,C,H,W]")
        if base_features.shape != heavy_features.shape:
            raise ValueError("base_features and heavy_features must have same shape")

        alpha = self.compute_alpha(boundary_proxy, uncertainty_proxy, like=base_features)
        fused = base_features + alpha * (heavy_features - base_features)
        return fused, alpha
