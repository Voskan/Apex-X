from __future__ import annotations

import torch
from torch import Tensor, nn


class CheapBlock(nn.Module):
    """Lightweight conv block: 1x1 conv + norm + ReGLU + optional residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_residual: bool = True,
        norm_groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if norm_groups <= 0:
            raise ValueError("norm_groups must be > 0")
        if (2 * out_channels) % norm_groups != 0:
            raise ValueError("norm_groups must divide 2*out_channels")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_residual = bool(use_residual)

        self.conv = nn.Conv2d(self.in_channels, 2 * self.out_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=2 * self.out_channels)

        if self.use_residual and self.in_channels != self.out_channels:
            self.residual_proj: nn.Module | None = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.residual_proj = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("input must be [B,C,H,W]")
        if x.shape[1] != self.in_channels:
            raise ValueError("input channel dimension does not match in_channels")

        y = self.conv(x)
        y = self.norm(y)
        a, b = torch.chunk(y, 2, dim=1)
        y = a * torch.relu(b)

        if self.use_residual:
            residual = x if self.residual_proj is None else self.residual_proj(x)
            y = y + residual
        return y
