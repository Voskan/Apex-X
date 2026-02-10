from __future__ import annotations

import torch
from torch import Tensor, nn


class TileRefineBlock(nn.Module):
    """Local refine block for packed tiles [B,K,C,t,t]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        kernel_size: int = 3,
        use_residual: bool = True,
        norm_groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        out_c = in_channels if out_channels is None else int(out_channels)
        if out_c <= 0:
            raise ValueError("out_channels must be > 0")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if norm_groups <= 0:
            raise ValueError("norm_groups must be > 0")
        if (2 * out_c) % norm_groups != 0:
            raise ValueError("norm_groups must divide 2*out_channels")

        self.in_channels = int(in_channels)
        self.out_channels = out_c
        self.use_residual = bool(use_residual)

        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            self.in_channels,
            2 * self.out_channels,
            kernel_size=1,
            bias=True,
        )
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

    def forward(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 5:
            raise ValueError("tiles must be [B,K,C,t,t]")
        if tiles.shape[2] != self.in_channels:
            raise ValueError("tiles channel dimension does not match in_channels")

        bsz, k_tiles, channels, tile_h, tile_w = tiles.shape
        tiles_flat = tiles.reshape(bsz * k_tiles, channels, tile_h, tile_w)

        y = self.depthwise(tiles_flat)
        y = self.pointwise(y)
        y = self.norm(y)
        a, b = torch.chunk(y, 2, dim=1)
        y = a * torch.relu(b)

        if self.use_residual:
            residual = tiles_flat if self.residual_proj is None else self.residual_proj(tiles_flat)
            y = y + residual

        return y.reshape(bsz, k_tiles, self.out_channels, tile_h, tile_w)
