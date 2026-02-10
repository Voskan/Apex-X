from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from .cheap_block import CheapBlock


class PVBackbone(nn.Module):
    """Peripheral-vision backbone producing pyramid features P3/P4/P5."""

    def __init__(
        self,
        in_channels: int = 3,
        p3_channels: int = 80,
        p4_channels: int = 160,
        p5_channels: int = 256,
        norm_groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if p3_channels <= 0 or p4_channels <= 0 or p5_channels <= 0:
            raise ValueError("p3_channels, p4_channels, and p5_channels must be > 0")
        if norm_groups <= 0:
            raise ValueError("norm_groups must be > 0")
        for channels in (p3_channels, p4_channels, p5_channels):
            if channels % norm_groups != 0:
                raise ValueError("norm_groups must divide each output channel count")

        self.in_channels = int(in_channels)
        self.p3_channels = int(p3_channels)
        self.p4_channels = int(p4_channels)
        self.p5_channels = int(p5_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.p3_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.p3_channels),
            nn.SiLU(inplace=False),
        )
        self.down_s4 = nn.Sequential(
            nn.Conv2d(
                self.p3_channels,
                self.p3_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.p3_channels),
            nn.SiLU(inplace=False),
        )
        self.down_s8 = nn.Sequential(
            nn.Conv2d(
                self.p3_channels,
                self.p3_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.p3_channels),
            nn.SiLU(inplace=False),
        )
        self.p3_block = CheapBlock(
            in_channels=self.p3_channels,
            out_channels=self.p3_channels,
            use_residual=True,
            norm_groups=norm_groups,
        )

        self.down_p4 = nn.Sequential(
            nn.Conv2d(
                self.p3_channels,
                self.p4_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.p4_channels),
            nn.SiLU(inplace=False),
        )
        self.p4_block = CheapBlock(
            in_channels=self.p4_channels,
            out_channels=self.p4_channels,
            use_residual=True,
            norm_groups=norm_groups,
        )

        self.down_p5 = nn.Sequential(
            nn.Conv2d(
                self.p4_channels,
                self.p5_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.p5_channels),
            nn.SiLU(inplace=False),
        )
        self.p5_block = CheapBlock(
            in_channels=self.p5_channels,
            out_channels=self.p5_channels,
            use_residual=True,
            norm_groups=norm_groups,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError("input must be [B,C,H,W]")
        if x.shape[1] != self.in_channels:
            raise ValueError("input channel dimension does not match in_channels")
        if x.shape[2] < 32 or x.shape[3] < 32:
            raise ValueError("input spatial size must be at least 32x32")

        s2 = self.stem(x)
        s4 = self.down_s4(s2)
        s8 = self.down_s8(s4)
        p3 = self.p3_block(s8)

        p4 = self.down_p4(p3)
        p4 = self.p4_block(p4)

        p5 = self.down_p5(p4)
        p5 = self.p5_block(p5)

        return {
            "P3": cast(Tensor, p3),
            "P4": cast(Tensor, p4),
            "P5": cast(Tensor, p5),
        }
