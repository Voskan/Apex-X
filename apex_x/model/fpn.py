from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as f

from .cheap_block import CheapBlock


@dataclass(frozen=True)
class DualPathFPNOutput:
    """Dual-path FPN outputs combining PV pyramid and FF high-resolution features."""

    pyramid: dict[str, Tensor]  # keys: P3/P4/P5
    ff_aligned: Tensor  # [B,C_out,H3,W3] projected FF feature at P3 resolution


class DualPathFPN(nn.Module):
    """FPN that fuses PV low-res features with FF high-res detail.

    Inputs:
    - pv_features: dict with keys P3/P4/P5 from PV stream
    - ff_high: [B,C_ff,Hf,Wf] high-resolution FF feature/detail map

    Output:
    - pyramid dict with fused P3/P4/P5 at `out_channels`
    """

    def __init__(
        self,
        pv_p3_channels: int,
        pv_p4_channels: int,
        pv_p5_channels: int,
        ff_channels: int,
        out_channels: int = 160,
        *,
        norm_groups: int = 1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("pv_p3_channels", pv_p3_channels),
            ("pv_p4_channels", pv_p4_channels),
            ("pv_p5_channels", pv_p5_channels),
            ("ff_channels", ff_channels),
            ("out_channels", out_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if norm_groups <= 0:
            raise ValueError("norm_groups must be > 0")
        if out_channels % norm_groups != 0:
            raise ValueError("norm_groups must divide out_channels")

        self.pv_p3_channels = int(pv_p3_channels)
        self.pv_p4_channels = int(pv_p4_channels)
        self.pv_p5_channels = int(pv_p5_channels)
        self.ff_channels = int(ff_channels)
        self.out_channels = int(out_channels)

        self.p3_lateral = nn.Conv2d(
            self.pv_p3_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.p4_lateral = nn.Conv2d(
            self.pv_p4_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.p5_lateral = nn.Conv2d(
            self.pv_p5_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.ff_lateral = nn.Conv2d(
            self.ff_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

        self.p3_smooth = CheapBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            use_residual=True,
            norm_groups=norm_groups,
        )
        self.p4_smooth = CheapBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            use_residual=True,
            norm_groups=norm_groups,
        )
        self.p5_smooth = CheapBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            use_residual=True,
            norm_groups=norm_groups,
        )

    def _validate_pv_features(
        self,
        pv_features: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        required = ("P3", "P4", "P5")
        missing = [key for key in required if key not in pv_features]
        if missing:
            raise ValueError(f"pv_features is missing required keys: {missing}")

        p3 = pv_features["P3"]
        p4 = pv_features["P4"]
        p5 = pv_features["P5"]
        for name, feature, channels in (
            ("P3", p3, self.pv_p3_channels),
            ("P4", p4, self.pv_p4_channels),
            ("P5", p5, self.pv_p5_channels),
        ):
            if feature.ndim != 4:
                raise ValueError(f"pv_features[{name!r}] must be [B,C,H,W]")
            if feature.shape[1] != channels:
                raise ValueError(f"pv_features[{name!r}] channel dim must be {channels}")

        bsz = p3.shape[0]
        if p4.shape[0] != bsz or p5.shape[0] != bsz:
            raise ValueError("all pv feature maps must share batch size")
        return p3, p4, p5

    def forward(self, pv_features: dict[str, Tensor], ff_high: Tensor) -> DualPathFPNOutput:
        p3_in, p4_in, p5_in = self._validate_pv_features(pv_features)
        if ff_high.ndim != 4:
            raise ValueError("ff_high must be [B,C,H,W]")
        if ff_high.shape[0] != p3_in.shape[0]:
            raise ValueError("ff_high batch dimension must match PV features")
        if ff_high.shape[1] != self.ff_channels:
            raise ValueError(f"ff_high channel dimension must be {self.ff_channels}")

        p5_lat = self.p5_lateral(p5_in)
        p4_lat = self.p4_lateral(p4_in)
        p3_lat = self.p3_lateral(p3_in)

        ff_proj = self.ff_lateral(ff_high)
        ff_aligned = f.interpolate(
            ff_proj,
            size=p3_lat.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        p5_td = p5_lat
        p4_td = p4_lat + f.interpolate(p5_td, size=p4_lat.shape[2:], mode="nearest")
        p3_td = p3_lat + f.interpolate(p4_td, size=p3_lat.shape[2:], mode="nearest") + ff_aligned

        p3 = self.p3_smooth(p3_td)
        p4 = self.p4_smooth(p4_td)
        p5 = self.p5_smooth(p5_td)

        return DualPathFPNOutput(
            pyramid={
                "P3": p3,
                "P4": p4,
                "P5": p5,
            },
            ff_aligned=ff_aligned,
        )
