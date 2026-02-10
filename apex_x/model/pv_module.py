from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .pv_backbone import PVBackbone
from .pv_coarse_heads import PVCoarseHeads, PVCoarseOutput


@dataclass(frozen=True)
class PVModuleOutput:
    """Output bundle for the PV stream."""

    features: dict[str, Tensor]  # keys: P3/P4/P5
    coarse: PVCoarseOutput
    proxy_maps: dict[str, Tensor]  # keys: objectness/uncertainty/boundary/variance


class PVModule(nn.Module):
    """PV module wiring backbone features to coarse proxy heads."""

    def __init__(
        self,
        in_channels: int = 3,
        p3_channels: int = 80,
        p4_channels: int = 160,
        p5_channels: int = 256,
        norm_groups: int = 1,
        coarse_level: str = "P4",
        coarse_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if coarse_level not in {"P3", "P4", "P5"}:
            raise ValueError("coarse_level must be one of: P3, P4, P5")

        self.coarse_level = coarse_level
        self.backbone = PVBackbone(
            in_channels=in_channels,
            p3_channels=p3_channels,
            p4_channels=p4_channels,
            p5_channels=p5_channels,
            norm_groups=norm_groups,
        )

        level_channels = {
            "P3": p3_channels,
            "P4": p4_channels,
            "P5": p5_channels,
        }
        self.coarse_heads = PVCoarseHeads(
            in_channels=int(level_channels[coarse_level]),
            hidden_channels=coarse_hidden_channels,
        )

    def forward(self, x: Tensor) -> PVModuleOutput:
        features = self.backbone(x)
        coarse = self.coarse_heads(features, level=self.coarse_level)
        proxy_maps = {
            "objectness": coarse.objectness,
            "uncertainty": coarse.uncertainty_proxy,
            "boundary": coarse.boundary_proxy,
            "variance": coarse.variance_proxy,
        }
        return PVModuleOutput(features=features, coarse=coarse, proxy_maps=proxy_maps)
