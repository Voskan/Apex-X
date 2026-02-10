from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PVCoarseOutput:
    """PV coarse maps used by routing and diagnostics."""

    objectness_logits: Tensor  # [B,1,H,W]
    objectness: Tensor  # [B,1,H,W], sigmoid(logits)
    boundary_proxy: Tensor  # [B,1,H,W], sigmoid(boundary_logits)
    variance_proxy: Tensor  # [B,1,H,W], softplus(var_logits)
    uncertainty_proxy: Tensor  # [B,1,H,W], 4*p*(1-p)


class PVCoarseHeads(nn.Module):
    """Coarse PV heads for objectness, boundary, variance, and uncertainty proxies."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        hidden = in_channels if hidden_channels is None else int(hidden_channels)
        if hidden <= 0:
            raise ValueError("hidden_channels must be > 0")

        self.in_channels = int(in_channels)
        self.hidden_channels = hidden

        self.trunk = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=hidden),
            nn.SiLU(inplace=False),
        )
        self.objectness_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
        self.boundary_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
        self.variance_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

    @staticmethod
    def uncertainty_from_objectness(objectness: Tensor) -> Tensor:
        """Uncertainty proxy from objectness probability.

        Definition:
          u_hat = 4 * p * (1 - p), where p in [0,1].
        This is normalized Bernoulli variance:
        - max uncertainty at p=0.5 -> u_hat=1
        - min uncertainty at p in {0,1} -> u_hat=0
        """
        return 4.0 * objectness * (1.0 - objectness)

    def _select_feature(self, features: dict[str, Tensor], level: str) -> Tensor:
        if level not in features:
            expected = ", ".join(sorted(features.keys()))
            raise ValueError(f"missing feature level {level!r}; available: {expected}")
        x = features[level]
        if x.ndim != 4:
            raise ValueError("selected feature map must be [B,C,H,W]")
        if x.shape[1] != self.in_channels:
            raise ValueError("feature channels do not match in_channels")
        return x

    def forward(self, features: dict[str, Tensor], level: str = "P4") -> PVCoarseOutput:
        if not isinstance(features, dict):
            raise ValueError("features must be a dict with keys like P3/P4/P5")

        x = self._select_feature(features, level=level)
        h = self.trunk(x)
        objectness_logits = self.objectness_head(h)
        boundary_logits = self.boundary_head(h)
        variance_logits = self.variance_head(h)

        objectness = torch.sigmoid(objectness_logits)
        boundary_proxy = torch.sigmoid(boundary_logits)
        variance_proxy = torch.nn.functional.softplus(variance_logits)
        uncertainty_proxy = self.uncertainty_from_objectness(objectness)

        return PVCoarseOutput(
            objectness_logits=objectness_logits,
            objectness=objectness,
            boundary_proxy=boundary_proxy,
            variance_proxy=variance_proxy,
            uncertainty_proxy=uncertainty_proxy,
        )
