from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RouterKANOutput:
    """Router outputs per tile: utility U, split utility S, optional temporal keep T."""

    U: Tensor  # [B, K]
    S: Tensor  # [B, K]
    T: Tensor | None = None  # [B, K] when temporal head enabled


class LightweightSplineActivation(nn.Module):
    """Per-feature piecewise-linear spline with fixed knot positions.

    This is a compact KAN-like placeholder focused on numerical stability:
    - bounded input domain via clamp
    - identity initialization
    - linear interpolation on a fixed grid
    """

    def __init__(
        self,
        features: int,
        num_knots: int = 8,
        x_min: float = -3.0,
        x_max: float = 3.0,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be > 0")
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if x_max <= x_min:
            raise ValueError("x_max must be > x_min")

        self.features = features
        self.num_knots = num_knots
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.delta = (self.x_max - self.x_min) / float(self.num_knots - 1)

        knot_x = torch.linspace(self.x_min, self.x_max, self.num_knots, dtype=torch.float32)
        self.register_buffer("knot_x", knot_x, persistent=False)

        # Identity init keeps early training stable.
        knot_y = knot_x.unsqueeze(0).repeat(self.features, 1)
        self.knot_y = nn.Parameter(knot_y)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.features:
            raise ValueError(f"x last dim must equal features={self.features}")

        x_clean = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=self.x_max,
            neginf=self.x_min,
        )
        x_clip = x_clean.clamp(self.x_min, self.x_max)

        flat = x_clip.reshape(-1, self.features)
        positions = (flat - self.x_min) / self.delta
        idx0 = torch.floor(positions).to(torch.long).clamp(0, self.num_knots - 2)
        idx1 = idx0 + 1
        frac = positions - idx0.to(positions.dtype)

        knot_y_expanded = self.knot_y.unsqueeze(0).expand(flat.shape[0], -1, -1)
        y0 = torch.take_along_dim(knot_y_expanded, idx0.unsqueeze(-1), dim=-1).squeeze(-1)
        y1 = torch.take_along_dim(knot_y_expanded, idx1.unsqueeze(-1), dim=-1).squeeze(-1)
        out = y0 + frac * (y1 - y0)

        return out.reshape_as(x_clip)


class RouterKANLike(nn.Module):
    """Small KAN-like router using lightweight spline activations."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        num_knots: int = 8,
        temporal_head: bool = False,
        logit_clip: float = 20.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if logit_clip <= 0.0:
            raise ValueError("logit_clip must be > 0")

        self.input_dim = input_dim
        self.temporal_head = temporal_head
        self.logit_clip = float(logit_clip)

        self.norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.spline = LightweightSplineActivation(features=hidden_dim, num_knots=num_knots)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.head_u = nn.Linear(hidden_dim, 1)
        self.head_s = nn.Linear(hidden_dim, 1)
        self.head_t = nn.Linear(hidden_dim, 1) if temporal_head else None

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.head_u.weight, gain=0.25)
        nn.init.zeros_(self.head_u.bias)
        nn.init.xavier_uniform_(self.head_s.weight, gain=0.25)
        nn.init.zeros_(self.head_s.bias)
        if self.head_t is not None:
            nn.init.xavier_uniform_(self.head_t.weight, gain=0.25)
            nn.init.zeros_(self.head_t.bias)

    def forward(self, x: Tensor) -> RouterKANOutput:
        if x.ndim != 3:
            raise ValueError("x must be [B,K,D]")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x last dim must equal input_dim={self.input_dim}")

        x_clean = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        h = self.norm(x_clean)
        h = self.in_proj(h)
        h = self.spline(h)
        h = self.out_proj(h)

        u = torch.clamp(self.head_u(h).squeeze(-1), -self.logit_clip, self.logit_clip)
        s = torch.clamp(self.head_s(h).squeeze(-1), -self.logit_clip, self.logit_clip)

        t: Tensor | None = None
        if self.head_t is not None:
            t = torch.clamp(self.head_t(h).squeeze(-1), -self.logit_clip, self.logit_clip)

        return RouterKANOutput(U=u, S=s, T=t)

    def predict_utilities(self, tile_signals: Sequence[float]) -> list[float]:
        if self.input_dim != 1:
            raise ValueError("predict_utilities requires RouterKANLike(input_dim=1)")

        values = [float(v) for v in tile_signals]
        if not values:
            return []

        x = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            out = self.forward(x)
        return [float(v) for v in out.U[0].detach().cpu().tolist()]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
