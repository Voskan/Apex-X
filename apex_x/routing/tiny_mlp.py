from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RouterTinyOutput:
    """Router outputs per tile: utility U, split utility S, optional temporal keep T."""

    U: Tensor  # [B, K]
    S: Tensor  # [B, K]
    T: Tensor | None = None  # [B, K] when temporal head enabled


class RouterTinyMLP(nn.Module):
    """Tiny MLP router that predicts U, S, and optional T from per-tile vectors."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        temporal_head: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.input_dim = input_dim
        self.temporal_head = temporal_head

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_u = nn.Linear(hidden_dim, 1)
        self.head_s = nn.Linear(hidden_dim, 1)
        self.head_t = nn.Linear(hidden_dim, 1) if temporal_head else None

    def forward(self, x: Tensor) -> RouterTinyOutput:
        if x.ndim != 3:
            raise ValueError("x must be [B,K,D]")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x last dim must equal input_dim={self.input_dim}")

        h = self.backbone(x)
        u = self.head_u(h).squeeze(-1)
        s = self.head_s(h).squeeze(-1)
        t = self.head_t(h).squeeze(-1) if self.head_t is not None else None
        return RouterTinyOutput(U=u, S=s, T=t)

    def predict_utilities(self, tile_signals: Sequence[float]) -> list[float]:
        if self.input_dim != 1:
            raise ValueError("predict_utilities requires RouterTinyMLP(input_dim=1)")

        values = [float(v) for v in tile_signals]
        if not values:
            return []

        x = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            out = self.forward(x)
        return [float(v) for v in out.U[0].detach().cpu().tolist()]
