from __future__ import annotations

import torch
from torch import Tensor, nn


def apply_film(tiles: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """Apply FiLM modulation to packed tiles.

    tiles: [B,K,C,t,t]
    gamma: [B,K,C]
    beta: [B,K,C]
    """
    if tiles.ndim != 5:
        raise ValueError("tiles must be [B,K,C,t,t]")
    if gamma.ndim != 3 or beta.ndim != 3:
        raise ValueError("gamma and beta must be [B,K,C]")
    if gamma.shape != beta.shape:
        raise ValueError("gamma and beta must have identical shapes")
    if tuple(gamma.shape) != tuple(tiles.shape[:3]):
        raise ValueError("gamma/beta shape must match tiles [B,K,C]")

    gamma_5d = gamma.unsqueeze(-1).unsqueeze(-1)
    beta_5d = beta.unsqueeze(-1).unsqueeze(-1)
    return (1.0 + gamma_5d) * tiles + beta_5d


class TileFiLM(nn.Module):
    """Generate FiLM parameters from tile tokens and modulate packed tiles."""

    def __init__(
        self,
        token_dim: int,
        tile_channels: int,
        *,
        hidden_dim: int | None = None,
        gamma_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if tile_channels <= 0:
            raise ValueError("tile_channels must be > 0")
        if gamma_limit <= 0.0:
            raise ValueError("gamma_limit must be > 0")

        self.token_dim = int(token_dim)
        self.tile_channels = int(tile_channels)
        self.gamma_limit = float(gamma_limit)
        hidden = max(self.token_dim, self.tile_channels) if hidden_dim is None else int(hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.token_norm = nn.LayerNorm(self.token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.token_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.tile_channels),
        )

    def film_params(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Compute FiLM parameters from tokens [B,K,Ct]."""
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B,K,Ct]")
        if tokens.shape[2] != self.token_dim:
            raise ValueError("tokens channel dimension does not match token_dim")

        tokens_clean = torch.nan_to_num(tokens, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        params = self.mlp(self.token_norm(tokens_clean))
        gamma_raw, beta = torch.chunk(params, 2, dim=-1)
        gamma = torch.tanh(gamma_raw) * self.gamma_limit
        return gamma, beta

    def forward(self, tokens: Tensor, tiles: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply FiLM modulation from tokens to packed tiles.

        tokens: [B,K,Ct]
        tiles: [B,K,C,t,t]
        """
        if tiles.ndim != 5:
            raise ValueError("tiles must be [B,K,C,t,t]")
        if tiles.shape[2] != self.tile_channels:
            raise ValueError("tiles channel dimension does not match tile_channels")
        if tokens.shape[0] != tiles.shape[0] or tokens.shape[1] != tiles.shape[1]:
            raise ValueError("tokens and tiles must match batch and tile dimensions")

        gamma, beta = self.film_params(tokens)
        modulated = apply_film(tiles, gamma, beta)
        return modulated, gamma, beta
