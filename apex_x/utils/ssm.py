from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SSMScanStats:
    """Complexity/operation summary for one scan call."""

    steps: int
    recurrent_updates: int
    pairwise_updates: int


class StableStateSpaceScan(nn.Module):
    """Stable state-space-like scan with constrained recurrent parameters."""

    def __init__(
        self,
        channels: int,
        *,
        min_decay: float = 1e-3,
        max_decay: float = 0.999,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if not (0.0 < min_decay < max_decay < 1.0):
            raise ValueError("decay bounds must satisfy 0 < min_decay < max_decay < 1")

        self.channels = int(channels)
        self.min_decay = float(min_decay)
        self.max_decay = float(max_decay)

        # Constrained via sigmoid to remain in (min_decay, max_decay).
        self.decay_logit = nn.Parameter(torch.zeros(self.channels))
        # Positive gains via softplus for stable scaling.
        self.input_gain_raw = nn.Parameter(torch.zeros(self.channels))
        self.output_gain_raw = nn.Parameter(torch.zeros(self.channels))
        self.state_bias = nn.Parameter(torch.zeros(self.channels))

    def constrained_decay(self) -> Tensor:
        unit = torch.sigmoid(self.decay_logit)
        return self.min_decay + (self.max_decay - self.min_decay) * unit

    def constrained_input_gain(self) -> Tensor:
        return torch.nn.functional.softplus(self.input_gain_raw) + 1e-6

    def constrained_output_gain(self) -> Tensor:
        return torch.nn.functional.softplus(self.output_gain_raw) + 1e-6

    def forward(
        self,
        tokens: Tensor,
        state: Tensor | None = None,
        *,
        return_stats: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, SSMScanStats]:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B,K,C]")
        batch, steps, channels = tokens.shape
        if channels != self.channels:
            raise ValueError(f"tokens channel dimension must equal channels={self.channels}")

        if state is None:
            state_t = torch.zeros((batch, channels), dtype=tokens.dtype, device=tokens.device)
        else:
            if state.ndim != 2 or state.shape != (batch, channels):
                raise ValueError("state must be [B,C] and match tokens batch/channels")
            state_t = state.to(dtype=tokens.dtype, device=tokens.device)

        tokens_sanitized = torch.nan_to_num(
            tokens,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).clamp(-1e4, 1e4)

        decay = self.constrained_decay().to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)
        one_minus_decay = 1.0 - decay
        input_gain = (
            self.constrained_input_gain().to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)
        )
        output_gain = (
            self.constrained_output_gain().to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)
        )
        state_bias = self.state_bias.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)

        outputs = torch.zeros_like(tokens_sanitized)
        for step in range(steps):
            driven = input_gain * tokens_sanitized[:, step, :] + state_bias
            state_t = decay * state_t + one_minus_decay * driven
            outputs[:, step, :] = output_gain * state_t

        if not return_stats:
            return outputs, state_t

        stats = SSMScanStats(
            steps=steps,
            recurrent_updates=batch * channels * steps,
            pairwise_updates=0,
        )
        return outputs, state_t, stats


class StableBidirectionalStateSpaceScan(nn.Module):
    """Bidirectional stable scan with constrained merge gate."""

    def __init__(
        self,
        channels: int,
        *,
        min_decay: float = 1e-3,
        max_decay: float = 0.999,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        self.channels = int(channels)
        self.forward_scan = StableStateSpaceScan(
            channels=channels,
            min_decay=min_decay,
            max_decay=max_decay,
        )
        self.backward_scan = StableStateSpaceScan(
            channels=channels,
            min_decay=min_decay,
            max_decay=max_decay,
        )
        self.merge_gate_logit = nn.Parameter(torch.zeros(channels))

    def merge_gate(self) -> Tensor:
        """Channel-wise merge gate in [0, 1]."""
        return torch.sigmoid(self.merge_gate_logit)

    def forward(
        self,
        tokens: Tensor,
        state_forward: Tensor | None = None,
        state_backward: Tensor | None = None,
        *,
        return_stats: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, SSMScanStats]:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B,K,C]")
        if tokens.shape[2] != self.channels:
            raise ValueError(f"tokens channel dimension must equal channels={self.channels}")

        forward_out, forward_state, stats_f = self.forward_scan(
            tokens,
            state=state_forward,
            return_stats=True,
        )

        backward_tokens = torch.flip(tokens, dims=(1,))
        backward_rev, backward_state, stats_b = self.backward_scan(
            backward_tokens,
            state=state_backward,
            return_stats=True,
        )
        backward_out = torch.flip(backward_rev, dims=(1,))

        gate = self.merge_gate().to(dtype=tokens.dtype, device=tokens.device).reshape(1, 1, -1)
        merged = gate * forward_out + (1.0 - gate) * backward_out

        if not return_stats:
            return merged, forward_state, backward_state

        stats = SSMScanStats(
            steps=stats_f.steps,
            recurrent_updates=stats_f.recurrent_updates + stats_b.recurrent_updates,
            pairwise_updates=0,
        )
        return merged, forward_state, backward_state, stats


def tile_ssm_scan(tokens: np.ndarray, alpha: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """Simple streaming scan placeholder for a Mamba-like Tile-SSM.

    Args:
        tokens: [B, K, C]
    Returns:
        mixed: [B, K, C]
        final_state: [B, C]
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B,K,C]")
    alpha_stable = float(np.clip(alpha, 1e-4, 0.999))

    bsz, steps, channels = tokens.shape
    mixed = np.zeros_like(tokens)
    state = np.zeros((bsz, channels), dtype=tokens.dtype)

    for t in range(steps):
        state = alpha_stable * state + (1.0 - alpha_stable) * tokens[:, t, :]
        mixed[:, t, :] = state
    return mixed, state
