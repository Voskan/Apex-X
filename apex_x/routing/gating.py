from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

GateMode = Literal["threshold", "bernoulli"]


def sigmoid_probabilities(utilities: Tensor, temperature: float = 1.0) -> Tensor:
    """Map utility logits to probabilities with numerical guards."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    utilities_clean = torch.nan_to_num(utilities, nan=0.0, posinf=60.0, neginf=-60.0)
    scaled = (utilities_clean / temperature).clamp(-60.0, 60.0)
    return torch.sigmoid(scaled)


def ste_hard_gate(
    probabilities: Tensor,
    *,
    mode: GateMode = "threshold",
    threshold: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply hard gating with straight-through gradients.

    Forward:
    - threshold mode: g = 1[p >= threshold]
    - bernoulli mode: g ~ Bernoulli(p)

    Backward:
    - dg/dp = 1 (straight-through estimator)
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")

    probs = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    if mode == "threshold":
        hard = (probs >= threshold).to(dtype=probs.dtype)
    elif mode == "bernoulli":
        hard = torch.bernoulli(probs, generator=generator)
    else:
        raise ValueError("mode must be one of {'threshold', 'bernoulli'}")

    return hard.detach() - probs.detach() + probs


def ste_gate_from_utilities(
    utilities: Tensor,
    *,
    threshold: float = 0.5,
    mode: GateMode = "threshold",
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute p = sigmoid(U) and hard gate g with STE gradients."""
    probabilities = sigmoid_probabilities(utilities, temperature=temperature)
    gate = ste_hard_gate(
        probabilities,
        mode=mode,
        threshold=threshold,
        generator=generator,
    )
    return probabilities, gate
