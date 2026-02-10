from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from logging import Logger

import torch
from torch import Tensor

from apex_x.utils import get_logger, log_event


@dataclass(slots=True)
class BudgetDualController:
    """Dual controller for continuous-budget optimization.

    Implements:
    - expected cost: E[C] = sum_i(p_i*C_h + (1-p_i)*C_c)
    - budget loss: L_budget = mu * (E[C] - B)
    - projected dual update with clamp: mu <- clamp(mu + lr*(E[C]-B), [mu_min, mu_max])
    """

    budget: float
    mu_init: float = 0.1
    mu_lr: float = 0.01
    mu_min: float = 0.0
    mu_max: float = 10.0
    logger_name: str = "routing.dual"
    mu: float = field(init=False)
    _logger: Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget) or self.budget <= 0.0:
            raise ValueError("budget must be finite and > 0")
        if not math.isfinite(self.mu_lr) or self.mu_lr <= 0.0:
            raise ValueError("mu_lr must be finite and > 0")
        if not math.isfinite(self.mu_min) or self.mu_min < 0.0:
            raise ValueError("mu_min must be finite and >= 0")
        if not math.isfinite(self.mu_max) or self.mu_max < self.mu_min:
            raise ValueError("mu_max must be finite and >= mu_min")
        if not math.isfinite(self.mu_init) or not (self.mu_min <= self.mu_init <= self.mu_max):
            raise ValueError("mu_init must be finite and within [mu_min, mu_max]")

        self.mu = float(self.mu_init)
        self._logger = get_logger(self.logger_name)

    def expected_cost(
        self,
        probabilities: Tensor | Sequence[float],
        c_heavy: float,
        c_cheap: float,
    ) -> Tensor | float:
        if not math.isfinite(c_cheap) or c_cheap < 0.0:
            raise ValueError("c_cheap must be finite and >= 0")
        if not math.isfinite(c_heavy) or c_heavy <= c_cheap:
            raise ValueError("c_heavy must be finite and > c_cheap")

        if isinstance(probabilities, Tensor):
            probs = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            return torch.sum(probs * c_heavy + (1.0 - probs) * c_cheap)

        total = 0.0
        for prob in probabilities:
            p = float(prob)
            if not math.isfinite(p) or p < 0.0 or p > 1.0:
                raise ValueError("probabilities must be finite and within [0, 1]")
            total += p * c_heavy + (1.0 - p) * c_cheap
        return total

    def budget_loss(
        self,
        expected_cost: Tensor | float,
        budget: float | None = None,
    ) -> Tensor | float:
        budget_target = self._resolve_budget(budget)
        if isinstance(expected_cost, Tensor):
            return expected_cost.new_tensor(self.mu) * (expected_cost - budget_target)
        return self.mu * (float(expected_cost) - budget_target)

    def update_mu(self, expected_cost: float, budget: float | None = None) -> float:
        if not math.isfinite(expected_cost):
            raise ValueError("expected_cost must be finite")

        budget_target = self._resolve_budget(budget)
        mu_prev = self.mu
        delta = self.mu_lr * (expected_cost - budget_target)
        mu_next = min(self.mu_max, max(self.mu_min, mu_prev + delta))
        clamped = mu_next != mu_prev + delta
        self.mu = float(mu_next)

        log_event(
            self._logger,
            "dual_mu_update",
            level="DEBUG",
            fields={
                "budget": budget_target,
                "expected_cost": float(expected_cost),
                "delta": float(delta),
                "mu_prev": float(mu_prev),
                "mu_next": float(self.mu),
                "clamped": clamped,
            },
        )
        return self.mu

    def _resolve_budget(self, budget: float | None) -> float:
        budget_target = self.budget if budget is None else float(budget)
        if not math.isfinite(budget_target) or budget_target <= 0.0:
            raise ValueError("budget must be finite and > 0")
        return budget_target
