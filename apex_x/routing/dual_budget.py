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
    adaptive_lr: bool = False
    lr_decay: float = 0.0
    delta_clip: float | None = None
    deadband_ratio: float = 0.0
    error_ema_beta: float = 0.9
    adaptive_lr_min_scale: float = 0.5
    adaptive_lr_max_scale: float = 3.0
    logger_name: str = "routing.dual"
    mu: float = field(init=False)
    update_count: int = field(init=False, default=0)
    error_ema: float = field(init=False, default=0.0)
    last_effective_lr: float = field(init=False, default=0.0)
    last_raw_delta: float = field(init=False, default=0.0)
    last_applied_delta: float = field(init=False, default=0.0)
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
        if not math.isfinite(self.lr_decay) or self.lr_decay < 0.0:
            raise ValueError("lr_decay must be finite and >= 0")
        if self.delta_clip is not None and (
            not math.isfinite(self.delta_clip) or self.delta_clip <= 0.0
        ):
            raise ValueError("delta_clip must be finite and > 0 when provided")
        if not math.isfinite(self.deadband_ratio) or not (0.0 <= self.deadband_ratio < 1.0):
            raise ValueError("deadband_ratio must be finite and in [0, 1)")
        if not math.isfinite(self.error_ema_beta) or not (0.0 <= self.error_ema_beta < 1.0):
            raise ValueError("error_ema_beta must be finite and in [0, 1)")
        if not math.isfinite(self.adaptive_lr_min_scale) or self.adaptive_lr_min_scale <= 0.0:
            raise ValueError("adaptive_lr_min_scale must be finite and > 0")
        if (
            not math.isfinite(self.adaptive_lr_max_scale)
            or self.adaptive_lr_max_scale < self.adaptive_lr_min_scale
        ):
            raise ValueError("adaptive_lr_max_scale must be finite and >= adaptive_lr_min_scale")

        self.mu = float(self.mu_init)
        self.last_effective_lr = float(self.mu_lr)
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
        budget_error = float(expected_cost - budget_target)
        normalized_error = float(budget_error / budget_target)

        self.error_ema = (
            self.error_ema_beta * self.error_ema + (1.0 - self.error_ema_beta) * normalized_error
        )

        base_lr = self.mu_lr / (1.0 + self.lr_decay * self.update_count)
        adaptive_scale = 1.0
        if self.adaptive_lr:
            adaptive_scale = max(
                self.adaptive_lr_min_scale,
                min(self.adaptive_lr_max_scale, 1.0 + abs(self.error_ema)),
            )
        effective_lr = float(base_lr * adaptive_scale)

        raw_delta = 0.0
        if abs(normalized_error) > self.deadband_ratio:
            raw_delta = float(effective_lr * budget_error)

        applied_delta = raw_delta
        if self.delta_clip is not None:
            clip = float(self.delta_clip)
            applied_delta = float(max(-clip, min(clip, raw_delta)))

        mu_candidate = mu_prev + applied_delta
        mu_next = min(self.mu_max, max(self.mu_min, mu_candidate))
        clamped = mu_next != mu_candidate

        self.mu = float(mu_next)
        self.update_count += 1
        self.last_effective_lr = effective_lr
        self.last_raw_delta = float(raw_delta)
        self.last_applied_delta = float(applied_delta)

        log_event(
            self._logger,
            "dual_mu_update",
            level="DEBUG",
            fields={
                "budget": budget_target,
                "expected_cost": float(expected_cost),
                "budget_error": budget_error,
                "normalized_error": normalized_error,
                "error_ema": float(self.error_ema),
                "effective_lr": effective_lr,
                "delta_raw": float(raw_delta),
                "delta_applied": float(applied_delta),
                "adaptive_lr": bool(self.adaptive_lr),
                "delta_clip": self.delta_clip,
                "deadband_ratio": self.deadband_ratio,
                "update_count": int(self.update_count),
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
