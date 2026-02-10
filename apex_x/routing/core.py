from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .inference_budget import deterministic_greedy_selection
from .interfaces import BudgetControllerProtocol, RouterProtocol


class IdentityRouter(RouterProtocol):
    """Reference router that forwards precomputed utility signals."""

    def predict_utilities(self, tile_signals: Sequence[float]) -> list[float]:
        return list(tile_signals)


class GreedyBudgetController(BudgetControllerProtocol):
    """Deterministic knapsack-lite controller for inference budgets."""

    def select(
        self,
        utilities: Sequence[float],
        costs: Sequence[float],
        budget: float,
        kmax: int,
    ) -> tuple[list[int], float]:
        return greedy_utility_per_cost(utilities, costs, budget, kmax)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ste_gate(probability: float, threshold: float = 0.5) -> int:
    """Straight-through gate in forward pass as a hard binary mask."""
    return int(probability >= threshold)


def delta_loss_oracle(loss_i0: float, loss_i1: float) -> float:
    """Delta utility from teacher distillation loss.

    Delta_i = L_distill(y(i=0), yT) - L_distill(y(i=1), yT)
    """
    return loss_i0 - loss_i1


def expected_cost(utilities: Sequence[float], c_heavy: float, c_cheap: float) -> float:
    """Expected compute cost under continuous gating."""
    total = 0.0
    for u in utilities:
        p_i = sigmoid(u)
        total += p_i * c_heavy + (1.0 - p_i) * c_cheap
    return total


def dual_update(mu: float, exp_cost: float, budget: float, dual_lr: float) -> float:
    """Projected dual ascent for compute-budget Lagrange multiplier."""
    mu_next = mu + dual_lr * (exp_cost - budget)
    return max(0.0, mu_next)


def stable_rank_tile_ids(scores: Sequence[float]) -> list[int]:
    """Stable tile ranking: primary score desc, secondary tile_id asc."""
    scored: list[tuple[int, float]] = []
    for tile_id, score in enumerate(scores):
        score_f = float(score)
        if not math.isfinite(score_f):
            raise ValueError("scores must be finite")
        scored.append((tile_id, score_f))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [tile_id for tile_id, _ in scored]


def greedy_utility_per_cost(
    utilities: Sequence[float],
    costs: Sequence[float],
    budget: float,
    kmax: int,
) -> tuple[list[int], float]:
    """Deterministic greedy selection by utility-per-cost."""
    result = deterministic_greedy_selection(
        utilities=[float(value) for value in utilities],
        delta_costs=[float(value) for value in costs],
        budget=float(budget),
        kmax=int(kmax),
    )
    return result.selected_indices, result.spent_budget


def hysteresis_update(
    utilities_t: Sequence[float],
    prev_mask: Sequence[int],
    theta_on: float,
    theta_off: float,
) -> list[int]:
    """Anti-flicker temporal rule with theta_on > theta_off."""
    if theta_on <= theta_off:
        raise ValueError("theta_on must be > theta_off")
    if len(utilities_t) != len(prev_mask):
        raise ValueError("utilities_t and prev_mask must have the same length")

    out: list[int] = []
    for u, prev in zip(utilities_t, prev_mask, strict=True):
        keep = u > theta_on or (u > theta_off and prev == 1)
        out.append(int(keep))
    return out


def hysteresis_rollout(
    utilities_sequence: Sequence[Sequence[float]],
    initial_mask: Sequence[int],
    theta_on: float,
    theta_off: float,
) -> list[list[int]]:
    """Apply hysteresis over time and return masks for all timesteps."""
    prev = [int(value) for value in initial_mask]
    outputs: list[list[int]] = []
    for utilities_t in utilities_sequence:
        current = hysteresis_update(utilities_t, prev, theta_on=theta_on, theta_off=theta_off)
        outputs.append(current)
        prev = current
    return outputs


def count_mask_toggles(mask_sequence: Sequence[Sequence[int]]) -> int:
    """Count total 0/1 state transitions across all timesteps and tiles."""
    if not mask_sequence:
        return 0

    first_len = len(mask_sequence[0])
    for mask in mask_sequence:
        if len(mask) != first_len:
            raise ValueError("all masks in sequence must have the same length")

    toggles = 0
    for prev, current in zip(mask_sequence, mask_sequence[1:], strict=False):
        toggles += sum(int(p != c) for p, c in zip(prev, current, strict=True))
    return toggles


def split_by_budget(
    split_utilities: Sequence[float],
    split_overheads: Sequence[float],
    budget_b2: float,
    max_splits: int,
) -> tuple[list[int], float]:
    """Choose split candidates by score S_i / O_split until B2."""
    return greedy_utility_per_cost(split_utilities, split_overheads, budget_b2, max_splits)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
