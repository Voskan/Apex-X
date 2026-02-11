from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

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


def hysteresis_update_with_budget(
    utilities_t: Sequence[float],
    prev_mask: Sequence[int],
    theta_on: float,
    theta_off: float,
    max_active: int,
) -> list[int]:
    """Apply hysteresis and enforce deterministic per-frame active budget.

    Priority when clipping is required:
    1. Keep previously-active tiles (`prev_mask=1`) to reduce flicker.
    2. Higher utility first.
    3. Lower tile id as deterministic tie-break.
    """
    if max_active < 0:
        raise ValueError("max_active must be >= 0")

    candidate = hysteresis_update(
        utilities_t=utilities_t,
        prev_mask=prev_mask,
        theta_on=theta_on,
        theta_off=theta_off,
    )
    if max_active >= len(candidate):
        return candidate

    active_indices = [idx for idx, value in enumerate(candidate) if value == 1]
    if len(active_indices) <= max_active:
        return candidate

    order = sorted(
        active_indices,
        key=lambda idx: (-int(prev_mask[idx] == 1), -float(utilities_t[idx]), idx),
    )
    keep = set(order[:max_active])
    return [1 if idx in keep else 0 for idx in range(len(candidate))]


def hysteresis_rollout(
    utilities_sequence: Sequence[Sequence[float]],
    initial_mask: Sequence[int],
    theta_on: float,
    theta_off: float,
    *,
    max_active: int | None = None,
) -> list[list[int]]:
    """Apply hysteresis over time and return masks for all timesteps."""
    if max_active is not None and max_active < 0:
        raise ValueError("max_active must be >= 0")

    prev = [int(value) for value in initial_mask]
    outputs: list[list[int]] = []
    for utilities_t in utilities_sequence:
        if max_active is None:
            current = hysteresis_update(utilities_t, prev, theta_on=theta_on, theta_off=theta_off)
        else:
            current = hysteresis_update_with_budget(
                utilities_t,
                prev,
                theta_on=theta_on,
                theta_off=theta_off,
                max_active=max_active,
            )
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


def tile_flip_rate(mask_sequence: Sequence[Sequence[int]]) -> float:
    """Average fraction of tiles that flip per temporal transition.

    Returns a value in [0, 1], where:
    - 0 means no tile-state changes across the sequence
    - 1 means every tile flips at every step
    """
    if not mask_sequence:
        return 0.0
    tile_count = len(mask_sequence[0])
    if tile_count == 0 or len(mask_sequence) < 2:
        return 0.0
    toggles = count_mask_toggles(mask_sequence)
    denom = (len(mask_sequence) - 1) * tile_count
    return float(toggles / denom) if denom > 0 else 0.0


def temporal_consistency(mask_sequence: Sequence[Sequence[int]]) -> float:
    """Temporal consistency score in [0, 1], defined as 1 - flip_rate."""
    return float(1.0 - tile_flip_rate(mask_sequence))


def mean_active_ratio(mask_sequence: Sequence[Sequence[int]]) -> float:
    """Mean active-tile ratio over all frames."""
    if not mask_sequence:
        return 0.0
    tile_count = len(mask_sequence[0])
    if tile_count == 0:
        return 0.0
    for mask in mask_sequence:
        if len(mask) != tile_count:
            raise ValueError("all masks in sequence must have the same length")
    total_active = sum(sum(int(value == 1) for value in mask) for mask in mask_sequence)
    return float(total_active / (len(mask_sequence) * tile_count))


@dataclass(frozen=True, slots=True)
class TemporalStabilityMetrics:
    total_toggles: int
    flip_rate: float
    temporal_consistency: float
    mean_active_ratio: float
    peak_active_tiles: int
    frame_count: int
    tile_count: int


def summarize_temporal_stability(
    mask_sequence: Sequence[Sequence[int]],
) -> TemporalStabilityMetrics:
    """Summarize temporal stability for routing-mask sequences."""
    if not mask_sequence:
        return TemporalStabilityMetrics(
            total_toggles=0,
            flip_rate=0.0,
            temporal_consistency=1.0,
            mean_active_ratio=0.0,
            peak_active_tiles=0,
            frame_count=0,
            tile_count=0,
        )
    tile_count = len(mask_sequence[0])
    for mask in mask_sequence:
        if len(mask) != tile_count:
            raise ValueError("all masks in sequence must have the same length")
    toggles = count_mask_toggles(mask_sequence)
    flip_rate = tile_flip_rate(mask_sequence)
    consistency = temporal_consistency(mask_sequence)
    mean_ratio = mean_active_ratio(mask_sequence)
    peak_active = max(sum(int(v == 1) for v in mask) for mask in mask_sequence)
    return TemporalStabilityMetrics(
        total_toggles=toggles,
        flip_rate=flip_rate,
        temporal_consistency=consistency,
        mean_active_ratio=mean_ratio,
        peak_active_tiles=int(peak_active),
        frame_count=len(mask_sequence),
        tile_count=tile_count,
    )


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
