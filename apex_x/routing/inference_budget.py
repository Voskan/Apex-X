from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from apex_x.tiles import l0_to_l1_children_indices, l1_grid_shape_from_l0, order_tile_indices

_EPS: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class GreedySelectionResult:
    """Result bundle for deterministic utility-per-cost selection."""

    selected_indices: list[int]
    spent_budget: float
    ordered_candidates: list[int]
    scores: list[float]
    kmax_buffer: list[int]
    valid_count: int


@dataclass(frozen=True, slots=True)
class TwoStageSelectionResult:
    """Result bundle for deterministic two-stage (L0->L1) selection."""

    l0: GreedySelectionResult
    split_parent_indices: list[int]
    split_parent_order: list[int]
    split_spent_budget: float
    l1_indices: list[int]
    l1_ordered_indices: list[int]
    l1_grid_h: int
    l1_grid_w: int
    l1_kmax_buffer: list[int]
    l1_valid_count: int


def build_kmax_buffer(selected_indices: list[int], kmax: int, pad_value: int = -1) -> list[int]:
    """Pack selected ids into fixed-length Kmax buffer."""
    if kmax < 0:
        raise ValueError("kmax must be >= 0")
    if pad_value < 0 and any(idx < 0 for idx in selected_indices):
        raise ValueError("selected_indices must be non-negative")

    limited = selected_indices[:kmax]
    padded = limited + [pad_value] * max(0, kmax - len(limited))
    return padded


def deterministic_greedy_selection(
    utilities: list[float],
    delta_costs: list[float],
    budget: float,
    kmax: int,
) -> GreedySelectionResult:
    """Deterministic greedy selection by score = U / delta_cost.

    Tie handling:
    - primary key: score desc
    - secondary key: tile_id asc
    """
    if budget < 0.0 or not math.isfinite(budget):
        raise ValueError("budget must be finite and >= 0")
    if kmax < 0:
        raise ValueError("kmax must be >= 0")
    if len(utilities) != len(delta_costs):
        raise ValueError("utilities and delta_costs must have the same length")

    scores: list[float] = []
    for utility, delta_cost in zip(utilities, delta_costs, strict=True):
        u = float(utility)
        c = float(delta_cost)
        if not math.isfinite(u):
            raise ValueError("utilities must be finite")
        if not math.isfinite(c):
            raise ValueError("delta_costs must be finite")
        denom = c if c > _EPS else _EPS
        scores.append(u / denom)

    ordered_candidates = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))

    selected_indices: list[int] = []
    spent_budget = 0.0
    for idx in ordered_candidates:
        if len(selected_indices) >= kmax:
            break
        cost = float(delta_costs[idx])
        if spent_budget + cost <= budget + _EPS:
            selected_indices.append(idx)
            spent_budget += cost

    kmax_buffer = build_kmax_buffer(selected_indices, kmax=kmax)
    return GreedySelectionResult(
        selected_indices=selected_indices,
        spent_budget=float(spent_budget),
        ordered_candidates=ordered_candidates,
        scores=scores,
        kmax_buffer=kmax_buffer,
        valid_count=len(selected_indices),
    )


def deterministic_two_stage_selection(
    l0_utilities: list[float],
    l0_delta_costs: list[float],
    split_utilities: list[float],
    split_overheads: list[float],
    *,
    budget_b1: float,
    budget_b2: float,
    kmax_l0: int,
    kmax_l1: int,
    l0_grid_h: int,
    l0_grid_w: int,
    l1_order_mode: str = "hilbert",
) -> TwoStageSelectionResult:
    """Deterministic two-stage selection for nesting depth=1.

    Stage 1:
    - select L0 tiles by `U / delta_cost` under `budget_b1` and `kmax_l0`

    Stage 2:
    - among selected L0 tiles, score split candidates by `S / O_split`
    - select parents under `budget_b2` and L1 capacity (`kmax_l1`)
    - expand selected parents into L1 child indices and deterministically order them
    """
    if budget_b2 < 0.0 or not math.isfinite(budget_b2):
        raise ValueError("budget_b2 must be finite and >= 0")
    if kmax_l1 < 0:
        raise ValueError("kmax_l1 must be >= 0")
    if len(l0_utilities) != len(l0_delta_costs):
        raise ValueError("l0_utilities and l0_delta_costs must have the same length")
    if len(split_utilities) != len(split_overheads):
        raise ValueError("split_utilities and split_overheads must have the same length")
    if len(split_utilities) != len(l0_utilities):
        raise ValueError("split arrays must align 1:1 with l0 utilities")

    l0_result = deterministic_greedy_selection(
        utilities=l0_utilities,
        delta_costs=l0_delta_costs,
        budget=budget_b1,
        kmax=kmax_l0,
    )
    selected_l0 = l0_result.selected_indices

    if not selected_l0 or kmax_l1 == 0 or budget_b2 == 0.0:
        l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
        return TwoStageSelectionResult(
            l0=l0_result,
            split_parent_indices=[],
            split_parent_order=[],
            split_spent_budget=0.0,
            l1_indices=[],
            l1_ordered_indices=[],
            l1_grid_h=l1_grid_h,
            l1_grid_w=l1_grid_w,
            l1_kmax_buffer=build_kmax_buffer([], kmax=kmax_l1),
            l1_valid_count=0,
        )

    split_scores: dict[int, float] = {}
    for idx in selected_l0:
        s = float(split_utilities[idx])
        o = float(split_overheads[idx])
        if not math.isfinite(s):
            raise ValueError("split_utilities must be finite")
        if not math.isfinite(o):
            raise ValueError("split_overheads must be finite")
        denom = o if o > _EPS else _EPS
        split_scores[idx] = s / denom

    split_parent_order = sorted(selected_l0, key=lambda idx: (-split_scores[idx], idx))

    split_parent_indices: list[int] = []
    split_spent_budget = 0.0
    l1_indices: list[int] = []

    for parent_idx in split_parent_order:
        overhead = float(split_overheads[parent_idx])
        children = l0_to_l1_children_indices(parent_idx, l0_grid_h, l0_grid_w).tolist()

        if split_spent_budget + overhead > budget_b2 + _EPS:
            continue
        if len(l1_indices) + len(children) > kmax_l1:
            continue

        split_parent_indices.append(parent_idx)
        split_spent_budget += overhead
        l1_indices.extend(int(child) for child in children)

    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    l1_ordered_indices = order_tile_indices(l1_indices, l1_grid_h, l1_grid_w, mode=l1_order_mode)
    l1_kmax_buffer = build_kmax_buffer(l1_ordered_indices, kmax=kmax_l1)

    return TwoStageSelectionResult(
        l0=l0_result,
        split_parent_indices=split_parent_indices,
        split_parent_order=split_parent_order,
        split_spent_budget=float(split_spent_budget),
        l1_indices=l1_indices,
        l1_ordered_indices=l1_ordered_indices,
        l1_grid_h=l1_grid_h,
        l1_grid_w=l1_grid_w,
        l1_kmax_buffer=l1_kmax_buffer,
        l1_valid_count=len(l1_ordered_indices),
    )
