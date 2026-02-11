from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from apex_x.tiles import (
    l0_to_l1_children_indices,
    l1_grid_shape_from_l0,
    l1_to_l2_children_indices,
    l2_grid_shape_from_l1,
    order_tile_indices,
)

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


@dataclass(frozen=True, slots=True)
class ThreeStageSelectionResult:
    """Result bundle for deterministic three-stage (L0->L1->L2) selection."""

    two_stage: TwoStageSelectionResult
    split_l1_parent_indices: list[int]
    split_l1_parent_order: list[int]
    split_l1_spent_budget: float
    l2_indices: list[int]
    l2_ordered_indices: list[int]
    l2_grid_h: int
    l2_grid_w: int
    l2_kmax_buffer: list[int]
    l2_valid_count: int


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


def deterministic_three_stage_selection(
    l0_utilities: list[float],
    l0_delta_costs: list[float],
    split_utilities_l0: list[float],
    split_overheads_l0: list[float],
    split_utilities_l1: list[float],
    split_overheads_l1: list[float],
    *,
    budget_b1: float,
    budget_b2: float,
    budget_b3: float,
    kmax_l0: int,
    kmax_l1: int,
    kmax_l2: int,
    l0_grid_h: int,
    l0_grid_w: int,
    l1_order_mode: str = "hilbert",
    l2_order_mode: str = "hilbert",
) -> ThreeStageSelectionResult:
    """Deterministic three-stage selection for nesting depth=2.

    Stage 1:
    - select L0 tiles by `U / delta_cost` under `budget_b1` and `kmax_l0`

    Stage 2:
    - among selected L0 tiles, score split candidates by `S / O_split` under `budget_b2`
    - expand selected L0 parents into L1 child indices

    Stage 3:
    - among selected L1 tiles, score split candidates by `S / O_split` under `budget_b3`
    - expand selected L1 parents into L2 child indices
    """
    if budget_b3 < 0.0 or not math.isfinite(budget_b3):
        raise ValueError("budget_b3 must be finite and >= 0")
    if kmax_l2 < 0:
        raise ValueError("kmax_l2 must be >= 0")

    two_stage = deterministic_two_stage_selection(
        l0_utilities=l0_utilities,
        l0_delta_costs=l0_delta_costs,
        split_utilities=split_utilities_l0,
        split_overheads=split_overheads_l0,
        budget_b1=budget_b1,
        budget_b2=budget_b2,
        kmax_l0=kmax_l0,
        kmax_l1=kmax_l1,
        l0_grid_h=l0_grid_h,
        l0_grid_w=l0_grid_w,
        l1_order_mode=l1_order_mode,
    )

    l1_grid_h, l1_grid_w = two_stage.l1_grid_h, two_stage.l1_grid_w
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)
    expected_l1_size = l1_grid_h * l1_grid_w
    if len(split_utilities_l1) != len(split_overheads_l1):
        raise ValueError("split_utilities_l1 and split_overheads_l1 must have the same length")
    if len(split_utilities_l1) != expected_l1_size:
        raise ValueError("split_utilities_l1/split_overheads_l1 must match full L1 grid size")

    selected_l1 = two_stage.l1_ordered_indices
    if not selected_l1 or kmax_l2 == 0 or budget_b3 == 0.0:
        return ThreeStageSelectionResult(
            two_stage=two_stage,
            split_l1_parent_indices=[],
            split_l1_parent_order=[],
            split_l1_spent_budget=0.0,
            l2_indices=[],
            l2_ordered_indices=[],
            l2_grid_h=l2_grid_h,
            l2_grid_w=l2_grid_w,
            l2_kmax_buffer=build_kmax_buffer([], kmax=kmax_l2),
            l2_valid_count=0,
        )

    split_scores_l1: dict[int, float] = {}
    for idx in selected_l1:
        s = float(split_utilities_l1[idx])
        o = float(split_overheads_l1[idx])
        if not math.isfinite(s):
            raise ValueError("split_utilities_l1 must be finite")
        if not math.isfinite(o):
            raise ValueError("split_overheads_l1 must be finite")
        denom = o if o > _EPS else _EPS
        split_scores_l1[idx] = s / denom

    split_l1_parent_order = sorted(selected_l1, key=lambda idx: (-split_scores_l1[idx], idx))

    split_l1_parent_indices: list[int] = []
    split_l1_spent_budget = 0.0
    l2_indices: list[int] = []

    for parent_idx in split_l1_parent_order:
        overhead = float(split_overheads_l1[parent_idx])
        children = l1_to_l2_children_indices(parent_idx, l1_grid_h, l1_grid_w).tolist()

        if split_l1_spent_budget + overhead > budget_b3 + _EPS:
            continue
        if len(l2_indices) + len(children) > kmax_l2:
            continue

        split_l1_parent_indices.append(parent_idx)
        split_l1_spent_budget += overhead
        l2_indices.extend(int(child) for child in children)

    l2_ordered_indices = order_tile_indices(l2_indices, l2_grid_h, l2_grid_w, mode=l2_order_mode)
    l2_kmax_buffer = build_kmax_buffer(l2_ordered_indices, kmax=kmax_l2)

    return ThreeStageSelectionResult(
        two_stage=two_stage,
        split_l1_parent_indices=split_l1_parent_indices,
        split_l1_parent_order=split_l1_parent_order,
        split_l1_spent_budget=float(split_l1_spent_budget),
        l2_indices=l2_indices,
        l2_ordered_indices=l2_ordered_indices,
        l2_grid_h=l2_grid_h,
        l2_grid_w=l2_grid_w,
        l2_kmax_buffer=l2_kmax_buffer,
        l2_valid_count=len(l2_ordered_indices),
    )
