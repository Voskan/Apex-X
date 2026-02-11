from __future__ import annotations

import math

import pytest

from apex_x.routing import deterministic_three_stage_selection


def test_three_stage_selection_l2_generation_from_selected_l1() -> None:
    l1_utilities = [0.0] * 16
    l1_overheads = [1.0] * 16
    l1_utilities[1] = 1.0
    l1_utilities[4] = 0.9
    l1_utilities[0] = 0.5
    l1_utilities[5] = 0.1

    result = deterministic_three_stage_selection(
        l0_utilities=[5.0, 4.0, 3.0, 2.0],
        l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
        split_utilities_l0=[1.0, 1.0, 0.0, 0.0],
        split_overheads_l0=[1.0, 1.0, 1.0, 1.0],
        split_utilities_l1=l1_utilities,
        split_overheads_l1=l1_overheads,
        budget_b1=2.0,
        budget_b2=1.0,
        budget_b3=1.0,
        kmax_l0=2,
        kmax_l1=8,
        kmax_l2=8,
        l0_grid_h=2,
        l0_grid_w=2,
        l1_order_mode="l2r",
        l2_order_mode="l2r",
    )

    assert result.two_stage.l0.selected_indices == [0, 1]
    assert result.two_stage.split_parent_indices == [0]
    assert result.two_stage.l1_indices == [0, 1, 4, 5]
    assert result.split_l1_parent_indices == [1]
    assert math.isclose(result.split_l1_spent_budget, 1.0, rel_tol=1e-9)
    assert result.l2_indices == [2, 3, 10, 11]
    assert result.l2_ordered_indices == [2, 3, 10, 11]
    assert result.l2_valid_count == 4


def test_three_stage_split_tie_break_prefers_lower_l1_parent_id() -> None:
    l1_utilities = [0.0] * 16
    l1_overheads = [1.0] * 16
    # Tie between parents 1 and 4.
    l1_utilities[1] = 2.0
    l1_utilities[4] = 2.0

    result = deterministic_three_stage_selection(
        l0_utilities=[5.0, 4.0, 3.0, 2.0],
        l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
        split_utilities_l0=[1.0, 1.0, 0.0, 0.0],
        split_overheads_l0=[1.0, 1.0, 1.0, 1.0],
        split_utilities_l1=l1_utilities,
        split_overheads_l1=l1_overheads,
        budget_b1=2.0,
        budget_b2=1.0,
        budget_b3=1.0,
        kmax_l0=2,
        kmax_l1=8,
        kmax_l2=8,
        l0_grid_h=2,
        l0_grid_w=2,
    )

    assert result.split_l1_parent_order[:2] == [1, 4]
    assert result.split_l1_parent_indices == [1]


def test_three_stage_selection_enforces_b3_and_kmax_l2_capacity() -> None:
    l1_utilities = [0.0] * 16
    l1_overheads = [0.8] * 16
    # Two-stage selects l1 set from parents 0 and 1 -> indices [0,1,4,5,2,3,6,7]
    for idx in [0, 1, 2, 3, 4, 5, 6, 7]:
        l1_utilities[idx] = 1.0

    result = deterministic_three_stage_selection(
        l0_utilities=[4.0, 3.0, 2.0, 1.0],
        l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
        split_utilities_l0=[3.0, 2.0, 0.0, 0.0],
        split_overheads_l0=[1.0, 1.0, 1.0, 1.0],
        split_utilities_l1=l1_utilities,
        split_overheads_l1=l1_overheads,
        budget_b1=2.0,
        budget_b2=2.0,
        budget_b3=5.0,
        kmax_l0=2,
        kmax_l1=12,
        kmax_l2=6,  # one L1 parent (4 children) fits; second would overflow.
        l0_grid_h=2,
        l0_grid_w=2,
        l1_order_mode="l2r",
        l2_order_mode="l2r",
    )

    assert result.two_stage.split_parent_indices == [0, 1]
    assert len(result.split_l1_parent_indices) == 1
    assert result.l2_valid_count == 4
    assert len(result.l2_kmax_buffer) == 6
    assert result.l2_kmax_buffer[4:] == [-1, -1]
    assert result.split_l1_spent_budget <= 5.0 + 1e-9


def test_three_stage_selection_is_deterministic_across_runs() -> None:
    l1_utilities = [0.0] * 16
    l1_overheads = [0.6] * 16
    l1_utilities[0] = 0.4
    l1_utilities[1] = 0.9
    l1_utilities[4] = 0.8
    l1_utilities[5] = 0.3

    kwargs = {
        "l0_utilities": [1.5, 1.4, 0.9, 0.7],
        "l0_delta_costs": [0.9, 1.1, 1.0, 1.2],
        "split_utilities_l0": [0.2, 0.8, 0.6, 0.1],
        "split_overheads_l0": [0.6, 0.7, 0.5, 0.9],
        "split_utilities_l1": l1_utilities,
        "split_overheads_l1": l1_overheads,
        "budget_b1": 2.1,
        "budget_b2": 1.4,
        "budget_b3": 1.2,
        "kmax_l0": 3,
        "kmax_l1": 8,
        "kmax_l2": 8,
        "l0_grid_h": 2,
        "l0_grid_w": 2,
        "l1_order_mode": "hilbert",
        "l2_order_mode": "hilbert",
    }

    first = deterministic_three_stage_selection(**kwargs)
    for _ in range(50):
        current = deterministic_three_stage_selection(**kwargs)
        assert current.two_stage.l0.selected_indices == first.two_stage.l0.selected_indices
        assert current.two_stage.split_parent_indices == first.two_stage.split_parent_indices
        assert current.two_stage.l1_ordered_indices == first.two_stage.l1_ordered_indices
        assert current.split_l1_parent_indices == first.split_l1_parent_indices
        assert current.l2_ordered_indices == first.l2_ordered_indices
        assert current.l2_kmax_buffer == first.l2_kmax_buffer


def test_three_stage_selection_validation_errors() -> None:
    with pytest.raises(ValueError, match="same length"):
        deterministic_three_stage_selection(
            l0_utilities=[1.0, 0.9, 0.8, 0.7],
            l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
            split_utilities_l0=[1.0, 1.0, 0.0, 0.0],
            split_overheads_l0=[1.0, 1.0, 1.0, 1.0],
            split_utilities_l1=[0.0] * 16,
            split_overheads_l1=[1.0] * 15,
            budget_b1=2.0,
            budget_b2=1.0,
            budget_b3=1.0,
            kmax_l0=2,
            kmax_l1=8,
            kmax_l2=8,
            l0_grid_h=2,
            l0_grid_w=2,
        )

    with pytest.raises(ValueError, match="full L1 grid size"):
        deterministic_three_stage_selection(
            l0_utilities=[1.0, 0.9, 0.8, 0.7],
            l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
            split_utilities_l0=[1.0, 1.0, 0.0, 0.0],
            split_overheads_l0=[1.0, 1.0, 1.0, 1.0],
            split_utilities_l1=[0.0] * 8,
            split_overheads_l1=[1.0] * 8,
            budget_b1=2.0,
            budget_b2=1.0,
            budget_b3=1.0,
            kmax_l0=2,
            kmax_l1=8,
            kmax_l2=8,
            l0_grid_h=2,
            l0_grid_w=2,
        )
