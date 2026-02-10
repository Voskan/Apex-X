from __future__ import annotations

import math

from apex_x.routing import deterministic_two_stage_selection


def test_two_stage_selection_l0_under_b1_and_l1_generation() -> None:
    result = deterministic_two_stage_selection(
        l0_utilities=[5.0, 4.0, 3.0, 2.0],
        l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
        split_utilities=[1.0, 1.0, 0.0, 0.0],
        split_overheads=[1.0, 1.0, 1.0, 1.0],
        budget_b1=2.0,
        budget_b2=1.0,
        kmax_l0=2,
        kmax_l1=8,
        l0_grid_h=2,
        l0_grid_w=2,
        l1_order_mode="l2r",
    )

    assert result.l0.selected_indices == [0, 1]
    assert math.isclose(result.l0.spent_budget, 2.0, rel_tol=1e-9)
    assert result.split_parent_indices == [0]
    assert math.isclose(result.split_spent_budget, 1.0, rel_tol=1e-9)
    assert result.l1_indices == [0, 1, 4, 5]
    assert result.l1_ordered_indices == [0, 1, 4, 5]


def test_two_stage_split_tie_handling_prefers_lower_parent_id() -> None:
    result = deterministic_two_stage_selection(
        l0_utilities=[0.0, 1.0, 1.0, 0.0],
        l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
        split_utilities=[0.0, 2.0, 2.0, 0.0],  # tie between parents 1 and 2
        split_overheads=[1.0, 1.0, 1.0, 1.0],
        budget_b1=2.0,
        budget_b2=1.0,
        kmax_l0=2,
        kmax_l1=8,
        l0_grid_h=2,
        l0_grid_w=2,
    )

    assert result.l0.selected_indices == [1, 2]
    assert result.split_parent_order == [1, 2]
    assert result.split_parent_indices == [1]


def test_two_stage_selection_enforces_b2_and_kmax_l1_capacity() -> None:
    result = deterministic_two_stage_selection(
        l0_utilities=[4.0, 3.0, 2.0, 1.0],
        l0_delta_costs=[1.0, 1.0, 1.0, 1.0],
        split_utilities=[3.0, 2.0, 1.0, 0.0],
        split_overheads=[0.8, 0.8, 0.8, 0.8],
        budget_b1=3.0,
        budget_b2=5.0,
        kmax_l0=3,
        kmax_l1=6,  # only one split parent can fit because each adds 4 children
        l0_grid_h=2,
        l0_grid_w=2,
    )

    assert result.l0.selected_indices == [0, 1, 2]
    assert len(result.split_parent_indices) == 1
    assert result.l1_valid_count == 4
    assert len(result.l1_kmax_buffer) == 6
    assert result.l1_kmax_buffer[4:] == [-1, -1]
    assert result.split_spent_budget <= 5.0 + 1e-9


def test_two_stage_selection_is_deterministic_across_runs() -> None:
    kwargs = {
        "l0_utilities": [1.5, 1.4, 0.9, 0.7],
        "l0_delta_costs": [0.9, 1.1, 1.0, 1.2],
        "split_utilities": [0.2, 0.8, 0.6, 0.1],
        "split_overheads": [0.6, 0.7, 0.5, 0.9],
        "budget_b1": 2.1,
        "budget_b2": 1.4,
        "kmax_l0": 3,
        "kmax_l1": 8,
        "l0_grid_h": 2,
        "l0_grid_w": 2,
        "l1_order_mode": "hilbert",
    }

    first = deterministic_two_stage_selection(**kwargs)
    for _ in range(50):
        current = deterministic_two_stage_selection(**kwargs)
        assert current.l0.selected_indices == first.l0.selected_indices
        assert current.split_parent_indices == first.split_parent_indices
        assert current.l1_ordered_indices == first.l1_ordered_indices
        assert current.l1_kmax_buffer == first.l1_kmax_buffer
