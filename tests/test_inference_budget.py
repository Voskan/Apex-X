from __future__ import annotations

import math

from apex_x.routing import build_kmax_buffer, deterministic_greedy_selection


def test_deterministic_greedy_selection_is_repeatable() -> None:
    utilities = [0.3, 1.1, 0.9, 0.2, 0.8]
    delta_costs = [1.0, 1.0, 2.0, 0.5, 1.5]

    first = deterministic_greedy_selection(utilities, delta_costs, budget=3.0, kmax=3)
    for _ in range(100):
        current = deterministic_greedy_selection(utilities, delta_costs, budget=3.0, kmax=3)
        assert current.selected_indices == first.selected_indices
        assert current.ordered_candidates == first.ordered_candidates
        assert math.isclose(current.spent_budget, first.spent_budget, rel_tol=1e-9)


def test_deterministic_greedy_selection_enforces_budget_and_kmax() -> None:
    result = deterministic_greedy_selection(
        utilities=[0.9, 0.5, 0.4, 0.3],
        delta_costs=[1.1, 1.1, 1.1, 1.1],
        budget=2.0,
        kmax=3,
    )

    assert len(result.selected_indices) == 1
    assert result.valid_count == 1
    assert result.spent_budget <= 2.0 + 1e-9
    assert len(result.kmax_buffer) == 3
    assert result.kmax_buffer[0] == result.selected_indices[0]
    assert result.kmax_buffer[1:] == [-1, -1]


def test_tie_handling_prefers_lower_tile_id() -> None:
    # Scores are all 2.0 -> tie-break by tile id.
    result = deterministic_greedy_selection(
        utilities=[2.0, 4.0, 1.0, 8.0],
        delta_costs=[1.0, 2.0, 0.5, 4.0],
        budget=4.0,
        kmax=3,
    )

    assert result.ordered_candidates == [0, 1, 2, 3]
    assert result.selected_indices == [0, 1, 2]


def test_build_kmax_buffer_padding_and_truncation() -> None:
    assert build_kmax_buffer([4, 2], kmax=4) == [4, 2, -1, -1]
    assert build_kmax_buffer([4, 2, 1], kmax=2) == [4, 2]
