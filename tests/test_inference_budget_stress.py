from __future__ import annotations

import math

from apex_x.routing import deterministic_greedy_selection


def test_equal_utility_ties_stay_stable_at_scale() -> None:
    tile_count = 128
    kmax = 32
    result = deterministic_greedy_selection(
        utilities=[1.0] * tile_count,
        delta_costs=[1.0] * tile_count,
        budget=128.0,
        kmax=kmax,
    )
    assert result.ordered_candidates == list(range(tile_count))
    assert result.selected_indices == list(range(kmax))
    assert result.valid_count == kmax


def test_zero_and_near_zero_delta_costs_are_handled_deterministically() -> None:
    utilities = [0.5, 0.4, 0.3, 0.2]
    delta_costs = [0.0, 1e-12, 0.15, 0.15]

    first = deterministic_greedy_selection(
        utilities=utilities,
        delta_costs=delta_costs,
        budget=0.30,
        kmax=4,
    )
    for _ in range(100):
        current = deterministic_greedy_selection(
            utilities=utilities,
            delta_costs=delta_costs,
            budget=0.30,
            kmax=4,
        )
        assert current.selected_indices == first.selected_indices
        assert current.ordered_candidates == first.ordered_candidates
        assert math.isclose(current.spent_budget, first.spent_budget, rel_tol=1e-12)

    assert all(math.isfinite(value) for value in first.scores)
    assert first.spent_budget <= 0.30 + 1e-9
    assert first.selected_indices[:2] == [0, 1]


def test_saturated_kmax_clips_selection_even_when_budget_is_large() -> None:
    result = deterministic_greedy_selection(
        utilities=[float(100 - i) for i in range(100)],
        delta_costs=[1.0] * 100,
        budget=1000.0,
        kmax=7,
    )
    assert result.selected_indices == list(range(7))
    assert result.valid_count == 7
    assert len(result.kmax_buffer) == 7


def test_adversarial_close_scores_are_repeatable() -> None:
    utilities = [
        1.0,
        0.9999999999,
        0.9999999998,
        0.9999999997,
        0.9999999996,
        0.9999999995,
    ]
    delta_costs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    first = deterministic_greedy_selection(
        utilities=utilities,
        delta_costs=delta_costs,
        budget=2.0,
        kmax=4,
    )
    for _ in range(200):
        current = deterministic_greedy_selection(
            utilities=utilities,
            delta_costs=delta_costs,
            budget=2.0,
            kmax=4,
        )
        assert current.selected_indices == first.selected_indices
        assert current.ordered_candidates == first.ordered_candidates
        assert math.isclose(current.spent_budget, first.spent_budget, rel_tol=1e-12)
