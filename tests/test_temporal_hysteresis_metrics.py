from __future__ import annotations

import math

import pytest

from apex_x.routing import (
    count_mask_toggles,
    hysteresis_rollout,
    hysteresis_update_with_budget,
    summarize_temporal_stability,
    temporal_consistency,
    tile_flip_rate,
)


def _threshold_with_budget(
    utilities_sequence: list[list[float]],
    *,
    threshold: float,
    max_active: int,
) -> list[list[int]]:
    masks: list[list[int]] = []
    for utilities in utilities_sequence:
        active = [idx for idx, value in enumerate(utilities) if value > threshold]
        order = sorted(active, key=lambda idx: (-float(utilities[idx]), idx))
        keep = set(order[:max_active])
        masks.append([1 if idx in keep else 0 for idx in range(len(utilities))])
    return masks


def test_hysteresis_update_with_budget_caps_and_prefers_previous_state() -> None:
    out = hysteresis_update_with_budget(
        utilities_t=[0.91, 0.90, 0.89],
        prev_mask=[1, 0, 1],
        theta_on=0.6,
        theta_off=0.4,
        max_active=2,
    )
    assert out == [1, 0, 1]


def test_hysteresis_rollout_with_budget_never_exceeds_active_limit() -> None:
    utilities_sequence = [
        [0.62, 0.58, 0.61],
        [0.54, 0.56, 0.53],
        [0.57, 0.52, 0.55],
        [0.51, 0.59, 0.50],
        [0.60, 0.48, 0.58],
        [0.49, 0.57, 0.47],
    ]
    max_active = 2
    hysteresis_masks = hysteresis_rollout(
        utilities_sequence=utilities_sequence,
        initial_mask=[0, 0, 0],
        theta_on=0.58,
        theta_off=0.50,
        max_active=max_active,
    )
    threshold_masks = _threshold_with_budget(
        utilities_sequence,
        threshold=0.54,
        max_active=max_active,
    )

    for mask in hysteresis_masks:
        assert sum(mask) <= max_active

    # Temporal quality gate: hysteresis should reduce or match toggle churn
    # while preserving frame-level budget constraints.
    assert count_mask_toggles(hysteresis_masks) <= count_mask_toggles(threshold_masks)


def test_temporal_stability_metrics_and_summary() -> None:
    masks = [
        [0, 1],
        [1, 1],
        [1, 0],
    ]
    assert math.isclose(tile_flip_rate(masks), 0.5)
    assert math.isclose(temporal_consistency(masks), 0.5)

    summary = summarize_temporal_stability(masks)
    assert summary.total_toggles == 2
    assert math.isclose(summary.flip_rate, 0.5)
    assert math.isclose(summary.temporal_consistency, 0.5)
    assert math.isclose(summary.mean_active_ratio, 4.0 / 6.0)
    assert summary.peak_active_tiles == 2
    assert summary.frame_count == 3
    assert summary.tile_count == 2


def test_hysteresis_budget_validation_errors() -> None:
    with pytest.raises(ValueError, match="max_active must be >= 0"):
        hysteresis_update_with_budget(
            utilities_t=[0.8],
            prev_mask=[0],
            theta_on=0.6,
            theta_off=0.4,
            max_active=-1,
        )

    with pytest.raises(ValueError, match="max_active must be >= 0"):
        hysteresis_rollout(
            utilities_sequence=[[0.8]],
            initial_mask=[0],
            theta_on=0.6,
            theta_off=0.4,
            max_active=-1,
        )
