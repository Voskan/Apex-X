from __future__ import annotations

import pytest

from apex_x.routing import count_mask_toggles, hysteresis_rollout, hysteresis_update


def _threshold_rollout(utilities_sequence: list[list[float]], threshold: float) -> list[list[int]]:
    return [[int(value > threshold) for value in frame] for frame in utilities_sequence]


def test_hysteresis_rollout_respects_previous_mask_inside_deadband() -> None:
    utilities_sequence = [[0.5], [0.5], [0.5], [0.5]]

    from_on = hysteresis_rollout(
        utilities_sequence=utilities_sequence,
        initial_mask=[1],
        theta_on=0.6,
        theta_off=0.4,
    )
    from_off = hysteresis_rollout(
        utilities_sequence=utilities_sequence,
        initial_mask=[0],
        theta_on=0.6,
        theta_off=0.4,
    )

    assert from_on == [[1], [1], [1], [1]]
    assert from_off == [[0], [0], [0], [0]]


def test_hysteresis_reduces_toggling_on_synthetic_noisy_sequence() -> None:
    utilities_sequence = [
        [0.56],
        [0.49],
        [0.51],
        [0.48],
        [0.52],
        [0.47],
        [0.53],
        [0.46],
        [0.54],
        [0.44],
        [0.56],
    ]

    hysteresis_masks = hysteresis_rollout(
        utilities_sequence=utilities_sequence,
        initial_mask=[0],
        theta_on=0.55,
        theta_off=0.45,
    )
    threshold_masks = _threshold_rollout(utilities_sequence, threshold=0.5)

    hysteresis_toggles = count_mask_toggles(hysteresis_masks)
    threshold_toggles = count_mask_toggles(threshold_masks)

    assert hysteresis_toggles < threshold_toggles
    assert hysteresis_toggles == 2
    assert threshold_toggles == 10


def test_hysteresis_validation_errors() -> None:
    with pytest.raises(ValueError, match="theta_on must be > theta_off"):
        hysteresis_update([0.6], [0], theta_on=0.5, theta_off=0.5)

    with pytest.raises(ValueError, match="same length"):
        hysteresis_update([0.6, 0.7], [0], theta_on=0.6, theta_off=0.4)

    with pytest.raises(ValueError, match="same length"):
        count_mask_toggles([[1, 0], [1]])
