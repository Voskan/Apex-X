from __future__ import annotations

from collections import Counter

import pytest

from apex_x.routing import sample_oracle_set


def test_oracle_sampler_is_deterministic_with_seed() -> None:
    u_hat = [0.05, 0.7, 0.2, 0.9, 0.4, 0.1, 0.3, 0.6]
    out_a = sample_oracle_set(
        u_hat=u_hat,
        random_fraction=0.25,
        uncertainty_fraction=0.25,
        seed=42,
    )
    out_b = sample_oracle_set(
        u_hat=u_hat,
        random_fraction=0.25,
        uncertainty_fraction=0.25,
        seed=42,
    )

    assert out_a == out_b
    assert out_a.indices == [0, 1, 6, 7]
    assert out_a.random_indices == [0, 1]
    assert out_a.uncertainty_indices == [6, 7]


def test_oracle_sampler_count_and_no_duplicates() -> None:
    u_hat = [0.1] * 20
    out = sample_oracle_set(
        u_hat=u_hat,
        random_fraction=0.2,
        uncertainty_fraction=0.3,
        seed=123,
    )
    assert len(out.random_indices) == 4
    assert len(out.uncertainty_indices) == 6
    assert len(out.indices) == 10
    assert len(set(out.indices)) == len(out.indices)
    assert set(out.random_indices).isdisjoint(out.uncertainty_indices)


def test_oracle_sampler_uncertainty_bias_distribution() -> None:
    # One tile has much higher uncertainty and should be sampled most often.
    u_hat = [0.01] * 9 + [1.0]
    hit_counter: Counter[int] = Counter()

    for seed in range(500):
        out = sample_oracle_set(
            u_hat=u_hat,
            random_fraction=0.0,
            uncertainty_fraction=0.2,
            seed=seed,
        )
        hit_counter.update(out.indices)

    high = hit_counter[9]
    low_max = max(hit_counter[i] for i in range(9))
    assert high > low_max


def test_oracle_sampler_validation_errors() -> None:
    with pytest.raises(ValueError, match="fractions must be within"):
        sample_oracle_set([0.1, 0.2], random_fraction=-0.1, uncertainty_fraction=0.1)
    with pytest.raises(ValueError, match="fractions must be within"):
        sample_oracle_set([0.1, 0.2], random_fraction=0.1, uncertainty_fraction=1.1)
    with pytest.raises(ValueError, match="weights must be finite"):
        sample_oracle_set([0.1, float("nan"), 0.2], random_fraction=0.0, uncertainty_fraction=0.5)
