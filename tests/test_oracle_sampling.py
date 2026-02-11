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
    assert out.long_tail_indices == []


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
    with pytest.raises(ValueError, match="fractions must be within"):
        sample_oracle_set(
            [0.1, 0.2],
            random_fraction=0.1,
            uncertainty_fraction=0.1,
            long_tail_fraction=1.2,
        )
    with pytest.raises(ValueError, match="length must match"):
        sample_oracle_set(
            [0.1, 0.2, 0.3],
            random_fraction=0.1,
            uncertainty_fraction=0.1,
            long_tail_fraction=0.1,
            long_tail_scores=[0.1, 0.2],
        )
    with pytest.raises(ValueError, match="long_tail_scores must be finite"):
        sample_oracle_set(
            [0.1, 0.2, 0.3],
            random_fraction=0.1,
            uncertainty_fraction=0.1,
            long_tail_fraction=0.2,
            long_tail_scores=[0.1, float("nan"), 0.3],
        )


def test_oracle_sampler_long_tail_component_selects_top_remaining_scores() -> None:
    u_hat = [0.1] * 8
    long_tail_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    out = sample_oracle_set(
        u_hat=u_hat,
        random_fraction=0.25,
        uncertainty_fraction=0.25,
        long_tail_fraction=0.25,
        long_tail_scores=long_tail_scores,
        seed=11,
    )

    assert len(out.long_tail_indices) == 2
    assert set(out.long_tail_indices).isdisjoint(out.random_indices)
    assert set(out.long_tail_indices).isdisjoint(out.uncertainty_indices)

    selected_seed = set(out.random_indices) | set(out.uncertainty_indices)
    remaining = [idx for idx in range(len(u_hat)) if idx not in selected_seed]
    expected = sorted(sorted(remaining, key=lambda idx: (-long_tail_scores[idx], idx))[:2])
    assert out.long_tail_indices == expected
