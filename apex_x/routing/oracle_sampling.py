from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


def _fraction_to_count(fraction: float, total: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fractions must be within [0, 1]")
    if total <= 0:
        return 0
    count = int(round(fraction * total))
    if fraction > 0.0 and count == 0:
        count = 1
    return min(count, total)


def _weighted_sample_without_replacement(
    candidates: list[int],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    if k <= 0 or not candidates:
        return []
    k = min(k, len(candidates))

    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("uncertainty weights must be finite and >= 0")

    positive_pairs = [
        (idx, weight) for idx, weight in zip(candidates, weights, strict=True) if weight > 0.0
    ]
    if not positive_pairs:
        return rng.sample(candidates, k)

    keys: list[tuple[float, int]] = []
    for idx, weight in positive_pairs:
        r = rng.random()
        while r <= 0.0:
            r = rng.random()
        key = math.log(r) / weight
        keys.append((key, idx))
    keys.sort(reverse=True)

    selected = [idx for _, idx in keys[:k]]
    if len(selected) < k:
        remaining = [idx for idx in candidates if idx not in selected]
        selected.extend(rng.sample(remaining, k - len(selected)))
    return selected


@dataclass(frozen=True)
class OracleSetSample:
    """Oracle subset S built from random and uncertainty-biased samples."""

    indices: list[int]
    random_indices: list[int]
    uncertainty_indices: list[int]


def sample_oracle_set(
    u_hat: Sequence[float],
    random_fraction: float = 0.1,
    uncertainty_fraction: float = 0.1,
    seed: int | None = None,
) -> OracleSetSample:
    """Sample oracle set S using random and uncertainty-biased components."""
    num_tiles = len(u_hat)
    if num_tiles == 0:
        return OracleSetSample(indices=[], random_indices=[], uncertainty_indices=[])

    rng = random.Random(seed)
    all_indices = list(range(num_tiles))

    random_count = _fraction_to_count(random_fraction, num_tiles)
    random_indices = rng.sample(all_indices, random_count) if random_count > 0 else []
    random_set = set(random_indices)

    remaining = [idx for idx in all_indices if idx not in random_set]
    remaining_unc = [float(u_hat[idx]) for idx in remaining]
    uncertainty_count = min(_fraction_to_count(uncertainty_fraction, num_tiles), len(remaining))
    uncertainty_indices = _weighted_sample_without_replacement(
        candidates=remaining,
        weights=remaining_unc,
        k=uncertainty_count,
        rng=rng,
    )

    selected = sorted(set(random_indices) | set(uncertainty_indices))
    return OracleSetSample(
        indices=selected,
        random_indices=sorted(random_indices),
        uncertainty_indices=sorted(uncertainty_indices),
    )
