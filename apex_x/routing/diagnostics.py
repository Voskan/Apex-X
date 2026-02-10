from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class HistogramSummary:
    counts: list[int]
    bin_edges: list[float]
    min_value: float
    max_value: float
    mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "bin_edges": self.bin_edges,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
        }


def _ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def utility_histogram(values: Sequence[float], bins: int = 8) -> HistogramSummary:
    if bins <= 0:
        raise ValueError("bins must be > 0")
    if not values:
        counts = [0 for _ in range(bins)]
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64).tolist()
        return HistogramSummary(
            counts=counts,
            bin_edges=[float(v) for v in edges],
            min_value=0.0,
            max_value=0.0,
            mean=0.0,
        )

    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("utility values must be finite")
    counts_arr, edges_arr = np.histogram(array, bins=bins)
    return HistogramSummary(
        counts=[int(v) for v in counts_arr.tolist()],
        bin_edges=[float(v) for v in edges_arr.tolist()],
        min_value=float(array.min()),
        max_value=float(array.max()),
        mean=float(array.mean()),
    )


def _budget_usage_dict(used: float, budget: float) -> dict[str, float]:
    used_f = float(used)
    budget_f = float(budget)
    if not math.isfinite(used_f) or used_f < 0.0:
        raise ValueError("used budget must be finite and >= 0")
    if not math.isfinite(budget_f) or budget_f < 0.0:
        raise ValueError("budget must be finite and >= 0")
    return {
        "used": used_f,
        "budget": budget_f,
        "ratio": (used_f / budget_f) if budget_f > 0.0 else 0.0,
    }


def build_routing_diagnostics(
    *,
    utilities_by_level: Mapping[str, Sequence[float]],
    selected_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
    budget_used: Mapping[str, float],
    budget_total: Mapping[str, float],
    mu_history: Sequence[float],
    histogram_bins: int = 8,
) -> dict[str, Any]:
    levels = sorted(set(total_counts.keys()) | set(selected_counts.keys()))

    selected_ratios: dict[str, float] = {}
    for level in levels:
        selected = int(selected_counts.get(level, 0))
        total = int(total_counts.get(level, 0))
        if selected < 0 or total < 0:
            raise ValueError("selected_counts and total_counts must be >= 0")
        if selected > total and total > 0:
            raise ValueError("selected count cannot exceed total count")
        selected_ratios[level] = _ratio(selected, total)

    histograms: dict[str, dict[str, Any]] = {}
    for level, values in utilities_by_level.items():
        histograms[str(level)] = utility_histogram(values, bins=histogram_bins).to_dict()

    budget_usage: dict[str, dict[str, float]] = {}
    budget_keys = sorted(set(budget_total.keys()) | set(budget_used.keys()))
    for name in budget_keys:
        budget_usage[str(name)] = _budget_usage_dict(
            used=float(budget_used.get(name, 0.0)),
            budget=float(budget_total.get(name, 0.0)),
        )

    mu_hist = [float(value) for value in mu_history]
    if any((not math.isfinite(value) or value < 0.0) for value in mu_hist):
        raise ValueError("mu_history values must be finite and >= 0")

    return {
        "selected_ratios": selected_ratios,
        "selected_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "total_counts": {str(k): int(v) for k, v in total_counts.items()},
        "utility_histograms": histograms,
        "budget_usage": budget_usage,
        "mu_history": mu_hist,
    }
