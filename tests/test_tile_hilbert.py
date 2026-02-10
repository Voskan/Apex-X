from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from apex_x.tiles import (
    hilbert_full_indices,
    hilbert_order_coords,
    hilbert_order_indices,
    order_idx,
)


def _load_hilbert_fixture(name: str) -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / name
    return cast(dict[str, Any], json.loads(fixture_path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("fixture_name", "grid_h", "grid_w"),
    [
        ("hilbert_2x2.json", 2, 2),
        ("hilbert_4x4.json", 4, 4),
        ("hilbert_8x8.json", 8, 8),
    ],
)
def test_hilbert_order_coords_matches_fixture_and_is_deterministic(
    fixture_name: str,
    grid_h: int,
    grid_w: int,
) -> None:
    fixture = _load_hilbert_fixture(fixture_name)
    expected_coords = [tuple(coord) for coord in fixture["coords"]]

    actual_a = hilbert_order_coords(grid_h, grid_w)
    actual_b = hilbert_order_coords(grid_h, grid_w)

    assert actual_a == expected_coords
    assert actual_b == expected_coords
    assert len(actual_a) == grid_h * grid_w
    assert set(actual_a) == {(ty, tx) for ty in range(grid_h) for tx in range(grid_w)}


@pytest.mark.parametrize(
    ("fixture_name", "grid_h", "grid_w"),
    [
        ("hilbert_2x2.json", 2, 2),
        ("hilbert_4x4.json", 4, 4),
        ("hilbert_8x8.json", 8, 8),
    ],
)
def test_hilbert_full_indices_matches_fixture_and_has_full_coverage(
    fixture_name: str,
    grid_h: int,
    grid_w: int,
) -> None:
    fixture = _load_hilbert_fixture(fixture_name)
    expected_indices = [int(idx) for idx in fixture["indices"]]

    actual = hilbert_full_indices(grid_h, grid_w)
    assert actual == expected_indices
    assert len(actual) == grid_h * grid_w
    assert set(actual) == set(range(grid_h * grid_w))


@pytest.mark.parametrize(("grid_h", "grid_w"), [(2, 2), (4, 4), (8, 8)])
def test_hilbert_order_indices_subset_is_stable(grid_h: int, grid_w: int) -> None:
    full = hilbert_full_indices(grid_h, grid_w)
    subset = list(reversed(full[::2]))
    expected = [idx for idx in full if idx in set(subset)]
    actual = hilbert_order_indices(subset, grid_h, grid_w)
    assert actual == expected
    assert order_idx(subset, grid_h, grid_w, mode="hilbert") == expected
