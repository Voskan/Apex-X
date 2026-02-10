from __future__ import annotations

import numpy as np
import pytest

from apex_x.routing import (
    aggregate_pv_maps_to_ff_tile_vectors,
    compute_ff_tile_bounds_on_pv_grid,
)


def test_compute_ff_tile_bounds_alignment_simple_ratio() -> None:
    bounds = compute_ff_tile_bounds_on_pv_grid(ff_h=8, ff_w=8, ff_tile_size=4, pv_h=4, pv_w=4)
    expected = np.asarray(
        [
            [0, 2, 0, 2],
            [0, 2, 2, 4],
            [2, 4, 0, 2],
            [2, 4, 2, 4],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(bounds, expected)


def test_aggregate_pv_maps_shape_and_stats() -> None:
    # Single-map deterministic values to verify mean/max/var on each tile.
    pv = np.asarray(
        [[[[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]]],
        dtype=np.float32,
    )  # [1,1,4,4]
    out = aggregate_pv_maps_to_ff_tile_vectors(
        pv_maps={"u_hat": pv},
        ff_h=8,
        ff_w=8,
        ff_tile_size=4,
    )
    assert out.vectors.shape == (1, 4, 3)
    assert out.feature_layout == ("u_hat:mean:c0", "u_hat:max:c0", "u_hat:var:c0")

    # Tile 0 covers [[0,1],[10,11]].
    expected_tile0 = np.asarray([5.5, 11.0, 25.25], dtype=np.float32)
    assert np.allclose(out.vectors[0, 0], expected_tile0, rtol=1e-6, atol=1e-6)

    # Tile 3 covers [[22,23],[32,33]].
    expected_tile3 = np.asarray([27.5, 33.0, 25.25], dtype=np.float32)
    assert np.allclose(out.vectors[0, 3], expected_tile3, rtol=1e-6, atol=1e-6)


def test_aggregate_multi_map_channels_and_determinism() -> None:
    obj = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    unc = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)

    out_a = aggregate_pv_maps_to_ff_tile_vectors(
        pv_maps={"u_hat": unc, "obj": obj},
        ff_h=8,
        ff_w=8,
        ff_tile_size=4,
    )
    out_b = aggregate_pv_maps_to_ff_tile_vectors(
        pv_maps={"obj": obj, "u_hat": unc},
        ff_h=8,
        ff_w=8,
        ff_tile_size=4,
    )
    # Deterministic map-name sorting should make output identical.
    assert np.array_equal(out_a.vectors, out_b.vectors)
    assert out_a.feature_layout == out_b.feature_layout
    assert out_a.vectors.shape == (1, 4, 9)  # (obj:1ch + u_hat:2ch) * 3 stats


def test_compute_bounds_non_integer_scale_hits_edges() -> None:
    bounds = compute_ff_tile_bounds_on_pv_grid(ff_h=10, ff_w=10, ff_tile_size=5, pv_h=3, pv_w=3)
    # 2x2 tiles
    assert bounds.shape == (4, 4)
    # Last tile should reach the PV max edge.
    assert tuple(bounds[-1].tolist()) == (1, 3, 1, 3)
    # All bounds are non-empty and valid.
    for y0, y1, x0, x1 in bounds.tolist():
        assert 0 <= y0 < y1 <= 3
        assert 0 <= x0 < x1 <= 3


def test_aggregation_validation_errors() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_pv_maps_to_ff_tile_vectors({}, ff_h=8, ff_w=8, ff_tile_size=4)
    with pytest.raises(ValueError, match="divisible"):
        compute_ff_tile_bounds_on_pv_grid(ff_h=7, ff_w=8, ff_tile_size=4, pv_h=4, pv_w=4)
    with pytest.raises(ValueError, match="unsupported stat"):
        aggregate_pv_maps_to_ff_tile_vectors(
            {"u_hat": np.zeros((1, 1, 4, 4), dtype=np.float32)},
            ff_h=8,
            ff_w=8,
            ff_tile_size=4,
            stat_order=("median",),
        )
