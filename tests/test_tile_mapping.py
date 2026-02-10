import numpy as np
import pytest

from apex_x.tiles import (
    l0_coords_to_indices,
    l0_grid_shape,
    l0_index_to_tile,
    l0_indices_to_coords,
    l0_tile_to_index,
)


def test_l0_grid_shape_and_bijection():
    grid_h, grid_w = l0_grid_shape(feature_h=16, feature_w=24, tile_size=8)
    assert (grid_h, grid_w) == (2, 3)

    for ty in range(grid_h):
        for tx in range(grid_w):
            idx = l0_tile_to_index(ty, tx, grid_h, grid_w)
            ty_rt, tx_rt = l0_index_to_tile(idx, grid_h, grid_w)
            assert (ty_rt, tx_rt) == (ty, tx)


def test_l0_grid_shape_invalid_sizes():
    with pytest.raises(ValueError, match="divisible"):
        l0_grid_shape(feature_h=15, feature_w=16, tile_size=8)
    with pytest.raises(ValueError, match="must be > 0"):
        l0_grid_shape(feature_h=0, feature_w=16, tile_size=8)
    with pytest.raises(ValueError, match="must be > 0"):
        l0_grid_shape(feature_h=16, feature_w=16, tile_size=0)


def test_l0_index_bounds():
    with pytest.raises(ValueError, match="out of bounds"):
        l0_index_to_tile(index=-1, grid_h=2, grid_w=3)
    with pytest.raises(ValueError, match="out of bounds"):
        l0_index_to_tile(index=6, grid_h=2, grid_w=3)


def test_l0_tile_bounds():
    with pytest.raises(ValueError, match="out of bounds"):
        l0_tile_to_index(ty=-1, tx=0, grid_h=2, grid_w=3)
    with pytest.raises(ValueError, match="out of bounds"):
        l0_tile_to_index(ty=0, tx=3, grid_h=2, grid_w=3)


def test_l0_batched_indices_coords_roundtrip():
    indices = np.asarray([[0, 1, 5], [2, 3, 4]], dtype=np.int64)
    coords = l0_indices_to_coords(indices, grid_h=2, grid_w=3)
    roundtrip = l0_coords_to_indices(coords, grid_h=2, grid_w=3)
    assert np.array_equal(indices, roundtrip)


def test_l0_indices_to_coords_validation():
    with pytest.raises(ValueError, match="rank-2"):
        l0_indices_to_coords(np.asarray([0, 1, 2], dtype=np.int64), grid_h=2, grid_w=3)
    with pytest.raises(TypeError, match="integer"):
        l0_indices_to_coords(np.asarray([[0.0, 1.0]], dtype=np.float32), grid_h=2, grid_w=3)
    with pytest.raises(ValueError, match="out of bounds"):
        l0_indices_to_coords(np.asarray([[0, 6]], dtype=np.int64), grid_h=2, grid_w=3)


def test_l0_coords_to_indices_validation():
    with pytest.raises(ValueError, match="rank-3"):
        l0_coords_to_indices(np.asarray([[0, 1]], dtype=np.int64), grid_h=2, grid_w=3)
    with pytest.raises(TypeError, match="integer"):
        l0_coords_to_indices(np.asarray([[[0.0, 1.0]]], dtype=np.float32), grid_h=2, grid_w=3)
    with pytest.raises(ValueError, match="out of bounds"):
        l0_coords_to_indices(np.asarray([[[2, 0]]], dtype=np.int64), grid_h=2, grid_w=3)
