import numpy as np
import pytest

from apex_x.tiles import (
    OVERLAP_PRIORITY_L0,
    OVERLAP_PRIORITY_L1,
    OVERLAP_PRIORITY_L2,
    order_idx,
    pack_tiles,
    tile_grid_shape,
    unpack_tiles,
)


def test_order_idx_modes_are_deterministic():
    indices = [0, 3, 1, 2]
    a = order_idx(indices, grid_h=2, grid_w=2, mode="hilbert")
    b = order_idx(indices, grid_h=2, grid_w=2, mode="hilbert")
    assert a == b


def test_pack_unpack_roundtrip_identity_when_tile_written_back():
    feat = np.arange(1 * 2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    idx = np.asarray([[0, 3]], dtype=np.int64)
    packed, meta = pack_tiles(feat, idx, tile_size=4)
    merged, _ = unpack_tiles(feat, packed, meta, level_priority=1)
    assert np.allclose(merged, feat)


def test_tile_grid_shape_rejects_non_divisible_feature_map():
    with pytest.raises(ValueError, match="divisible"):
        tile_grid_shape(height=7, width=8, tile_size=4)


def test_unpack_overlap_priority_contract_l2_over_l1_over_l0() -> None:
    base = np.zeros((1, 1, 4, 4), dtype=np.float32)
    meta = {
        "indices": np.asarray([[0]], dtype=np.int64),
        "origins": np.asarray([[[0, 0]]], dtype=np.int64),
        "tile_size": np.asarray(4),
        "grid": np.asarray([1, 1], dtype=np.int64),
    }

    p0 = np.full((1, 1, 1, 4, 4), 1.0, dtype=np.float32)
    p1 = np.full((1, 1, 1, 4, 4), 2.0, dtype=np.float32)
    p2 = np.full((1, 1, 1, 4, 4), 3.0, dtype=np.float32)

    out0, pri = unpack_tiles(base, p0, meta, level_priority=OVERLAP_PRIORITY_L0)
    out1, pri = unpack_tiles(out0, p1, meta, level_priority=OVERLAP_PRIORITY_L1, priority_map=pri)
    out2, _ = unpack_tiles(out1, p2, meta, level_priority=OVERLAP_PRIORITY_L2, priority_map=pri)
    assert float(out2[0, 0, 0, 0]) == 3.0
