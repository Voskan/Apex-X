from __future__ import annotations

import pytest

from apex_x.tiles import (
    inverse_scan_mode,
    l0_index_to_tile,
    l0_tile_to_index,
    normalize_order_mode,
    order_idx,
    order_tile_indices,
    scan_order_coords,
    scan_order_indices,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("scan_lr", "l2r"),
        ("lr", "l2r"),
        ("L2R", "l2r"),
        ("scan_rl", "r2l"),
        ("RL", "r2l"),
        ("u2d", "u2d"),
        ("DU", "d2u"),
    ],
)
def test_normalize_order_mode_aliases(mode: str, expected: str) -> None:
    assert normalize_order_mode(mode) == expected


def test_scan_order_coords_is_deterministic_and_complete() -> None:
    grid_h, grid_w = 4, 5
    for mode in ("l2r", "r2l", "u2d", "d2u"):
        a = scan_order_coords(grid_h, grid_w, mode)
        b = scan_order_coords(grid_h, grid_w, mode)
        assert a == b
        assert len(a) == grid_h * grid_w
        assert set(a) == {(ty, tx) for ty in range(grid_h) for tx in range(grid_w)}


def test_scan_order_indices_is_deterministic_and_stable_for_duplicates() -> None:
    indices = [6, 1, 6, 0, 9, 1]
    ordered_a = scan_order_indices(indices, grid_h=4, grid_w=5, mode="l2r")
    ordered_b = scan_order_indices(indices, grid_h=4, grid_w=5, mode="l2r")
    assert ordered_a == ordered_b
    assert ordered_a == [0, 1, 1, 6, 6, 9]


@pytest.mark.parametrize(("mode", "inverse"), [("l2r", "r2l"), ("u2d", "d2u")])
def test_scan_mode_reversible_mapping(mode: str, inverse: str) -> None:
    grid_h, grid_w = 4, 5
    all_indices = list(range(grid_h * grid_w))

    ordered = order_tile_indices(all_indices, grid_h, grid_w, mode=mode)
    ordered_inv = order_tile_indices(all_indices, grid_h, grid_w, mode=inverse)
    inv_pos = {idx: pos for pos, idx in enumerate(ordered_inv)}

    for pos, idx in enumerate(ordered):
        ty, tx = l0_index_to_tile(idx, grid_h, grid_w)
        if mode == "l2r":
            mirrored = l0_tile_to_index(ty, grid_w - 1 - tx, grid_h, grid_w)
        else:
            mirrored = l0_tile_to_index(grid_h - 1 - ty, tx, grid_h, grid_w)
        assert inv_pos[mirrored] == pos


def test_inverse_scan_mode_pairs() -> None:
    assert inverse_scan_mode("l2r") == "r2l"
    assert inverse_scan_mode("r2l") == "l2r"
    assert inverse_scan_mode("u2d") == "d2u"
    assert inverse_scan_mode("d2u") == "u2d"
    with pytest.raises(ValueError, match="not a scan ordering mode"):
        inverse_scan_mode("hilbert")


def test_dispatcher_matches_ops_order_idx_for_scan_modes() -> None:
    indices = [8, 1, 9, 0, 2, 4]
    for mode in ("L2R", "R2L", "U2D", "D2U"):
        from_dispatcher = order_tile_indices(indices, grid_h=4, grid_w=5, mode=mode)
        from_ops = order_idx(indices, grid_h=4, grid_w=5, mode=mode)
        assert from_dispatcher == from_ops
