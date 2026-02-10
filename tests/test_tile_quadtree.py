from __future__ import annotations

import numpy as np
import pytest

from apex_x.tiles import (
    OVERLAP_PRIORITY_L0,
    OVERLAP_PRIORITY_L1,
    OVERLAP_PRIORITY_L2,
    build_l0_l1_l2_quadtree_meta,
    build_l0_l1_quadtree_meta,
    build_l1_l2_quadtree_meta,
    l0_l1_grid_shapes_from_feature,
    l0_l1_l2_grid_shapes_from_feature,
    l0_to_l1_children_coords,
    l0_to_l1_children_indices,
    l0_to_l2_descendant_indices,
    l1_grid_shape_from_l0,
    l1_l2_grid_shapes_from_feature,
    l1_to_l0_parent_coord,
    l1_to_l0_parent_index,
    l1_to_l2_children_coords,
    l1_to_l2_children_indices,
    l2_grid_shape_from_l1,
    l2_to_l1_parent_coord,
    l2_to_l1_parent_index,
    overlap_priority_for_level,
)


@pytest.mark.parametrize(
    ("feature_h", "feature_w", "tile_size_l0", "tile_size_l1", "expected_l0", "expected_l1"),
    [
        (32, 32, 8, 4, (4, 4), (8, 8)),
        (48, 64, 16, 8, (3, 4), (6, 8)),
        (96, 160, 16, 8, (6, 10), (12, 20)),
    ],
)
def test_l0_l1_grid_shapes_from_feature_multiple_configs(
    feature_h: int,
    feature_w: int,
    tile_size_l0: int,
    tile_size_l1: int,
    expected_l0: tuple[int, int],
    expected_l1: tuple[int, int],
) -> None:
    l0_grid, l1_grid = l0_l1_grid_shapes_from_feature(
        feature_h=feature_h,
        feature_w=feature_w,
        tile_size_l0=tile_size_l0,
        tile_size_l1=tile_size_l1,
    )
    assert l0_grid == expected_l0
    assert l1_grid == expected_l1


def test_l0_to_l1_boundary_tile_mapping_bottom_right() -> None:
    l0_grid_h, l0_grid_w = 3, 4
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    assert (l1_grid_h, l1_grid_w) == (6, 8)

    children = l0_to_l1_children_coords(l0_ty=2, l0_tx=3, l0_grid_h=l0_grid_h, l0_grid_w=l0_grid_w)
    assert children == ((4, 6), (4, 7), (5, 6), (5, 7))

    idx = l0_to_l1_children_indices(l0_index=11, l0_grid_h=l0_grid_h, l0_grid_w=l0_grid_w)
    expected = np.asarray([38, 39, 46, 47], dtype=np.int64)
    assert np.array_equal(idx, expected)


@pytest.mark.parametrize(("l0_grid_h", "l0_grid_w"), [(2, 2), (3, 5), (6, 4)])
def test_l1_to_l0_parent_reverse_mapping_covers_all_children(
    l0_grid_h: int,
    l0_grid_w: int,
) -> None:
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)

    for l1_ty in range(l1_grid_h):
        for l1_tx in range(l1_grid_w):
            p_ty, p_tx = l1_to_l0_parent_coord(l1_ty, l1_tx, l0_grid_h, l0_grid_w)
            assert p_ty == l1_ty // 2
            assert p_tx == l1_tx // 2

            l1_index = l1_ty * l1_grid_w + l1_tx
            p_index = l1_to_l0_parent_index(l1_index, l0_grid_h, l0_grid_w)
            assert p_index == p_ty * l0_grid_w + p_tx


def test_build_l0_l1_quadtree_meta_shapes_and_consistency() -> None:
    meta = build_l0_l1_quadtree_meta(parent_indices=[0, 5, 11], l0_grid_h=3, l0_grid_w=4)
    assert meta.l1_grid_h == 6
    assert meta.l1_grid_w == 8
    assert meta.parent_indices.shape == (3,)
    assert meta.parent_coords.shape == (3, 2)
    assert meta.child_indices.shape == (3, 4)
    assert meta.child_coords.shape == (3, 4, 2)

    # Parent 11 is bottom-right tile in 3x4 L0.
    assert tuple(meta.parent_coords[2].tolist()) == (2, 3)
    assert tuple(meta.child_coords[2, 0].tolist()) == (4, 6)
    assert tuple(meta.child_coords[2, 3].tolist()) == (5, 7)
    assert meta.child_indices[2, 0] == 38
    assert meta.child_indices[2, 3] == 47


def test_quadtree_validation_errors() -> None:
    with pytest.raises(ValueError, match="exactly tile_size_l0/2"):
        l0_l1_grid_shapes_from_feature(feature_h=32, feature_w=32, tile_size_l0=8, tile_size_l1=3)
    with pytest.raises(ValueError, match="divisible"):
        l0_l1_grid_shapes_from_feature(feature_h=30, feature_w=32, tile_size_l0=8, tile_size_l1=4)
    with pytest.raises(ValueError, match="out of bounds"):
        l0_to_l1_children_indices(l0_index=12, l0_grid_h=3, l0_grid_w=4)
    with pytest.raises(ValueError, match="out of bounds"):
        l1_to_l0_parent_coord(l1_ty=6, l1_tx=0, l0_grid_h=3, l0_grid_w=4)


@pytest.mark.parametrize(
    ("feature_h", "feature_w", "tile_size_l1", "tile_size_l2", "expected_l1", "expected_l2"),
    [
        (32, 32, 4, 2, (8, 8), (16, 16)),
        (48, 64, 8, 4, (6, 8), (12, 16)),
        (96, 160, 8, 4, (12, 20), (24, 40)),
    ],
)
def test_l1_l2_grid_shapes_from_feature_multiple_configs(
    feature_h: int,
    feature_w: int,
    tile_size_l1: int,
    tile_size_l2: int,
    expected_l1: tuple[int, int],
    expected_l2: tuple[int, int],
) -> None:
    l1_grid, l2_grid = l1_l2_grid_shapes_from_feature(
        feature_h=feature_h,
        feature_w=feature_w,
        tile_size_l1=tile_size_l1,
        tile_size_l2=tile_size_l2,
    )
    assert l1_grid == expected_l1
    assert l2_grid == expected_l2


def test_l1_to_l2_boundary_tile_mapping_bottom_right() -> None:
    l1_grid_h, l1_grid_w = 6, 8
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)
    assert (l2_grid_h, l2_grid_w) == (12, 16)

    children = l1_to_l2_children_coords(l1_ty=5, l1_tx=7, l1_grid_h=l1_grid_h, l1_grid_w=l1_grid_w)
    assert children == ((10, 14), (10, 15), (11, 14), (11, 15))

    idx = l1_to_l2_children_indices(l1_index=47, l1_grid_h=l1_grid_h, l1_grid_w=l1_grid_w)
    expected = np.asarray([174, 175, 190, 191], dtype=np.int64)
    assert np.array_equal(idx, expected)


@pytest.mark.parametrize(("l1_grid_h", "l1_grid_w"), [(2, 2), (3, 5), (6, 4)])
def test_l2_to_l1_parent_reverse_mapping_covers_all_children(
    l1_grid_h: int,
    l1_grid_w: int,
) -> None:
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)

    for l2_ty in range(l2_grid_h):
        for l2_tx in range(l2_grid_w):
            p_ty, p_tx = l2_to_l1_parent_coord(l2_ty, l2_tx, l1_grid_h, l1_grid_w)
            assert p_ty == l2_ty // 2
            assert p_tx == l2_tx // 2

            l2_index = l2_ty * l2_grid_w + l2_tx
            p_index = l2_to_l1_parent_index(l2_index, l1_grid_h, l1_grid_w)
            assert p_index == p_ty * l1_grid_w + p_tx


def test_l0_to_l2_descendants_index_correctness() -> None:
    descendants = l0_to_l2_descendant_indices(l0_index=11, l0_grid_h=3, l0_grid_w=4)
    expected = np.asarray(
        [140, 141, 156, 157, 142, 143, 158, 159, 172, 173, 188, 189, 174, 175, 190, 191],
        dtype=np.int64,
    )
    assert np.array_equal(descendants, expected)


def test_build_l1_l2_and_l0_l1_l2_meta_shapes_and_consistency() -> None:
    l1_l2_meta = build_l1_l2_quadtree_meta(parent_indices=[0, 47], l1_grid_h=6, l1_grid_w=8)
    assert l1_l2_meta.l2_grid_h == 12
    assert l1_l2_meta.l2_grid_w == 16
    assert l1_l2_meta.child_indices.shape == (2, 4)
    assert tuple(l1_l2_meta.child_coords[1, 0].tolist()) == (10, 14)
    assert tuple(l1_l2_meta.child_coords[1, 3].tolist()) == (11, 15)

    combo = build_l0_l1_l2_quadtree_meta(l0_parent_indices=[11], l0_grid_h=3, l0_grid_w=4)
    assert combo.l0_l1.child_indices.shape == (1, 4)
    assert combo.l1_l2.child_indices.shape == (4, 4)
    flattened = combo.l1_l2.child_indices.reshape(-1)
    assert np.array_equal(flattened, l0_to_l2_descendant_indices(11, 3, 4))


def test_overlap_priority_contract_l2_larger_than_l1_and_l0() -> None:
    assert OVERLAP_PRIORITY_L0 < OVERLAP_PRIORITY_L1 < OVERLAP_PRIORITY_L2
    assert overlap_priority_for_level("l0") == OVERLAP_PRIORITY_L0
    assert overlap_priority_for_level("L1") == OVERLAP_PRIORITY_L1
    assert overlap_priority_for_level("l2") == OVERLAP_PRIORITY_L2
    with pytest.raises(ValueError, match="Unsupported tile level"):
        overlap_priority_for_level("l3")


def test_l0_l1_l2_grid_shape_validation_errors() -> None:
    with pytest.raises(ValueError, match="exactly tile_size_l1/2"):
        l1_l2_grid_shapes_from_feature(feature_h=32, feature_w=32, tile_size_l1=4, tile_size_l2=3)
    with pytest.raises(ValueError, match="divisible"):
        l1_l2_grid_shapes_from_feature(feature_h=30, feature_w=32, tile_size_l1=4, tile_size_l2=2)
    with pytest.raises(ValueError, match="exactly tile_size_l1/2"):
        l0_l1_l2_grid_shapes_from_feature(
            feature_h=32,
            feature_w=32,
            tile_size_l0=8,
            tile_size_l1=4,
            tile_size_l2=3,
        )
