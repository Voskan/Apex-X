from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from .mapping import l0_grid_shape, l0_index_to_tile, l0_tile_to_index

TileLevel = Literal["l0", "l1", "l2"]

# Overlap priority contract: L2 > L1 > L0.
OVERLAP_PRIORITY_L0 = 1
OVERLAP_PRIORITY_L1 = 2
OVERLAP_PRIORITY_L2 = 3
OVERLAP_PRIORITY_BY_LEVEL: dict[TileLevel, int] = {
    "l0": OVERLAP_PRIORITY_L0,
    "l1": OVERLAP_PRIORITY_L1,
    "l2": OVERLAP_PRIORITY_L2,
}


@dataclass(frozen=True)
class L0L1QuadtreeMeta:
    """Metadata for deterministic L0->L1 quadtree expansion."""

    l0_grid_h: int
    l0_grid_w: int
    l1_grid_h: int
    l1_grid_w: int
    parent_indices: np.ndarray  # [K]
    parent_coords: np.ndarray  # [K, 2]
    child_indices: np.ndarray  # [K, 4]
    child_coords: np.ndarray  # [K, 4, 2]


@dataclass(frozen=True)
class L1L2QuadtreeMeta:
    """Metadata for deterministic L1->L2 quadtree expansion."""

    l1_grid_h: int
    l1_grid_w: int
    l2_grid_h: int
    l2_grid_w: int
    parent_indices: np.ndarray  # [K]
    parent_coords: np.ndarray  # [K, 2]
    child_indices: np.ndarray  # [K, 4]
    child_coords: np.ndarray  # [K, 4, 2]


@dataclass(frozen=True)
class L0L1L2QuadtreeMeta:
    """Combined metadata bundle for nesting depth=2."""

    l0_l1: L0L1QuadtreeMeta
    l1_l2: L1L2QuadtreeMeta


def _normalize_level(level: str) -> TileLevel:
    normalized = level.strip().lower()
    if normalized not in OVERLAP_PRIORITY_BY_LEVEL:
        raise ValueError(f"Unsupported tile level: {level}")
    return cast(TileLevel, normalized)


def overlap_priority_for_level(level: str) -> int:
    return OVERLAP_PRIORITY_BY_LEVEL[_normalize_level(level)]


def l1_grid_shape_from_l0(l0_grid_h: int, l0_grid_w: int) -> tuple[int, int]:
    if l0_grid_h <= 0 or l0_grid_w <= 0:
        raise ValueError("l0_grid_h and l0_grid_w must be > 0")
    return 2 * l0_grid_h, 2 * l0_grid_w


def l2_grid_shape_from_l1(l1_grid_h: int, l1_grid_w: int) -> tuple[int, int]:
    if l1_grid_h <= 0 or l1_grid_w <= 0:
        raise ValueError("l1_grid_h and l1_grid_w must be > 0")
    return 2 * l1_grid_h, 2 * l1_grid_w


def l0_l1_grid_shapes_from_feature(
    feature_h: int,
    feature_w: int,
    tile_size_l0: int,
    tile_size_l1: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if tile_size_l0 <= 0 or tile_size_l1 <= 0:
        raise ValueError("tile_size_l0 and tile_size_l1 must be > 0")
    if tile_size_l0 != 2 * tile_size_l1:
        raise ValueError("tile_size_l1 must be exactly tile_size_l0/2")

    l0_grid = l0_grid_shape(feature_h, feature_w, tile_size_l0)
    l1_grid = l0_grid_shape(feature_h, feature_w, tile_size_l1)
    expected_l1 = l1_grid_shape_from_l0(*l0_grid)
    if l1_grid != expected_l1:
        raise ValueError("l1 grid must be exactly 2x l0 grid")
    return l0_grid, l1_grid


def l1_l2_grid_shapes_from_feature(
    feature_h: int,
    feature_w: int,
    tile_size_l1: int,
    tile_size_l2: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if tile_size_l1 <= 0 or tile_size_l2 <= 0:
        raise ValueError("tile_size_l1 and tile_size_l2 must be > 0")
    if tile_size_l1 != 2 * tile_size_l2:
        raise ValueError("tile_size_l2 must be exactly tile_size_l1/2")

    l1_grid = l0_grid_shape(feature_h, feature_w, tile_size_l1)
    l2_grid = l0_grid_shape(feature_h, feature_w, tile_size_l2)
    expected_l2 = l2_grid_shape_from_l1(*l1_grid)
    if l2_grid != expected_l2:
        raise ValueError("l2 grid must be exactly 2x l1 grid")
    return l1_grid, l2_grid


def l0_l1_l2_grid_shapes_from_feature(
    feature_h: int,
    feature_w: int,
    tile_size_l0: int,
    tile_size_l1: int,
    tile_size_l2: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    l0_grid, l1_grid = l0_l1_grid_shapes_from_feature(
        feature_h=feature_h,
        feature_w=feature_w,
        tile_size_l0=tile_size_l0,
        tile_size_l1=tile_size_l1,
    )
    l1_grid_check, l2_grid = l1_l2_grid_shapes_from_feature(
        feature_h=feature_h,
        feature_w=feature_w,
        tile_size_l1=tile_size_l1,
        tile_size_l2=tile_size_l2,
    )
    if l1_grid_check != l1_grid:
        raise ValueError("l1 grid mismatch between l0->l1 and l1->l2 shape checks")
    return l0_grid, l1_grid, l2_grid


def l0_to_l1_children_coords(
    l0_ty: int,
    l0_tx: int,
    l0_grid_h: int,
    l0_grid_w: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    _ = l0_tile_to_index(l0_ty, l0_tx, l0_grid_h, l0_grid_w)
    base_ty = 2 * l0_ty
    base_tx = 2 * l0_tx
    return (
        (base_ty, base_tx),
        (base_ty, base_tx + 1),
        (base_ty + 1, base_tx),
        (base_ty + 1, base_tx + 1),
    )


def l0_to_l1_children_indices(
    l0_index: int,
    l0_grid_h: int,
    l0_grid_w: int,
) -> np.ndarray:
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    l0_ty, l0_tx = l0_index_to_tile(l0_index, l0_grid_h, l0_grid_w)
    children = l0_to_l1_children_coords(l0_ty, l0_tx, l0_grid_h, l0_grid_w)
    return np.asarray(
        [l0_tile_to_index(ty, tx, l1_grid_h, l1_grid_w) for ty, tx in children],
        dtype=np.int64,
    )


def l1_to_l2_children_coords(
    l1_ty: int,
    l1_tx: int,
    l1_grid_h: int,
    l1_grid_w: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    _ = l0_tile_to_index(l1_ty, l1_tx, l1_grid_h, l1_grid_w)
    base_ty = 2 * l1_ty
    base_tx = 2 * l1_tx
    return (
        (base_ty, base_tx),
        (base_ty, base_tx + 1),
        (base_ty + 1, base_tx),
        (base_ty + 1, base_tx + 1),
    )


def l1_to_l2_children_indices(
    l1_index: int,
    l1_grid_h: int,
    l1_grid_w: int,
) -> np.ndarray:
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)
    l1_ty, l1_tx = l0_index_to_tile(l1_index, l1_grid_h, l1_grid_w)
    children = l1_to_l2_children_coords(l1_ty, l1_tx, l1_grid_h, l1_grid_w)
    return np.asarray(
        [l0_tile_to_index(ty, tx, l2_grid_h, l2_grid_w) for ty, tx in children],
        dtype=np.int64,
    )


def l1_to_l0_parent_coord(
    l1_ty: int,
    l1_tx: int,
    l0_grid_h: int,
    l0_grid_w: int,
) -> tuple[int, int]:
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    _ = l0_tile_to_index(l1_ty, l1_tx, l1_grid_h, l1_grid_w)
    return l1_ty // 2, l1_tx // 2


def l1_to_l0_parent_index(
    l1_index: int,
    l0_grid_h: int,
    l0_grid_w: int,
) -> int:
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    l1_ty, l1_tx = l0_index_to_tile(l1_index, l1_grid_h, l1_grid_w)
    p_ty, p_tx = l1_to_l0_parent_coord(l1_ty, l1_tx, l0_grid_h, l0_grid_w)
    return l0_tile_to_index(p_ty, p_tx, l0_grid_h, l0_grid_w)


def l2_to_l1_parent_coord(
    l2_ty: int,
    l2_tx: int,
    l1_grid_h: int,
    l1_grid_w: int,
) -> tuple[int, int]:
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)
    _ = l0_tile_to_index(l2_ty, l2_tx, l2_grid_h, l2_grid_w)
    return l2_ty // 2, l2_tx // 2


def l2_to_l1_parent_index(
    l2_index: int,
    l1_grid_h: int,
    l1_grid_w: int,
) -> int:
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)
    l2_ty, l2_tx = l0_index_to_tile(l2_index, l2_grid_h, l2_grid_w)
    p_ty, p_tx = l2_to_l1_parent_coord(l2_ty, l2_tx, l1_grid_h, l1_grid_w)
    return l0_tile_to_index(p_ty, p_tx, l1_grid_h, l1_grid_w)


def l0_to_l2_descendant_indices(
    l0_index: int,
    l0_grid_h: int,
    l0_grid_w: int,
) -> np.ndarray:
    l1_children = l0_to_l1_children_indices(l0_index, l0_grid_h, l0_grid_w)
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    out = np.zeros((16,), dtype=np.int64)
    cursor = 0
    for l1_index in l1_children.tolist():
        l2_children = l1_to_l2_children_indices(l1_index, l1_grid_h, l1_grid_w)
        out[cursor : cursor + 4] = l2_children
        cursor += 4
    return out


def build_l0_l1_quadtree_meta(
    parent_indices: Iterable[int],
    l0_grid_h: int,
    l0_grid_w: int,
) -> L0L1QuadtreeMeta:
    l1_grid_h, l1_grid_w = l1_grid_shape_from_l0(l0_grid_h, l0_grid_w)
    parent_list = [int(i) for i in parent_indices]
    k = len(parent_list)

    parent_idx_arr = np.asarray(parent_list, dtype=np.int64)
    parent_coords = np.zeros((k, 2), dtype=np.int64)
    child_indices = np.zeros((k, 4), dtype=np.int64)
    child_coords = np.zeros((k, 4, 2), dtype=np.int64)

    for row, l0_index in enumerate(parent_list):
        p_ty, p_tx = l0_index_to_tile(l0_index, l0_grid_h, l0_grid_w)
        parent_coords[row] = np.asarray([p_ty, p_tx], dtype=np.int64)

        children_xy = l0_to_l1_children_coords(p_ty, p_tx, l0_grid_h, l0_grid_w)
        for col, (c_ty, c_tx) in enumerate(children_xy):
            child_coords[row, col] = np.asarray([c_ty, c_tx], dtype=np.int64)
            child_indices[row, col] = l0_tile_to_index(c_ty, c_tx, l1_grid_h, l1_grid_w)

    return L0L1QuadtreeMeta(
        l0_grid_h=l0_grid_h,
        l0_grid_w=l0_grid_w,
        l1_grid_h=l1_grid_h,
        l1_grid_w=l1_grid_w,
        parent_indices=parent_idx_arr,
        parent_coords=parent_coords,
        child_indices=child_indices,
        child_coords=child_coords,
    )


def build_l1_l2_quadtree_meta(
    parent_indices: Iterable[int],
    l1_grid_h: int,
    l1_grid_w: int,
) -> L1L2QuadtreeMeta:
    l2_grid_h, l2_grid_w = l2_grid_shape_from_l1(l1_grid_h, l1_grid_w)
    parent_list = [int(i) for i in parent_indices]
    k = len(parent_list)

    parent_idx_arr = np.asarray(parent_list, dtype=np.int64)
    parent_coords = np.zeros((k, 2), dtype=np.int64)
    child_indices = np.zeros((k, 4), dtype=np.int64)
    child_coords = np.zeros((k, 4, 2), dtype=np.int64)

    for row, l1_index in enumerate(parent_list):
        p_ty, p_tx = l0_index_to_tile(l1_index, l1_grid_h, l1_grid_w)
        parent_coords[row] = np.asarray([p_ty, p_tx], dtype=np.int64)

        children_xy = l1_to_l2_children_coords(p_ty, p_tx, l1_grid_h, l1_grid_w)
        for col, (c_ty, c_tx) in enumerate(children_xy):
            child_coords[row, col] = np.asarray([c_ty, c_tx], dtype=np.int64)
            child_indices[row, col] = l0_tile_to_index(c_ty, c_tx, l2_grid_h, l2_grid_w)

    return L1L2QuadtreeMeta(
        l1_grid_h=l1_grid_h,
        l1_grid_w=l1_grid_w,
        l2_grid_h=l2_grid_h,
        l2_grid_w=l2_grid_w,
        parent_indices=parent_idx_arr,
        parent_coords=parent_coords,
        child_indices=child_indices,
        child_coords=child_coords,
    )


def build_l0_l1_l2_quadtree_meta(
    l0_parent_indices: Iterable[int],
    l0_grid_h: int,
    l0_grid_w: int,
) -> L0L1L2QuadtreeMeta:
    l0_l1 = build_l0_l1_quadtree_meta(
        parent_indices=l0_parent_indices,
        l0_grid_h=l0_grid_h,
        l0_grid_w=l0_grid_w,
    )
    l1_parents = l0_l1.child_indices.reshape(-1)
    l1_l2 = build_l1_l2_quadtree_meta(
        parent_indices=l1_parents.tolist(),
        l1_grid_h=l0_l1.l1_grid_h,
        l1_grid_w=l0_l1.l1_grid_w,
    )
    return L0L1L2QuadtreeMeta(l0_l1=l0_l1, l1_l2=l1_l2)
