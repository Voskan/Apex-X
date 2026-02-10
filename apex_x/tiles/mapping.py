from __future__ import annotations

import numpy as np


def l0_grid_shape(feature_h: int, feature_w: int, tile_size: int) -> tuple[int, int]:
    """Compute L0 tile grid shape with strict divisibility checks."""
    if feature_h <= 0 or feature_w <= 0:
        raise ValueError("feature_h and feature_w must be > 0")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if feature_h % tile_size != 0 or feature_w % tile_size != 0:
        raise ValueError("feature map dimensions must be divisible by tile_size")
    return feature_h // tile_size, feature_w // tile_size


def l0_tile_to_index(ty: int, tx: int, grid_h: int, grid_w: int) -> int:
    """Convert tile coordinates (ty, tx) to linear tile index."""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be > 0")
    if ty < 0 or ty >= grid_h or tx < 0 or tx >= grid_w:
        raise ValueError("tile coordinates out of bounds")
    return ty * grid_w + tx


def l0_index_to_tile(index: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    """Convert linear tile index to tile coordinates (ty, tx)."""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be > 0")
    if index < 0 or index >= grid_h * grid_w:
        raise ValueError("tile index out of bounds")
    return index // grid_w, index % grid_w


def l0_indices_to_coords(indices: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Map batched tile indices [B,K] to tile coordinates [B,K,2]."""
    if indices.ndim != 2:
        raise ValueError("indices must be rank-2 [B,K]")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must have integer dtype")

    bsz, k = indices.shape
    coords = np.zeros((bsz, k, 2), dtype=np.int64)
    for b in range(bsz):
        for j in range(k):
            ty, tx = l0_index_to_tile(int(indices[b, j]), grid_h, grid_w)
            coords[b, j, 0] = ty
            coords[b, j, 1] = tx
    return coords


def l0_coords_to_indices(coords: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Map batched tile coordinates [B,K,2] to tile indices [B,K]."""
    if coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError("coords must be rank-3 [B,K,2]")
    if not np.issubdtype(coords.dtype, np.integer):
        raise TypeError("coords must have integer dtype")

    bsz, k, _ = coords.shape
    indices = np.zeros((bsz, k), dtype=np.int64)
    for b in range(bsz):
        for j in range(k):
            ty = int(coords[b, j, 0])
            tx = int(coords[b, j, 1])
            indices[b, j] = l0_tile_to_index(ty, tx, grid_h, grid_w)
    return indices
