from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .interfaces import TileMeta, TilePackerProtocol, TileUnpackerProtocol
from .mapping import l0_grid_shape
from .ordering import order_tile_indices


class NumpyTileCodec(TilePackerProtocol, TileUnpackerProtocol):
    """Reference NumPy implementation of tile pack/unpack interfaces."""

    def pack(
        self,
        feature_map: np.ndarray,
        indices: np.ndarray,
        tile_size: int,
        order_mode: str = "hilbert",
    ) -> tuple[np.ndarray, TileMeta]:
        return pack_tiles(feature_map, indices, tile_size, order_mode)

    def unpack(
        self,
        base_map: np.ndarray,
        packed_out: np.ndarray,
        meta: TileMeta,
        level_priority: int = 1,
        priority_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return unpack_tiles(base_map, packed_out, meta, level_priority, priority_map)


def tile_grid_shape(height: int, width: int, tile_size: int) -> tuple[int, int]:
    return l0_grid_shape(height, width, tile_size)


def order_idx(indices: Iterable[int], grid_h: int, grid_w: int, mode: str = "hilbert") -> list[int]:
    return order_tile_indices(indices, grid_h, grid_w, mode=mode)


def pack_tiles(
    feature_map: np.ndarray,
    indices: np.ndarray,
    tile_size: int,
    order_mode: str = "hilbert",
) -> tuple[np.ndarray, TileMeta]:
    """Gather selected tiles into contiguous [B, K, C, t, t] tensor."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [B,C,H,W]")
    if indices.ndim != 2:
        raise ValueError("indices must be [B,K]")

    bsz, channels, height, width = feature_map.shape
    grid_h, grid_w = tile_grid_shape(height, width, tile_size)
    kmax = indices.shape[1]

    packed = np.zeros((bsz, kmax, channels, tile_size, tile_size), dtype=feature_map.dtype)
    sorted_idx = np.zeros_like(indices)
    origins = np.zeros((bsz, kmax, 2), dtype=np.int64)

    for b in range(bsz):
        ordered = order_idx(indices[b].tolist(), grid_h, grid_w, order_mode)
        sorted_idx[b] = np.asarray(ordered, dtype=np.int64)
        for k, idx in enumerate(ordered):
            y = (idx // grid_w) * tile_size
            x = (idx % grid_w) * tile_size
            origins[b, k, 0] = y
            origins[b, k, 1] = x
            packed[b, k] = feature_map[b, :, y : y + tile_size, x : x + tile_size]

    meta: TileMeta = {
        "indices": sorted_idx,
        "origins": origins,
        "tile_size": np.asarray(tile_size),
        "grid": np.asarray([grid_h, grid_w], dtype=np.int64),
    }
    return packed, meta


def unpack_tiles(
    base_map: np.ndarray,
    packed_out: np.ndarray,
    meta: TileMeta,
    level_priority: int = 1,
    priority_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter heavy tiles into base map; larger priority wins at overlap."""
    merged = base_map.copy()
    bsz, kmax, channels, tile_size, _ = packed_out.shape
    if priority_map is None:
        priority_map = np.zeros(
            (base_map.shape[0], base_map.shape[2], base_map.shape[3]),
            dtype=np.int8,
        )

    origins = meta["origins"]
    for b in range(bsz):
        for k in range(kmax):
            y = int(origins[b, k, 0])
            x = int(origins[b, k, 1])
            patch_priority = priority_map[b, y : y + tile_size, x : x + tile_size]
            mask = level_priority >= patch_priority
            if np.any(mask):
                for c in range(channels):
                    patch = merged[b, c, y : y + tile_size, x : x + tile_size]
                    patch[mask] = packed_out[b, k, c][mask]
                    merged[b, c, y : y + tile_size, x : x + tile_size] = patch
                priority_map[b, y : y + tile_size, x : x + tile_size][mask] = level_priority

    return merged, priority_map
