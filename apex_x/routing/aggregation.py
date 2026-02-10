from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PVTileAggregation:
    """Aggregated PV statistics aligned to FF tile grid."""

    vectors: np.ndarray  # [B, K, D]
    tile_bounds_pv: np.ndarray  # [K, 4] as (y0, y1, x0, x1)
    feature_layout: tuple[str, ...]  # length D entries: "{map}:{stat}:c{channel}"


def _validate_pv_maps(pv_maps: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    if not pv_maps:
        raise ValueError("pv_maps must not be empty")

    batch_size: int | None = None
    pv_h: int | None = None
    pv_w: int | None = None

    for name, fmap in pv_maps.items():
        if fmap.ndim != 4:
            raise ValueError(f"pv map {name!r} must be [B,C,H,W]")
        bsz, _ch, h, w = fmap.shape
        if bsz <= 0 or h <= 0 or w <= 0:
            raise ValueError("pv map shapes must be positive")
        if batch_size is None:
            batch_size = bsz
            pv_h = h
            pv_w = w
        else:
            if bsz != batch_size:
                raise ValueError("all pv maps must share the same batch size")
            if h != pv_h or w != pv_w:
                raise ValueError("all pv maps must share the same spatial shape")

    assert batch_size is not None and pv_h is not None and pv_w is not None
    return batch_size, pv_h, pv_w


def compute_ff_tile_bounds_on_pv_grid(
    ff_h: int,
    ff_w: int,
    ff_tile_size: int,
    pv_h: int,
    pv_w: int,
) -> np.ndarray:
    """Map each FF tile to aligned PV bounds."""
    if ff_h <= 0 or ff_w <= 0 or ff_tile_size <= 0:
        raise ValueError("ff_h, ff_w, and ff_tile_size must be > 0")
    if pv_h <= 0 or pv_w <= 0:
        raise ValueError("pv_h and pv_w must be > 0")
    if ff_h % ff_tile_size != 0 or ff_w % ff_tile_size != 0:
        raise ValueError("ff_h and ff_w must be divisible by ff_tile_size")

    grid_h = ff_h // ff_tile_size
    grid_w = ff_w // ff_tile_size
    bounds = np.zeros((grid_h * grid_w, 4), dtype=np.int64)

    k = 0
    for ty in range(grid_h):
        y0_ff = ty * ff_tile_size
        y1_ff = y0_ff + ff_tile_size
        y0 = int(math.floor(y0_ff * pv_h / ff_h))
        y1 = int(math.ceil(y1_ff * pv_h / ff_h))
        y0 = max(0, min(y0, pv_h - 1))
        y1 = max(y0 + 1, min(y1, pv_h))

        for tx in range(grid_w):
            x0_ff = tx * ff_tile_size
            x1_ff = x0_ff + ff_tile_size
            x0 = int(math.floor(x0_ff * pv_w / ff_w))
            x1 = int(math.ceil(x1_ff * pv_w / ff_w))
            x0 = max(0, min(x0, pv_w - 1))
            x1 = max(x0 + 1, min(x1, pv_w))

            bounds[k] = np.asarray([y0, y1, x0, x1], dtype=np.int64)
            k += 1

    return bounds


def aggregate_pv_maps_to_ff_tile_vectors(
    pv_maps: Mapping[str, np.ndarray],
    ff_h: int,
    ff_w: int,
    ff_tile_size: int,
    stat_order: Sequence[str] = ("mean", "max", "var"),
) -> PVTileAggregation:
    """Aggregate mean/max/var on PV maps aligned to FF tile grid."""
    if not stat_order:
        raise ValueError("stat_order must not be empty")
    valid_stats = {"mean", "max", "var"}
    for stat in stat_order:
        if stat not in valid_stats:
            raise ValueError(f"unsupported stat {stat!r}; expected mean/max/var")

    bsz, pv_h, pv_w = _validate_pv_maps(pv_maps)
    bounds = compute_ff_tile_bounds_on_pv_grid(ff_h, ff_w, ff_tile_size, pv_h, pv_w)
    k_tiles = bounds.shape[0]

    ordered_names = sorted(pv_maps.keys())
    dim = 0
    layout: list[str] = []
    for name in ordered_names:
        channels = pv_maps[name].shape[1]
        for stat in stat_order:
            for c in range(channels):
                layout.append(f"{name}:{stat}:c{c}")
                dim += 1

    vectors = np.zeros((bsz, k_tiles, dim), dtype=np.float32)
    for k, (y0, y1, x0, x1) in enumerate(bounds.tolist()):
        cursor = 0
        for name in ordered_names:
            patch = pv_maps[name][:, :, y0:y1, x0:x1]  # [B, C, ph, pw]
            for stat in stat_order:
                if stat == "mean":
                    stat_values = patch.mean(axis=(2, 3), dtype=np.float32)
                elif stat == "max":
                    stat_values = patch.max(axis=(2, 3))
                else:
                    stat_values = patch.var(axis=(2, 3), dtype=np.float32)
                channels = stat_values.shape[1]
                vectors[:, k, cursor : cursor + channels] = stat_values
                cursor += channels

    return PVTileAggregation(
        vectors=vectors,
        tile_bounds_pv=bounds,
        feature_layout=tuple(layout),
    )
