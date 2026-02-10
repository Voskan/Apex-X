from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

from .mapping import l0_index_to_tile, l0_tile_to_index

ScanOrderMode = Literal["l2r", "r2l", "u2d", "d2u"]
OrderMode = Literal["hilbert", "multi_direction", "l2r", "r2l", "u2d", "d2u"]


_MODE_ALIASES: dict[str, OrderMode] = {
    "hilbert": "hilbert",
    "multi_direction": "multi_direction",
    "scan_lr": "l2r",
    "lr": "l2r",
    "l2r": "l2r",
    "scan_rl": "r2l",
    "rl": "r2l",
    "r2l": "r2l",
    "scan_ud": "u2d",
    "ud": "u2d",
    "u2d": "u2d",
    "scan_du": "d2u",
    "du": "d2u",
    "d2u": "d2u",
}

_SCAN_INVERSES: dict[ScanOrderMode, ScanOrderMode] = {
    "l2r": "r2l",
    "r2l": "l2r",
    "u2d": "d2u",
    "d2u": "u2d",
}


def normalize_order_mode(mode: str) -> OrderMode:
    normalized = mode.strip().lower()
    if normalized not in _MODE_ALIASES:
        raise ValueError(f"Unsupported order mode: {mode}")
    return _MODE_ALIASES[normalized]


def _to_scan_mode(mode: OrderMode) -> ScanOrderMode:
    if mode in _SCAN_INVERSES:
        return cast(ScanOrderMode, mode)
    raise ValueError(f"Mode is not a scan ordering mode: {mode}")


def inverse_scan_mode(mode: str) -> ScanOrderMode:
    normalized = normalize_order_mode(mode)
    scan_mode = _to_scan_mode(normalized)
    return _SCAN_INVERSES[scan_mode]


def _validate_grid(grid_h: int, grid_w: int) -> None:
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be > 0")


def _next_pow2(x: int) -> int:
    if x <= 0:
        raise ValueError("x must be > 0")
    out = 1
    while out < x:
        out <<= 1
    return out


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def hilbert_distance(tx: int, ty: int, order_n: int) -> int:
    """Return Hilbert curve distance for tile coordinate (ty, tx)."""
    if order_n <= 0:
        raise ValueError("order_n must be > 0")
    if tx < 0 or ty < 0:
        raise ValueError("tile coordinates must be >= 0")
    if tx >= order_n or ty >= order_n:
        raise ValueError("tile coordinates must be < order_n")

    d = 0
    s = order_n // 2
    x = tx
    y = ty
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s //= 2
    return d


def hilbert_order_coords(grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    """Return full-grid tile coordinates in deterministic Hilbert order."""
    _validate_grid(grid_h, grid_w)
    order_n = _next_pow2(max(grid_h, grid_w))
    coords = [(ty, tx) for ty in range(grid_h) for tx in range(grid_w)]
    return sorted(coords, key=lambda c: hilbert_distance(c[1], c[0], order_n))


def hilbert_order_indices(indices: Iterable[int], grid_h: int, grid_w: int) -> list[int]:
    """Sort tile indices by Hilbert rank over their (ty, tx) coordinates."""
    _validate_grid(grid_h, grid_w)
    order_n = _next_pow2(max(grid_h, grid_w))
    idx_list = [int(i) for i in indices]

    def rank(index: int) -> int:
        ty, tx = l0_index_to_tile(index, grid_h, grid_w)
        return hilbert_distance(tx, ty, order_n)

    return sorted(idx_list, key=rank)


def hilbert_full_indices(grid_h: int, grid_w: int) -> list[int]:
    """Return deterministic Hilbert traversal for all indices in the grid."""
    coords = hilbert_order_coords(grid_h, grid_w)
    return [l0_tile_to_index(ty, tx, grid_h, grid_w) for ty, tx in coords]


def _scan_rank(
    ty: int,
    tx: int,
    grid_h: int,
    grid_w: int,
    mode: ScanOrderMode,
) -> tuple[int, int]:
    if mode == "l2r":
        return ty, tx
    if mode == "r2l":
        return ty, grid_w - 1 - tx
    if mode == "u2d":
        return tx, ty
    return tx, grid_h - 1 - ty


def scan_order_coords(grid_h: int, grid_w: int, mode: str) -> list[tuple[int, int]]:
    normalized = normalize_order_mode(mode)
    scan_mode = _to_scan_mode(normalized)
    _validate_grid(grid_h, grid_w)

    coords = [(ty, tx) for ty in range(grid_h) for tx in range(grid_w)]
    return sorted(coords, key=lambda c: _scan_rank(c[0], c[1], grid_h, grid_w, scan_mode))


def scan_order_indices(indices: Iterable[int], grid_h: int, grid_w: int, mode: str) -> list[int]:
    normalized = normalize_order_mode(mode)
    scan_mode = _to_scan_mode(normalized)
    _validate_grid(grid_h, grid_w)

    idx_with_pos = [(int(idx), pos) for pos, idx in enumerate(indices)]
    return [
        idx
        for idx, _ in sorted(
            idx_with_pos,
            key=lambda pair: (
                *_scan_rank(*l0_index_to_tile(pair[0], grid_h, grid_w), grid_h, grid_w, scan_mode),
                pair[1],
            ),
        )
    ]


def _multi_direction_rank(ty: int, tx: int, grid_h: int, grid_w: int) -> int:
    l2r_rank = ty * grid_w + tx
    r2l_rank = ty * grid_w + (grid_w - 1 - tx)
    u2d_rank = tx * grid_h + ty
    d2u_rank = tx * grid_h + (grid_h - 1 - ty)
    return l2r_rank + r2l_rank + u2d_rank + d2u_rank


def order_tile_indices(
    indices: Iterable[int],
    grid_h: int,
    grid_w: int,
    mode: str = "hilbert",
) -> list[int]:
    normalized = normalize_order_mode(mode)
    idx_list = [int(i) for i in indices]
    if normalized == "hilbert":
        return hilbert_order_indices(idx_list, grid_h, grid_w)
    if normalized == "multi_direction":
        idx_with_pos = [(idx, pos) for pos, idx in enumerate(idx_list)]
        return [
            idx
            for idx, _ in sorted(
                idx_with_pos,
                key=lambda pair: (
                    _multi_direction_rank(
                        *l0_index_to_tile(pair[0], grid_h, grid_w),
                        grid_h,
                        grid_w,
                    ),
                    pair[1],
                ),
            )
        ]
    return scan_order_indices(idx_list, grid_h, grid_w, normalized)
