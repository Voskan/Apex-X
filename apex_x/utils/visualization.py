from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from apex_x.tiles import l0_index_to_tile


def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("batch image input must have batch size 1")
        image = image[0]

    if image.ndim != 3:
        raise ValueError("image must be HWC, CHW, or batch-size-1 variant")

    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] != 3:
        raise ValueError("image must have 3 channels")

    if np.issubdtype(image.dtype, np.floating):
        image_float = np.asarray(image, dtype=np.float32)
        max_val = float(np.max(image_float)) if image_float.size > 0 else 1.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        image_float = np.clip(image_float * scale, 0.0, 255.0)
        return cast(np.ndarray, image_float.astype(np.uint8))

    return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)


def _alpha_blend_inplace(region: np.ndarray, color: tuple[int, int, int], alpha_255: int) -> None:
    if alpha_255 <= 0:
        return
    if alpha_255 >= 255:
        region[...] = np.asarray(color, dtype=np.uint8)
        return

    src = region.astype(np.uint16)
    color_arr = np.asarray(color, dtype=np.uint16).reshape(1, 1, 3)
    inv = 255 - alpha_255
    blended = (src * inv + color_arr * alpha_255 + 127) // 255
    region[...] = blended.astype(np.uint8)


def _normalize_indices(indices: Sequence[int], max_index: int) -> list[int]:
    unique_sorted = sorted({int(idx) for idx in indices})
    for idx in unique_sorted:
        if idx < 0 or idx >= max_index:
            raise ValueError("tile index out of bounds for grid")
    return unique_sorted


def draw_selected_tiles_overlay(
    image: np.ndarray,
    selected_indices: Sequence[int],
    grid_h: int,
    grid_w: int,
    tile_size: int,
    fill_color: tuple[int, int, int] = (255, 64, 64),
    edge_color: tuple[int, int, int] = (255, 255, 255),
    fill_alpha: float = 0.35,
    edge_alpha: float = 1.0,
    line_width: int = 1,
) -> np.ndarray:
    """Draw selected tile overlays on an image and return HWC uint8 output."""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be > 0")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if line_width <= 0:
        raise ValueError("line_width must be > 0")
    if not (0.0 <= fill_alpha <= 1.0):
        raise ValueError("fill_alpha must be in [0,1]")
    if not (0.0 <= edge_alpha <= 1.0):
        raise ValueError("edge_alpha must be in [0,1]")

    base = _as_hwc_uint8(image)
    h, w, _ = base.shape
    if h < grid_h * tile_size or w < grid_w * tile_size:
        raise ValueError("image is too small for provided grid_h/grid_w/tile_size")

    overlay = base.copy()
    max_index = grid_h * grid_w
    indices = _normalize_indices(selected_indices, max_index)

    fill_alpha_255 = int(round(fill_alpha * 255.0))
    edge_alpha_255 = int(round(edge_alpha * 255.0))

    for idx in indices:
        ty, tx = l0_index_to_tile(idx, grid_h, grid_w)
        y0 = ty * tile_size
        x0 = tx * tile_size
        y1 = y0 + tile_size
        x1 = x0 + tile_size

        _alpha_blend_inplace(overlay[y0:y1, x0:x1], fill_color, fill_alpha_255)

        bw = min(line_width, tile_size)
        _alpha_blend_inplace(overlay[y0 : y0 + bw, x0:x1], edge_color, edge_alpha_255)
        _alpha_blend_inplace(overlay[y1 - bw : y1, x0:x1], edge_color, edge_alpha_255)
        _alpha_blend_inplace(overlay[y0:y1, x0 : x0 + bw], edge_color, edge_alpha_255)
        _alpha_blend_inplace(overlay[y0:y1, x1 - bw : x1], edge_color, edge_alpha_255)

    return overlay


def save_overlay_ppm(image: np.ndarray, path: str | Path) -> Path:
    """Save HWC/CHW image as binary PPM for dependency-free debug dumps."""
    out_path = Path(path)
    if out_path.suffix.lower() != ".ppm":
        raise ValueError("overlay path must use .ppm extension")

    image_u8 = _as_hwc_uint8(image)
    h, w, c = image_u8.shape
    if c != 3:
        raise ValueError("image must have 3 channels")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    out_path.write_bytes(header + image_u8.tobytes(order="C"))
    return out_path


def draw_and_save_selected_tiles_overlay(
    image: np.ndarray,
    selected_indices: Sequence[int],
    grid_h: int,
    grid_w: int,
    tile_size: int,
    output_path: str | Path,
    fill_color: tuple[int, int, int] = (255, 64, 64),
    edge_color: tuple[int, int, int] = (255, 255, 255),
    fill_alpha: float = 0.35,
    edge_alpha: float = 1.0,
    line_width: int = 1,
) -> np.ndarray:
    overlay = draw_selected_tiles_overlay(
        image=image,
        selected_indices=selected_indices,
        grid_h=grid_h,
        grid_w=grid_w,
        tile_size=tile_size,
        fill_color=fill_color,
        edge_color=edge_color,
        fill_alpha=fill_alpha,
        edge_alpha=edge_alpha,
        line_width=line_width,
    )
    save_overlay_ppm(overlay, output_path)
    return overlay
