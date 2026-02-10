from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from apex_x.utils import draw_selected_tiles_overlay, save_overlay_ppm


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_draw_selected_tiles_overlay_is_deterministic_shape_and_hash() -> None:
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    indices_a = [15, 0, 10, 5]
    indices_b = [5, 10, 15, 0, 10]

    overlay_a = draw_selected_tiles_overlay(
        image=image,
        selected_indices=indices_a,
        grid_h=4,
        grid_w=4,
        tile_size=8,
        fill_alpha=0.25,
        edge_alpha=1.0,
        line_width=1,
    )
    overlay_b = draw_selected_tiles_overlay(
        image=image,
        selected_indices=indices_b,
        grid_h=4,
        grid_w=4,
        tile_size=8,
        fill_alpha=0.25,
        edge_alpha=1.0,
        line_width=1,
    )

    assert overlay_a.shape == (32, 32, 3)
    assert overlay_a.dtype == np.uint8
    assert np.array_equal(overlay_a, overlay_b)
    assert _sha256_hex(overlay_a.tobytes()) == (
        "b12b4139b38a8d8b3f5aa8435e9271a35b58a27741fd2132b6dde25a67ca3009"
    )


def test_save_overlay_ppm_is_deterministic_and_hash_stable(tmp_path: Path) -> None:
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    overlay = draw_selected_tiles_overlay(
        image=image,
        selected_indices=[1, 2, 6],
        grid_h=4,
        grid_w=4,
        tile_size=8,
        fill_alpha=0.35,
        edge_alpha=1.0,
        line_width=2,
    )

    out_path = tmp_path / "overlay.ppm"
    save_overlay_ppm(overlay, out_path)
    file_bytes = out_path.read_bytes()

    assert file_bytes.startswith(b"P6\n32 32\n255\n")
    assert _sha256_hex(file_bytes) == (
        "9934395ff46d983fa08798cf3548e77be0ce48078e862033d4983ca2f6cdaeab"
    )


def test_save_overlay_ppm_requires_ppm_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".ppm"):
        save_overlay_ppm(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "overlay.png")
