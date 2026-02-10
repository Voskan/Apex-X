from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex_x.tiles import TileSelection, TileSelectionTrace


def test_tile_selection_roundtrip_dict() -> None:
    selection = TileSelection(
        level="L1",
        indices=[3, 1, 2],
        ordered_indices=[1, 2, 3],
        meta={"grid": np.asarray([4, 4], dtype=np.int64), "name": "debug"},
        budgets_used={"b2": 3.5, "b_total": 7.0},
    )
    payload = selection.to_dict()
    loaded = TileSelection.from_dict(payload)

    assert loaded.level == "l1"
    assert loaded.indices == [3, 1, 2]
    assert loaded.ordered_indices == [1, 2, 3]
    assert loaded.meta["grid"] == [4, 4]
    assert loaded.budgets_used["b2"] == 3.5


def test_tile_selection_save_load_json(tmp_path: Path) -> None:
    path = tmp_path / "l0_selection.json"
    selection = TileSelection(
        level="l0",
        indices=[0, 2],
        ordered_indices=[2, 0],
        meta={"router": {"threshold": 0.5}, "kmax": 2},
        budgets_used={"b1": 2.0},
    )
    selection.save_json(path)
    loaded = TileSelection.load_json(path)
    assert loaded.to_dict() == selection.to_dict()


def test_tile_selection_validation_errors() -> None:
    with pytest.raises(ValueError, match="one of"):
        TileSelection(level="l3", indices=[0], ordered_indices=[0], meta={}, budgets_used={})
    with pytest.raises(ValueError, match="non-negative"):
        TileSelection(level="l0", indices=[-1], ordered_indices=[-1], meta={}, budgets_used={})
    with pytest.raises(ValueError, match="permutation"):
        TileSelection(level="l0", indices=[0, 1], ordered_indices=[0, 2], meta={}, budgets_used={})
    with pytest.raises(ValueError, match="finite"):
        TileSelection(
            level="l0",
            indices=[0],
            ordered_indices=[0],
            meta={},
            budgets_used={"b1": -1.0},
        )


def test_tile_selection_trace_roundtrip_and_lookup(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    s0 = TileSelection(
        level="l0",
        indices=[0, 1],
        ordered_indices=[1, 0],
        meta={"tag": "root"},
        budgets_used={"b1": 1.0},
    )
    s1 = TileSelection(
        level="l1",
        indices=[4, 5],
        ordered_indices=[4, 5],
        meta={"tag": "split"},
        budgets_used={"b2": 0.5},
    )
    trace = TileSelectionTrace(selections=[s0, s1], run_meta={"frame_id": 7})
    trace.save_json(path)
    loaded = TileSelectionTrace.load_json(path)

    assert loaded.for_level("l0") is not None
    assert loaded.for_level("L1") is not None
    assert loaded.for_level("l2") is None
    assert loaded.to_dict() == trace.to_dict()


def test_tile_selection_trace_requires_non_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TileSelectionTrace(selections=[], run_meta={})
