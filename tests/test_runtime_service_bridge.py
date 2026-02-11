from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex_x.runtime.service_bridge import (
    _parse_items,
    _reshape_flat_input,
    _run_bridge,
    _score_from_arrays,
)


def test_reshape_flat_input_respects_static_shape() -> None:
    flat = np.arange(12, dtype=np.float32)
    shaped = _reshape_flat_input(flat, [1, 3, 2, 2])
    assert shaped.shape == (1, 3, 2, 2)


def test_parse_items_rejects_missing_request_id() -> None:
    with pytest.raises(ValueError, match="request_id"):
        _parse_items([{"budget_profile": "balanced", "input": [1.0]}])


def test_score_from_arrays_clips_to_unit_interval() -> None:
    score = _score_from_arrays([np.asarray([100.0], dtype=np.float32)])
    assert score == 1.0


def test_run_bridge_rejects_unknown_backend(tmp_path: Path) -> None:
    artifact = tmp_path / "dummy.bin"
    artifact.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported backend"):
        _run_bridge(
            {
                "backend": "unknown",
                "artifact_path": str(artifact),
                "requests": [{"request_id": "r1", "budget_profile": "balanced", "input": [0.1]}],
            }
        )
