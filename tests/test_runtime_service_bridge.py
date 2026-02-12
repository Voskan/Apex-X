from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from apex_x.runtime.service_bridge import (
    BridgePayload,
    BridgeRequestItem,
    _reshape_flat_input,
    _run_bridge,
    _score_from_arrays,
)


def test_reshape_flat_input_respects_static_shape() -> None:
    flat = np.arange(12, dtype=np.float32)
    shaped = _reshape_flat_input(flat, [1, 3, 2, 2])
    assert shaped.shape == (1, 3, 2, 2)


def test_reshape_flat_input_resolves_dynamic_square_dims() -> None:
    flat = np.arange(3 * 8 * 8, dtype=np.float32)
    shaped = _reshape_flat_input(flat, [-1, 3, -1, -1])
    assert shaped.shape == (1, 3, 8, 8)


def test_reshape_flat_input_resolves_single_dynamic_dim() -> None:
    flat = np.arange(12, dtype=np.float32)
    shaped = _reshape_flat_input(flat, [1, -1])
    assert shaped.shape == (1, 12)


def test_reshape_flat_input_falls_back_when_dynamic_dims_are_unresolvable() -> None:
    flat = np.arange(13, dtype=np.float32)
    shaped = _reshape_flat_input(flat, [-1, 3, -1, -1])
    assert shaped.shape == (1, 13)


def test_bridge_request_item_validation() -> None:
    # Valid - input can be list
    item = BridgeRequestItem(request_id="r1", input=[1.0, 2.0])
    assert item.request_id == "r1"
    assert item.input_values == [1.0, 2.0]
    assert item.budget_profile == "balanced"  # default

    # Missing request_id (Pydantic raises ValidationError)
    with pytest.raises(ValidationError):
        BridgeRequestItem(input=[1.0]) # type: ignore

    # Empty request_id
    with pytest.raises(ValidationError):
         BridgeRequestItem(request_id="", input=[1.0])

    # Numpy input conversion (handled by validator)
    item_np = BridgeRequestItem(
        request_id="r2", 
        input=np.array([1.0, 2.0], dtype=np.float32)
    )
    assert item_np.input_values == [1.0, 2.0]


def test_score_from_arrays_clips_to_unit_interval() -> None:
    score = _score_from_arrays([np.asarray([100.0], dtype=np.float32)])
    assert score == 1.0


def test_bridge_payload_validation(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.touch()
    
    # Valid payload
    payload = BridgePayload(
        backend="onnxruntime",
        artifact_path=str(artifact),
        requests=[{"request_id": "r1", "input": [1.0]}]
    )
    assert payload.backend == "onnxruntime"
    assert len(payload.requests) == 1

    # Invalid backend enum
    with pytest.raises(ValidationError):
        BridgePayload(
            backend="unknown_backend", # type: ignore
            artifact_path=str(artifact)
        )

    # Missing artifact path
    with pytest.raises(ValidationError):
        BridgePayload(
            backend="onnxruntime",
            artifact_path=""
        )


def test_run_bridge_validates_path_existence(tmp_path: Path) -> None:
    # Path that doesn't exist
    non_existent = tmp_path / "gone.onnx"
    
    payload_dict = {
        "backend": "onnxruntime",
        "artifact_path": str(non_existent),
        "requests": []
    }
    
    
    with pytest.raises(FileNotFoundError):
        _run_bridge(payload_dict)


def test_bridge_health_check() -> None:
    payload = BridgePayload(backend="health")
    assert payload.backend == "health"
    # Artifact path should be optional/ignored for health
    assert payload.artifact_path == "" 
    
    result = _run_bridge({"backend": "health"})
    assert result["status"] == "ok"
    assert "backends" in result
    assert "onnxruntime" in result["backends"]

