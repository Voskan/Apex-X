from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex_x.runtime import TensorRTEngineExecutor
from apex_x.runtime.tensorrt import executor as executor_module


def test_tensorrt_executor_requires_python_package_when_unavailable(tmp_path: Path) -> None:
    if executor_module.trt is not None:
        pytest.skip("TensorRT Python is available; covered by integration tests")

    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake-engine")
    executor = TensorRTEngineExecutor(engine_path=engine_path)

    with pytest.raises(RuntimeError, match="TensorRT Python package is not available"):
        executor.run(input_batch=np.zeros((1, 3, 8, 8), dtype=np.float32))


def test_resolve_named_inputs_infers_missing_key_for_multi_input() -> None:
    image = np.zeros((1, 3, 8, 8), dtype=np.float32)
    centers = np.zeros((12, 2), dtype=np.float32)
    strides = np.ones((12,), dtype=np.float32)
    resolved, primary = executor_module._resolve_named_inputs(
        input_names=("image", "centers", "strides"),
        input_batch=image,
        input_name=None,
        input_tensors={"centers": centers, "strides": strides},
    )
    assert primary == "image"
    assert set(resolved) == {"image", "centers", "strides"}
    assert resolved["image"].shape == (1, 3, 8, 8)


def test_resolve_named_inputs_rejects_ambiguous_multi_input_without_name() -> None:
    with pytest.raises(ValueError, match="multiple inputs"):
        executor_module._resolve_named_inputs(
            input_names=("image", "centers", "strides"),
            input_batch=np.zeros((1, 3, 8, 8), dtype=np.float32),
            input_name=None,
            input_tensors={"centers": np.zeros((12, 2), dtype=np.float32)},
        )


def test_resolve_named_inputs_rejects_unknown_extra_input_name() -> None:
    with pytest.raises(ValueError, match="unknown TensorRT input tensors supplied"):
        executor_module._resolve_named_inputs(
            input_names=("image",),
            input_batch=np.zeros((1, 3, 8, 8), dtype=np.float32),
            input_name=None,
            input_tensors={"unexpected": np.zeros((1,), dtype=np.float32)},
        )
