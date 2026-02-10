from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from apex_x.runtime.caps import detect_runtime_caps
from apex_x.runtime.tensorrt import TensorRTEngineBuildConfig, TensorRTEngineBuilder


def _tiny_network_builder(network: Any, trt_mod: Any) -> None:
    x = network.add_input("x", trt_mod.DataType.FLOAT, (1, 8))
    identity = network.add_identity(x)
    out = identity.get_output(0)
    out.name = "y"
    network.mark_output(out)


def test_tensorrt_builder_requires_tensorrt_when_not_installed(tmp_path: Path) -> None:
    if importlib.util.find_spec("tensorrt") is not None:
        pytest.skip("TensorRT is installed in this environment")

    builder = TensorRTEngineBuilder()
    with pytest.raises(RuntimeError, match="tensorrt Python package is not available"):
        builder.build_from_network(
            network_builder=_tiny_network_builder,
            engine_path=tmp_path / "tiny.engine",
            build=TensorRTEngineBuildConfig(
                strict_plugin_check=False,
                expected_plugins=(),
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensorrt_builder_fp16_tiny_network_smoke(tmp_path: Path) -> None:
    caps = detect_runtime_caps()
    if not caps.tensorrt.python_available:
        pytest.skip("TensorRT Python is unavailable")

    builder = TensorRTEngineBuilder()
    engine_path = tmp_path / "tiny_fp16.engine"
    result = builder.build_from_network(
        network_builder=_tiny_network_builder,
        engine_path=engine_path,
        build=TensorRTEngineBuildConfig(
            enable_fp16=True,
            enable_int8=False,
            strict_plugin_check=False,
            expected_plugins=(),
        ),
    )

    assert result.engine_path == engine_path.resolve()
    assert result.engine_path.exists()
    assert result.used_fp16 is True
    assert result.used_int8 is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensorrt_builder_int8_tiny_network_smoke(tmp_path: Path) -> None:
    caps = detect_runtime_caps()
    if not caps.tensorrt.python_available:
        pytest.skip("TensorRT Python is unavailable")
    if not caps.tensorrt.int8_available:
        pytest.skip("TensorRT INT8 build is unavailable on this environment")

    calibration_batches = [
        np.random.RandomState(7).randn(1, 8).astype(np.float32),
        np.random.RandomState(9).randn(1, 8).astype(np.float32),
    ]

    cache_path = tmp_path / "int8.cache"
    builder = TensorRTEngineBuilder()
    engine_path = tmp_path / "tiny_int8.engine"
    result = builder.build_from_network(
        network_builder=_tiny_network_builder,
        engine_path=engine_path,
        build=TensorRTEngineBuildConfig(
            enable_fp16=True,
            enable_int8=True,
            strict_plugin_check=False,
            expected_plugins=(),
            calibration_cache_path=cache_path,
            router_fp16_layer_keywords=("router",),
        ),
        calibration_batches=calibration_batches,
    )

    assert result.engine_path == engine_path.resolve()
    assert result.engine_path.exists()
    assert result.used_int8 is True
    assert result.calibration_cache_path == cache_path
