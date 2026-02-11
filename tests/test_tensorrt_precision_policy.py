from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from apex_x.runtime.caps import CudaCaps, FP8Caps, RuntimeCaps, TensorRTCaps, TritonCaps
from apex_x.runtime.tensorrt import TensorRTEngineBuildConfig, TensorRTEngineBuilder


class _FakeTRT:
    float16 = object()

    class Logger:
        WARNING = 1

        def __init__(self, level: int) -> None:
            self.level = level

    class BuilderFlag:
        FP16 = "FP16"
        INT8 = "INT8"
        PREFER_PRECISION_CONSTRAINTS = "PREFER_PRECISION_CONSTRAINTS"

    class MemoryPoolType:
        WORKSPACE = "WORKSPACE"


@dataclass
class _FakeLayer:
    name: str
    num_outputs: int = 1
    with_precision: bool = True
    with_output_type: bool = True

    def __post_init__(self) -> None:
        self.output_constraints: list[tuple[int, Any]] = []
        if self.with_precision:
            self.precision = None
        if self.with_output_type:
            self.set_output_type = self._set_output_type  # type: ignore[method-assign]

    def _set_output_type(self, output_idx: int, dtype: Any) -> None:
        self.output_constraints.append((int(output_idx), dtype))


@dataclass(frozen=True)
class _FakeInput:
    name: str


class _FakeNetwork:
    def __init__(self, layers: list[_FakeLayer], *, input_names: tuple[str, ...] = ("x",)) -> None:
        self._layers = layers
        self._inputs = [_FakeInput(name=value) for value in input_names]
        self.num_layers = len(layers)
        self.num_inputs = len(self._inputs)

    def get_layer(self, index: int) -> _FakeLayer:
        return self._layers[int(index)]

    def get_input(self, index: int) -> _FakeInput:
        return self._inputs[int(index)]


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.flags: list[Any] = []
        self.memory_pool_limits: dict[Any, int] = {}
        self.int8_calibrator: Any | None = None

    def set_flag(self, flag: Any) -> None:
        self.flags.append(flag)

    def set_memory_pool_limit(self, pool_type: Any, value: int) -> None:
        self.memory_pool_limits[pool_type] = int(value)


class _FakeBuilder:
    def __init__(self) -> None:
        self.config = _FakeBuilderConfig()

    def create_builder_config(self) -> _FakeBuilderConfig:
        return self.config


class _DummyCalibrator:
    def __init__(self, batches: Any, *, config: Any) -> None:
        self.batches = batches
        self.config = config


def _runtime_caps_cuda_tensorrt_available() -> RuntimeCaps:
    return RuntimeCaps(
        cuda=CudaCaps(
            available=True,
            device_count=1,
            device_name="Fake CUDA",
            compute_capability=(9, 0),
            reason=None,
        ),
        triton=TritonCaps(available=False, version=None, reason="triton_not_installed"),
        tensorrt=TensorRTCaps(
            python_available=True,
            python_version="10.0.0",
            python_reason=None,
            headers_available=True,
            header_path="/fake/include/NvInfer.h",
            int8_available=True,
            int8_reason=None,
        ),
        fp8=FP8Caps(
            available=True,
            dtype_available=True,
            supported_dtypes=("float8_e4m3fn",),
            reason=None,
        ),
    )


def test_mark_router_layers_fp16_applies_constraints_for_sensitive_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apex_x.runtime.tensorrt import builder as builder_module

    monkeypatch.setattr(builder_module, "trt", _FakeTRT)
    builder = TensorRTEngineBuilder(runtime_caps=_runtime_caps_cuda_tensorrt_available())
    layers = [
        _FakeLayer("router_block"),
        _FakeLayer("det_head"),
        _FakeLayer("kan_gate"),
    ]
    network = _FakeNetwork(layers)

    statuses = builder._mark_router_layers_fp16(  # pyright: ignore[reportPrivateUsage]
        network,
        ("router", "kan"),
        strict=True,
    )

    assert [status.layer_name for status in statuses] == ["router_block", "kan_gate"]
    assert all(status.precision_applied for status in statuses)
    assert all(status.output_constraints_applied == 1 for status in statuses)
    assert layers[0].precision is _FakeTRT.float16
    assert layers[1].precision is None
    assert layers[2].precision is _FakeTRT.float16


def test_mark_router_layers_fp16_strict_mode_rejects_unconstrainable_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apex_x.runtime.tensorrt import builder as builder_module

    monkeypatch.setattr(builder_module, "trt", _FakeTRT)
    builder = TensorRTEngineBuilder(runtime_caps=_runtime_caps_cuda_tensorrt_available())
    network = _FakeNetwork(
        [_FakeLayer("router_missing", with_precision=False, with_output_type=False)]
    )

    with pytest.raises(RuntimeError, match="unable to enforce FP16 precision constraint"):
        builder._mark_router_layers_fp16(  # pyright: ignore[reportPrivateUsage]
            network,
            ("router",),
            strict=True,
        )


def test_prepare_builder_config_int8_emits_layer_precision_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apex_x.runtime.tensorrt import builder as builder_module

    monkeypatch.setattr(builder_module, "trt", _FakeTRT)
    monkeypatch.setattr(builder_module, "TensorRTEntropyCalibrator", _DummyCalibrator)

    builder = TensorRTEngineBuilder(runtime_caps=_runtime_caps_cuda_tensorrt_available())
    fake_builder = _FakeBuilder()
    network = _FakeNetwork([_FakeLayer("router_gate"), _FakeLayer("head_out")])
    build_cfg = TensorRTEngineBuildConfig(
        enable_fp16=True,
        enable_int8=True,
        strict_plugin_check=False,
        expected_plugins=(),
        router_fp16_layer_keywords=("router",),
    )
    calibration_batches = [np.ones((1, 8), dtype=np.float32)]

    config_obj, used_fp16, used_int8, layer_status = (
        builder._prepare_builder_config(  # pyright: ignore[reportPrivateUsage]
            builder=fake_builder,
            network=network,
            build=build_cfg,
            calibration_batches=calibration_batches,
        )
    )

    assert used_fp16 is True
    assert used_int8 is True
    assert config_obj.flags == [
        _FakeTRT.BuilderFlag.FP16,
        _FakeTRT.BuilderFlag.INT8,
        _FakeTRT.BuilderFlag.PREFER_PRECISION_CONSTRAINTS,
    ]
    assert isinstance(config_obj.int8_calibrator, _DummyCalibrator)
    assert len(layer_status) == 1
    assert layer_status[0].layer_name == "router_gate"
    assert layer_status[0].precision_applied is True
    assert layer_status[0].output_constraints_applied == 1
