from __future__ import annotations

from dataclasses import dataclass

import pytest

from apex_x.runtime.caps import CudaCaps, FP8Caps, RuntimeCaps, TensorRTCaps, TritonCaps
from apex_x.runtime.tensorrt import TensorRTEngineBuildConfig, TensorRTEngineBuilder


@dataclass(frozen=True, slots=True)
class _FakePluginField:
    name: str


@dataclass(slots=True)
class _FakePluginCreator:
    name: str
    plugin_version: str
    plugin_namespace: str
    field_names: tuple[_FakePluginField, ...]


@dataclass(slots=True)
class _FakePluginRegistry:
    plugin_creator_list: tuple[_FakePluginCreator, ...]

    def get_plugin_creator(
        self,
        name: str,
        version: str,
        namespace: str,
    ) -> _FakePluginCreator | None:
        for creator in self.plugin_creator_list:
            if (
                creator.name == name
                and creator.plugin_version == version
                and creator.plugin_namespace == namespace
            ):
                return creator
        return None


class _FakeTRTModule:
    class Logger:
        WARNING = 1

        def __init__(self, level: int) -> None:
            self.level = level

    def __init__(self, creators: tuple[_FakePluginCreator, ...]) -> None:
        self._registry = _FakePluginRegistry(plugin_creator_list=creators)

    def get_plugin_registry(self) -> _FakePluginRegistry:
        return self._registry


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


def _default_creators() -> tuple[_FakePluginCreator, ...]:
    return (
        _FakePluginCreator(
            name="TilePack",
            plugin_version="1",
            plugin_namespace="apexx",
            field_names=(_FakePluginField("tile_size"),),
        ),
        _FakePluginCreator(
            name="TileSSMScan",
            plugin_version="1",
            plugin_namespace="apexx",
            field_names=(
                _FakePluginField("direction"),
                _FakePluginField("clamp_value"),
            ),
        ),
        _FakePluginCreator(
            name="TileUnpackFusion",
            plugin_version="1",
            plugin_namespace="apexx",
            field_names=(),
        ),
    )


def _make_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    creators: tuple[_FakePluginCreator, ...],
) -> TensorRTEngineBuilder:
    from apex_x.runtime.tensorrt import builder as builder_module

    monkeypatch.setattr(builder_module, "trt", _FakeTRTModule(creators))
    return TensorRTEngineBuilder(runtime_caps=_runtime_caps_cuda_tensorrt_available())


def test_plugin_contract_status_matches_for_default_required_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _make_builder(monkeypatch, creators=_default_creators())
    build_cfg = TensorRTEngineBuildConfig(strict_plugin_check=True)

    contracts = builder._resolve_plugin_contracts(build=build_cfg)  # noqa: SLF001
    statuses = builder._plugin_statuses(plugin_contracts=contracts)  # noqa: SLF001

    assert len(statuses) == 3
    assert all(status.found for status in statuses)
    assert all(status.version_match is True for status in statuses)
    assert all(status.namespace_match is True for status in statuses)
    assert all(status.field_signature_match is True for status in statuses)
    builder._ensure_plugin_status(statuses, strict=True)  # noqa: SLF001


def test_plugin_contract_strict_fails_on_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creators = (
        _FakePluginCreator(
            name="TilePack",
            plugin_version="2",
            plugin_namespace="apexx",
            field_names=(_FakePluginField("tile_size"),),
        ),
        *_default_creators()[1:],
    )
    builder = _make_builder(monkeypatch, creators=creators)
    contracts = builder._resolve_plugin_contracts(build=TensorRTEngineBuildConfig())  # noqa: SLF001
    statuses = builder._plugin_statuses(plugin_contracts=contracts)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="mismatched required plugin contracts"):
        builder._ensure_plugin_status(statuses, strict=True)  # noqa: SLF001


def test_plugin_contract_strict_fails_on_missing_field_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creators = (
        _default_creators()[0],
        _FakePluginCreator(
            name="TileSSMScan",
            plugin_version="1",
            plugin_namespace="apexx",
            field_names=(_FakePluginField("direction"),),
        ),
        _default_creators()[2],
    )
    builder = _make_builder(monkeypatch, creators=creators)
    contracts = builder._resolve_plugin_contracts(build=TensorRTEngineBuildConfig())  # noqa: SLF001
    statuses = builder._plugin_statuses(plugin_contracts=contracts)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="field_signature"):
        builder._ensure_plugin_status(statuses, strict=True)  # noqa: SLF001


def test_plugin_contract_non_strict_allows_mismatch_for_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creators = (
        _FakePluginCreator(
            name="TilePack",
            plugin_version="1",
            plugin_namespace="wrong_ns",
            field_names=(_FakePluginField("tile_size"),),
        ),
        *_default_creators()[1:],
    )
    builder = _make_builder(monkeypatch, creators=creators)
    contracts = builder._resolve_plugin_contracts(build=TensorRTEngineBuildConfig())  # noqa: SLF001
    statuses = builder._plugin_statuses(plugin_contracts=contracts)  # noqa: SLF001

    builder._ensure_plugin_status(statuses, strict=False)  # noqa: SLF001
    tilepack = next(status for status in statuses if status.name == "TilePack")
    assert tilepack.namespace_match is False
