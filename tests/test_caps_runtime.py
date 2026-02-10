from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apex_x.runtime import caps as caps_module


def test_detect_runtime_caps_cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(caps_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(caps_module.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(caps_module, "_find_spec", lambda module_name: None)
    monkeypatch.setattr(caps_module, "_supported_fp8_dtype_names", lambda: ("float8_e4m3fn",))

    caps = caps_module.detect_runtime_caps(header_search_paths=[tmp_path])

    assert caps.cuda.available is False
    assert caps.cuda.reason == "cuda_unavailable"
    assert caps.triton.available is False
    assert caps.triton.reason == "triton_not_installed"
    assert caps.tensorrt.python_available is False
    assert caps.tensorrt.headers_available is False
    assert caps.tensorrt.int8_available is False
    assert caps.fp8.available is False
    assert caps.fp8.reason == "fp8_requires_cuda"
    assert caps.any_gpu_runtime is False

    as_dict = caps.to_dict()
    assert as_dict["cuda"]["available"] is False
    assert as_dict["any_gpu_runtime"] is False


def test_detect_runtime_caps_mocked_full_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _mock_find_spec(module_name: str) -> object | None:
        if module_name in {"triton", "tensorrt"}:
            return object()
        return None

    class _MockBuilderFlag:
        INT8 = object()

    trt_module = SimpleNamespace(BuilderFlag=_MockBuilderFlag, __version__="10.0.0")
    triton_module = SimpleNamespace(__version__="3.0.0")

    def _mock_safe_import(module_name: str) -> object | None:
        if module_name == "tensorrt":
            return trt_module
        if module_name == "triton":
            return triton_module
        return None

    def _mock_package_version(package_name: str) -> str | None:
        versions = {"triton": "3.0.0", "tensorrt": "10.0.0"}
        return versions.get(package_name)

    monkeypatch.setattr(caps_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(caps_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(caps_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(caps_module.torch.cuda, "get_device_name", lambda idx: "Mock GPU")
    monkeypatch.setattr(caps_module.torch.cuda, "get_device_capability", lambda idx: (9, 0))
    monkeypatch.setattr(caps_module, "_find_spec", _mock_find_spec)
    monkeypatch.setattr(caps_module, "_safe_import", _mock_safe_import)
    monkeypatch.setattr(caps_module, "_package_version", _mock_package_version)
    monkeypatch.setattr(caps_module, "_supported_fp8_dtype_names", lambda: ("float8_e4m3fn",))

    (tmp_path / "NvInfer.h").write_text("// mock header\n", encoding="utf-8")
    caps = caps_module.detect_runtime_caps(header_search_paths=[tmp_path])

    assert caps.cuda.available is True
    assert caps.cuda.device_name == "Mock GPU"
    assert caps.cuda.compute_capability == (9, 0)
    assert caps.triton.available is True
    assert caps.triton.version == "3.0.0"
    assert caps.tensorrt.python_available is True
    assert caps.tensorrt.python_version == "10.0.0"
    assert caps.tensorrt.headers_available is True
    assert caps.tensorrt.header_path is not None
    assert caps.tensorrt.int8_available is True
    assert caps.fp8.available is True
    assert caps.any_gpu_runtime is True

