from __future__ import annotations

from pathlib import Path

import pytest

from apex_x.runtime import caps as caps_module
from apex_x.runtime.caps import CudaCaps


def test_detect_tensorrt_caps_import_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(caps_module, "_find_spec", lambda module_name: object())
    monkeypatch.setattr(caps_module, "_safe_import", lambda module_name: None)

    cuda = CudaCaps(
        available=True,
        device_count=1,
        device_name="GPU",
        compute_capability=(8, 0),
        reason=None,
    )
    caps = caps_module.detect_tensorrt_caps(cuda=cuda, header_search_paths=[tmp_path])

    assert caps.python_available is False
    assert caps.python_reason == caps_module.TENSORRT_PYTHON_REASON_IMPORT_FAILED
    assert caps.headers_available is False
    assert caps.int8_available is False
    assert caps.int8_reason == caps_module.TENSORRT_INT8_REASON_PYTHON_UNAVAILABLE


def test_detect_fp8_caps_below_sm90(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caps_module, "_supported_fp8_dtype_names", lambda: ("float8_e4m3fn",))
    cuda = CudaCaps(
        available=True,
        device_count=1,
        device_name="GPU",
        compute_capability=(8, 9),
        reason=None,
    )
    caps = caps_module.detect_fp8_caps(cuda=cuda)

    assert caps.available is False
    assert caps.dtype_available is True
    assert caps.reason == caps_module.FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90


def test_detect_tensorrt_headers_found(tmp_path: Path) -> None:
    include_dir = tmp_path / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "NvInferRuntime.h").write_text("// runtime header\n", encoding="utf-8")

    cuda = CudaCaps(
        available=False,
        device_count=0,
        device_name=None,
        compute_capability=None,
        reason="cuda_unavailable",
    )
    caps = caps_module.detect_tensorrt_caps(cuda=cuda, header_search_paths=[include_dir])

    assert caps.headers_available is True
    assert caps.header_path is not None
    assert caps.int8_available is False
    assert caps.int8_reason in {
        caps_module.TENSORRT_INT8_REASON_PYTHON_UNAVAILABLE,
        caps_module.TENSORRT_INT8_REASON_CUDA_REQUIRED,
    }
