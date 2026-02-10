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
    assert caps.python_reason == "tensorrt_python_import_failed"
    assert caps.headers_available is False
    assert caps.int8_available is False
    assert caps.int8_reason == "tensorrt_python_unavailable"


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
    assert caps.reason == "compute_capability_8_9_below_sm90"


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
    assert caps.int8_reason in {"tensorrt_python_unavailable", "cuda_required_for_tensorrt_int8"}
