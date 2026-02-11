from __future__ import annotations

from types import SimpleNamespace

import pytest

from apex_x.bench.trt_engine_sweep import (
    TRTShapeSweepConfig,
    _parse_case_shapes,
    render_trt_shape_sweep_markdown,
    run_trt_engine_shape_sweep,
)
from apex_x.runtime import FP8Caps, RuntimeCaps, TensorRTCaps, TritonCaps
from apex_x.runtime.caps import CudaCaps


def _caps_cpu_only() -> RuntimeCaps:
    return RuntimeCaps(
        cuda=CudaCaps(
            available=False,
            device_count=0,
            device_name=None,
            compute_capability=None,
            reason="cuda_unavailable",
        ),
        triton=TritonCaps(available=False, version=None, reason="triton_not_installed"),
        tensorrt=TensorRTCaps(
            python_available=False,
            python_version=None,
            python_reason="tensorrt_python_not_installed",
            headers_available=False,
            header_path=None,
            int8_available=False,
            int8_reason="tensorrt_python_unavailable",
        ),
        fp8=FP8Caps(
            available=False,
            dtype_available=False,
            supported_dtypes=(),
            reason="fp8_requires_cuda",
        ),
    )


def _caps_trt_ready() -> RuntimeCaps:
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


def test_parse_case_shapes_normalizes_entries() -> None:
    parsed = _parse_case_shapes(" image=1x3x128x128 ; centers=1024x2 ; strides=1024 ")
    assert parsed == ("image=1x3x128x128", "centers=1024x2", "strides=1024")


def test_parse_case_shapes_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="invalid shape case entry"):
        _parse_case_shapes("image:1x3x128x128")
    with pytest.raises(ValueError, match="tensor dims must be > 0"):
        _parse_case_shapes("image=1x3x0x128")


def test_run_trt_engine_shape_sweep_skips_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "apex_x.bench.trt_engine_sweep.detect_runtime_caps",
        lambda: _caps_cpu_only(),
    )
    report = run_trt_engine_shape_sweep(
        TRTShapeSweepConfig(
            trt_engine_path="artifacts/trt/model.engine",
            shape_cases=(),
        )
    )
    assert report["status"] == "skipped"
    assert report["reason"] == "cuda_unavailable"


def test_run_trt_engine_shape_sweep_aggregates_case_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "apex_x.bench.trt_engine_sweep.detect_runtime_caps",
        lambda: _caps_trt_ready(),
    )

    seen_shapes: list[tuple[str, ...]] = []

    def _fake_bench(cfg: object, *, device: object) -> dict[str, object]:
        del device
        cfg_ns = SimpleNamespace(**getattr(cfg, "__dict__", {}))
        shape_specs = tuple(getattr(cfg, "trt_input_shapes", ()))
        seen_shapes.append(shape_specs)
        if not shape_specs:
            return {
                "status": "ok",
                "mode": "modern_io_tensors",
                "metrics": {"p50_ms": 2.0, "p95_ms": 2.5, "frames_per_s": 500.0},
            }
        if "input=1x3x256x256" in shape_specs:
            return {"status": "skipped", "reason": "unresolved_tensor_shape:output"}
        assert cfg_ns is not None
        return {
            "status": "ok",
            "mode": "modern_io_tensors",
            "metrics": {"p50_ms": 3.0, "p95_ms": 3.7, "frames_per_s": 330.0},
        }

    monkeypatch.setattr("apex_x.bench.trt_engine_sweep._bench_tensorrt_engine", _fake_bench)

    report = run_trt_engine_shape_sweep(
        TRTShapeSweepConfig(
            trt_engine_path="artifacts/trt/model.engine",
            shape_cases=(
                "input=1x3x128x128",
                "input=1x3x256x256",
            ),
            warmup=1,
            iters=2,
            seed=7,
        )
    )
    assert report["status"] == "ok"
    summary = report["summary"]
    assert summary["ok_count"] == 2
    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["p50_ms_min"] == pytest.approx(2.0, abs=1e-6)
    assert summary["p50_ms_max"] == pytest.approx(3.0, abs=1e-6)
    assert seen_shapes[0] == ()
    assert seen_shapes[1] == ("input=1x3x128x128",)
    assert seen_shapes[2] == ("input=1x3x256x256",)

    markdown = render_trt_shape_sweep_markdown(report)
    assert "TensorRT Engine Shape Sweep" in markdown
    assert "case_001_1_3_128_128" in markdown
