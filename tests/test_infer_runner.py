from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apex_x.config import ApexXConfig
from apex_x.infer import (
    evaluate_model_dataset,
    extract_routing_diagnostics,
    load_eval_dataset_npz,
    load_eval_images_npz,
    run_model_inference,
)
from apex_x.model import ApexXModel
from apex_x.runtime import FP8Caps, RuntimeCaps, TensorRTCaps, TritonCaps, detect_runtime_caps
from apex_x.runtime.caps import CudaCaps


def _dummy_input() -> np.ndarray:
    return np.random.RandomState(123).rand(1, 3, 128, 128).astype(np.float32)


def _runtime_caps_tensorrt_available() -> RuntimeCaps:
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


def test_extract_routing_diagnostics_handles_missing_or_invalid_payload() -> None:
    assert extract_routing_diagnostics(None) == {}
    assert extract_routing_diagnostics({"routing_diagnostics": []}) == {}
    assert extract_routing_diagnostics({"routing_diagnostics": {"ok": 1}}) == {"ok": 1}


def test_run_model_inference_cpu_schema() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="cpu",
        selected_backend="cpu",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
    )
    assert result.selected_tiles >= 0
    assert isinstance(result.routing_diagnostics, dict)
    assert result.runtime.execution_backend == "cpu"
    assert result.runtime.precision_profile == cfg.runtime.precision_profile
    assert result.runtime.execution_fallback_reason is None
    assert "total" in result.runtime.latency_ms
    assert "backend_execute" in result.runtime.latency_ms
    assert "backend_preflight" in result.runtime.latency_ms
    assert result.runtime.latency_ms["total"] >= 0.0


def test_run_model_inference_strict_rejects_unimplemented_backend() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    with pytest.raises(
        RuntimeError,
        match=(
            "triton backend is unavailable for execution|"
            "execution path is not implemented in CLI runtime"
        ),
    ):
        run_model_inference(
            model=model,
            input_batch=_dummy_input(),
            requested_backend="triton",
            selected_backend="triton",
            fallback_policy="strict",
            precision_profile=cfg.runtime.precision_profile,
            selection_fallback_reason=None,
            runtime_caps=detect_runtime_caps(),
        )


def test_run_model_inference_triton_permissive_falls_back_without_caps() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="triton",
        selected_backend="triton",
        fallback_policy="permissive",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
    )
    assert result.runtime.execution_backend == "cpu"
    assert result.runtime.execution_fallback_reason is not None


def test_run_model_inference_tensorrt_permissive_falls_back_without_caps() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="tensorrt",
        selected_backend="tensorrt",
        fallback_policy="permissive",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
    )
    assert result.runtime.execution_backend == "cpu"
    assert result.runtime.execution_fallback_reason is not None


def test_run_model_inference_tensorrt_strict_requires_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    monkeypatch.delenv("APEXX_EXPORT_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("APEXX_TRT_ENGINE_PATH", raising=False)
    with pytest.raises(RuntimeError, match="tensorrt_preflight_failed"):
        run_model_inference(
            model=model,
            input_batch=_dummy_input(),
            requested_backend="tensorrt",
            selected_backend="tensorrt",
            fallback_policy="strict",
            precision_profile=cfg.runtime.precision_profile,
            selection_fallback_reason=None,
            runtime_caps=_runtime_caps_tensorrt_available(),
        )


def test_run_model_inference_tensorrt_permissive_with_manifest_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    import json

    from apex_x.utils import hash_file_sha256

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "format": "onnx",
        "shape_mode": "static",
        "profile": "small",
        "artifacts": {
            "onnx_path": str(onnx_path),
            "onnx_sha256": hash_file_sha256(onnx_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    monkeypatch.setenv("APEXX_EXPORT_MANIFEST_PATH", str(manifest_path))
    monkeypatch.delenv("APEXX_TRT_ENGINE_PATH", raising=False)

    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="tensorrt",
        selected_backend="tensorrt",
        fallback_policy="permissive",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=_runtime_caps_tensorrt_available(),
    )
    assert result.runtime.execution_backend == "cpu"
    assert result.runtime.execution_fallback_reason == (
        "tensorrt_engine_path_missing_reference_fallback"
    )


def test_run_model_inference_tensorrt_executes_when_runner_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake-engine")
    monkeypatch.setenv("APEXX_TRT_ENGINE_PATH", str(engine_path))
    monkeypatch.delenv("APEXX_EXPORT_MANIFEST_PATH", raising=False)

    observed: dict[str, Path | None] = {}

    def _fake_run_tensorrt_inference(
        *,
        input_batch: np.ndarray,
        manifest_path: Path | None,
        engine_path: Path | None,
    ) -> dict[str, Any]:
        del input_batch
        observed["manifest_path"] = manifest_path
        observed["engine_path"] = engine_path
        return {
            "selected_tiles": [1, 2, 3],
            "routing_diagnostics": {"backend": "tensorrt"},
            "det": {
                "boxes": np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                "scores": np.asarray([0.87], dtype=np.float32),
                "class_ids": np.asarray([0], dtype=np.int64),
            },
        }

    monkeypatch.setattr(
        "apex_x.infer.runner._run_tensorrt_inference",
        _fake_run_tensorrt_inference,
    )
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="tensorrt",
        selected_backend="tensorrt",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=_runtime_caps_tensorrt_available(),
    )
    assert observed["manifest_path"] is None
    assert observed["engine_path"] == engine_path.resolve()
    assert result.runtime.execution_backend == "tensorrt"
    assert result.runtime.execution_fallback_reason is None
    assert result.det_score == pytest.approx(0.87, abs=1e-6)
    assert result.selected_tiles == 3
    assert result.routing_diagnostics == {"backend": "tensorrt"}


def test_run_model_inference_tensorrt_maps_det_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    engine_path = tmp_path / "det.engine"
    engine_path.write_bytes(b"fake-engine")
    monkeypatch.setenv("APEXX_TRT_ENGINE_PATH", str(engine_path))
    monkeypatch.delenv("APEXX_EXPORT_MANIFEST_PATH", raising=False)

    class _FakeExecutor:
        def __init__(
            self,
            *,
            engine_path: str | Path,
            plugin_library_paths: tuple[str | Path, ...] = (),
        ) -> None:
            del plugin_library_paths
            self._engine_path = Path(engine_path).resolve()

        def run(
            self,
            *,
            input_batch: np.ndarray | None = None,
            input_name: str | None = None,
            input_tensors: dict[str, np.ndarray] | None = None,
        ) -> SimpleNamespace:
            del input_batch, input_name, input_tensors
            return SimpleNamespace(
                engine_path=self._engine_path,
                input_name="image",
                output_names=("out_boxes", "out_scores", "out_class_ids", "out_valid"),
                outputs={
                    "out_boxes": np.asarray(
                        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
                        dtype=np.float32,
                    ),
                    "out_scores": np.asarray([[0.91, 0.42]], dtype=np.float32),
                    "out_class_ids": np.asarray([[2, 7]], dtype=np.int32),
                    "out_valid": np.asarray([1], dtype=np.int32),
                },
            )

    monkeypatch.setattr("apex_x.infer.runner.TensorRTEngineExecutor", _FakeExecutor)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="tensorrt",
        selected_backend="tensorrt",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=_runtime_caps_tensorrt_available(),
    )
    assert result.runtime.execution_backend == "tensorrt"
    assert result.runtime.execution_fallback_reason is None
    assert result.det_score == pytest.approx(0.91, abs=1e-6)
    det = result.model_output["det"]
    assert isinstance(det, dict)
    assert det["boxes"].shape == (1, 4)
    assert det["scores"].shape == (1,)
    assert det["class_ids"].shape == (1,)


def test_run_model_inference_tensorrt_loads_extra_inputs_npz(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    engine_path = tmp_path / "multi.engine"
    engine_path.write_bytes(b"fake-engine")
    extra_path = tmp_path / "extra_inputs.npz"
    np.savez(
        extra_path,
        centers=np.zeros((16, 2), dtype=np.float32),
        strides=np.ones((16,), dtype=np.float32),
    )
    monkeypatch.setenv("APEXX_TRT_ENGINE_PATH", str(engine_path))
    monkeypatch.setenv("APEXX_TRT_EXTRA_INPUTS_NPZ", str(extra_path))
    monkeypatch.setenv("APEXX_TRT_INPUT_NAME", "image")
    monkeypatch.delenv("APEXX_EXPORT_MANIFEST_PATH", raising=False)

    observed: dict[str, Any] = {}

    class _FakeExecutor:
        def __init__(
            self,
            *,
            engine_path: str | Path,
            plugin_library_paths: tuple[str | Path, ...] = (),
        ) -> None:
            del plugin_library_paths
            self._engine_path = Path(engine_path).resolve()

        def run(
            self,
            *,
            input_batch: np.ndarray | None = None,
            input_name: str | None = None,
            input_tensors: dict[str, np.ndarray] | None = None,
        ) -> SimpleNamespace:
            observed["input_name"] = input_name
            observed["input_batch_shape"] = None if input_batch is None else input_batch.shape
            observed["input_tensors"] = {} if input_tensors is None else dict(input_tensors)
            return SimpleNamespace(
                engine_path=self._engine_path,
                input_name="image",
                output_names=("output",),
                outputs={"output": np.asarray([[[[0.73]]]], dtype=np.float32)},
            )

    monkeypatch.setattr("apex_x.infer.runner.TensorRTEngineExecutor", _FakeExecutor)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="tensorrt",
        selected_backend="tensorrt",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=_runtime_caps_tensorrt_available(),
    )
    assert result.runtime.execution_backend == "tensorrt"
    assert observed["input_name"] == "image"
    assert observed["input_batch_shape"] == (1, 3, 128, 128)
    input_tensors = observed["input_tensors"]
    assert set(input_tensors.keys()) == {"centers", "strides"}
    assert input_tensors["centers"].shape == (16, 2)
    assert input_tensors["strides"].shape == (16,)


def test_run_model_inference_permissive_backend_falls_back_to_cpu() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="triton",
        selected_backend="triton",
        fallback_policy="permissive",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
    )
    assert result.runtime.execution_backend == "cpu"
    assert result.runtime.execution_fallback_reason is not None


def test_run_model_inference_torch_backend_executes() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    result = run_model_inference(
        model=model,
        input_batch=_dummy_input(),
        requested_backend="torch",
        selected_backend="torch",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
    )
    assert result.runtime.execution_backend == "torch"
    assert result.runtime.execution_fallback_reason is None
    assert result.selected_tiles >= 0
    assert isinstance(result.routing_diagnostics, dict)


def test_load_eval_images_npz_validates_shape_and_channels(tmp_path: Path) -> None:
    images = np.random.RandomState(7).rand(3, 3, 128, 128).astype(np.float32)
    path = tmp_path / "dataset.npz"
    np.savez(path, images=images)
    loaded = load_eval_images_npz(path=path, expected_height=128, expected_width=128)
    assert loaded.shape == images.shape

    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, images=np.random.RandomState(8).rand(3, 1, 128, 128).astype(np.float32))
    with pytest.raises(ValueError, match="channel dimension must be 3"):
        load_eval_images_npz(path=bad_path, expected_height=128, expected_width=128)


def test_load_eval_dataset_npz_parses_det_score_target(tmp_path: Path) -> None:
    images = np.random.RandomState(11).rand(4, 3, 128, 128).astype(np.float32)
    target = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    selected_target = np.asarray([2, 3, 4, 5], dtype=np.int64)
    path = tmp_path / "dataset_with_target.npz"
    np.savez(
        path,
        images=images,
        det_score_target=target,
        selected_tiles_target=selected_target,
    )

    dataset = load_eval_dataset_npz(path=path, expected_height=128, expected_width=128)
    assert dataset.images.shape == images.shape
    assert dataset.det_score_target is not None
    assert dataset.det_score_target.shape == (4,)
    assert dataset.selected_tiles_target is not None
    assert dataset.selected_tiles_target.shape == (4,)


def test_load_eval_dataset_npz_rejects_target_length_mismatch(tmp_path: Path) -> None:
    images = np.random.RandomState(12).rand(3, 3, 128, 128).astype(np.float32)
    target = np.asarray([0.1, 0.2], dtype=np.float32)
    path = tmp_path / "dataset_bad_target.npz"
    np.savez(path, images=images, det_score_target=target)

    with pytest.raises(ValueError, match="length mismatch"):
        load_eval_dataset_npz(path=path, expected_height=128, expected_width=128)


def test_load_eval_dataset_npz_rejects_selected_tiles_target_length_mismatch(
    tmp_path: Path,
) -> None:
    images = np.random.RandomState(13).rand(3, 3, 128, 128).astype(np.float32)
    selected_target = np.asarray([1, 2], dtype=np.int64)
    path = tmp_path / "dataset_bad_selected_target.npz"
    np.savez(path, images=images, selected_tiles_target=selected_target)

    with pytest.raises(ValueError, match="selected_tiles_target length mismatch"):
        load_eval_dataset_npz(path=path, expected_height=128, expected_width=128)


def test_evaluate_model_dataset_returns_summary() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    images = np.random.RandomState(99).rand(4, 3, 128, 128).astype(np.float32)
    summary = evaluate_model_dataset(
        model=model,
        images=images,
        requested_backend="cpu",
        selected_backend="cpu",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
        max_samples=3,
    )
    assert summary.num_samples == 3
    assert summary.det_score_min <= summary.det_score_mean <= summary.det_score_max
    assert summary.selected_tiles_mean >= 0.0


def test_evaluate_model_dataset_with_target_metrics() -> None:
    cfg = ApexXConfig()
    model = ApexXModel(config=cfg)
    images = np.random.RandomState(100).rand(4, 3, 128, 128).astype(np.float32)
    target = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    summary = evaluate_model_dataset(
        model=model,
        images=images,
        requested_backend="cpu",
        selected_backend="cpu",
        fallback_policy="strict",
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=detect_runtime_caps(),
        det_score_target=target,
        selected_tiles_target=np.asarray([1, 1, 1, 1], dtype=np.int64),
        max_samples=4,
    )
    assert summary.det_score_target_metrics is not None
    assert "mae" in summary.det_score_target_metrics
    assert "rmse" in summary.det_score_target_metrics
    assert summary.selected_tiles_target_metrics is not None
    assert "exact_match_rate" in summary.selected_tiles_target_metrics
