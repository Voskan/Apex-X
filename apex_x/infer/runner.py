from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apex_x.config import ApexXConfig
from apex_x.model import ApexXModel, FFModule, TeacherModelV3
from apex_x.runtime import RuntimeCaps, TensorRTEngineExecutor, load_export_manifest
from apex_x.infer.tta import TestTimeAugmentation


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    requested_backend: str
    selected_backend: str
    execution_backend: str
    fallback_policy: str
    precision_profile: str
    selection_fallback_reason: str | None
    execution_fallback_reason: str | None
    runtime_caps: dict[str, Any]
    latency_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "execution_backend": self.execution_backend,
            "fallback_policy": self.fallback_policy,
            "precision_profile": self.precision_profile,
            "selection_fallback_reason": self.selection_fallback_reason,
            "execution_fallback_reason": self.execution_fallback_reason,
            "runtime_caps": self.runtime_caps,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class InferenceRunResult:
    model_output: dict[str, Any]
    routing_diagnostics: dict[str, Any]
    selected_tiles: int
    det_score: float
    runtime: RuntimeMetadata


@dataclass(frozen=True, slots=True)
class ModelDatasetEvalSummary:
    num_samples: int
    det_score_mean: float
    det_score_std: float
    det_score_min: float
    det_score_max: float
    selected_tiles_mean: float
    selected_tiles_p95: float
    execution_backend: str
    precision_profile: str
    source: str
    det_score_target_metrics: dict[str, float | None] | None = None
    selected_tiles_target_metrics: dict[str, float | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "num_samples": self.num_samples,
            "det_score": {
                "mean": self.det_score_mean,
                "std": self.det_score_std,
                "min": self.det_score_min,
                "max": self.det_score_max,
            },
            "selected_tiles": {
                "mean": self.selected_tiles_mean,
                "p95": self.selected_tiles_p95,
            },
            "execution_backend": self.execution_backend,
            "precision_profile": self.precision_profile,
        }
        if self.det_score_target_metrics is not None:
            payload["det_score_target"] = self.det_score_target_metrics
        if self.selected_tiles_target_metrics is not None:
            payload["selected_tiles_target"] = self.selected_tiles_target_metrics
        return payload


@dataclass(frozen=True, slots=True)
class EvalDataset:
    images: np.ndarray
    det_score_target: np.ndarray | None
    selected_tiles_target: np.ndarray | None


def extract_routing_diagnostics(model_output: dict[str, Any] | None = None) -> dict[str, Any]:
    if model_output is None:
        return {}
    diagnostics = model_output.get("routing_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return {}
    return diagnostics


def _downsample_mean_torch(x: torch.Tensor, stride: int) -> torch.Tensor:
    bsz, channels, height, width = x.shape
    hs = height // stride
    ws = width // stride
    x_crop = x[:, :, : hs * stride, : ws * stride]
    x_view = x_crop.reshape(bsz, channels, hs, stride, ws, stride)
    return x_view.mean(dim=(3, 5))


def _tile_utilities_torch(ff: torch.Tensor, tile_size: int) -> torch.Tensor:
    bsz, channels, height, width = ff.shape
    if height % tile_size != 0 or width % tile_size != 0:
        raise ValueError("ff feature map must be divisible by tile_size")
    grid_h = height // tile_size
    grid_w = width // tile_size
    tiles = (
        ff.reshape(bsz, channels, grid_h, tile_size, grid_w, tile_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(bsz, grid_h * grid_w, channels, tile_size, tile_size)
    )
    tile_var = tiles.var(dim=(2, 3, 4), unbiased=False)
    tile_mean_abs = tiles.abs().mean(dim=(2, 3, 4))
    return tile_var + tile_mean_abs


def _torch_model_forward(
    model: ApexXModel | TeacherModelV3,
    input_batch: np.ndarray,
    *,
    use_runtime_plugins: bool,
    use_tta: bool = False,
) -> dict[str, Any]:
    cfg: ApexXConfig = model.config
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 1. Full SOTA Teacher Inference (Masks + Boxes)
    if isinstance(model, TeacherModelV3) or use_tta:
        model.to(device)
        model.eval()
        
        with torch.inference_mode():
            images_t = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            
            if use_tta:
                tta_wrapper = TestTimeAugmentation(model)
                out_list = tta_wrapper(images_t)
                out = out_list[0]
            else:
                out = model(images_t)
                
            # Convert to runner-compatible format
            det = {
                "boxes": out["boxes"].cpu().numpy(),
                "scores": out["scores"].cpu().numpy(),
                "class_ids": out.get("classes", out.get("labels", torch.zeros_like(out["scores"]))).cpu().numpy(),
            }
            
            result = {
                "det": det,
                "routing_diagnostics": out.get("routing_diagnostics", {}),
                "selected_tiles": out.get("selected_tiles", []),
                "feature_toggles": out.get("feature_toggles", {}),
            }
            
            if "masks" in out:
                result["masks"] = out["masks"].cpu().numpy()
                
            return result

    # 2. Router-only / Legacy FFModule Path
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
        pv16 = _downsample_mean_torch(x, cfg.model.pv_stride)
        ff8 = _downsample_mean_torch(x, cfg.model.ff_primary_stride)

        utilities = _tile_utilities_torch(ff8, cfg.model.tile_size_l0)
        split_utilities = utilities
        boundary_proxy = ff8.abs().mean(dim=1, keepdim=True)
        centered = ff8 - ff8.mean(dim=(2, 3), keepdim=True)
        uncertainty_proxy = centered.abs().mean(dim=1, keepdim=True)

        exec_cfg = ApexXConfig.from_dict(cfg.to_dict())
        exec_cfg.runtime.enable_runtime_plugins = bool(use_runtime_plugins)
        ff_module = FFModule(channels=int(ff8.shape[1]), config=exec_cfg).to(device)
        ff_module.eval()
        ff_out = ff_module.forward_infer(
            dense_features=ff8,
            utilities=utilities,
            split_utilities=split_utilities,
            boundary_proxy=boundary_proxy,
            uncertainty_proxy=uncertainty_proxy,
        )

        merged = ff_out.heavy_features
        det_score = float(torch.clamp(merged.max(), min=0.0, max=1.0).item())
        selected_l0 = ff_out.selected_l0[0] if ff_out.selected_l0 else []
        scores = np.asarray([det_score], dtype=np.float32)
        det_box = np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32)

        return {
            "pv16": pv16.detach().cpu().numpy(),
            "ff8": ff8.detach().cpu().numpy(),
            "merged": merged.detach().cpu().numpy(),
            "selected_tiles": [int(idx) for idx in selected_l0],
            "hysteresis_mask": [1 for _ in range(len(selected_l0))],
            "routing_diagnostics": ff_out.diagnostics,
            "feature_toggles": {
                "router_off": not cfg.model.router_enabled(),
                "no_nesting": bool(cfg.model.disable_nesting),
                "no_ssm": not cfg.model.ssm_enabled(),
                "no_distill": bool(cfg.train.disable_distill),
                "no_pcgradpp": not cfg.train.pcgradpp_enabled(),
            },
            "ssm_state": np.zeros((1, int(ff8.shape[1])), dtype=np.float32),
            "det": {
                "boxes": det_box,
                "scores": scores,
                "class_ids": np.asarray([0], dtype=np.int64),
            },
        }


def _is_truthy_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _tensorrt_preflight_from_env() -> tuple[Path | None, Path | None]:
    manifest_raw = os.environ.get("APEXX_EXPORT_MANIFEST_PATH")
    engine_raw = os.environ.get("APEXX_TRT_ENGINE_PATH")
    verify_hash = _is_truthy_env(
        os.environ.get("APEXX_TRT_VERIFY_MANIFEST_HASH"),
        default=True,
    )

    manifest_path: Path | None = None
    if manifest_raw:
        manifest = load_export_manifest(
            manifest_path=manifest_raw,
            verify_onnx_hash=verify_hash,
        )
        manifest_path = manifest.manifest_path

    engine_path: Path | None = None
    if engine_raw:
        candidate = Path(engine_raw).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {candidate}")
        engine_path = candidate

    if manifest_path is None and engine_path is None:
        raise RuntimeError(
            "tensorrt_artifacts_missing: set APEXX_TRT_ENGINE_PATH or APEXX_EXPORT_MANIFEST_PATH"
        )
    return manifest_path, engine_path


class _TensorRTExecutionConfigError(RuntimeError):
    """Raised when TensorRT execution configuration is incomplete."""


def _tensorrt_plugin_libraries_from_env() -> tuple[Path, ...]:
    raw = os.environ.get("APEXX_TRT_PLUGIN_LIB", "").strip()
    if not raw:
        return ()
    paths: list[Path] = []
    for item in raw.split(os.pathsep):
        value = item.strip()
        if not value:
            continue
        candidate = Path(value).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"TensorRT plugin library not found: {candidate}")
        paths.append(candidate)
    return tuple(paths)


def _tensorrt_extra_inputs_from_env() -> dict[str, np.ndarray]:
    raw = os.environ.get("APEXX_TRT_EXTRA_INPUTS_NPZ", "").strip()
    if not raw:
        return {}
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"TensorRT extra inputs file not found: {path}")
    if path.suffix.lower() not in {".npz", ".npy"}:
        raise ValueError("APEXX_TRT_EXTRA_INPUTS_NPZ must point to .npz or .npy")
    if path.suffix.lower() == ".npy":
        raise ValueError(
            "APEXX_TRT_EXTRA_INPUTS_NPZ must be .npz with named arrays for multi-input engines"
        )

    resolved: dict[str, np.ndarray] = {}
    with np.load(path) as archive:
        for key in archive.files:
            arr = np.asarray(archive[key])
            if arr.size <= 0:
                raise ValueError(f"TensorRT extra input '{key}' must be non-empty")
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                raise ValueError(f"TensorRT extra input '{key}' must contain finite values")
            resolved[key] = arr
    if not resolved:
        raise ValueError("TensorRT extra inputs archive is empty")
    return resolved


def _normalize_trt_output_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_tensorrt_output_name(
    *,
    outputs: dict[str, np.ndarray],
    env_var: str,
    aliases: tuple[str, ...],
) -> str | None:
    preferred = os.environ.get(env_var, "").strip()
    if preferred:
        if preferred not in outputs:
            raise ValueError(
                f"{env_var}={preferred!r} is not present in TensorRT outputs {list(outputs.keys())}"
            )
        return preferred

    alias_map = {_normalize_trt_output_key(alias): alias for alias in aliases}
    for name in outputs:
        normalized = _normalize_trt_output_key(name)
        if normalized in alias_map:
            return name
    return None


def _extract_batch_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return np.asarray(arr[0])


def _extract_batch_boxes(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[-1] == 4:
        return arr
    if arr.ndim >= 3 and arr.shape[-1] == 4:
        return np.asarray(arr[0])
    raise ValueError("TensorRT DET boxes output must have trailing dimension 4")


def _extract_tensorrt_det(
    *,
    outputs: dict[str, np.ndarray],
    fallback_score: float,
) -> tuple[dict[str, np.ndarray], float]:
    boxes_name = _resolve_tensorrt_output_name(
        outputs=outputs,
        env_var="APEXX_TRT_DET_BOXES_NAME",
        aliases=("out_boxes", "boxes", "det_boxes", "bboxes"),
    )
    scores_name = _resolve_tensorrt_output_name(
        outputs=outputs,
        env_var="APEXX_TRT_DET_SCORES_NAME",
        aliases=("out_scores", "scores", "det_scores"),
    )
    class_ids_name = _resolve_tensorrt_output_name(
        outputs=outputs,
        env_var="APEXX_TRT_DET_CLASS_IDS_NAME",
        aliases=("out_class_ids", "class_ids", "classes", "labels"),
    )
    valid_name = _resolve_tensorrt_output_name(
        outputs=outputs,
        env_var="APEXX_TRT_DET_VALID_NAME",
        aliases=("out_valid", "valid", "num_valid", "num_dets"),
    )

    if boxes_name is None or scores_name is None:
        return (
            {
                "boxes": np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
                "scores": np.asarray([fallback_score], dtype=np.float32),
                "class_ids": np.asarray([0], dtype=np.int64),
            },
            fallback_score,
        )

    boxes_arr = np.asarray(outputs[boxes_name])
    scores_arr = np.asarray(outputs[scores_name])
    boxes = _extract_batch_boxes(boxes_arr).astype(np.float32, copy=False)
    scores = _extract_batch_rows(scores_arr).astype(np.float32, copy=False)

    if class_ids_name is not None:
        class_ids_arr = np.asarray(outputs[class_ids_name])
        class_ids = _extract_batch_rows(class_ids_arr).astype(np.int64, copy=False)
    else:
        class_ids = np.zeros((scores.shape[0],), dtype=np.int64)

    valid_count: int | None = None
    if valid_name is not None:
        valid_arr = np.asarray(outputs[valid_name])
        if valid_arr.ndim == 0:
            valid_count = int(valid_arr.item())
        elif valid_arr.size > 0:
            valid_count = int(valid_arr.reshape(-1)[0])
        else:
            valid_count = 0

    limit = min(int(boxes.shape[0]), int(scores.shape[0]), int(class_ids.shape[0]))
    if valid_count is not None:
        limit = min(limit, max(valid_count, 0))
    if limit <= 0:
        return (
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "class_ids": np.zeros((0,), dtype=np.int64),
            },
            0.0,
        )

    det_boxes = boxes[:limit]
    det_scores = scores[:limit]
    det_class_ids = class_ids[:limit]
    det_score = float(det_scores[0]) if det_scores.size > 0 else 0.0
    return (
        {
            "boxes": det_boxes,
            "scores": det_scores,
            "class_ids": det_class_ids,
        },
        det_score,
    )


def _select_tensorrt_primary_output(outputs: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    if not outputs:
        raise RuntimeError("TensorRT engine produced no outputs")
    requested = os.environ.get("APEXX_TRT_PRIMARY_OUTPUT_NAME")
    if requested:
        name = requested.strip()
        if name and name in outputs:
            return name, outputs[name]
        available = ",".join(outputs.keys())
        raise ValueError(
            f"APEXX_TRT_PRIMARY_OUTPUT_NAME={requested!r} is not in engine outputs [{available}]"
        )
    for name, tensor in outputs.items():
        if np.issubdtype(tensor.dtype, np.floating):
            return name, tensor
    first_name = next(iter(outputs))
    return first_name, outputs[first_name]


def _tensorrt_outputs_to_model_output(
    *,
    input_batch: np.ndarray,
    outputs: dict[str, np.ndarray],
    engine_path: Path,
    manifest_path: Path | None,
) -> dict[str, Any]:
    primary_name, primary_tensor = _select_tensorrt_primary_output(outputs)
    primary = np.asarray(primary_tensor, dtype=np.float32)
    finite = primary[np.isfinite(primary)]
    primary_max = float(finite.max()) if finite.size > 0 else 0.0
    fallback_score = float(np.clip(primary_max, 0.0, 1.0))
    det, det_score = _extract_tensorrt_det(
        outputs=outputs,
        fallback_score=fallback_score,
    )

    model_output: dict[str, Any] = {
        "selected_tiles": [],
        "hysteresis_mask": [],
        "routing_diagnostics": {},
        "det": det,
        "tensorrt": {
            "engine_path": str(engine_path),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "output_names": list(outputs.keys()),
            "primary_output_name": primary_name,
        },
    }
    if primary.ndim == 4:
        model_output["merged"] = primary
        model_output["ff8"] = primary
        model_output["pv16"] = np.asarray(input_batch, dtype=np.float32)
    return model_output


def _run_tensorrt_inference(
    *,
    input_batch: np.ndarray,
    manifest_path: Path | None,
    engine_path: Path | None,
) -> dict[str, Any]:
    if engine_path is None:
        raise _TensorRTExecutionConfigError(
            "tensorrt_engine_path_missing: set APEXX_TRT_ENGINE_PATH for runtime execution"
        )
    plugin_paths = _tensorrt_plugin_libraries_from_env()
    extra_inputs = _tensorrt_extra_inputs_from_env()
    input_name = os.environ.get("APEXX_TRT_INPUT_NAME")
    resolved_input_name = input_name.strip() if input_name and input_name.strip() else None
    executor = TensorRTEngineExecutor(
        engine_path=engine_path,
        plugin_library_paths=plugin_paths,
    )
    execution = executor.run(
        input_batch=np.asarray(input_batch, dtype=np.float32),
        input_name=resolved_input_name,
        input_tensors=extra_inputs if extra_inputs else None,
    )
    return _tensorrt_outputs_to_model_output(
        input_batch=input_batch,
        outputs=execution.outputs,
        engine_path=execution.engine_path,
        manifest_path=manifest_path,
    )


def run_model_inference(
    *,
    model: ApexXModel,
    input_batch: np.ndarray,
    requested_backend: str,
    selected_backend: str,
    fallback_policy: str,
    precision_profile: str,
    selection_fallback_reason: str | None,
    runtime_caps: RuntimeCaps,
    use_tta: bool = False,
) -> InferenceRunResult:
    total_started = time.perf_counter()
    backend_preflight_latency_ms = 0.0
    backend_execute_latency_ms = 0.0

    def _run_with_timing(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        nonlocal backend_execute_latency_ms
        started = time.perf_counter()
        result = fn()
        backend_execute_latency_ms += (time.perf_counter() - started) * 1000.0
        return result

    execution_backend = selected_backend
    execution_fallback_reason: str | None = None
    if selected_backend == "cpu":
        model_output = _run_with_timing(lambda: model.forward(input_batch))
    elif selected_backend == "torch":
        config_tta = getattr(model.config.train, "tta_enabled", False)
        effective_tta = use_tta or config_tta
        try:
            model_output = _run_with_timing(
                lambda: _torch_model_forward(
                    model,
                    input_batch,
                    use_runtime_plugins=False,
                    use_tta=effective_tta,
                )
            )
        except Exception as exc:
            if fallback_policy == "strict":
                raise
            model_output = _run_with_timing(lambda: model.forward(input_batch))
            execution_backend = "cpu"
            execution_fallback_reason = f"torch_executor_error:{type(exc).__name__}"
    elif selected_backend == "triton":
        triton_execution_enabled = _is_truthy_env(
            os.environ.get("APEXX_ENABLE_TRITON_EXECUTION"),
            default=False,
        )
        if not triton_execution_enabled:
            if fallback_policy == "strict":
                raise RuntimeError("triton backend is unavailable for execution")
            model_output = _run_with_timing(lambda: model.forward(input_batch))
            execution_backend = "cpu"
            execution_fallback_reason = "triton_execution_not_enabled_reference_fallback"
        elif not runtime_caps.cuda.available or not runtime_caps.triton.available:
            if fallback_policy == "strict":
                raise RuntimeError("triton backend is unavailable for execution")
            model_output = _run_with_timing(lambda: model.forward(input_batch))
            execution_backend = "cpu"
            execution_fallback_reason = "triton_backend_unavailable_reference_fallback"
        else:
            try:
                model_output = _run_with_timing(
                    lambda: _torch_model_forward(
                        model,
                        input_batch,
                        use_runtime_plugins=True,
                    )
                )
            except Exception as exc:
                if fallback_policy == "strict":
                    raise
                model_output = _run_with_timing(lambda: model.forward(input_batch))
                execution_backend = "cpu"
                execution_fallback_reason = f"triton_executor_error:{type(exc).__name__}"
    elif selected_backend == "tensorrt":
        if not runtime_caps.cuda.available or not runtime_caps.tensorrt.python_available:
            if fallback_policy == "strict":
                raise RuntimeError("tensorrt backend is unavailable for execution")
            model_output = _run_with_timing(lambda: model.forward(input_batch))
            execution_backend = "cpu"
            execution_fallback_reason = "tensorrt_backend_unavailable_reference_fallback"
        else:
            try:
                preflight_started = time.perf_counter()
                try:
                    manifest_path, engine_path = _tensorrt_preflight_from_env()
                finally:
                    backend_preflight_latency_ms += (
                        time.perf_counter() - preflight_started
                    ) * 1000.0
            except Exception as exc:
                if fallback_policy == "strict":
                    raise RuntimeError(
                        f"tensorrt_preflight_failed:{type(exc).__name__}:{exc}"
                    ) from exc
                model_output = _run_with_timing(lambda: model.forward(input_batch))
                execution_backend = "cpu"
                execution_fallback_reason = f"tensorrt_preflight_error:{type(exc).__name__}"
            else:
                try:
                    model_output = _run_with_timing(
                        lambda: _run_tensorrt_inference(
                            input_batch=input_batch,
                            manifest_path=manifest_path,
                            engine_path=engine_path,
                        )
                    )
                except _TensorRTExecutionConfigError as exc:
                    if fallback_policy == "strict":
                        raise RuntimeError(
                            f"tensorrt_runtime_execution_failed:{type(exc).__name__}:{exc}"
                        ) from exc
                    model_output = _run_with_timing(lambda: model.forward(input_batch))
                    execution_backend = "cpu"
                    execution_fallback_reason = "tensorrt_engine_path_missing_reference_fallback"
                except Exception as exc:
                    if fallback_policy == "strict":
                        raise RuntimeError(
                            f"tensorrt_runtime_execution_failed:{type(exc).__name__}:{exc}"
                        ) from exc
                    model_output = _run_with_timing(lambda: model.forward(input_batch))
                    execution_backend = "cpu"
                    execution_fallback_reason = f"tensorrt_executor_error:{type(exc).__name__}"
    else:
        if fallback_policy == "strict":
            raise RuntimeError(
                f"selected backend '{selected_backend}' execution path is not implemented "
                "in CLI runtime"
            )
        model_output = _run_with_timing(lambda: model.forward(input_batch))
        execution_backend = "cpu"
        execution_fallback_reason = "selected_backend_execution_not_implemented_reference_fallback"

    routing_diagnostics = extract_routing_diagnostics(model_output)
    selected_tiles_raw = model_output.get("selected_tiles", [])
    selected_tiles = len(selected_tiles_raw) if isinstance(selected_tiles_raw, list) else 0

    det_score = 0.0
    det = model_output.get("det", {})
    if isinstance(det, dict):
        scores = det.get("scores")
        if isinstance(scores, np.ndarray) and scores.size > 0:
            det_score = float(scores[0])

    runtime = RuntimeMetadata(
        requested_backend=requested_backend,
        selected_backend=selected_backend,
        execution_backend=execution_backend,
        fallback_policy=fallback_policy,
        precision_profile=precision_profile,
        selection_fallback_reason=selection_fallback_reason,
        execution_fallback_reason=execution_fallback_reason,
        runtime_caps=runtime_caps.to_dict(),
        latency_ms={
            "total": (time.perf_counter() - total_started) * 1000.0,
            "backend_execute": backend_execute_latency_ms,
            "backend_preflight": backend_preflight_latency_ms,
        },
    )
    return InferenceRunResult(
        model_output=model_output,
        routing_diagnostics=routing_diagnostics,
        selected_tiles=selected_tiles,
        det_score=det_score,
        runtime=runtime,
    )


def _normalize_det_score_target(
    *,
    raw: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    det_target = np.asarray(raw, dtype=np.float32)
    if det_target.ndim == 2 and det_target.shape[1] == 1:
        det_target = det_target[:, 0]
    if det_target.ndim != 1:
        raise ValueError("det_score_target must be shape [N] or [N,1]")
    if det_target.shape[0] != num_samples:
        raise ValueError(
            "det_score_target length mismatch: "
            f"expected N={num_samples}, got N={det_target.shape[0]}"
        )
    if not np.isfinite(det_target).all():
        raise ValueError("det_score_target must be finite")
    return det_target


def _normalize_selected_tiles_target(
    *,
    raw: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    selected_target = np.asarray(raw)
    if selected_target.ndim == 2 and selected_target.shape[1] == 1:
        selected_target = selected_target[:, 0]
    if selected_target.ndim != 1:
        raise ValueError("selected_tiles_target must be shape [N] or [N,1]")
    if selected_target.shape[0] != num_samples:
        raise ValueError(
            "selected_tiles_target length mismatch: "
            f"expected N={num_samples}, got N={selected_target.shape[0]}"
        )
    if np.issubdtype(selected_target.dtype, np.floating) and not np.isfinite(selected_target).all():
        raise ValueError("selected_tiles_target must be finite")
    selected_int = selected_target.astype(np.int64, copy=False)
    if np.any(selected_int < 0):
        raise ValueError("selected_tiles_target must be >= 0")
    return selected_int


def load_eval_dataset_npz(
    *,
    path: str | Path,
    expected_height: int,
    expected_width: int,
) -> EvalDataset:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    det_score_target: np.ndarray | None = None
    selected_tiles_target: np.ndarray | None = None
    if suffix == ".npy":
        arr = np.load(dataset_path)
    elif suffix == ".npz":
        with np.load(dataset_path) as archive:
            if "images" not in archive:
                raise ValueError("dataset npz must contain key 'images'")
            arr = archive["images"]
            if "det_score_target" in archive:
                det_score_target = np.asarray(archive["det_score_target"])
            elif "det_scores_target" in archive:
                det_score_target = np.asarray(archive["det_scores_target"])
            if "selected_tiles_target" in archive:
                selected_tiles_target = np.asarray(archive["selected_tiles_target"])
            elif "selected_tiles_targets" in archive:
                selected_tiles_target = np.asarray(archive["selected_tiles_targets"])
    else:
        raise ValueError("dataset path must be .npy or .npz")

    images = np.asarray(arr, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError("dataset images must be [N,3,H,W]")
    if images.shape[1] != 3:
        raise ValueError("dataset images channel dimension must be 3")
    if images.shape[0] <= 0:
        raise ValueError("dataset must contain at least one sample")
    if images.shape[2] != expected_height or images.shape[3] != expected_width:
        raise ValueError(
            "dataset image shape mismatch: expected "
            f"H={expected_height}, W={expected_width}; got H={images.shape[2]}, W={images.shape[3]}"
        )
    if not np.isfinite(images).all():
        raise ValueError("dataset images must be finite")
    normalized_target = None
    if det_score_target is not None:
        normalized_target = _normalize_det_score_target(
            raw=det_score_target,
            num_samples=int(images.shape[0]),
        )
    normalized_selected_tiles_target = None
    if selected_tiles_target is not None:
        normalized_selected_tiles_target = _normalize_selected_tiles_target(
            raw=selected_tiles_target,
            num_samples=int(images.shape[0]),
        )
    return EvalDataset(
        images=images,
        det_score_target=normalized_target,
        selected_tiles_target=normalized_selected_tiles_target,
    )


def load_eval_images_npz(
    *,
    path: str | Path,
    expected_height: int,
    expected_width: int,
) -> np.ndarray:
    return load_eval_dataset_npz(
        path=path,
        expected_height=expected_height,
        expected_width=expected_width,
    ).images


def evaluate_model_dataset(
    *,
    model: ApexXModel,
    images: np.ndarray,
    requested_backend: str,
    selected_backend: str,
    fallback_policy: str,
    precision_profile: str,
    selection_fallback_reason: str | None,
    runtime_caps: RuntimeCaps,
    det_score_target: np.ndarray | None = None,
    selected_tiles_target: np.ndarray | None = None,
    max_samples: int | None = None,
    use_tta: bool = False,
) -> ModelDatasetEvalSummary:
    if images.ndim != 4:
        raise ValueError("images must be [N,3,H,W]")
    if images.shape[0] <= 0:
        raise ValueError("images must contain at least one sample")

    total_samples = int(images.shape[0])
    limit = total_samples if max_samples is None else min(int(max_samples), total_samples)
    if limit <= 0:
        raise ValueError("max_samples must be >= 1 when provided")

    det_scores: list[float] = []
    selected_tiles: list[int] = []
    execution_backend = "cpu"
    for idx in range(limit):
        batch = images[idx : idx + 1]
        result = run_model_inference(
            model=model,
            input_batch=batch,
            requested_backend=requested_backend,
            selected_backend=selected_backend,
            fallback_policy=fallback_policy,
            precision_profile=precision_profile,
            selection_fallback_reason=selection_fallback_reason,
            runtime_caps=runtime_caps,
            use_tta=use_tta,
        )
        det_scores.append(float(result.det_score))
        selected_tiles.append(int(result.selected_tiles))
        execution_backend = result.runtime.execution_backend

    det_arr = np.asarray(det_scores, dtype=np.float64)
    tiles_arr = np.asarray(selected_tiles, dtype=np.float64)
    det_score_target_metrics: dict[str, float | None] | None = None
    if det_score_target is not None:
        target_arr = _normalize_det_score_target(
            raw=np.asarray(det_score_target),
            num_samples=total_samples,
        ).astype(np.float64, copy=False)
        target_subset = target_arr[:limit]
        error = det_arr - target_subset
        mae = float(np.abs(error).mean())
        rmse = float(np.sqrt(np.mean(error * error)))
        bias = float(error.mean())

        ss_res = float(np.sum(error * error))
        centered = target_subset - float(target_subset.mean())
        ss_tot = float(np.sum(centered * centered))
        r2: float | None = None if ss_tot <= 0.0 else float(1.0 - (ss_res / ss_tot))

        corr: float | None = None
        pred_std = float(det_arr.std())
        tgt_std = float(target_subset.std())
        if pred_std > 0.0 and tgt_std > 0.0:
            corr = float(np.corrcoef(det_arr, target_subset)[0, 1])
        det_score_target_metrics = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "pearson_corr": corr,
        }

    selected_tiles_target_metrics: dict[str, float | None] | None = None
    if selected_tiles_target is not None:
        tiles_target_arr = _normalize_selected_tiles_target(
            raw=np.asarray(selected_tiles_target),
            num_samples=total_samples,
        ).astype(np.float64, copy=False)
        tiles_target_subset = tiles_target_arr[:limit]
        tiles_error = tiles_arr - tiles_target_subset
        selected_tiles_target_metrics = {
            "mae": float(np.abs(tiles_error).mean()),
            "rmse": float(np.sqrt(np.mean(tiles_error * tiles_error))),
            "bias": float(tiles_error.mean()),
            "exact_match_rate": float(np.mean(tiles_arr == tiles_target_subset)),
        }

    return ModelDatasetEvalSummary(
        num_samples=int(limit),
        det_score_mean=float(det_arr.mean()),
        det_score_std=float(det_arr.std()),
        det_score_min=float(det_arr.min()),
        det_score_max=float(det_arr.max()),
        selected_tiles_mean=float(tiles_arr.mean()),
        selected_tiles_p95=float(np.percentile(tiles_arr, 95)),
        execution_backend=execution_backend,
        precision_profile=precision_profile,
        source="model_dataset_npz",
        det_score_target_metrics=det_score_target_metrics,
        selected_tiles_target_metrics=selected_tiles_target_metrics,
    )


__all__ = [
    "RuntimeMetadata",
    "InferenceRunResult",
    "ModelDatasetEvalSummary",
    "EvalDataset",
    "extract_routing_diagnostics",
    "run_model_inference",
    "load_eval_dataset_npz",
    "load_eval_images_npz",
    "evaluate_model_dataset",
]
