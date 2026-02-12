from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

from apex_x.observability.metrics import (
    INFERENCE_LATENCY_SECONDS,
    INFERENCE_REQUESTS_TOTAL,
    start_metrics_server,
)
from apex_x.observability.tracing import configure_tracing, start_span
from apex_x.utils.logging import get_logger, log_event

LOGGER = get_logger("service_bridge")

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ort = None

TensorRTEngineExecutor: Any
try:
    from apex_x.runtime.tensorrt import TensorRTEngineExecutor as _TensorRTEngineExecutor

    TensorRTEngineExecutor = _TensorRTEngineExecutor
except Exception:  # pragma: no cover - optional dependency
    TensorRTEngineExecutor = None


class BridgeRequestItem(BaseModel):
    request_id: str = Field(..., min_length=1)
    budget_profile: str = Field(default="balanced")
    input_values: list[float] = Field(default_factory=list, alias="input")

    @field_validator("input_values", mode="before")
    @classmethod
    def parse_input_values(cls, v: Any) -> list[float]:
        # Allow numpy arrays or other iterables to be passed in test contexts,
        # but in production JSON it will be a list.
        if isinstance(v, np.ndarray):
            return v.flatten().tolist()
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return []


class BridgePayload(BaseModel):
    backend: Literal["onnxruntime", "ort", "tensorrt", "trt", "health"]
    artifact_path: str = Field(default="", description="Required for inference backends")
    requests: list[BridgeRequestItem] = Field(default_factory=list)

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, v: str, info: Any) -> str:
        # Pydantic v2 validation info access might differ slightly depending on version,
        # but accessing .data is standard for cross-field validation.
        # Fallback to simple check if context logic is complex.
        if info.data.get("backend") == "health":
            return v
        if not v or not v.strip():
             raise ValueError("artifact_path is required for inference backends")
        return v


def _selected_tiles_from_profile(profile: str) -> int:
    normalized = str(profile).strip().lower()
    if normalized == "quality":
        return 64
    if normalized == "edge":
        return 16
    return 32


def _positive_dim(dim: Any) -> int | None:
    if isinstance(dim, bool):
        return None
    if isinstance(dim, int) and dim > 0:
        return dim
    return None


def _reshape_flat_input(flat: np.ndarray, shape_spec: Any) -> np.ndarray:
    flat_vec = np.asarray(flat, dtype=np.float32).reshape(-1)
    if shape_spec is None:
        return flat_vec.reshape(1, flat_vec.size)

    if isinstance(shape_spec, (tuple, list)):
        dims = [_positive_dim(dim) for dim in shape_spec]
        if dims:
            # TensorRT commonly reports dynamic dims as -1. Resolve them against
            # provided flat input length when possible.
            resolved = list(dims)
            if resolved[0] is None:
                resolved[0] = 1
            dynamic_positions = [idx for idx, dim in enumerate(resolved) if dim is None]
            known_product = 1
            for dim in resolved:
                if dim is not None:
                    known_product *= int(dim)
            if known_product > 0 and flat_vec.size % known_product == 0:
                remaining = int(flat_vec.size // known_product)
                if len(dynamic_positions) == 0 and remaining == 1:
                    return flat_vec.reshape(tuple(int(cast(int, dim)) for dim in resolved))
                if len(dynamic_positions) == 1:
                    resolved[dynamic_positions[0]] = remaining
                    return flat_vec.reshape(tuple(int(cast(int, dim)) for dim in resolved))
                if len(dynamic_positions) == 2:
                    side = int(np.sqrt(float(remaining)))
                    if side * side == remaining:
                        resolved[dynamic_positions[0]] = side
                        resolved[dynamic_positions[1]] = side
                        return flat_vec.reshape(tuple(int(cast(int, dim)) for dim in resolved))

        if len(dims) >= 2 and all(dim is not None for dim in dims[1:]):
            tail_dims = [cast(int, dim) for dim in dims[1:]]
            expected = int(np.prod(np.asarray(tail_dims, dtype=np.int64)))
            if expected == flat_vec.size:
                return flat_vec.reshape(tuple([1, *tail_dims]))
        if len(dims) == 2 and dims[1] is not None and int(dims[1]) == flat_vec.size:
            return flat_vec.reshape(1, int(dims[1]))
        if len(dims) == 1 and dims[0] is not None and int(dims[0]) == flat_vec.size:
            return flat_vec.reshape(int(dims[0]))
    return flat_vec.reshape(1, flat_vec.size)


def _score_from_arrays(values: list[np.ndarray]) -> float:
    if not values:
        return 0.0
    first = np.asarray(values[0], dtype=np.float32)
    if first.size <= 0:
        return 0.0
    score = float(np.mean(np.abs(first)))
    return float(np.clip(score, 0.0, 1.0))


def _predict_onnxruntime(
    *,
    artifact_path: Path,
    items: list[BridgeRequestItem],
) -> list[dict[str, Any]]:
    if ort is None:
        raise RuntimeError("onnxruntime python package is not available")
    session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
    model_inputs = list(session.get_inputs())
    if not model_inputs:
        raise RuntimeError("onnx model has no inputs")

    results: list[dict[str, Any]] = []
    
    start_time = time.perf_counter()
    try:
        for item in items:
            feed: dict[str, np.ndarray] = {}
            input_np = np.asarray(item.input_values, dtype=np.float32)
            for model_input in model_inputs:
                shaped = _reshape_flat_input(input_np, getattr(model_input, "shape", None))
                feed[str(model_input.name)] = shaped.astype(np.float32, copy=False)
            outputs = session.run(None, feed)
            output_arrays = [np.asarray(value) for value in outputs]
            score = _score_from_arrays(output_arrays)
            results.append(
                {
                    "request_id": item.request_id,
                    "scores": [score],
                    "selected_tiles": _selected_tiles_from_profile(item.budget_profile),
                    "backend": "onnxruntime-python-bridge",
                }
            )
        
        duration = time.perf_counter() - start_time
        INFERENCE_LATENCY_SECONDS.labels(model_version="onnx", precision="fp32").observe(duration)
        INFERENCE_REQUESTS_TOTAL.labels(status="success").inc(len(items))

    except Exception:
        INFERENCE_REQUESTS_TOTAL.labels(status="error").inc(len(items))
        raise

    return results


def _predict_tensorrt(
    *,
    artifact_path: Path,
    items: list[BridgeRequestItem],
) -> list[dict[str, Any]]:
    if TensorRTEngineExecutor is None:
        raise RuntimeError("TensorRT Python executor is not available")
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(f"TensorRT Python package is not available: {exc}") from exc

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(artifact_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {artifact_path}")

    input_shapes: dict[str, tuple[int, ...]] = {}
    if hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name"):
        for idx in range(int(engine.num_io_tensors)):
            name = str(engine.get_tensor_name(idx))
            mode = engine.get_tensor_mode(name)
            if hasattr(trt, "TensorIOMode"):
                is_input = bool(mode == trt.TensorIOMode.INPUT)
            else:  # pragma: no cover - TRT API compatibility path
                is_input = "INPUT" in str(mode).upper()
            if is_input:
                input_shapes[name] = tuple(int(dim) for dim in engine.get_tensor_shape(name))
    elif hasattr(engine, "num_bindings"):  # pragma: no cover - TRT API compatibility path
        for idx in range(int(engine.num_bindings)):
            if bool(engine.binding_is_input(idx)):
                name = str(engine.get_binding_name(idx))
                input_shapes[name] = tuple(int(dim) for dim in engine.get_binding_shape(idx))

    if not input_shapes:
        raise RuntimeError("TensorRT engine has no inputs")

    executor = TensorRTEngineExecutor(engine_path=artifact_path)
    results: list[dict[str, Any]] = []
    
    start_time = time.perf_counter()
    try:
        for item in items:
            input_tensors: dict[str, np.ndarray] = {}
            input_np = np.asarray(item.input_values, dtype=np.float32)
            for name, shape_spec in input_shapes.items():
                shaped = _reshape_flat_input(input_np, shape_spec)
                input_tensors[name] = shaped.astype(np.float32, copy=False)
            execution = executor.run(input_tensors=input_tensors)
            output_arrays = [np.asarray(value) for value in execution.outputs.values()]
            score = _score_from_arrays(output_arrays)
            results.append(
                {
                    "request_id": item.request_id,
                    "scores": [score],
                    "selected_tiles": _selected_tiles_from_profile(item.budget_profile),
                    "backend": "tensorrt-python-bridge",
                }
            )
        duration = time.perf_counter() - start_time
        INFERENCE_LATENCY_SECONDS.labels(model_version="trt", precision="fp16").observe(duration)
        INFERENCE_REQUESTS_TOTAL.labels(status="success").inc(len(items))

    except Exception:
        INFERENCE_REQUESTS_TOTAL.labels(status="error").inc(len(items))
        raise

    return results


def _run_bridge(raw_payload: dict[str, Any]) -> dict[str, Any]:
    # Pydantic validation handles parsing and basic checks
    payload = BridgePayload.model_validate(raw_payload)

    if payload.backend == "health":
        return {
            "status": "ok",
            "backends": {
                "onnxruntime": bool(ort is not None),
                "tensorrt": bool(TensorRTEngineExecutor is not None),
            }
        }
    
    artifact_path = Path(payload.artifact_path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact path not found: {artifact_path}")

    if payload.backend in {"onnxruntime", "ort"}:
        return {"results": _predict_onnxruntime(artifact_path=artifact_path, items=payload.requests)}
    if payload.backend in {"tensorrt", "trt"}:
        return {"results": _predict_tensorrt(artifact_path=artifact_path, items=payload.requests)}
    
    # improved Pydantic validation should theoretically catch this before, but keeping for safety
    raise ValueError(f"unsupported backend: {payload.backend}")


def main() -> int:
    # Start Prometheus Metrics Server
    start_metrics_server(8000)
    log_event(LOGGER, "service_started", fields={"port": 8000})

    configure_tracing()

    try:
        # Read from stdin with a span
        with start_span("read_stdin"):
             input_data = sys.stdin.read()
        
        if not input_data.strip():
             raise ValueError("empty input")
        
        payload_dict = json.loads(input_data)
        if not isinstance(payload_dict, dict):
            raise ValueError("bridge payload must be a JSON object")
            
        # Extract trace context if passed in payload headers/metadata?
        # Assuming payload might have 'trace_context' or similar if passed from Go
        # For now, we wrap the processing
        trace_context = payload_dict.get("trace_context")
        
        with start_span("process_request", context_carrier=trace_context, attributes={"item_count": len(payload_dict.get("requests", []))}):
            result = _run_bridge(payload_dict)
        
        log_event(LOGGER, "request_processed", fields={"item_count": len(payload_dict.get("requests", []))})
    except ValidationError as ve:
        # Structured validation errors
        err_payload = {
            "results": [], 
            "error": "ValidationError",
            "details": ve.errors(include_url=False, include_context=False)
        }
        sys.stdout.write(json.dumps(err_payload))
        log_event(LOGGER, "validation_error", fields={"details": str(ve)})
        INFERENCE_REQUESTS_TOTAL.labels(status="validation_error").inc()
        return 1
    except Exception as exc:  # pragma: no cover - exercised by integration boundary
        err_payload = {"results": [], "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(err_payload))
        log_event(LOGGER, "internal_error", fields={"error": str(exc)}, level="ERROR")
        INFERENCE_REQUESTS_TOTAL.labels(status="internal_error").inc()
        return 1

    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
