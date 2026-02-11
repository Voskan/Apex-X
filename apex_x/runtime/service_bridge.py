from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import onnxruntime as ort  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    ort = None

TensorRTEngineExecutor: Any
try:
    from apex_x.runtime.tensorrt import TensorRTEngineExecutor as _TensorRTEngineExecutor
    TensorRTEngineExecutor = _TensorRTEngineExecutor
except Exception:  # pragma: no cover - optional dependency
    TensorRTEngineExecutor = None


@dataclass(frozen=True, slots=True)
class BridgeRequestItem:
    request_id: str
    budget_profile: str
    input_values: np.ndarray


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


def _parse_items(raw_items: Any) -> list[BridgeRequestItem]:
    if not isinstance(raw_items, list):
        raise ValueError("requests must be a list")
    out: list[BridgeRequestItem] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"request at index {idx} must be an object")
        request_id = str(item.get("request_id", "")).strip()
        if not request_id:
            raise ValueError(f"request at index {idx} must provide request_id")
        budget_profile = str(item.get("budget_profile", "balanced")).strip() or "balanced"
        input_values = np.asarray(item.get("input", []), dtype=np.float32).reshape(-1)
        out.append(
            BridgeRequestItem(
                request_id=request_id,
                budget_profile=budget_profile,
                input_values=input_values,
            )
        )
    return out


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
    for item in items:
        feed: dict[str, np.ndarray] = {}
        for model_input in model_inputs:
            shaped = _reshape_flat_input(item.input_values, getattr(model_input, "shape", None))
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
    return results


def _predict_tensorrt(
    *,
    artifact_path: Path,
    items: list[BridgeRequestItem],
) -> list[dict[str, Any]]:
    if TensorRTEngineExecutor is None:
        raise RuntimeError("TensorRT Python executor is not available")
    executor = TensorRTEngineExecutor(engine_path=artifact_path)
    results: list[dict[str, Any]] = []
    for item in items:
        # TensorRT runtime bridge currently maps flat vectors to [1, F] by default.
        input_batch = _reshape_flat_input(item.input_values, None)
        execution = executor.run(input_batch=input_batch.astype(np.float32, copy=False))
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
    return results


def _run_bridge(raw_payload: dict[str, Any]) -> dict[str, Any]:
    backend = str(raw_payload.get("backend", "")).strip().lower()
    artifact_path_raw = raw_payload.get("artifact_path")
    if not backend:
        raise ValueError("backend is required")
    if not isinstance(artifact_path_raw, str) or not artifact_path_raw.strip():
        raise ValueError("artifact_path is required")
    artifact_path = Path(artifact_path_raw).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact path not found: {artifact_path}")
    items = _parse_items(raw_payload.get("requests", []))

    if backend in {"onnxruntime", "ort"}:
        return {"results": _predict_onnxruntime(artifact_path=artifact_path, items=items)}
    if backend in {"tensorrt", "trt"}:
        return {"results": _predict_tensorrt(artifact_path=artifact_path, items=items)}
    raise ValueError(f"unsupported backend: {backend}")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("bridge payload must be a JSON object")
        result = _run_bridge(payload)
    except Exception as exc:  # pragma: no cover - exercised by integration boundary
        err_payload = {"results": [], "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(err_payload))
        return 1

    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
