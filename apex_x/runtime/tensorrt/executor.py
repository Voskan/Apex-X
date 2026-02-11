from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apex_x.utils.logging import get_logger

try:
    import tensorrt as trt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    trt = None


@dataclass(frozen=True, slots=True)
class TensorRTExecutionResult:
    engine_path: Path
    input_name: str
    output_names: tuple[str, ...]
    outputs: dict[str, np.ndarray]


def _shape_to_tuple(raw_shape: Any) -> tuple[int, ...]:
    if isinstance(raw_shape, tuple):
        return tuple(int(v) for v in raw_shape)
    if isinstance(raw_shape, list):
        return tuple(int(v) for v in raw_shape)
    return tuple(int(v) for v in raw_shape)


def _dtype_to_torch_dtype(*, trt_mod: Any, dtype_obj: Any) -> torch.dtype:
    if dtype_obj in {getattr(trt_mod, "float16", object()), getattr(trt_mod, "HALF", object())}:
        return torch.float16
    if dtype_obj in {getattr(trt_mod, "float32", object()), getattr(trt_mod, "FLOAT", object())}:
        return torch.float32
    if dtype_obj in {getattr(trt_mod, "int32", object()), getattr(trt_mod, "INT32", object())}:
        return torch.int32
    if dtype_obj in {getattr(trt_mod, "int8", object()), getattr(trt_mod, "INT8", object())}:
        return torch.int8
    if dtype_obj in {getattr(trt_mod, "bool", object()), getattr(trt_mod, "BOOL", object())}:
        return torch.bool

    dtype_name = str(dtype_obj).lower()
    if "float16" in dtype_name or "half" in dtype_name:
        return torch.float16
    if "float32" in dtype_name or "float" in dtype_name:
        return torch.float32
    if "int32" in dtype_name:
        return torch.int32
    if "int8" in dtype_name:
        return torch.int8
    if "bool" in dtype_name:
        return torch.bool
    return torch.float32


def _resolve_named_inputs(
    *,
    input_names: tuple[str, ...],
    input_batch: np.ndarray | None,
    input_name: str | None,
    input_tensors: dict[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], str]:
    if not input_names:
        raise RuntimeError("TensorRT engine has no inputs")

    resolved: dict[str, np.ndarray] = {}
    if input_tensors:
        for name, arr in input_tensors.items():
            key = str(name).strip()
            if not key:
                raise ValueError("TensorRT input tensor name must be non-empty")
            resolved[key] = np.asarray(arr)

    explicit_name = input_name.strip() if input_name is not None else None
    primary_input_name: str | None = None
    input_arr = np.asarray(input_batch) if input_batch is not None else None
    if input_arr is not None:
        if explicit_name:
            if explicit_name in resolved:
                raise ValueError(
                    "TensorRT input "
                    f"'{explicit_name}' provided both in input_batch and input_tensors"
                )
            resolved[explicit_name] = input_arr
            primary_input_name = explicit_name
        elif len(input_names) == 1:
            only_name = input_names[0]
            if only_name in resolved:
                raise ValueError(
                    f"TensorRT input '{only_name}' provided both in input_batch and input_tensors"
                )
            resolved[only_name] = input_arr
            primary_input_name = only_name
        else:
            missing = [name for name in input_names if name not in resolved]
            if len(missing) == 1:
                resolved[missing[0]] = input_arr
                primary_input_name = missing[0]
            elif len(missing) == 0:
                raise ValueError(
                    "input_batch provided but all TensorRT inputs are "
                    "already supplied in input_tensors"
                )
            else:
                raise ValueError(
                    "TensorRT engine has multiple inputs; set input_name or provide input_tensors "
                    f"for all but one input. available={input_names}"
                )
    elif explicit_name:
        if explicit_name not in resolved:
            raise ValueError(
                f"input_name={explicit_name!r} was provided but input_batch is missing and "
                "input_tensors does not contain that key"
            )
        primary_input_name = explicit_name

    extra = [name for name in resolved if name not in input_names]
    if extra:
        raise ValueError(f"unknown TensorRT input tensors supplied: {extra}")
    missing_required = [name for name in input_names if name not in resolved]
    if missing_required:
        raise ValueError(f"missing TensorRT input tensors: {missing_required}")

    if primary_input_name is None:
        primary_input_name = input_names[0]
    return resolved, primary_input_name


class TensorRTEngineExecutor:
    """Run TensorRT engine inference from a serialized engine file."""

    def __init__(
        self,
        *,
        engine_path: str | Path,
        plugin_library_paths: tuple[str | Path, ...] = (),
    ) -> None:
        self._logger = get_logger(__name__)
        self._engine_path = Path(engine_path).expanduser().resolve()
        self._plugin_library_paths: tuple[Path, ...] = tuple(
            Path(path).expanduser().resolve() for path in plugin_library_paths
        )
        self._loaded_plugin_libraries: list[ctypes.CDLL] = []
        if trt is None:
            self._trt_logger = None
        else:
            self._trt_logger = trt.Logger(trt.Logger.WARNING)

    @property
    def engine_path(self) -> Path:
        return self._engine_path

    def load_plugin_libraries(self) -> tuple[Path, ...]:
        loaded: list[Path] = []
        dlopen_mode = int(getattr(ctypes, "RTLD_GLOBAL", 0))
        for path in self._plugin_library_paths:
            if not path.exists():
                raise FileNotFoundError(f"TensorRT plugin library not found: {path}")
            handle = ctypes.CDLL(str(path), mode=dlopen_mode)
            self._loaded_plugin_libraries.append(handle)
            loaded.append(path)
            self._logger.info("loaded_tensorrt_runtime_plugin_library path=%s", path)
        return tuple(loaded)

    def _require_trt(self) -> Any:
        if trt is None:
            raise RuntimeError("TensorRT Python package is not available")
        assert self._trt_logger is not None
        return trt

    def _deserialize_engine(self, *, trt_mod: Any) -> Any:
        if not self._engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self._engine_path}")
        runtime = trt_mod.Runtime(self._trt_logger)
        engine = runtime.deserialize_cuda_engine(self._engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self._engine_path}")
        return engine

    def _is_tensor_input(self, *, trt_mod: Any, mode: Any) -> bool:
        if hasattr(trt_mod, "TensorIOMode"):
            return bool(mode == trt_mod.TensorIOMode.INPUT)
        normalized = str(mode).upper()
        return "INPUT" in normalized and "OUTPUT" not in normalized

    def _collect_io_names(
        self,
        *,
        trt_mod: Any,
        engine: Any,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name"):
            input_names: list[str] = []
            output_names: list[str] = []
            for idx in range(int(engine.num_io_tensors)):
                name = str(engine.get_tensor_name(idx))
                mode = engine.get_tensor_mode(name)
                if self._is_tensor_input(trt_mod=trt_mod, mode=mode):
                    input_names.append(name)
                else:
                    output_names.append(name)
            return tuple(input_names), tuple(output_names)

        input_names = []
        output_names = []
        if not hasattr(engine, "num_bindings"):
            raise RuntimeError("TensorRT engine does not expose IO tensor or binding APIs")
        for idx in range(int(engine.num_bindings)):
            name = str(engine.get_binding_name(idx))
            if bool(engine.binding_is_input(idx)):
                input_names.append(name)
            else:
                output_names.append(name)
        return tuple(input_names), tuple(output_names)

    def _binding_index(self, *, engine: Any, name: str) -> int | None:
        if not hasattr(engine, "get_binding_index"):
            return None
        index = int(engine.get_binding_index(name))
        if index < 0:
            return None
        return index

    def _set_input_shape(
        self,
        *,
        context: Any,
        input_name: str,
        binding_index: int | None,
        shape: tuple[int, ...],
    ) -> None:
        if hasattr(context, "set_input_shape"):
            ok = context.set_input_shape(input_name, shape)
            if ok is False:
                raise RuntimeError(f"failed to set TensorRT input shape for tensor '{input_name}'")
            return
        if binding_index is None:
            return
        if hasattr(context, "set_binding_shape"):
            ok = context.set_binding_shape(binding_index, shape)
            if ok is False:
                raise RuntimeError(
                    f"failed to set TensorRT binding shape for tensor '{input_name}'"
                )

    def _tensor_shape(
        self,
        *,
        context: Any,
        engine: Any,
        name: str,
        binding_index: int | None,
    ) -> tuple[int, ...]:
        shape: tuple[int, ...]
        if hasattr(context, "get_tensor_shape"):
            shape = _shape_to_tuple(context.get_tensor_shape(name))
        elif binding_index is not None and hasattr(context, "get_binding_shape"):
            shape = _shape_to_tuple(context.get_binding_shape(binding_index))
        elif hasattr(engine, "get_tensor_shape"):
            shape = _shape_to_tuple(engine.get_tensor_shape(name))
        elif binding_index is not None and hasattr(engine, "get_binding_shape"):
            shape = _shape_to_tuple(engine.get_binding_shape(binding_index))
        else:
            raise RuntimeError(f"unable to resolve TensorRT shape for tensor '{name}'")

        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"dynamic shape remained unresolved for tensor '{name}': {shape}")
        return shape

    def _tensor_dtype(
        self,
        *,
        trt_mod: Any,
        engine: Any,
        name: str,
        binding_index: int | None,
    ) -> torch.dtype:
        if hasattr(engine, "get_tensor_dtype"):
            dtype_obj = engine.get_tensor_dtype(name)
            return _dtype_to_torch_dtype(trt_mod=trt_mod, dtype_obj=dtype_obj)
        if binding_index is not None and hasattr(engine, "get_binding_dtype"):
            dtype_obj = engine.get_binding_dtype(binding_index)
            return _dtype_to_torch_dtype(trt_mod=trt_mod, dtype_obj=dtype_obj)
        return torch.float32

    def _execute(
        self,
        *,
        engine: Any,
        context: Any,
        tensor_map: dict[str, torch.Tensor],
    ) -> None:
        if hasattr(context, "set_tensor_address"):
            for name, tensor in tensor_map.items():
                context.set_tensor_address(name, int(tensor.data_ptr()))
            if hasattr(context, "execute_async_v3"):
                stream_ptr = int(torch.cuda.current_stream().cuda_stream)
                ok = context.execute_async_v3(stream_ptr)
                if ok is False:
                    raise RuntimeError("TensorRT execute_async_v3 returned false")
                torch.cuda.synchronize()
                return

        if not hasattr(engine, "num_bindings") or not hasattr(context, "execute_v2"):
            raise RuntimeError("TensorRT engine/context does not support executable binding API")

        bindings = [0] * int(engine.num_bindings)
        for name, tensor in tensor_map.items():
            idx = self._binding_index(engine=engine, name=name)
            if idx is None:
                raise RuntimeError(f"TensorRT binding not found for tensor '{name}'")
            bindings[idx] = int(tensor.data_ptr())
        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        torch.cuda.synchronize()

    def run(
        self,
        *,
        input_batch: np.ndarray | None = None,
        input_name: str | None = None,
        input_tensors: dict[str, np.ndarray] | None = None,
    ) -> TensorRTExecutionResult:
        trt_mod = self._require_trt()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT engine execution")

        self.load_plugin_libraries()
        engine = self._deserialize_engine(trt_mod=trt_mod)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT create_execution_context returned None")

        input_names, output_names = self._collect_io_names(trt_mod=trt_mod, engine=engine)
        if not output_names:
            raise RuntimeError("TensorRT engine has no outputs")

        resolved_inputs, primary_input_name = _resolve_named_inputs(
            input_names=input_names,
            input_batch=input_batch,
            input_name=input_name,
            input_tensors=input_tensors,
        )
        tensors: dict[str, torch.Tensor] = {}
        for name in input_names:
            input_arr = np.asarray(resolved_inputs[name])
            if input_arr.ndim <= 0:
                raise ValueError(f"TensorRT input '{name}' must have rank >= 1")
            binding_index = self._binding_index(engine=engine, name=name)
            self._set_input_shape(
                context=context,
                input_name=name,
                binding_index=binding_index,
                shape=tuple(int(v) for v in input_arr.shape),
            )
            input_dtype = self._tensor_dtype(
                trt_mod=trt_mod,
                engine=engine,
                name=name,
                binding_index=binding_index,
            )
            tensors[name] = (
                torch.from_numpy(input_arr)
                .contiguous()
                .to(
                    device="cuda",
                    dtype=input_dtype,
                )
            )

        for name in output_names:
            binding_index = self._binding_index(engine=engine, name=name)
            shape = self._tensor_shape(
                context=context,
                engine=engine,
                name=name,
                binding_index=binding_index,
            )
            dtype = self._tensor_dtype(
                trt_mod=trt_mod,
                engine=engine,
                name=name,
                binding_index=binding_index,
            )
            tensors[name] = torch.empty(shape, device="cuda", dtype=dtype)

        self._execute(engine=engine, context=context, tensor_map=tensors)
        outputs = {name: tensors[name].detach().cpu().numpy().copy() for name in output_names}
        return TensorRTExecutionResult(
            engine_path=self._engine_path,
            input_name=primary_input_name,
            output_names=output_names,
            outputs=outputs,
        )


__all__ = ["TensorRTExecutionResult", "TensorRTEngineExecutor"]
