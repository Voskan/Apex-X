from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CUDA_REASON_CUDA_UNAVAILABLE = "cuda_unavailable"
CUDA_REASON_DEVICE_NOT_FOUND = "cuda_device_not_found"
CUDA_REASON_DEVICE_INDEX_OUT_OF_RANGE = "cuda_device_index_out_of_range"
CUDA_REASON_QUERY_FAILED = "cuda_query_failed"

TRITON_REASON_NOT_INSTALLED = "triton_not_installed"

TENSORRT_PYTHON_REASON_NOT_INSTALLED = "tensorrt_python_not_installed"
TENSORRT_PYTHON_REASON_IMPORT_FAILED = "tensorrt_python_import_failed"
TENSORRT_INT8_REASON_PYTHON_UNAVAILABLE = "tensorrt_python_unavailable"
TENSORRT_INT8_REASON_CUDA_REQUIRED = "cuda_required_for_tensorrt_int8"
TENSORRT_INT8_REASON_BUILDER_FLAG_MISSING = "tensorrt_int8_builder_flag_missing"

FP8_REASON_TORCH_DTYPE_MISSING = "torch_build_missing_fp8_dtype"
FP8_REASON_CUDA_REQUIRED = "fp8_requires_cuda"
FP8_REASON_COMPUTE_CAPABILITY_UNKNOWN = "cuda_compute_capability_unknown"
FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90 = "compute_capability_below_sm90"


def runtime_reason_catalog() -> dict[str, tuple[str, ...]]:
    """Return canonical reason-code sets for runtime capability reporting."""

    return {
        "cuda": (
            CUDA_REASON_CUDA_UNAVAILABLE,
            CUDA_REASON_DEVICE_NOT_FOUND,
            CUDA_REASON_DEVICE_INDEX_OUT_OF_RANGE,
            CUDA_REASON_QUERY_FAILED,
        ),
        "triton": (TRITON_REASON_NOT_INSTALLED,),
        "tensorrt_python": (
            TENSORRT_PYTHON_REASON_NOT_INSTALLED,
            TENSORRT_PYTHON_REASON_IMPORT_FAILED,
        ),
        "tensorrt_int8": (
            TENSORRT_INT8_REASON_PYTHON_UNAVAILABLE,
            TENSORRT_INT8_REASON_CUDA_REQUIRED,
            TENSORRT_INT8_REASON_BUILDER_FLAG_MISSING,
        ),
        "fp8": (
            FP8_REASON_TORCH_DTYPE_MISSING,
            FP8_REASON_CUDA_REQUIRED,
            FP8_REASON_COMPUTE_CAPABILITY_UNKNOWN,
            FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90,
        ),
    }


def _find_spec(module_name: str) -> object | None:
    return importlib.util.find_spec(module_name)


def _safe_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _supported_fp8_dtype_names() -> tuple[str, ...]:
    dtype_names: list[str] = []
    for attr_name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(torch, attr_name, None)
        if isinstance(dtype, torch.dtype):
            dtype_names.append(attr_name)
    return tuple(dtype_names)


@dataclass(frozen=True, slots=True)
class CudaCaps:
    available: bool
    device_count: int
    device_name: str | None
    compute_capability: tuple[int, int] | None
    reason: str | None

    @property
    def compute_capability_str(self) -> str | None:
        if self.compute_capability is None:
            return None
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"


@dataclass(frozen=True, slots=True)
class TritonCaps:
    available: bool
    version: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class TensorRTCaps:
    python_available: bool
    python_version: str | None
    python_reason: str | None
    headers_available: bool
    header_path: str | None
    int8_available: bool
    int8_reason: str | None


@dataclass(frozen=True, slots=True)
class FP8Caps:
    available: bool
    dtype_available: bool
    supported_dtypes: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True, slots=True)
class RuntimeCaps:
    cuda: CudaCaps
    triton: TritonCaps
    tensorrt: TensorRTCaps
    fp8: FP8Caps

    @property
    def any_gpu_runtime(self) -> bool:
        return self.cuda.available and (self.triton.available or self.tensorrt.python_available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda": {
                "available": self.cuda.available,
                "device_count": self.cuda.device_count,
                "device_name": self.cuda.device_name,
                "compute_capability": self.cuda.compute_capability_str,
                "reason": self.cuda.reason,
            },
            "triton": {
                "available": self.triton.available,
                "version": self.triton.version,
                "reason": self.triton.reason,
            },
            "tensorrt": {
                "python_available": self.tensorrt.python_available,
                "python_version": self.tensorrt.python_version,
                "python_reason": self.tensorrt.python_reason,
                "headers_available": self.tensorrt.headers_available,
                "header_path": self.tensorrt.header_path,
                "int8_available": self.tensorrt.int8_available,
                "int8_reason": self.tensorrt.int8_reason,
            },
            "fp8": {
                "available": self.fp8.available,
                "dtype_available": self.fp8.dtype_available,
                "supported_dtypes": list(self.fp8.supported_dtypes),
                "reason": self.fp8.reason,
            },
            "any_gpu_runtime": self.any_gpu_runtime,
        }


def detect_cuda_caps(*, device_index: int | None = None) -> CudaCaps:
    if not torch.cuda.is_available():
        return CudaCaps(
            available=False,
            device_count=0,
            device_name=None,
            compute_capability=None,
            reason=CUDA_REASON_CUDA_UNAVAILABLE,
        )

    count = int(torch.cuda.device_count())
    if count <= 0:
        return CudaCaps(
            available=False,
            device_count=0,
            device_name=None,
            compute_capability=None,
            reason=CUDA_REASON_DEVICE_NOT_FOUND,
        )

    idx = int(torch.cuda.current_device() if device_index is None else device_index)
    if idx < 0 or idx >= count:
        return CudaCaps(
            available=False,
            device_count=count,
            device_name=None,
            compute_capability=None,
            reason=CUDA_REASON_DEVICE_INDEX_OUT_OF_RANGE,
        )

    try:
        name = str(torch.cuda.get_device_name(idx))
        capability = torch.cuda.get_device_capability(idx)
        cc = (int(capability[0]), int(capability[1]))
        return CudaCaps(
            available=True,
            device_count=count,
            device_name=name,
            compute_capability=cc,
            reason=None,
        )
    except Exception:  # pragma: no cover - defensive only
        return CudaCaps(
            available=False,
            device_count=count,
            device_name=None,
            compute_capability=None,
            reason=CUDA_REASON_QUERY_FAILED,
        )


def detect_triton_caps() -> TritonCaps:
    if _find_spec("triton") is None:
        return TritonCaps(available=False, version=None, reason=TRITON_REASON_NOT_INSTALLED)

    version = _package_version("triton")
    if version is None:
        module = _safe_import("triton")
        if module is not None:
            version_attr = getattr(module, "__version__", None)
            if isinstance(version_attr, str):
                version = version_attr

    return TritonCaps(available=True, version=version, reason=None)


def _candidate_tensorrt_include_paths() -> list[Path]:
    candidates: list[Path] = []

    env_vars = (
        "TENSORRT_INCLUDE_DIR",
        "TRT_INCLUDE_DIR",
        "TENSORRT_ROOT",
        "TRT_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
    )
    for var in env_vars:
        value = os.environ.get(var)
        if not value:
            continue
        root = Path(value).expanduser()
        candidates.append(root)
        candidates.append(root / "include")

    candidates.extend(
        [
            Path("/usr/include"),
            Path("/usr/local/include"),
            Path("/usr/include/x86_64-linux-gnu"),
            Path("/usr/local/TensorRT/include"),
            Path("/opt/tensorrt/include"),
            Path("/opt/homebrew/include"),
        ]
    )

    return candidates


def _find_tensorrt_header(
    search_paths: list[Path] | None = None,
) -> Path | None:
    header_names = ("NvInfer.h", "NvInferRuntime.h")
    candidates = search_paths or _candidate_tensorrt_include_paths()
    checked: set[Path] = set()

    for base in candidates:
        base_resolved = base.expanduser()
        if base_resolved in checked:
            continue
        checked.add(base_resolved)
        for header_name in header_names:
            header_path = base_resolved / header_name
            if header_path.is_file():
                return header_path
    return None


def detect_tensorrt_caps(
    *,
    cuda: CudaCaps,
    header_search_paths: list[str | Path] | None = None,
) -> TensorRTCaps:
    spec_found = _find_spec("tensorrt") is not None
    trt_module = _safe_import("tensorrt") if spec_found else None

    python_available = trt_module is not None
    python_version: str | None = None
    python_reason: str | None = None
    if python_available:
        python_version = _package_version("tensorrt")
        if python_version is None:
            version_attr = getattr(trt_module, "__version__", None)
            if isinstance(version_attr, str):
                python_version = version_attr
    else:
        python_reason = TENSORRT_PYTHON_REASON_NOT_INSTALLED
        if spec_found:
            python_reason = TENSORRT_PYTHON_REASON_IMPORT_FAILED

    header_paths = None
    if header_search_paths is not None:
        header_paths = [Path(p) for p in header_search_paths]
    header = _find_tensorrt_header(search_paths=header_paths)
    headers_available = header is not None
    header_path = str(header) if header is not None else None

    has_int8_builder_flag = False
    if trt_module is not None:
        builder_flag = getattr(trt_module, "BuilderFlag", None)
        has_int8_builder_flag = builder_flag is not None and hasattr(builder_flag, "INT8")

    if not python_available:
        int8_available = False
        int8_reason = TENSORRT_INT8_REASON_PYTHON_UNAVAILABLE
    elif not cuda.available:
        int8_available = False
        int8_reason = TENSORRT_INT8_REASON_CUDA_REQUIRED
    elif not has_int8_builder_flag:
        int8_available = False
        int8_reason = TENSORRT_INT8_REASON_BUILDER_FLAG_MISSING
    else:
        int8_available = True
        int8_reason = None

    return TensorRTCaps(
        python_available=python_available,
        python_version=python_version,
        python_reason=python_reason,
        headers_available=headers_available,
        header_path=header_path,
        int8_available=int8_available,
        int8_reason=int8_reason,
    )


def detect_fp8_caps(*, cuda: CudaCaps) -> FP8Caps:
    dtype_names = _supported_fp8_dtype_names()

    if not dtype_names:
        return FP8Caps(
            available=False,
            dtype_available=False,
            supported_dtypes=(),
            reason=FP8_REASON_TORCH_DTYPE_MISSING,
        )
    if not cuda.available:
        return FP8Caps(
            available=False,
            dtype_available=True,
            supported_dtypes=dtype_names,
            reason=FP8_REASON_CUDA_REQUIRED,
        )
    if cuda.compute_capability is None:
        return FP8Caps(
            available=False,
            dtype_available=True,
            supported_dtypes=dtype_names,
            reason=FP8_REASON_COMPUTE_CAPABILITY_UNKNOWN,
        )
    major, minor = cuda.compute_capability
    if major < 9:
        return FP8Caps(
            available=False,
            dtype_available=True,
            supported_dtypes=dtype_names,
            reason=FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90,
        )
    return FP8Caps(
        available=True,
        dtype_available=True,
        supported_dtypes=dtype_names,
        reason=None,
    )


def detect_runtime_caps(
    *,
    device_index: int | None = None,
    header_search_paths: list[str | Path] | None = None,
) -> RuntimeCaps:
    cuda = detect_cuda_caps(device_index=device_index)
    triton = detect_triton_caps()
    tensorrt = detect_tensorrt_caps(cuda=cuda, header_search_paths=header_search_paths)
    fp8 = detect_fp8_caps(cuda=cuda)
    return RuntimeCaps(cuda=cuda, triton=triton, tensorrt=tensorrt, fp8=fp8)


__all__ = [
    "CudaCaps",
    "TritonCaps",
    "TensorRTCaps",
    "FP8Caps",
    "RuntimeCaps",
    "detect_cuda_caps",
    "detect_triton_caps",
    "detect_tensorrt_caps",
    "detect_fp8_caps",
    "detect_runtime_caps",
    "runtime_reason_catalog",
    "CUDA_REASON_CUDA_UNAVAILABLE",
    "CUDA_REASON_DEVICE_NOT_FOUND",
    "CUDA_REASON_DEVICE_INDEX_OUT_OF_RANGE",
    "CUDA_REASON_QUERY_FAILED",
    "TRITON_REASON_NOT_INSTALLED",
    "TENSORRT_PYTHON_REASON_NOT_INSTALLED",
    "TENSORRT_PYTHON_REASON_IMPORT_FAILED",
    "TENSORRT_INT8_REASON_PYTHON_UNAVAILABLE",
    "TENSORRT_INT8_REASON_CUDA_REQUIRED",
    "TENSORRT_INT8_REASON_BUILDER_FLAG_MISSING",
    "FP8_REASON_TORCH_DTYPE_MISSING",
    "FP8_REASON_CUDA_REQUIRED",
    "FP8_REASON_COMPUTE_CAPABILITY_UNKNOWN",
    "FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90",
]
