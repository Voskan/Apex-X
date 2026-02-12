from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

from apex_x.config import ApexXConfig

from .caps import (
    FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90,
    FP8_REASON_COMPUTE_CAPABILITY_UNKNOWN,
    FP8_REASON_CUDA_REQUIRED,
    FP8_REASON_TORCH_DTYPE_MISSING,
)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _resolve_device(
    device: str | torch.device | None,
    *,
    preferred_backend: str | None = None,
) -> torch.device:
    if device is None:
        if preferred_backend == "cpu":
            return torch.device("cpu")
        if preferred_backend in {"torch", "triton", "tensorrt"} and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if preferred_backend is None and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(device)


def _preferred_fp8_dtype() -> torch.dtype | None:
    candidate = getattr(torch, "float8_e4m3fn", None)
    if isinstance(candidate, torch.dtype):
        return candidate
    return None


def _detect_cuda_fp8_support(device: torch.device) -> tuple[bool, str | None]:
    if device.type != "cuda":
        return False, FP8_REASON_CUDA_REQUIRED
    if not torch.cuda.is_available():
        return False, FP8_REASON_CUDA_REQUIRED
    if _preferred_fp8_dtype() is None:
        return False, FP8_REASON_TORCH_DTYPE_MISSING

    idx = 0 if device.index is None else int(device.index)
    try:
        major, _minor = torch.cuda.get_device_capability(idx)
    except Exception:  # pragma: no cover - defensive fallback
        return False, FP8_REASON_COMPUTE_CAPABILITY_UNKNOWN
    # Conservative gate: native FP8 tensor-core support starts with Hopper (sm90+).
    if major < 9:
        return False, FP8_REASON_COMPUTE_CAPABILITY_BELOW_SM90
    return True, None


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    profile: str
    device: str
    heavy_ops_dtype: torch.dtype
    fp8_requested: bool
    fp8_enabled: bool
    fallback_reason: str | None
    router_dtype: torch.dtype
    kan_dtype: torch.dtype

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "profile": self.profile,
            "device": self.device,
            "heavy_ops_dtype": dtype_name(self.heavy_ops_dtype),
            "fp8_requested": self.fp8_requested,
            "fp8_enabled": self.fp8_enabled,
            "fallback_reason": self.fallback_reason,
            "router_dtype": dtype_name(self.router_dtype),
            "kan_dtype": dtype_name(self.kan_dtype),
        }


def resolve_precision_policy(
    config: ApexXConfig,
    *,
    device: str | torch.device | None = None,
) -> PrecisionPolicy:
    resolved_device = _resolve_device(device, preferred_backend=config.runtime.backend)
    profile = config.runtime.precision_profile
    fp8_requested = profile == "balanced" or bool(config.train.qat_fp8)

    heavy_ops_dtype = torch.float16
    fp8_enabled = False
    fallback_reason: str | None = None

    if fp8_requested:
        fp8_ok, reason = _detect_cuda_fp8_support(resolved_device)
        fp8_dtype = _preferred_fp8_dtype()
        if fp8_ok and fp8_dtype is not None:
            heavy_ops_dtype = fp8_dtype
            fp8_enabled = True
        else:
            heavy_ops_dtype = torch.float16
            fp8_enabled = False
            fallback_reason = reason or "fp8_not_available"

    return PrecisionPolicy(
        profile=profile,
        device=str(resolved_device),
        heavy_ops_dtype=heavy_ops_dtype,
        fp8_requested=fp8_requested,
        fp8_enabled=fp8_enabled,
        fallback_reason=fallback_reason,
        router_dtype=torch.float16,
        kan_dtype=torch.float16,
    )


def heavy_ops_autocast_context(policy: PrecisionPolicy) -> AbstractContextManager[object]:
    if policy.heavy_ops_dtype is torch.float16:
        if policy.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        # CPU float16 autocast is unstable and can trigger dtype/inplace issues
        # in training paths that are expected to be CPU-safe.
        if policy.device == "cpu":
            return nullcontext()
    # FP8 is expected to be consumed by specialized kernels/plugins.
    return nullcontext()


__all__ = [
    "PrecisionPolicy",
    "dtype_name",
    "resolve_precision_policy",
    "heavy_ops_autocast_context",
]
