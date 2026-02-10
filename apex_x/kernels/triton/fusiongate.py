from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

BackendKind = Literal["reference", "triton"]


_fusiongate_alpha_kernel: Any | None = None
_fusiongate_fuse_kernel: Any | None = None
triton: Any
tl: Any

try:
    triton = __import__("triton")
    tl = __import__("triton.language", fromlist=["language"])
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class TritonFusionGateAvailability:
    triton_installed: bool
    cuda_available: bool
    cuda_device_count: int
    reason: str | None

    @property
    def available(self) -> bool:
        return self.triton_installed and self.cuda_available and self.cuda_device_count > 0


@dataclass(frozen=True, slots=True)
class FusionGateDispatchResult:
    alpha: Tensor
    fused: Tensor | None
    backend: BackendKind
    fallback_reason: str | None


def get_triton_fusiongate_availability() -> TritonFusionGateAvailability:
    triton_installed = importlib.util.find_spec("triton") is not None
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    reason: str | None = None
    if not triton_installed:
        reason = "triton_not_installed"
    elif _TRITON_IMPORT_ERROR is not None:
        reason = f"triton_import_failed:{_TRITON_IMPORT_ERROR}"
    elif not cuda_available:
        reason = "cuda_unavailable"
    elif device_count <= 0:
        reason = "cuda_device_not_found"

    return TritonFusionGateAvailability(
        triton_installed=triton_installed,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        reason=reason,
    )


def _prepare_proxy(proxy: Tensor, *, name: str) -> Tensor:
    if proxy.ndim == 3:
        proxy = proxy.unsqueeze(1)
    if proxy.ndim != 4:
        raise ValueError(f"{name} must be [B,1,H,W] or [B,H,W]")
    if proxy.shape[1] != 1:
        raise ValueError(f"{name} channel dimension must be 1")
    return torch.nan_to_num(proxy, nan=0.0, posinf=1.0, neginf=0.0).contiguous()


def _validate_fusion_inputs(
    *,
    base_features: Tensor | None,
    detail_features: Tensor | None,
    alpha: Tensor,
) -> None:
    if (base_features is None) != (detail_features is None):
        raise ValueError("base_features and detail_features must be provided together")
    if base_features is None or detail_features is None:
        return
    if base_features.ndim != 4 or detail_features.ndim != 4:
        raise ValueError("base_features and detail_features must be [B,C,H,W]")
    if base_features.shape != detail_features.shape:
        raise ValueError("base_features and detail_features must have the same shape")
    if alpha.shape[0] != base_features.shape[0] or alpha.shape[2:] != base_features.shape[2:]:
        raise ValueError("alpha must match base/detail batch and spatial shape")


def _softplus_weight(value: float | Tensor, *, device: torch.device) -> float:
    weight = torch.nn.functional.softplus(
        torch.as_tensor(float(value), dtype=torch.float32, device=device)
    )
    return float(weight.item())


def fusiongate_alpha_reference(
    boundary_proxy: Tensor,
    uncertainty_proxy: Tensor,
    *,
    boundary_log_weight: float = 1.0,
    uncertainty_log_weight: float = 1.0,
    bias: float = 0.0,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    boundary = _prepare_proxy(boundary_proxy, name="boundary_proxy")
    uncertainty = _prepare_proxy(uncertainty_proxy, name="uncertainty_proxy")
    if boundary.shape != uncertainty.shape:
        raise ValueError("boundary_proxy and uncertainty_proxy must match in shape")

    dtype = out_dtype or boundary.dtype
    boundary = boundary.to(dtype=dtype)
    uncertainty = uncertainty.to(dtype=dtype, device=boundary.device)

    boundary_w = _softplus_weight(boundary_log_weight, device=boundary.device)
    uncertainty_w = _softplus_weight(uncertainty_log_weight, device=boundary.device)
    logits = boundary_w * boundary + uncertainty_w * uncertainty + float(bias)
    return torch.sigmoid(logits).contiguous()


def fusiongate_fuse_reference(
    *,
    base_features: Tensor,
    detail_features: Tensor,
    alpha: Tensor,
    inplace: bool = False,
) -> Tensor:
    _validate_fusion_inputs(
        base_features=base_features,
        detail_features=detail_features,
        alpha=alpha,
    )
    if inplace:
        out = base_features
        out.copy_(base_features + alpha * (detail_features - base_features))
        return out
    return (base_features + alpha * (detail_features - base_features)).contiguous()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fusiongate_alpha_kernel(
        boundary_ptr: Any,
        uncertainty_ptr: Any,
        alpha_ptr: Any,
        n_elements: Any,
        boundary_w: Any,
        uncertainty_w: Any,
        bias: Any,
        BLOCK_SIZE: Any,
    ) -> None:
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        boundary = tl.load(boundary_ptr + offs, mask=mask, other=0).to(tl.float32)
        uncertainty = tl.load(uncertainty_ptr + offs, mask=mask, other=0).to(tl.float32)

        logits = boundary_w * boundary + uncertainty_w * uncertainty + bias
        alpha = tl.sigmoid(logits)
        tl.store(alpha_ptr + offs, alpha, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fusiongate_fuse_kernel(
        base_ptr: Any,
        detail_ptr: Any,
        alpha_ptr: Any,
        out_ptr: Any,
        n_elements: Any,
        channels: Any,
        hw: Any,
        BLOCK_SIZE: Any,
    ) -> None:
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        base = tl.load(base_ptr + offs, mask=mask, other=0).to(tl.float32)
        detail = tl.load(detail_ptr + offs, mask=mask, other=0).to(tl.float32)

        bc = offs // hw
        b = bc // channels
        spatial = offs - bc * hw
        alpha_offs = b * hw + spatial
        alpha = tl.load(alpha_ptr + alpha_offs, mask=mask, other=0).to(tl.float32)

        out = base + alpha * (detail - base)
        tl.store(out_ptr + offs, out, mask=mask)
else:
    _fusiongate_alpha_kernel = None
    _fusiongate_fuse_kernel = None


def fusiongate_alpha_triton(
    boundary_proxy: Tensor,
    uncertainty_proxy: Tensor,
    *,
    boundary_log_weight: float = 1.0,
    uncertainty_log_weight: float = 1.0,
    bias: float = 0.0,
) -> Tensor:
    if triton is None or _fusiongate_alpha_kernel is None:
        raise RuntimeError("Triton is not available")

    boundary = _prepare_proxy(boundary_proxy, name="boundary_proxy")
    uncertainty = _prepare_proxy(uncertainty_proxy, name="uncertainty_proxy")
    if boundary.shape != uncertainty.shape:
        raise ValueError("boundary_proxy and uncertainty_proxy must match in shape")
    if boundary.device.type != "cuda" or uncertainty.device.type != "cuda":
        raise ValueError("fusiongate_alpha_triton requires CUDA tensors")
    if boundary.dtype != uncertainty.dtype:
        raise ValueError("boundary_proxy and uncertainty_proxy must have same dtype")
    if boundary.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("fusiongate_alpha_triton supports fp16/bf16 only")

    boundary = boundary.contiguous()
    uncertainty = uncertainty.to(device=boundary.device, dtype=boundary.dtype).contiguous()
    alpha = torch.empty_like(boundary)

    boundary_w = _softplus_weight(boundary_log_weight, device=boundary.device)
    uncertainty_w = _softplus_weight(uncertainty_log_weight, device=boundary.device)
    n_elements = int(boundary.numel())

    grid = (triton.cdiv(n_elements, 1024),)
    _fusiongate_alpha_kernel[grid](
        boundary,
        uncertainty,
        alpha,
        n_elements,
        boundary_w,
        uncertainty_w,
        float(bias),
    )
    return alpha.contiguous()


def fusiongate_fuse_triton(
    *,
    base_features: Tensor,
    detail_features: Tensor,
    alpha: Tensor,
    inplace: bool = False,
) -> Tensor:
    if triton is None or _fusiongate_fuse_kernel is None:
        raise RuntimeError("Triton is not available")
    _validate_fusion_inputs(
        base_features=base_features,
        detail_features=detail_features,
        alpha=alpha,
    )
    if base_features.device.type != "cuda" or detail_features.device.type != "cuda":
        raise ValueError("fusiongate_fuse_triton requires CUDA base/detail tensors")
    if alpha.device.type != "cuda":
        raise ValueError("fusiongate_fuse_triton requires CUDA alpha tensor")
    if base_features.dtype != detail_features.dtype:
        raise ValueError("base_features and detail_features must have same dtype")
    if alpha.dtype != base_features.dtype:
        raise ValueError("alpha dtype must match base/detail dtype")
    if base_features.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("fusiongate_fuse_triton supports fp16/bf16 only")

    base = base_features.contiguous()
    detail = detail_features.contiguous()
    alpha_c = alpha.contiguous()

    out = base if inplace else torch.empty_like(base)
    n_elements = int(base.numel())
    channels = int(base.shape[1])
    hw = int(base.shape[2] * base.shape[3])

    grid = (triton.cdiv(n_elements, 1024),)
    _fusiongate_fuse_kernel[grid](
        base,
        detail,
        alpha_c,
        out,
        n_elements,
        channels,
        hw,
    )
    return out.contiguous()


def fusiongate_dispatch(
    boundary_proxy: Tensor,
    uncertainty_proxy: Tensor,
    *,
    base_features: Tensor | None = None,
    detail_features: Tensor | None = None,
    boundary_log_weight: float = 1.0,
    uncertainty_log_weight: float = 1.0,
    bias: float = 0.0,
    apply_fusion: bool = False,
    inplace_fusion: bool = False,
    prefer_triton: bool = True,
    allow_fallback: bool = True,
    inference_only: bool = True,
) -> FusionGateDispatchResult:
    if apply_fusion and (base_features is None or detail_features is None):
        raise ValueError(
            "base_features and detail_features are required when apply_fusion=True"
        )
    requires_grad = boundary_proxy.requires_grad or uncertainty_proxy.requires_grad
    if base_features is not None:
        requires_grad = requires_grad or base_features.requires_grad
    if detail_features is not None:
        requires_grad = requires_grad or detail_features.requires_grad

    if inference_only and requires_grad:
        alpha = fusiongate_alpha_reference(
            boundary_proxy,
            uncertainty_proxy,
            boundary_log_weight=boundary_log_weight,
            uncertainty_log_weight=uncertainty_log_weight,
            bias=bias,
            out_dtype=(base_features.dtype if base_features is not None else None),
        )
        fused: Tensor | None = None
        if apply_fusion and base_features is not None and detail_features is not None:
            fused = fusiongate_fuse_reference(
                base_features=base_features,
                detail_features=detail_features,
                alpha=alpha,
                inplace=inplace_fusion,
            )
        return FusionGateDispatchResult(
            alpha=alpha,
            fused=fused,
            backend="reference",
            fallback_reason="autograd_not_supported_for_triton_fusiongate",
        )

    availability = get_triton_fusiongate_availability()
    if prefer_triton and availability.available:
        try:
            alpha = fusiongate_alpha_triton(
                boundary_proxy,
                uncertainty_proxy,
                boundary_log_weight=boundary_log_weight,
                uncertainty_log_weight=uncertainty_log_weight,
                bias=bias,
            )
            fused = None
            if apply_fusion and base_features is not None and detail_features is not None:
                fused = fusiongate_fuse_triton(
                    base_features=base_features,
                    detail_features=detail_features,
                    alpha=alpha,
                    inplace=inplace_fusion,
                )
            return FusionGateDispatchResult(
                alpha=alpha,
                fused=fused,
                backend="triton",
                fallback_reason=None,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            alpha = fusiongate_alpha_reference(
                boundary_proxy,
                uncertainty_proxy,
                boundary_log_weight=boundary_log_weight,
                uncertainty_log_weight=uncertainty_log_weight,
                bias=bias,
                out_dtype=(base_features.dtype if base_features is not None else None),
            )
            fused = None
            if apply_fusion and base_features is not None and detail_features is not None:
                fused = fusiongate_fuse_reference(
                    base_features=base_features,
                    detail_features=detail_features,
                    alpha=alpha,
                    inplace=inplace_fusion,
                )
            return FusionGateDispatchResult(
                alpha=alpha,
                fused=fused,
                backend="reference",
                fallback_reason=f"triton_error:{type(exc).__name__}",
            )

    alpha = fusiongate_alpha_reference(
        boundary_proxy,
        uncertainty_proxy,
        boundary_log_weight=boundary_log_weight,
        uncertainty_log_weight=uncertainty_log_weight,
        bias=bias,
        out_dtype=(base_features.dtype if base_features is not None else None),
    )
    fused = None
    if apply_fusion and base_features is not None and detail_features is not None:
        fused = fusiongate_fuse_reference(
            base_features=base_features,
            detail_features=detail_features,
            alpha=alpha,
            inplace=inplace_fusion,
        )
    fallback_reason = availability.reason if prefer_triton else None
    return FusionGateDispatchResult(
        alpha=alpha,
        fused=fused,
        backend="reference",
        fallback_reason=fallback_reason,
    )


__all__ = [
    "BackendKind",
    "TritonFusionGateAvailability",
    "FusionGateDispatchResult",
    "get_triton_fusiongate_availability",
    "fusiongate_alpha_reference",
    "fusiongate_fuse_reference",
    "fusiongate_alpha_triton",
    "fusiongate_fuse_triton",
    "fusiongate_dispatch",
]
