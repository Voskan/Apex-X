from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from apex_x.tiles import OverlapMode, TilePackTorch, TileUnpackTorch

BackendKind = Literal["reference", "triton"]


@dataclass(frozen=True, slots=True)
class TritonAvailability:
    triton_installed: bool
    cuda_available: bool
    cuda_device_count: int
    reason: str | None

    @property
    def available(self) -> bool:
        return self.triton_installed and self.cuda_available and self.cuda_device_count > 0


@dataclass(frozen=True, slots=True)
class FusedTileScatterResult:
    merged: Tensor
    priority_map: Tensor
    alpha_map: Tensor
    backend: BackendKind
    meta: dict[str, Tensor]
    fallback_reason: str | None


def get_triton_availability() -> TritonAvailability:
    triton_installed = importlib.util.find_spec("triton") is not None

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    reason: str | None = None
    if not triton_installed:
        reason = "triton_not_installed"
    elif not cuda_available:
        reason = "cuda_unavailable"
    elif cuda_device_count <= 0:
        reason = "cuda_device_not_found"
    return TritonAvailability(
        triton_installed=triton_installed,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        reason=reason,
    )


def _prepare_proxy(proxy: Tensor, *, name: str, like: Tensor) -> Tensor:
    if proxy.ndim == 3:
        proxy = proxy.unsqueeze(1)
    if proxy.ndim != 4:
        raise ValueError(f"{name} must be [B,1,H,W] or [B,H,W]")
    if proxy.shape[1] != 1:
        raise ValueError(f"{name} channel dimension must be 1")
    if proxy.shape[0] != like.shape[0] or proxy.shape[2:] != like.shape[2:]:
        raise ValueError(f"{name} must match base/heavy batch and spatial shape")

    return torch.nan_to_num(
        proxy.to(dtype=like.dtype, device=like.device),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def _compute_alpha_map(
    *,
    boundary_proxy: Tensor,
    uncertainty_proxy: Tensor,
    like: Tensor,
    boundary_weight: float,
    uncertainty_weight: float,
    bias: float,
) -> Tensor:
    boundary = _prepare_proxy(boundary_proxy, name="boundary_proxy", like=like)
    uncertainty = _prepare_proxy(uncertainty_proxy, name="uncertainty_proxy", like=like)
    boundary_w = torch.nn.functional.softplus(
        torch.tensor(float(boundary_weight), dtype=like.dtype, device=like.device)
    )
    uncertainty_w = torch.nn.functional.softplus(
        torch.tensor(float(uncertainty_weight), dtype=like.dtype, device=like.device)
    )
    logits = boundary_w * boundary + uncertainty_w * uncertainty + float(bias)
    return torch.sigmoid(logits)


def _gather_from_origins(feature_map: Tensor, origins: Tensor, tile_size: int) -> Tensor:
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [B,C,H,W]")
    if origins.ndim != 3 or origins.shape[2] != 2:
        raise ValueError("origins must be [B,K,2]")
    bsz, channels, height, width = feature_map.shape
    if origins.shape[0] != bsz:
        raise ValueError("origins batch must match feature_map batch")

    per_batch: list[Tensor] = []
    for b in range(bsz):
        tiles: list[Tensor] = []
        for k in range(origins.shape[1]):
            y = int(origins[b, k, 0].item())
            x = int(origins[b, k, 1].item())
            if y < 0 or x < 0 or y + tile_size > height or x + tile_size > width:
                raise ValueError("origins contain out-of-bounds tile coordinates")
            tiles.append(feature_map[b, :, y : y + tile_size, x : x + tile_size])
        if tiles:
            per_batch.append(torch.stack(tiles, dim=0))
        else:
            per_batch.append(feature_map.new_empty((0, channels, tile_size, tile_size)))
    return torch.stack(per_batch, dim=0)


def gather_gate_scatter_reference(
    *,
    base_map: Tensor,
    heavy_map: Tensor,
    indices: Tensor,
    tile_size: int,
    boundary_proxy: Tensor,
    uncertainty_proxy: Tensor,
    level_priority: int = 1,
    priority_map: Tensor | None = None,
    overlap_mode: OverlapMode = "override",
    blend_alpha: float = 0.5,
    order_mode: str = "hilbert",
    boundary_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    gate_bias: float = 0.0,
) -> FusedTileScatterResult:
    if base_map.shape != heavy_map.shape:
        raise ValueError("base_map and heavy_map must have same shape")

    alpha_map = _compute_alpha_map(
        boundary_proxy=boundary_proxy,
        uncertainty_proxy=uncertainty_proxy,
        like=base_map,
        boundary_weight=boundary_weight,
        uncertainty_weight=uncertainty_weight,
        bias=gate_bias,
    )

    packer = TilePackTorch()
    unpacker = TileUnpackTorch()
    heavy_packed, meta = packer.pack(
        feature_map=heavy_map,
        indices=indices,
        tile_size=tile_size,
        order_mode=order_mode,
    )
    origins = meta["origins"]
    base_packed = _gather_from_origins(base_map, origins, tile_size)
    alpha_packed = _gather_from_origins(alpha_map, origins, tile_size)
    fused_packed = base_packed + alpha_packed * (heavy_packed - base_packed)

    merged, next_priority = unpacker.unpack(
        base_map=base_map,
        packed_out=fused_packed,
        meta=meta,
        level_priority=level_priority,
        priority_map=priority_map,
        overlap_mode=overlap_mode,
        blend_alpha=blend_alpha,
    )
    return FusedTileScatterResult(
        merged=merged,
        priority_map=next_priority,
        alpha_map=alpha_map,
        backend="reference",
        meta=meta,
        fallback_reason=None,
    )


def _triton_fused_kernel_stub(**_: object) -> FusedTileScatterResult:
    raise NotImplementedError(
        "Triton fused gather+gate+scatter kernel is not available in this environment."
    )


def gather_gate_scatter(
    *,
    base_map: Tensor,
    heavy_map: Tensor,
    indices: Tensor,
    tile_size: int,
    boundary_proxy: Tensor,
    uncertainty_proxy: Tensor,
    level_priority: int = 1,
    priority_map: Tensor | None = None,
    overlap_mode: OverlapMode = "override",
    blend_alpha: float = 0.5,
    order_mode: str = "hilbert",
    boundary_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    gate_bias: float = 0.0,
    prefer_triton: bool = True,
    allow_fallback: bool = True,
) -> FusedTileScatterResult:
    availability = get_triton_availability()
    if prefer_triton and availability.available:
        try:
            return _triton_fused_kernel_stub(
                base_map=base_map,
                heavy_map=heavy_map,
                indices=indices,
                tile_size=tile_size,
                boundary_proxy=boundary_proxy,
                uncertainty_proxy=uncertainty_proxy,
                level_priority=level_priority,
                priority_map=priority_map,
                overlap_mode=overlap_mode,
                blend_alpha=blend_alpha,
                order_mode=order_mode,
                boundary_weight=boundary_weight,
                uncertainty_weight=uncertainty_weight,
                gate_bias=gate_bias,
            )
        except NotImplementedError as exc:
            if not allow_fallback:
                raise
            ref = gather_gate_scatter_reference(
                base_map=base_map,
                heavy_map=heavy_map,
                indices=indices,
                tile_size=tile_size,
                boundary_proxy=boundary_proxy,
                uncertainty_proxy=uncertainty_proxy,
                level_priority=level_priority,
                priority_map=priority_map,
                overlap_mode=overlap_mode,
                blend_alpha=blend_alpha,
                order_mode=order_mode,
                boundary_weight=boundary_weight,
                uncertainty_weight=uncertainty_weight,
                gate_bias=gate_bias,
            )
            return FusedTileScatterResult(
                merged=ref.merged,
                priority_map=ref.priority_map,
                alpha_map=ref.alpha_map,
                backend="reference",
                meta=ref.meta,
                fallback_reason=str(exc),
            )

    ref = gather_gate_scatter_reference(
        base_map=base_map,
        heavy_map=heavy_map,
        indices=indices,
        tile_size=tile_size,
        boundary_proxy=boundary_proxy,
        uncertainty_proxy=uncertainty_proxy,
        level_priority=level_priority,
        priority_map=priority_map,
        overlap_mode=overlap_mode,
        blend_alpha=blend_alpha,
        order_mode=order_mode,
        boundary_weight=boundary_weight,
        uncertainty_weight=uncertainty_weight,
        gate_bias=gate_bias,
    )
    fallback_reason = None
    if prefer_triton:
        fallback_reason = availability.reason or "triton_path_not_selected"
    return FusedTileScatterResult(
        merged=ref.merged,
        priority_map=ref.priority_map,
        alpha_map=ref.alpha_map,
        backend="reference",
        meta=ref.meta,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "BackendKind",
    "TritonAvailability",
    "FusedTileScatterResult",
    "get_triton_availability",
    "gather_gate_scatter_reference",
    "gather_gate_scatter",
]
