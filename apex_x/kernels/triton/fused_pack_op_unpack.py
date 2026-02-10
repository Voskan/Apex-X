from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .tilepack import TorchTileMeta, tilepack_reference
from .tileunpack import tileunpack_reference

BackendKind = Literal["reference", "triton"]


_fused_pack_op_unpack_kernel: Any | None = None
triton: Any
tl: Any

try:
    triton = __import__("triton")
    tl = __import__("triton.language", fromlist=["language"])
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised by CPU-only environments
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class TritonFusedStage1Availability:
    triton_installed: bool
    cuda_available: bool
    cuda_device_count: int
    reason: str | None

    @property
    def available(self) -> bool:
        return self.triton_installed and self.cuda_available and self.cuda_device_count > 0


@dataclass(frozen=True, slots=True)
class FusedPackOpUnpackDispatchResult:
    merged: Tensor
    meta: TorchTileMeta
    backend: BackendKind
    fallback_reason: str | None


def get_triton_fused_stage1_availability() -> TritonFusedStage1Availability:
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

    return TritonFusedStage1Availability(
        triton_installed=triton_installed,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        reason=reason,
    )


def _validate_inputs(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
    *,
    require_unique_indices: bool,
) -> tuple[int, int, int, int]:
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [B,C,H,W]")
    if indices.ndim != 2:
        raise ValueError("indices must be [B,K]")
    if feature_map.shape[0] != indices.shape[0]:
        raise ValueError("feature_map and indices batch dimensions must match")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if feature_map.shape[2] % tile_size != 0 or feature_map.shape[3] % tile_size != 0:
        raise ValueError("H and W must be divisible by tile_size")
    if feature_map.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("feature_map dtype must be one of float16, bfloat16, float32")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("indices must be int32 or int64")

    batch, channels, height, width = feature_map.shape
    grid_h = height // tile_size
    grid_w = width // tile_size
    max_index = grid_h * grid_w - 1
    idx_i64 = indices.to(dtype=torch.int64)
    if idx_i64.numel() > 0:
        idx_min = int(idx_i64.min().item())
        idx_max = int(idx_i64.max().item())
        if idx_min < 0 or idx_max > max_index:
            raise ValueError("indices contain out-of-bounds tile ids")

    if require_unique_indices and idx_i64.numel() > 0:
        for b in range(batch):
            row = idx_i64[b]
            if torch.unique(row).numel() != row.numel():
                raise ValueError(
                    "indices must be unique per batch for deterministic fused overwrite semantics"
                )
    return batch, channels, height, width


def _build_meta(indices: Tensor, tile_size: int, height: int, width: int) -> TorchTileMeta:
    grid_h = height // tile_size
    grid_w = width // tile_size
    idx_i64 = indices.to(dtype=torch.int64)
    origins_y = (idx_i64 // grid_w) * tile_size
    origins_x = (idx_i64 % grid_w) * tile_size
    origins = torch.stack((origins_y, origins_x), dim=-1)
    return {
        "indices": idx_i64,
        "origins": origins,
        "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=idx_i64.device),
        "grid": torch.tensor([grid_h, grid_w], dtype=torch.int64, device=idx_i64.device),
    }


def apply_pointwise_affine_reglu(
    tensor: Tensor,
    *,
    value_scale: float,
    value_bias: float,
    gate_scale: float,
    gate_bias: float,
) -> Tensor:
    value = tensor * float(value_scale) + float(value_bias)
    gate = tensor * float(gate_scale) + float(gate_bias)
    return value * torch.relu(gate)


def separate_pack_op_unpack_reference(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
    *,
    value_scale: float = 1.0,
    value_bias: float = 0.0,
    gate_scale: float = 1.0,
    gate_bias: float = 0.0,
    require_unique_indices: bool = True,
) -> tuple[Tensor, TorchTileMeta]:
    _, _, height, width = _validate_inputs(
        feature_map,
        indices,
        tile_size,
        require_unique_indices=require_unique_indices,
    )
    packed, meta = tilepack_reference(feature_map, indices, tile_size)
    packed_out = apply_pointwise_affine_reglu(
        packed,
        value_scale=value_scale,
        value_bias=value_bias,
        gate_scale=gate_scale,
        gate_bias=gate_bias,
    )
    merged = tileunpack_reference(base_map=feature_map, packed_out=packed_out, meta=meta)
    normalized_meta = _build_meta(meta["indices"], tile_size, height, width)
    return merged.contiguous(), normalized_meta


def fused_pack_op_unpack_reference(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
    *,
    value_scale: float = 1.0,
    value_bias: float = 0.0,
    gate_scale: float = 1.0,
    gate_bias: float = 0.0,
    require_unique_indices: bool = True,
) -> tuple[Tensor, TorchTileMeta]:
    return separate_pack_op_unpack_reference(
        feature_map,
        indices,
        tile_size,
        value_scale=value_scale,
        value_bias=value_bias,
        gate_scale=gate_scale,
        gate_bias=gate_bias,
        require_unique_indices=require_unique_indices,
    )


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_PIX": 64}, num_warps=4),
            triton.Config({"BLOCK_PIX": 128}, num_warps=4),
            triton.Config({"BLOCK_PIX": 256}, num_warps=8),
        ],
        key=["tile_pixels"],
    )
    @triton.jit
    def _fused_pack_op_unpack_kernel(
        in_ptr: Any,  # [B,C,H,W] contiguous
        idx_ptr: Any,  # [B,K] int32
        out_ptr: Any,  # [B,C,H,W] contiguous
        value_scale: Any,
        value_bias: Any,
        gate_scale: Any,
        gate_bias: Any,
        batch: Any,
        channels: Any,
        height: Any,
        width: Any,
        kmax: Any,
        tile_size: Any,
        grid_w: Any,
        tile_pixels: Any,
        BLOCK_PIX: Any,
    ) -> None:
        pid_bk = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_blk = tl.program_id(2)

        b = pid_bk // kmax
        k = pid_bk - b * kmax
        c = pid_c
        if (b >= batch) or (c >= channels):
            return

        tile_idx = tl.load(idx_ptr + b * kmax + k).to(tl.int32)
        base_y = (tile_idx // grid_w) * tile_size
        base_x = (tile_idx % grid_w) * tile_size

        offs = pid_blk * BLOCK_PIX + tl.arange(0, BLOCK_PIX).to(tl.int32)
        valid = offs < tile_pixels
        local_y = offs // tile_size
        local_x = offs - local_y * tile_size
        y = base_y + local_y
        x = base_x + local_x

        valid = valid & (y >= 0) & (x >= 0) & (y < height) & (x < width)
        offsets = ((b * channels + c) * height + y) * width + x
        values = tl.load(in_ptr + offsets, mask=valid, other=0)
        value = values * value_scale + value_bias
        gate = values * gate_scale + gate_bias
        transformed = value * tl.maximum(gate, 0)
        tl.store(out_ptr + offsets, transformed, mask=valid)

else:
    _fused_pack_op_unpack_kernel = None


def fused_pack_op_unpack_triton(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
    *,
    value_scale: float = 1.0,
    value_bias: float = 0.0,
    gate_scale: float = 1.0,
    gate_bias: float = 0.0,
    require_unique_indices: bool = True,
) -> tuple[Tensor, TorchTileMeta]:
    if triton is None or _fused_pack_op_unpack_kernel is None:
        raise RuntimeError("Triton is not available")

    batch, channels, height, width = _validate_inputs(
        feature_map,
        indices,
        tile_size,
        require_unique_indices=require_unique_indices,
    )
    if feature_map.device.type != "cuda":
        raise ValueError("fused_pack_op_unpack_triton requires CUDA tensor input")
    if feature_map.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("fused_pack_op_unpack_triton supports fp16/bf16 inputs only")

    feature_contig = feature_map.contiguous()
    idx_i32 = indices.to(dtype=torch.int32, device=feature_contig.device).contiguous()
    out = feature_contig.clone()
    kmax = int(idx_i32.shape[1])
    grid_w = width // tile_size
    tile_pixels = tile_size * tile_size
    pix_blocks = triton.cdiv(tile_pixels, 256)

    grid = (batch * kmax, channels, pix_blocks)
    _fused_pack_op_unpack_kernel[grid](
        feature_contig,
        idx_i32,
        out,
        float(value_scale),
        float(value_bias),
        float(gate_scale),
        float(gate_bias),
        batch,
        channels,
        height,
        width,
        kmax,
        tile_size,
        grid_w,
        tile_pixels,
    )
    out = out.contiguous()
    meta = _build_meta(idx_i32, tile_size, height, width)
    return out, meta


def fused_pack_op_unpack_dispatch(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
    *,
    value_scale: float = 1.0,
    value_bias: float = 0.0,
    gate_scale: float = 1.0,
    gate_bias: float = 0.0,
    require_unique_indices: bool = True,
    prefer_triton: bool = True,
    allow_fallback: bool = True,
    inference_only: bool = True,
) -> FusedPackOpUnpackDispatchResult:
    if inference_only and feature_map.requires_grad:
        merged, meta = fused_pack_op_unpack_reference(
            feature_map,
            indices,
            tile_size,
            value_scale=value_scale,
            value_bias=value_bias,
            gate_scale=gate_scale,
            gate_bias=gate_bias,
            require_unique_indices=require_unique_indices,
        )
        return FusedPackOpUnpackDispatchResult(
            merged=merged,
            meta=meta,
            backend="reference",
            fallback_reason="autograd_not_supported_for_triton_fused_stage1",
        )

    availability = get_triton_fused_stage1_availability()
    if prefer_triton and availability.available:
        try:
            merged, meta = fused_pack_op_unpack_triton(
                feature_map,
                indices,
                tile_size,
                value_scale=value_scale,
                value_bias=value_bias,
                gate_scale=gate_scale,
                gate_bias=gate_bias,
                require_unique_indices=require_unique_indices,
            )
            return FusedPackOpUnpackDispatchResult(
                merged=merged,
                meta=meta,
                backend="triton",
                fallback_reason=None,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            merged, meta = fused_pack_op_unpack_reference(
                feature_map,
                indices,
                tile_size,
                value_scale=value_scale,
                value_bias=value_bias,
                gate_scale=gate_scale,
                gate_bias=gate_bias,
                require_unique_indices=require_unique_indices,
            )
            return FusedPackOpUnpackDispatchResult(
                merged=merged,
                meta=meta,
                backend="reference",
                fallback_reason=f"triton_error:{type(exc).__name__}",
            )

    merged, meta = fused_pack_op_unpack_reference(
        feature_map,
        indices,
        tile_size,
        value_scale=value_scale,
        value_bias=value_bias,
        gate_scale=gate_scale,
        gate_bias=gate_bias,
        require_unique_indices=require_unique_indices,
    )
    fallback_reason = None
    if prefer_triton:
        fallback_reason = availability.reason or "triton_path_not_selected"
    return FusedPackOpUnpackDispatchResult(
        merged=merged,
        meta=meta,
        backend="reference",
        fallback_reason=fallback_reason,
    )


__all__ = [
    "BackendKind",
    "TritonFusedStage1Availability",
    "FusedPackOpUnpackDispatchResult",
    "get_triton_fused_stage1_availability",
    "apply_pointwise_affine_reglu",
    "separate_pack_op_unpack_reference",
    "fused_pack_op_unpack_reference",
    "fused_pack_op_unpack_triton",
    "fused_pack_op_unpack_dispatch",
]
