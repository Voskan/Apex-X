from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .autotune_registry import (
    build_shape_bucket,
    get_cached_triton_config,
    record_triton_autotune_selection,
    resolve_triton_launch_config,
)

BackendKind = Literal["reference", "triton"]
TorchTileMeta = dict[str, Tensor]


_tilepack_kernel: Any | None = None
triton: Any
tl: Any

try:
    triton = __import__("triton")
    tl = __import__("triton.language", fromlist=["language"])
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised by CPU-only test environments
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class TritonTilePackAvailability:
    triton_installed: bool
    cuda_available: bool
    cuda_device_count: int
    reason: str | None

    @property
    def available(self) -> bool:
        return self.triton_installed and self.cuda_available and self.cuda_device_count > 0


@dataclass(frozen=True, slots=True)
class TilePackDispatchResult:
    packed: Tensor
    meta: TorchTileMeta
    backend: BackendKind
    fallback_reason: str | None


def get_triton_tilepack_availability() -> TritonTilePackAvailability:
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

    return TritonTilePackAvailability(
        triton_installed=triton_installed,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        reason=reason,
    )


def _validate_inputs(
    feature_map: Tensor, indices: Tensor, tile_size: int
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


def _tilepack_block_heuristic(tile_pixels: int) -> int:
    if tile_pixels <= 64:
        return 64
    if tile_pixels <= 128:
        return 128
    return 256


def tilepack_reference(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
) -> tuple[Tensor, TorchTileMeta]:
    batch, channels, height, width = _validate_inputs(feature_map, indices, tile_size)
    feature_contig = feature_map.contiguous()
    idx_i64 = indices.to(dtype=torch.int64, device=feature_contig.device).contiguous()
    kmax = int(idx_i64.shape[1])

    grid_w = width // tile_size
    tile_area = tile_size * tile_size

    tile_y = idx_i64 // grid_w
    tile_x = idx_i64 % grid_w

    dy = torch.arange(tile_size, device=feature_contig.device, dtype=torch.int64).view(1, 1, -1, 1)
    dx = torch.arange(tile_size, device=feature_contig.device, dtype=torch.int64).view(1, 1, 1, -1)
    gather_y = tile_y.view(batch, kmax, 1, 1) * tile_size + dy
    gather_x = tile_x.view(batch, kmax, 1, 1) * tile_size + dx
    linear = (gather_y * width + gather_x).reshape(batch, kmax, tile_area)

    flat = feature_contig.reshape(batch, channels, height * width).unsqueeze(1)
    gather_idx = linear.unsqueeze(2).expand(batch, kmax, channels, tile_area)
    gathered = torch.gather(
        flat.expand(batch, kmax, channels, height * width),
        dim=3,
        index=gather_idx,
    )

    packed = gathered.reshape(batch, kmax, channels, tile_size, tile_size).contiguous()
    meta = _build_meta(idx_i64, tile_size, height, width)
    return packed, meta


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
    def _tilepack_kernel(
        f_ptr,  # [B,C,H,W] contiguous
        idx_ptr,  # [B,K] int32
        out_ptr,  # [B,K,C,t,t] contiguous
        batch,
        channels,
        height,
        width,
        kmax,
        tile_size,
        grid_w,
        tile_pixels,
        BLOCK_PIX: tl.constexpr,
    ) -> None:
        pid_bk = tl.program_id(0)
        pid_c = tl.program_id(1)

        b = pid_bk // kmax
        k = pid_bk - b * kmax
        c = pid_c

        if (b >= batch) or (c >= channels):
            return

        tile_idx = tl.load(idx_ptr + b * kmax + k).to(tl.int32)
        base_y = (tile_idx // grid_w) * tile_size
        base_x = (tile_idx % grid_w) * tile_size

        offs = tl.arange(0, BLOCK_PIX).to(tl.int32)
        valid = offs < tile_pixels

        local_y = offs // tile_size
        local_x = offs - local_y * tile_size
        y = base_y + local_y
        x = base_x + local_x

        in_offsets = ((b * channels + c) * height + y) * width + x
        values = tl.load(f_ptr + in_offsets, mask=valid, other=0)

        out_offsets = (((b * kmax + k) * channels + c) * tile_size + local_y) * tile_size + local_x
        tl.store(out_ptr + out_offsets, values, mask=valid)

else:
    _tilepack_kernel = None


def tilepack_triton(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
) -> tuple[Tensor, TorchTileMeta]:
    if triton is None or _tilepack_kernel is None:
        raise RuntimeError("Triton is not available")

    batch, channels, height, width = _validate_inputs(feature_map, indices, tile_size)
    if feature_map.device.type != "cuda":
        raise ValueError("tilepack_triton requires CUDA tensor input")
    if feature_map.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("tilepack_triton supports fp16/bf16 inputs only")

    feature_contig = feature_map.contiguous()
    idx_i32 = indices.to(dtype=torch.int32, device=feature_contig.device).contiguous()
    kmax = int(idx_i32.shape[1])
    grid_w = width // tile_size
    tile_pixels = tile_size * tile_size
    shape_bucket = build_shape_bucket(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        kmax=kmax,
        tile_size=tile_size,
        dtype=str(feature_contig.dtype).replace("torch.", ""),
    )

    selected_config = get_cached_triton_config(op_name="tilepack", shape_bucket=shape_bucket)
    selection_source = "registry_cache"
    if selected_config is None:
        selected_config, selection_source = resolve_triton_launch_config(
            kernel=_tilepack_kernel,
            fallback_config={"BLOCK_PIX": _tilepack_block_heuristic(tile_pixels)},
        )

    out = torch.empty(
        (batch, kmax, channels, tile_size, tile_size),
        dtype=feature_contig.dtype,
        device=feature_contig.device,
    )

    grid = (batch * kmax, channels)
    _tilepack_kernel[grid](
        feature_contig,
        idx_i32,
        out,
        batch,
        channels,
        height,
        width,
        kmax,
        tile_size,
        grid_w,
        tile_pixels,
    )

    record_triton_autotune_selection(
        op_name="tilepack",
        kernel_name="_tilepack_kernel",
        shape_bucket=shape_bucket,
        selected_config=selected_config,
        selection_source=selection_source,
    )
    out = out.contiguous()
    meta = _build_meta(idx_i32, tile_size, height, width)
    return out, meta


def tilepack_dispatch(
    feature_map: Tensor,
    indices: Tensor,
    tile_size: int,
    *,
    prefer_triton: bool = True,
    allow_fallback: bool = True,
    inference_only: bool = True,
) -> TilePackDispatchResult:
    if inference_only and feature_map.requires_grad:
        packed, meta = tilepack_reference(feature_map, indices, tile_size)
        return TilePackDispatchResult(
            packed=packed,
            meta=meta,
            backend="reference",
            fallback_reason="autograd_not_supported_for_triton_tilepack",
        )

    availability = get_triton_tilepack_availability()
    if prefer_triton and availability.available:
        try:
            packed, meta = tilepack_triton(feature_map, indices, tile_size)
            return TilePackDispatchResult(
                packed=packed,
                meta=meta,
                backend="triton",
                fallback_reason=None,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            packed, meta = tilepack_reference(feature_map, indices, tile_size)
            return TilePackDispatchResult(
                packed=packed,
                meta=meta,
                backend="reference",
                fallback_reason=f"triton_error:{type(exc).__name__}",
            )

    packed, meta = tilepack_reference(feature_map, indices, tile_size)
    fallback_reason = None
    if prefer_triton:
        fallback_reason = availability.reason or "triton_path_not_selected"
    return TilePackDispatchResult(
        packed=packed,
        meta=meta,
        backend="reference",
        fallback_reason=fallback_reason,
    )


__all__ = [
    "BackendKind",
    "TorchTileMeta",
    "TritonTilePackAvailability",
    "TilePackDispatchResult",
    "get_triton_tilepack_availability",
    "tilepack_reference",
    "tilepack_triton",
    "tilepack_dispatch",
]
