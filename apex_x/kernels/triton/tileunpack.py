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
OverlapMode = Literal["override", "blend"]
TorchTileMeta = dict[str, Tensor]


_tileunpack_priority_kernel: Any | None = None
_tileunpack_scatter_kernel: Any | None = None
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
class TritonTileUnpackAvailability:
    triton_installed: bool
    cuda_available: bool
    cuda_device_count: int
    reason: str | None

    @property
    def available(self) -> bool:
        return self.triton_installed and self.cuda_available and self.cuda_device_count > 0


@dataclass(frozen=True, slots=True)
class TileUnpackDispatchResult:
    merged: Tensor
    backend: BackendKind
    fallback_reason: str | None


def get_triton_tileunpack_availability() -> TritonTileUnpackAvailability:
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

    return TritonTileUnpackAvailability(
        triton_installed=triton_installed,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        reason=reason,
    )


def _validate_base_and_packed(
    base_map: Tensor, packed_out: Tensor
) -> tuple[int, int, int, int, int, int]:
    if base_map.ndim != 4:
        raise ValueError("base_map must be [B,C,H,W]")
    if packed_out.ndim != 5:
        raise ValueError("packed_out must be [B,K,C,t,t]")
    if base_map.shape[0] != packed_out.shape[0] or base_map.shape[1] != packed_out.shape[2]:
        raise ValueError("base_map and packed_out batch/channel dimensions must match")
    if packed_out.shape[3] != packed_out.shape[4]:
        raise ValueError("packed_out tiles must be square [t,t]")

    batch, channels, height, width = base_map.shape
    kmax = int(packed_out.shape[1])
    tile_size = int(packed_out.shape[3])
    if tile_size <= 0:
        raise ValueError("tile_size inferred from packed_out must be > 0")
    if height % tile_size != 0 or width % tile_size != 0:
        raise ValueError("H and W must be divisible by tile_size")
    return batch, channels, height, width, kmax, tile_size


def _build_meta_from_indices(
    indices: Tensor, tile_size: int, height: int, width: int
) -> TorchTileMeta:
    if indices.ndim != 2:
        raise ValueError("indices must be [B,K]")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("indices must be int32 or int64")

    grid_h = height // tile_size
    grid_w = width // tile_size
    max_index = grid_h * grid_w - 1
    idx_i64 = indices.to(dtype=torch.int64)
    if idx_i64.numel() > 0:
        idx_min = int(idx_i64.min().item())
        idx_max = int(idx_i64.max().item())
        if idx_min < 0 or idx_max > max_index:
            raise ValueError("indices contain out-of-bounds tile ids")

    origins_y = (idx_i64 // grid_w) * tile_size
    origins_x = (idx_i64 % grid_w) * tile_size
    origins = torch.stack((origins_y, origins_x), dim=-1)
    return {
        "indices": idx_i64,
        "origins": origins,
        "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=idx_i64.device),
        "grid": torch.tensor([grid_h, grid_w], dtype=torch.int64, device=idx_i64.device),
    }


def _normalize_meta_or_indices(
    *,
    base_map: Tensor,
    packed_out: Tensor,
    indices: Tensor | None,
    meta: TorchTileMeta | None,
) -> TorchTileMeta:
    _, _, height, width, kmax, tile_size = _validate_base_and_packed(base_map, packed_out)
    if meta is not None:
        if "origins" not in meta:
            raise ValueError("meta must contain 'origins'")
        origins = meta["origins"]
        if origins.ndim != 3 or origins.shape != (base_map.shape[0], kmax, 2):
            raise ValueError("meta['origins'] must have shape [B,K,2]")

        grid_w = width // tile_size
        if "indices" in meta:
            idx_i64 = meta["indices"].to(dtype=torch.int64, device=base_map.device)
        else:
            y_tiles = origins[..., 0].to(dtype=torch.int64, device=base_map.device) // tile_size
            x_tiles = origins[..., 1].to(dtype=torch.int64, device=base_map.device) // tile_size
            idx_i64 = y_tiles * grid_w + x_tiles

        return {
            "indices": idx_i64,
            "origins": origins.to(dtype=torch.int64, device=base_map.device),
            "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=base_map.device),
            "grid": torch.tensor(
                [height // tile_size, width // tile_size],
                dtype=torch.int64,
                device=base_map.device,
            ),
        }

    if indices is None:
        raise ValueError("either indices or meta must be provided")
    return _build_meta_from_indices(
        indices.to(device=base_map.device),
        tile_size=tile_size,
        height=height,
        width=width,
    )


def _compute_priority_keys(
    *,
    batch: int,
    kmax: int,
    device: torch.device,
    levels: Tensor | None,
    assume_priority_sorted: bool,
) -> Tensor:
    order = torch.arange(kmax, dtype=torch.int64, device=device).view(1, kmax).expand(batch, kmax)
    if levels is None:
        if not assume_priority_sorted:
            raise ValueError("levels must be provided when assume_priority_sorted=False")
        return order.to(dtype=torch.int32)

    if levels.ndim != 2 or levels.shape != (batch, kmax):
        raise ValueError("levels must have shape [B,K]")
    if levels.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("levels must be an integer tensor")

    levels_i64 = levels.to(dtype=torch.int64, device=device)
    min_level = int(levels_i64.min().item()) if levels_i64.numel() > 0 else 0
    levels_adj = levels_i64 - min_level
    order_bits = max(1, (kmax - 1).bit_length())

    key_i64 = (levels_adj << order_bits) + order
    max_key = int(key_i64.max().item()) if key_i64.numel() > 0 else 0
    if max_key > int(torch.iinfo(torch.int32).max):
        raise ValueError("priority key overflow for int32; reduce level range or K")
    return key_i64.to(dtype=torch.int32)


def tileunpack_reference(
    base_map: Tensor,
    packed_out: Tensor,
    *,
    indices: Tensor | None = None,
    meta: TorchTileMeta | None = None,
    levels: Tensor | None = None,
    assume_priority_sorted: bool = True,
    overlap_mode: OverlapMode = "override",
    blend_alpha: float = 0.5,
) -> Tensor:
    batch, channels, height, width, _, tile_size = _validate_base_and_packed(base_map, packed_out)
    if overlap_mode not in {"override", "blend"}:
        raise ValueError("overlap_mode must be 'override' or 'blend'")
    if not (0.0 <= blend_alpha <= 1.0):
        raise ValueError("blend_alpha must be within [0, 1]")

    normalized_meta = _normalize_meta_or_indices(
        base_map=base_map,
        packed_out=packed_out,
        indices=indices,
        meta=meta,
    )
    idx_i64 = normalized_meta["indices"].to(dtype=torch.int64, device=base_map.device)
    kmax = int(idx_i64.shape[1])
    keys = _compute_priority_keys(
        batch=batch,
        kmax=kmax,
        device=base_map.device,
        levels=levels,
        assume_priority_sorted=assume_priority_sorted,
    )
    origins = normalized_meta["origins"].to(dtype=torch.int64, device=base_map.device)

    merged = base_map.contiguous().clone()
    alpha = torch.tensor(blend_alpha, dtype=base_map.dtype, device=base_map.device)

    for b in range(batch):
        order = torch.argsort(keys[b], dim=0, descending=False, stable=True)
        for k_idx_t in order:
            k_idx = int(k_idx_t.item())
            y = int(origins[b, k_idx, 0].item())
            x = int(origins[b, k_idx, 1].item())
            if y < 0 or x < 0 or y + tile_size > height or x + tile_size > width:
                raise ValueError("meta origins contain out-of-bounds tile coordinates")

            current_patch = merged[b, :, y : y + tile_size, x : x + tile_size]
            incoming_patch = packed_out[b, k_idx]
            if overlap_mode == "override":
                merged[b, :, y : y + tile_size, x : x + tile_size] = incoming_patch
            else:
                merged[b, :, y : y + tile_size, x : x + tile_size] = (
                    1.0 - alpha
                ) * current_patch + alpha * incoming_patch

    return merged.contiguous()


def _tileunpack_blend_ordered(
    *,
    base_map: Tensor,
    packed_out: Tensor,
    origins: Tensor,
    keys: Tensor,
    tile_size: int,
    blend_alpha: float,
) -> Tensor:
    batch, _, height, width = base_map.shape
    merged = base_map.contiguous().clone()
    alpha = torch.tensor(blend_alpha, dtype=base_map.dtype, device=base_map.device)
    one_minus_alpha = 1.0 - alpha

    for b in range(batch):
        order = torch.argsort(keys[b], dim=0, descending=False, stable=True)
        for k_idx_t in order:
            k_idx = int(k_idx_t.item())
            y = int(origins[b, k_idx, 0].item())
            x = int(origins[b, k_idx, 1].item())
            if y < 0 or x < 0 or y + tile_size > height or x + tile_size > width:
                raise ValueError("meta origins contain out-of-bounds tile coordinates")

            current_patch = merged[b, :, y : y + tile_size, x : x + tile_size]
            incoming_patch = packed_out[b, k_idx]
            merged[b, :, y : y + tile_size, x : x + tile_size] = (
                one_minus_alpha * current_patch + alpha * incoming_patch
            )

    return merged.contiguous()


def _tileunpack_block_heuristic(tile_pixels: int) -> int:
    if tile_pixels <= 64:
        return 64
    if tile_pixels <= 128:
        return 128
    return 256


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
    def _tileunpack_priority_kernel(
        keys_ptr,  # [B,K] int32
        origins_ptr,  # [B,K,2] int32
        winner_ptr,  # [B,H,W] int32
        batch,
        height,
        width,
        kmax,
        tile_size,
        tile_pixels,
        BLOCK_PIX: tl.constexpr,
    ) -> None:
        pid_bk = tl.program_id(0)
        pid_blk = tl.program_id(1)

        b = pid_bk // kmax
        k = pid_bk - b * kmax
        if b >= batch:
            return

        key = tl.load(keys_ptr + b * kmax + k).to(tl.int32)
        origin_base = (b * kmax + k) * 2
        base_y = tl.load(origins_ptr + origin_base).to(tl.int32)
        base_x = tl.load(origins_ptr + origin_base + 1).to(tl.int32)

        offs = pid_blk * BLOCK_PIX + tl.arange(0, BLOCK_PIX).to(tl.int32)
        valid = offs < tile_pixels
        local_y = offs // tile_size
        local_x = offs - local_y * tile_size
        y = base_y + local_y
        x = base_x + local_x

        valid = valid & (y >= 0) & (x >= 0) & (y < height) & (x < width)
        win_offsets = (b * height + y) * width + x
        tl.atomic_max(winner_ptr + win_offsets, key, mask=valid)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_PIX": 64}, num_warps=4),
            triton.Config({"BLOCK_PIX": 128}, num_warps=4),
            triton.Config({"BLOCK_PIX": 256}, num_warps=8),
        ],
        key=["tile_pixels"],
    )
    @triton.jit
    def _tileunpack_scatter_kernel(
        packed_ptr,  # [B,K,C,t,t] contiguous
        keys_ptr,  # [B,K] int32
        origins_ptr,  # [B,K,2] int32
        winner_ptr,  # [B,H,W] int32
        out_ptr,  # [B,C,H,W] contiguous
        batch,
        channels,
        height,
        width,
        kmax,
        tile_size,
        tile_pixels,
        BLOCK_PIX: tl.constexpr,
    ) -> None:
        pid_bk = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_blk = tl.program_id(2)

        b = pid_bk // kmax
        k = pid_bk - b * kmax
        c = pid_c
        if (b >= batch) or (c >= channels):
            return

        key = tl.load(keys_ptr + b * kmax + k).to(tl.int32)
        origin_base = (b * kmax + k) * 2
        base_y = tl.load(origins_ptr + origin_base).to(tl.int32)
        base_x = tl.load(origins_ptr + origin_base + 1).to(tl.int32)

        offs = pid_blk * BLOCK_PIX + tl.arange(0, BLOCK_PIX).to(tl.int32)
        valid = offs < tile_pixels
        local_y = offs // tile_size
        local_x = offs - local_y * tile_size
        y = base_y + local_y
        x = base_x + local_x

        valid = valid & (y >= 0) & (x >= 0) & (y < height) & (x < width)
        winner_offsets = (b * height + y) * width + x
        winner = tl.load(winner_ptr + winner_offsets, mask=valid, other=-2147483648)
        write_mask = valid & (winner == key)

        in_offsets = (((b * kmax + k) * channels + c) * tile_size + local_y) * tile_size + local_x
        values = tl.load(packed_ptr + in_offsets, mask=valid, other=0)

        out_offsets = ((b * channels + c) * height + y) * width + x
        tl.store(out_ptr + out_offsets, values, mask=write_mask)

else:
    _tileunpack_priority_kernel = None
    _tileunpack_scatter_kernel = None


def tileunpack_triton(
    base_map: Tensor,
    packed_out: Tensor,
    *,
    indices: Tensor | None = None,
    meta: TorchTileMeta | None = None,
    levels: Tensor | None = None,
    assume_priority_sorted: bool = True,
    overlap_mode: OverlapMode = "override",
    blend_alpha: float = 0.5,
) -> Tensor:
    batch, channels, height, width, kmax, tile_size = _validate_base_and_packed(
        base_map, packed_out
    )
    if base_map.device.type != "cuda" or packed_out.device.type != "cuda":
        raise ValueError("tileunpack_triton requires CUDA tensors for base_map and packed_out")
    if base_map.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("tileunpack_triton supports fp16/bf16 inputs only")
    if packed_out.dtype != base_map.dtype:
        raise ValueError("base_map and packed_out must use same dtype")
    if overlap_mode not in {"override", "blend"}:
        raise ValueError("overlap_mode must be 'override' or 'blend'")
    if not (0.0 <= blend_alpha <= 1.0):
        raise ValueError("blend_alpha must be within [0, 1]")

    normalized_meta = _normalize_meta_or_indices(
        base_map=base_map,
        packed_out=packed_out,
        indices=indices,
        meta=meta,
    )
    keys_i32 = _compute_priority_keys(
        batch=batch,
        kmax=kmax,
        device=base_map.device,
        levels=levels,
        assume_priority_sorted=assume_priority_sorted,
    ).contiguous()
    origins_i32 = (
        normalized_meta["origins"].to(dtype=torch.int32, device=base_map.device).contiguous()
    )

    if overlap_mode == "blend":
        return _tileunpack_blend_ordered(
            base_map=base_map,
            packed_out=packed_out,
            origins=origins_i32.to(dtype=torch.int64),
            keys=keys_i32.to(dtype=torch.int64),
            tile_size=tile_size,
            blend_alpha=blend_alpha,
        )

    if triton is None or _tileunpack_priority_kernel is None or _tileunpack_scatter_kernel is None:
        raise RuntimeError("Triton is not available")

    out = base_map.contiguous().clone()
    packed_contig = packed_out.contiguous()
    tile_pixels = tile_size * tile_size
    pix_blocks = triton.cdiv(tile_pixels, 256)
    shape_bucket = build_shape_bucket(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        kmax=kmax,
        tile_size=tile_size,
        dtype=str(base_map.dtype).replace("torch.", ""),
    )

    priority_config = get_cached_triton_config(
        op_name="tileunpack_priority",
        shape_bucket=shape_bucket,
    )
    priority_source = "registry_cache"
    if priority_config is None:
        priority_config, priority_source = resolve_triton_launch_config(
            kernel=_tileunpack_priority_kernel,
            fallback_config={"BLOCK_PIX": _tileunpack_block_heuristic(tile_pixels)},
        )

    scatter_config = get_cached_triton_config(
        op_name="tileunpack_scatter",
        shape_bucket=shape_bucket,
    )
    scatter_source = "registry_cache"
    if scatter_config is None:
        scatter_config, scatter_source = resolve_triton_launch_config(
            kernel=_tileunpack_scatter_kernel,
            fallback_config={"BLOCK_PIX": _tileunpack_block_heuristic(tile_pixels)},
        )
    winner = torch.full(
        (batch, height, width),
        fill_value=int(torch.iinfo(torch.int32).min),
        dtype=torch.int32,
        device=base_map.device,
    )

    grid_priority = (batch * kmax, pix_blocks)
    _tileunpack_priority_kernel[grid_priority](
        keys_i32,
        origins_i32,
        winner,
        batch,
        height,
        width,
        kmax,
        tile_size,
        tile_pixels,
    )
    record_triton_autotune_selection(
        op_name="tileunpack_priority",
        kernel_name="_tileunpack_priority_kernel",
        shape_bucket=shape_bucket,
        selected_config=priority_config,
        selection_source=priority_source,
    )

    grid_scatter = (batch * kmax, channels, pix_blocks)
    _tileunpack_scatter_kernel[grid_scatter](
        packed_contig,
        keys_i32,
        origins_i32,
        winner,
        out,
        batch,
        channels,
        height,
        width,
        kmax,
        tile_size,
        tile_pixels,
    )
    record_triton_autotune_selection(
        op_name="tileunpack_scatter",
        kernel_name="_tileunpack_scatter_kernel",
        shape_bucket=shape_bucket,
        selected_config=scatter_config,
        selection_source=scatter_source,
    )

    return out.contiguous()


def tileunpack_dispatch(
    base_map: Tensor,
    packed_out: Tensor,
    *,
    indices: Tensor | None = None,
    meta: TorchTileMeta | None = None,
    levels: Tensor | None = None,
    assume_priority_sorted: bool = True,
    overlap_mode: OverlapMode = "override",
    blend_alpha: float = 0.5,
    prefer_triton: bool = True,
    allow_fallback: bool = True,
    inference_only: bool = True,
) -> TileUnpackDispatchResult:
    normalized_meta = _normalize_meta_or_indices(
        base_map=base_map,
        packed_out=packed_out,
        indices=indices,
        meta=meta,
    )

    if inference_only and (base_map.requires_grad or packed_out.requires_grad):
        merged = tileunpack_reference(
            base_map,
            packed_out,
            meta=normalized_meta,
            levels=levels,
            assume_priority_sorted=assume_priority_sorted,
            overlap_mode=overlap_mode,
            blend_alpha=blend_alpha,
        )
        return TileUnpackDispatchResult(
            merged=merged,
            backend="reference",
            fallback_reason="autograd_not_supported_for_triton_tileunpack",
        )

    availability = get_triton_tileunpack_availability()
    if prefer_triton and availability.available:
        try:
            merged = tileunpack_triton(
                base_map,
                packed_out,
                meta=normalized_meta,
                levels=levels,
                assume_priority_sorted=assume_priority_sorted,
                overlap_mode=overlap_mode,
                blend_alpha=blend_alpha,
            )
            return TileUnpackDispatchResult(merged=merged, backend="triton", fallback_reason=None)
        except Exception as exc:
            if not allow_fallback:
                raise
            merged = tileunpack_reference(
                base_map,
                packed_out,
                meta=normalized_meta,
                levels=levels,
                assume_priority_sorted=assume_priority_sorted,
                overlap_mode=overlap_mode,
                blend_alpha=blend_alpha,
            )
            return TileUnpackDispatchResult(
                merged=merged,
                backend="reference",
                fallback_reason=f"triton_error:{type(exc).__name__}",
            )

    merged = tileunpack_reference(
        base_map,
        packed_out,
        meta=normalized_meta,
        levels=levels,
        assume_priority_sorted=assume_priority_sorted,
        overlap_mode=overlap_mode,
        blend_alpha=blend_alpha,
    )
    fallback_reason = None
    if prefer_triton:
        fallback_reason = availability.reason or "triton_path_not_selected"
    return TileUnpackDispatchResult(
        merged=merged,
        backend="reference",
        fallback_reason=fallback_reason,
    )


__all__ = [
    "BackendKind",
    "OverlapMode",
    "TorchTileMeta",
    "TritonTileUnpackAvailability",
    "TileUnpackDispatchResult",
    "get_triton_tileunpack_availability",
    "tileunpack_reference",
    "tileunpack_triton",
    "tileunpack_dispatch",
]
