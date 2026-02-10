from __future__ import annotations

from typing import Literal

import torch

from .ops import order_idx, tile_grid_shape

TorchTileMeta = dict[str, torch.Tensor]
OverlapMode = Literal["override", "blend"]


class TilePackTorch:
    """Torch tile packer: [B,C,H,W] + idx[B,K] -> [B,K,C,t,t] + meta."""

    def pack(
        self,
        feature_map: torch.Tensor,
        indices: torch.Tensor,
        tile_size: int,
        order_mode: str = "hilbert",
    ) -> tuple[torch.Tensor, TorchTileMeta]:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must be [B,C,H,W]")
        if indices.ndim != 2:
            raise ValueError("indices must be [B,K]")
        if feature_map.shape[0] != indices.shape[0]:
            raise ValueError("feature_map and indices batch dimensions must match")
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("indices must be an integer tensor")

        batch, channels, height, width = feature_map.shape
        grid_h, grid_w = tile_grid_shape(height, width, tile_size)
        max_index = grid_h * grid_w - 1
        idx64 = indices.to(dtype=torch.int64)
        if idx64.numel() > 0:
            idx_min = int(torch.min(idx64).item())
            idx_max = int(torch.max(idx64).item())
            if idx_min < 0 or idx_max > max_index:
                raise ValueError("indices contain out-of-bounds tile ids")

        kmax = idx64.shape[1]
        packed = feature_map.new_zeros((batch, kmax, channels, tile_size, tile_size))
        sorted_idx = torch.empty((batch, kmax), dtype=torch.int64, device=idx64.device)
        origins = torch.empty((batch, kmax, 2), dtype=torch.int64, device=idx64.device)

        for b in range(batch):
            ordered = order_idx(idx64[b].tolist(), grid_h=grid_h, grid_w=grid_w, mode=order_mode)
            ordered_t = torch.tensor(ordered, dtype=torch.int64, device=idx64.device)
            sorted_idx[b] = ordered_t
            for k, tile_idx_t in enumerate(ordered_t):
                tile_idx = int(tile_idx_t.item())
                y = (tile_idx // grid_w) * tile_size
                x = (tile_idx % grid_w) * tile_size
                origins[b, k, 0] = y
                origins[b, k, 1] = x
                packed[b, k] = feature_map[b, :, y : y + tile_size, x : x + tile_size]

        packed = packed.contiguous()
        meta: TorchTileMeta = {
            "indices": sorted_idx,
            "origins": origins,
            "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=idx64.device),
            "grid": torch.tensor([grid_h, grid_w], dtype=torch.int64, device=idx64.device),
        }
        return packed, meta


class TileUnpackTorch:
    """Torch tile unpacker with overlap priority handling."""

    def unpack(
        self,
        base_map: torch.Tensor,
        packed_out: torch.Tensor,
        meta: TorchTileMeta,
        level_priority: int = 1,
        priority_map: torch.Tensor | None = None,
        overlap_mode: OverlapMode = "override",
        blend_alpha: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_map.ndim != 4:
            raise ValueError("base_map must be [B,C,H,W]")
        if packed_out.ndim != 5:
            raise ValueError("packed_out must be [B,K,C,t,t]")
        if "origins" not in meta:
            raise ValueError("meta must contain 'origins'")
        if overlap_mode not in {"override", "blend"}:
            raise ValueError("overlap_mode must be 'override' or 'blend'")
        if not (0.0 <= blend_alpha <= 1.0):
            raise ValueError("blend_alpha must be within [0, 1]")

        batch, channels, height, width = base_map.shape
        p_batch, kmax, p_channels, tile_h, tile_w = packed_out.shape
        if p_batch != batch or p_channels != channels:
            raise ValueError("base_map and packed_out shapes are incompatible")
        if tile_h != tile_w:
            raise ValueError("packed_out tile size must be square")

        origins = meta["origins"]
        if origins.shape != (batch, kmax, 2):
            raise ValueError("meta['origins'] must have shape [B,K,2]")

        if priority_map is None:
            priority_map = torch.zeros(
                (batch, height, width),
                dtype=torch.int8,
                device=base_map.device,
            )
        else:
            if priority_map.shape != (batch, height, width):
                raise ValueError("priority_map must be [B,H,W]")
            priority_map = priority_map.to(device=base_map.device)

        merged = base_map.clone()
        level_priority_t = torch.tensor(
            level_priority,
            dtype=priority_map.dtype,
            device=priority_map.device,
        )
        alpha = torch.tensor(blend_alpha, dtype=base_map.dtype, device=base_map.device)

        for b in range(batch):
            for k in range(kmax):
                y = int(origins[b, k, 0].item())
                x = int(origins[b, k, 1].item())
                if y < 0 or x < 0 or y + tile_h > height or x + tile_w > width:
                    raise ValueError("meta origins contain out-of-bounds tile coordinates")

                priority_patch = priority_map[b, y : y + tile_h, x : x + tile_w]
                allow_mask = level_priority_t >= priority_patch
                if not torch.any(allow_mask):
                    continue

                current_patch = merged[b, :, y : y + tile_h, x : x + tile_w]
                incoming_patch = packed_out[b, k]
                allow_mask_c = allow_mask.unsqueeze(0)

                if overlap_mode == "override":
                    updated_patch = torch.where(allow_mask_c, incoming_patch, current_patch)
                else:
                    mixed_patch = (1.0 - alpha) * current_patch + alpha * incoming_patch
                    updated_patch = torch.where(allow_mask_c, mixed_patch, current_patch)

                merged[b, :, y : y + tile_h, x : x + tile_w] = updated_patch
                priority_map[b, y : y + tile_h, x : x + tile_w] = torch.where(
                    allow_mask,
                    level_priority_t,
                    priority_patch,
                )

        return merged, priority_map


def pack_tiles_torch(
    feature_map: torch.Tensor,
    indices: torch.Tensor,
    tile_size: int,
    order_mode: str = "hilbert",
) -> tuple[torch.Tensor, TorchTileMeta]:
    return TilePackTorch().pack(feature_map, indices, tile_size, order_mode)


def unpack_tiles_torch(
    base_map: torch.Tensor,
    packed_out: torch.Tensor,
    meta: TorchTileMeta,
    level_priority: int = 1,
    priority_map: torch.Tensor | None = None,
    overlap_mode: OverlapMode = "override",
    blend_alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    return TileUnpackTorch().unpack(
        base_map=base_map,
        packed_out=packed_out,
        meta=meta,
        level_priority=level_priority,
        priority_map=priority_map,
        overlap_mode=overlap_mode,
        blend_alpha=blend_alpha,
    )
