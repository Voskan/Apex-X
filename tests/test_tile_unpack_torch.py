from __future__ import annotations

import numpy as np
import torch

from apex_x.tiles import (
    TilePackTorch,
    TileUnpackTorch,
    pack_tiles,
    pack_tiles_torch,
    unpack_tiles,
    unpack_tiles_torch,
)


def _single_tile_meta_torch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "indices": torch.tensor([[0]], dtype=torch.int64, device=device),
        "origins": torch.tensor([[[0, 0]]], dtype=torch.int64, device=device),
        "tile_size": torch.tensor(4, dtype=torch.int64, device=device),
        "grid": torch.tensor([1, 1], dtype=torch.int64, device=device),
    }


def test_tile_unpack_torch_override_matches_numpy() -> None:
    feature_np = np.arange(1 * 2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    base_np = np.zeros_like(feature_np)
    idx_np = np.asarray([[0, 3]], dtype=np.int64)
    packed_np, meta_np = pack_tiles(feature_np, idx_np, tile_size=4, order_mode="hilbert")
    merged_np, pri_np = unpack_tiles(base_np, packed_np, meta_np, level_priority=2)

    feature_t = torch.from_numpy(feature_np)
    base_t = torch.from_numpy(base_np)
    idx_t = torch.from_numpy(idx_np)
    packed_t, meta_t = pack_tiles_torch(feature_t, idx_t, tile_size=4, order_mode="hilbert")
    merged_t, pri_t = unpack_tiles_torch(base_t, packed_t, meta_t, level_priority=2)

    assert np.allclose(merged_t.detach().cpu().numpy(), merged_np)
    assert np.array_equal(pri_t.cpu().numpy(), pri_np)


def test_tile_unpack_torch_overlap_priority_and_blend_mode() -> None:
    unpacker = TileUnpackTorch()
    base = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    meta = _single_tile_meta_torch(base.device)

    low = torch.full((1, 1, 1, 4, 4), 1.0, dtype=torch.float32)
    high = torch.full((1, 1, 1, 4, 4), 3.0, dtype=torch.float32)

    out_low, pri = unpacker.unpack(base, low, meta, level_priority=2, overlap_mode="override")
    out_ignored, pri = unpacker.unpack(
        out_low,
        high,
        meta,
        level_priority=1,
        priority_map=pri,
        overlap_mode="override",
    )
    assert float(out_ignored[0, 0, 0, 0].item()) == 1.0

    out_blend, pri = unpacker.unpack(
        out_low,
        high,
        meta,
        level_priority=3,
        priority_map=pri,
        overlap_mode="blend",
        blend_alpha=0.25,
    )
    assert torch.allclose(out_blend, torch.full_like(out_blend, 1.5))
    assert int(pri[0, 0, 0].item()) == 3


def test_tile_unpack_torch_blend_gradient_flow() -> None:
    unpacker = TileUnpackTorch()
    base = torch.zeros((1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
    packed = torch.ones((1, 1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
    meta = _single_tile_meta_torch(base.device)

    merged, _ = unpacker.unpack(
        base,
        packed,
        meta,
        level_priority=1,
        overlap_mode="blend",
        blend_alpha=0.25,
    )
    loss = merged.sum()
    loss.backward()

    assert base.grad is not None
    assert packed.grad is not None
    assert torch.isfinite(base.grad).all()
    assert torch.isfinite(packed.grad).all()
    assert torch.allclose(base.grad, torch.full_like(base.grad, 0.75))
    assert torch.allclose(packed.grad, torch.full_like(packed.grad, 0.25))


def test_tile_unpack_torch_pack_unpack_roundtrip_helper() -> None:
    feature = torch.arange(1 * 1 * 8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    idx = torch.tensor([[0, 3]], dtype=torch.int64)

    packed, meta = TilePackTorch().pack(feature, idx, tile_size=4)
    merged, _ = unpack_tiles_torch(
        torch.zeros_like(feature),
        packed,
        meta,
        level_priority=1,
        overlap_mode="override",
    )
    unpacked_direct, _ = TileUnpackTorch().unpack(
        torch.zeros_like(feature),
        packed,
        meta,
        level_priority=1,
        overlap_mode="override",
    )

    assert torch.allclose(merged, unpacked_direct)
