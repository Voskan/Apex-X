from __future__ import annotations

import numpy as np
import torch

from apex_x.tiles import TilePackTorch, pack_tiles, pack_tiles_torch


def test_tile_pack_torch_matches_numpy_and_is_contiguous() -> None:
    feature_np = np.arange(1 * 2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    idx_np = np.asarray([[0, 3]], dtype=np.int64)

    packed_np, meta_np = pack_tiles(feature_np, idx_np, tile_size=4, order_mode="hilbert")

    feature_t = torch.from_numpy(feature_np)
    idx_t = torch.from_numpy(idx_np)
    packed_t, meta_t = TilePackTorch().pack(feature_t, idx_t, tile_size=4, order_mode="hilbert")
    packed_t_fn, _ = pack_tiles_torch(feature_t, idx_t, tile_size=4, order_mode="hilbert")

    assert packed_t.is_contiguous()
    assert packed_t_fn.is_contiguous()
    assert packed_t.shape == (1, 2, 2, 4, 4)
    assert np.allclose(packed_t.detach().cpu().numpy(), packed_np)
    assert np.array_equal(meta_t["indices"].cpu().numpy(), meta_np["indices"])
    assert np.array_equal(meta_t["origins"].cpu().numpy(), meta_np["origins"])
    assert np.array_equal(meta_t["grid"].cpu().numpy(), meta_np["grid"])
    assert int(meta_t["tile_size"].item()) == int(meta_np["tile_size"])


def test_tile_pack_torch_gradient_flow() -> None:
    feature = torch.ones((1, 1, 8, 8), dtype=torch.float32, requires_grad=True)
    idx = torch.tensor([[0, 3]], dtype=torch.int64)

    packed, _ = TilePackTorch().pack(feature, idx, tile_size=4, order_mode="hilbert")
    loss = packed.sum()
    loss.backward()

    grad = feature.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0

    grad_hw = grad[0, 0]
    assert torch.all(grad_hw[:4, :4] == 1.0)
    assert torch.all(grad_hw[4:, 4:] == 1.0)
    assert torch.all(grad_hw[:4, 4:] == 0.0)
    assert torch.all(grad_hw[4:, :4] == 0.0)
