from __future__ import annotations

import torch

from apex_x.model import TileRefineBlock


def test_tile_refine_block_shape_and_projection() -> None:
    block = TileRefineBlock(in_channels=4, out_channels=6, use_residual=True)
    tiles = torch.randn(2, 3, 4, 8, 8)

    out = block(tiles)

    assert out.shape == (2, 3, 6, 8, 8)


def test_tile_refine_block_residual_identity_when_main_path_zeroed() -> None:
    block = TileRefineBlock(in_channels=5, out_channels=5, use_residual=True)
    tiles = torch.randn(1, 2, 5, 6, 6)

    with torch.no_grad():
        block.depthwise.weight.zero_()
        block.depthwise.bias.zero_()
        block.pointwise.weight.zero_()
        block.pointwise.bias.zero_()
        block.norm.weight.fill_(1.0)
        block.norm.bias.zero_()

    out = block(tiles)
    assert torch.allclose(out, tiles, atol=1e-6)


def test_tile_refine_block_operates_per_tile_without_cross_tile_mixing() -> None:
    torch.manual_seed(3)
    block = TileRefineBlock(in_channels=3, out_channels=3, use_residual=False)
    tiles_a = torch.randn(1, 2, 3, 5, 5)
    tiles_b = tiles_a.clone()
    tiles_b[:, 0] = tiles_b[:, 0] + 10.0

    out_a = block(tiles_a)
    out_b = block(tiles_b)

    assert torch.allclose(out_a[:, 1], out_b[:, 1], atol=1e-6)


def test_tile_refine_block_gradient_flow() -> None:
    torch.manual_seed(13)
    block = TileRefineBlock(in_channels=4, out_channels=4, use_residual=True)
    tiles = torch.randn(2, 3, 4, 6, 6, requires_grad=True)

    out = block(tiles)
    loss = out.square().mean()
    loss.backward()

    assert tiles.grad is not None
    assert torch.isfinite(tiles.grad).all()
    assert float(tiles.grad.abs().sum().item()) > 0.0

    grad_sum = sum(
        float(param.grad.abs().sum().item())
        for param in block.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0.0
