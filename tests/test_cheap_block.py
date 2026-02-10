from __future__ import annotations

import torch

from apex_x.model import CheapBlock


def test_cheap_block_output_shape_and_projection_residual() -> None:
    block = CheapBlock(in_channels=3, out_channels=8, use_residual=True)
    x = torch.randn(2, 3, 16, 16)

    y = block(x)

    assert y.shape == (2, 8, 16, 16)


def test_cheap_block_residual_identity_when_main_path_zeroed() -> None:
    block = CheapBlock(in_channels=4, out_channels=4, use_residual=True)
    x = torch.randn(1, 4, 8, 8)

    with torch.no_grad():
        block.conv.weight.zero_()
        block.conv.bias.zero_()
        block.norm.weight.fill_(1.0)
        block.norm.bias.zero_()

    y = block(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_cheap_block_no_residual_outputs_zero_when_main_path_zeroed() -> None:
    block = CheapBlock(in_channels=4, out_channels=4, use_residual=False)
    x = torch.randn(1, 4, 8, 8)

    with torch.no_grad():
        block.conv.weight.zero_()
        block.conv.bias.zero_()
        block.norm.weight.fill_(1.0)
        block.norm.bias.zero_()

    y = block(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_cheap_block_gradient_flow() -> None:
    torch.manual_seed(5)
    block = CheapBlock(in_channels=6, out_channels=6, use_residual=True)
    x = torch.randn(2, 6, 8, 8, requires_grad=True)

    y = block(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum().item()) > 0.0

    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in block.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0.0
