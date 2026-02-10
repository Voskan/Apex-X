from __future__ import annotations

import torch

from apex_x.model import TileFiLM, apply_film


def test_film_apply_formula_and_shapes() -> None:
    torch.manual_seed(17)
    film = TileFiLM(token_dim=6, tile_channels=4, hidden_dim=8, gamma_limit=0.35)

    tokens = torch.randn(2, 5, 6)
    tiles = torch.randn(2, 5, 4, 4, 4)
    modulated, gamma, beta = film(tokens, tiles)

    expected = (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) * tiles + beta.unsqueeze(-1).unsqueeze(-1)
    assert modulated.shape == tiles.shape
    assert gamma.shape == (2, 5, 4)
    assert beta.shape == (2, 5, 4)
    assert torch.allclose(modulated, expected)
    assert torch.all(gamma <= 0.35 + 1e-6)
    assert torch.all(gamma >= -0.35 - 1e-6)


def test_film_is_deterministic_for_fixed_inputs() -> None:
    torch.manual_seed(5)
    film = TileFiLM(token_dim=3, tile_channels=2, hidden_dim=6, gamma_limit=0.5)
    tokens = torch.randn(1, 4, 3)
    tiles = torch.randn(1, 4, 2, 8, 8)

    out1, gamma1, beta1 = film(tokens, tiles)
    out2, gamma2, beta2 = film(tokens, tiles)

    assert torch.allclose(out1, out2)
    assert torch.allclose(gamma1, gamma2)
    assert torch.allclose(beta1, beta2)


def test_film_gradient_flow() -> None:
    torch.manual_seed(11)
    film = TileFiLM(token_dim=4, tile_channels=4, hidden_dim=10, gamma_limit=0.75)
    tokens = torch.randn(2, 3, 4, requires_grad=True)
    tiles = torch.randn(2, 3, 4, 4, 4, requires_grad=True)

    out, gamma, beta = film(tokens, tiles)
    loss = out.square().mean() + gamma.square().mean() + beta.square().mean()
    loss.backward()

    assert tokens.grad is not None
    assert tiles.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert torch.isfinite(tiles.grad).all()
    assert float(tokens.grad.abs().sum().item()) > 0.0
    assert float(tiles.grad.abs().sum().item()) > 0.0

    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in film.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0.0


def test_apply_film_shape_validation() -> None:
    tiles = torch.randn(1, 2, 3, 4, 4)
    gamma = torch.randn(1, 2, 3)
    beta = torch.randn(1, 2, 3)

    out = apply_film(tiles, gamma, beta)
    assert out.shape == tiles.shape
