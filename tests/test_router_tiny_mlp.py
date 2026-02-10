from __future__ import annotations

import pytest
import torch

from apex_x.routing import RouterProtocol
from apex_x.routing.tiny_mlp import RouterTinyMLP


def test_router_tiny_mlp_shapes_without_temporal_head() -> None:
    model = RouterTinyMLP(input_dim=6, hidden_dim=16, num_layers=2, temporal_head=False)
    x = torch.randn(2, 5, 6)

    out = model(x)

    assert out.U.shape == (2, 5)
    assert out.S.shape == (2, 5)
    assert out.T is None


def test_router_tiny_mlp_shapes_with_temporal_head() -> None:
    model = RouterTinyMLP(input_dim=6, hidden_dim=16, num_layers=2, temporal_head=True)
    x = torch.randn(2, 5, 6)

    out = model(x)

    assert out.U.shape == (2, 5)
    assert out.S.shape == (2, 5)
    assert out.T is not None
    assert out.T.shape == (2, 5)


def test_router_tiny_mlp_is_deterministic_for_fixed_seed() -> None:
    torch.manual_seed(1234)
    model_a = RouterTinyMLP(input_dim=4, hidden_dim=8, num_layers=2, temporal_head=True)
    x = torch.randn(3, 7, 4)
    out_a = model_a(x)

    torch.manual_seed(1234)
    model_b = RouterTinyMLP(input_dim=4, hidden_dim=8, num_layers=2, temporal_head=True)
    out_b = model_b(x)

    assert torch.allclose(out_a.U, out_b.U)
    assert torch.allclose(out_a.S, out_b.S)
    assert out_a.T is not None
    assert out_b.T is not None
    assert torch.allclose(out_a.T, out_b.T)


def test_router_tiny_mlp_allows_gradient_flow() -> None:
    torch.manual_seed(7)
    model = RouterTinyMLP(input_dim=5, hidden_dim=12, num_layers=3, temporal_head=True)
    x = torch.randn(2, 4, 5, requires_grad=True)

    out = model(x)
    assert out.T is not None

    loss = out.U.square().mean() + out.S.square().mean() + out.T.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0

    grad_sums = [
        float(param.grad.abs().sum()) for param in model.parameters() if param.grad is not None
    ]
    assert grad_sums
    assert max(grad_sums) > 0.0


def test_predict_utilities_returns_per_tile_values() -> None:
    torch.manual_seed(99)
    model = RouterTinyMLP(input_dim=1, hidden_dim=8, num_layers=1, temporal_head=False)

    values = model.predict_utilities([0.5, 1.5, -2.0, 0.0])

    assert len(values) == 4
    assert all(isinstance(v, float) for v in values)


def test_predict_utilities_requires_input_dim_one() -> None:
    model = RouterTinyMLP(input_dim=2, hidden_dim=8, num_layers=1, temporal_head=False)

    with pytest.raises(ValueError, match="input_dim=1"):
        model.predict_utilities([0.1, 0.2])


def test_router_tiny_mlp_conforms_to_router_protocol() -> None:
    model = RouterTinyMLP(input_dim=1, hidden_dim=8, num_layers=1, temporal_head=False)
    assert isinstance(model, RouterProtocol)
