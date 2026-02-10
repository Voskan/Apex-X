from __future__ import annotations

import torch

from apex_x.routing import RouterKANLike, RouterProtocol
from apex_x.routing.kan_like import LightweightSplineActivation


def test_spline_activation_is_finite_on_extreme_values() -> None:
    act = LightweightSplineActivation(features=4, num_knots=8, x_min=-2.0, x_max=2.0)
    x = torch.tensor(
        [[float("-inf"), -1e6, float("nan"), 1e6], [1e8, -1e8, 0.5, float("inf")]],
        dtype=torch.float32,
    )

    y = act(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_spline_activation_has_finite_gradients() -> None:
    torch.manual_seed(11)
    act = LightweightSplineActivation(features=3, num_knots=6)
    x = (torch.randn(5, 3) * 1000.0).requires_grad_()

    y = act(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert act.knot_y.grad is not None
    assert torch.isfinite(act.knot_y.grad).all()


def test_router_kan_like_shapes_and_finite_outputs() -> None:
    torch.manual_seed(123)
    router = RouterKANLike(input_dim=6, hidden_dim=16, num_knots=8, temporal_head=True)
    x = torch.randn(2, 7, 6) * 1e6

    out = router(x)

    assert out.U.shape == (2, 7)
    assert out.S.shape == (2, 7)
    assert out.T is not None
    assert out.T.shape == (2, 7)
    assert torch.isfinite(out.U).all()
    assert torch.isfinite(out.S).all()
    assert torch.isfinite(out.T).all()


def test_router_kan_like_is_deterministic_for_fixed_seed() -> None:
    torch.manual_seed(21)
    router_a = RouterKANLike(input_dim=5, hidden_dim=12, num_knots=6, temporal_head=True)
    x = torch.randn(2, 6, 5)
    out_a = router_a(x)

    torch.manual_seed(21)
    router_b = RouterKANLike(input_dim=5, hidden_dim=12, num_knots=6, temporal_head=True)
    out_b = router_b(x)

    assert torch.allclose(out_a.U, out_b.U)
    assert torch.allclose(out_a.S, out_b.S)
    assert out_a.T is not None
    assert out_b.T is not None
    assert torch.allclose(out_a.T, out_b.T)


def test_router_kan_like_gradients_are_finite() -> None:
    torch.manual_seed(77)
    router = RouterKANLike(input_dim=4, hidden_dim=10, num_knots=6, temporal_head=True)
    x = (torch.randn(3, 8, 4) * 500.0).requires_grad_()

    out = router(x)
    assert out.T is not None
    loss = out.U.square().mean() + out.S.square().mean() + out.T.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [parameter.grad for parameter in router.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_router_kan_like_parameter_count_stays_small() -> None:
    router = RouterKANLike(input_dim=24, hidden_dim=16, num_knots=8, temporal_head=True)
    assert router.parameter_count() < 5000


def test_router_kan_like_predict_utilities_and_protocol() -> None:
    router = RouterKANLike(input_dim=1, hidden_dim=8, num_knots=6, temporal_head=False)
    values = router.predict_utilities([0.0, 1.0, -1.0])

    assert isinstance(router, RouterProtocol)
    assert len(values) == 3
    assert all(isinstance(value, float) for value in values)
