from __future__ import annotations

import torch

from apex_x.routing import sigmoid_probabilities, ste_gate_from_utilities, ste_hard_gate


def test_ste_threshold_gate_is_hard_and_has_nonzero_grad_wrt_u() -> None:
    u = torch.tensor([-2.0, -0.4, 0.0, 0.7, 2.0], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 0.75], dtype=torch.float32)

    _, gate = ste_gate_from_utilities(u, mode="threshold", threshold=0.5)
    assert torch.all((gate == 0.0) | (gate == 1.0))

    loss = (gate * weights).sum()
    loss.backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert float(u.grad.abs().sum()) > 0.0


def test_ste_bernoulli_gate_is_hard_and_has_nonzero_grad_wrt_u() -> None:
    u = torch.tensor([-1.0, 0.0, 1.0, -0.5, 0.5], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0, 1.5, 0.5], dtype=torch.float32)
    gen = torch.Generator().manual_seed(123)

    _, gate = ste_gate_from_utilities(
        u,
        mode="bernoulli",
        threshold=0.5,
        generator=gen,
    )
    assert torch.all((gate == 0.0) | (gate == 1.0))

    loss = (gate * weights).sum()
    loss.backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert float(u.grad.abs().sum()) > 0.0


def test_ste_backward_matches_sigmoid_derivative_for_linear_loss() -> None:
    u = torch.tensor([-1.2, -0.2, 0.3, 1.1], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([0.6, 1.2, 2.1, 0.4], dtype=torch.float32)
    temperature = 1.0

    p = sigmoid_probabilities(u, temperature=temperature)
    gate = ste_hard_gate(p, mode="threshold", threshold=0.5)
    loss = (gate * weights).sum()
    loss.backward()

    assert u.grad is not None
    expected = weights * p.detach() * (1.0 - p.detach()) / temperature
    assert torch.allclose(u.grad, expected, atol=1e-6, rtol=1e-5)


def test_sigmoid_probabilities_stays_finite_with_extreme_input() -> None:
    u = torch.tensor(
        [float("-inf"), -1e9, float("nan"), 0.0, 1e9, float("inf")],
        dtype=torch.float32,
    )
    p = sigmoid_probabilities(u)
    assert torch.isfinite(p).all()
    assert torch.all((p >= 0.0) & (p <= 1.0))
