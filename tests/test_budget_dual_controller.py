from __future__ import annotations

import math

import torch

from apex_x.routing import BudgetDualController


def test_expected_cost_and_budget_loss_for_sequence() -> None:
    controller = BudgetDualController(
        budget=2.0,
        mu_init=0.5,
        mu_lr=0.1,
        mu_min=0.0,
        mu_max=5.0,
    )

    expected = controller.expected_cost([0.0, 0.5, 1.0], c_heavy=2.0, c_cheap=0.0)
    assert isinstance(expected, float)
    assert math.isclose(expected, 3.0, rel_tol=1e-9)

    loss = controller.budget_loss(expected)
    assert isinstance(loss, float)
    assert math.isclose(loss, 0.5 * (3.0 - 2.0), rel_tol=1e-9)


def test_mu_update_moves_in_correct_direction() -> None:
    controller = BudgetDualController(
        budget=10.0,
        mu_init=1.0,
        mu_lr=0.2,
        mu_min=0.0,
        mu_max=5.0,
    )

    mu_up = controller.update_mu(expected_cost=12.0)
    assert math.isclose(mu_up, 1.4, rel_tol=1e-9)

    mu_down = controller.update_mu(expected_cost=9.0)
    assert math.isclose(mu_down, 1.2, rel_tol=1e-9)


def test_mu_update_respects_clamp_bounds() -> None:
    controller = BudgetDualController(
        budget=5.0,
        mu_init=0.4,
        mu_lr=1.0,
        mu_min=0.1,
        mu_max=0.6,
    )

    assert math.isclose(controller.update_mu(expected_cost=10.0), 0.6, rel_tol=1e-9)
    assert math.isclose(controller.update_mu(expected_cost=0.0), 0.1, rel_tol=1e-9)


def test_budget_loss_backpropagates_for_tensor_expected_cost() -> None:
    controller = BudgetDualController(
        budget=1.5,
        mu_init=0.3,
        mu_lr=0.1,
        mu_min=0.0,
        mu_max=5.0,
    )

    probs = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32, requires_grad=True)
    expected = controller.expected_cost(probs, c_heavy=2.0, c_cheap=0.5)
    assert isinstance(expected, torch.Tensor)

    loss = controller.budget_loss(expected)
    assert isinstance(loss, torch.Tensor)
    loss.backward()

    assert probs.grad is not None
    assert torch.isfinite(probs.grad).all()
    assert float(probs.grad.abs().sum()) > 0.0
