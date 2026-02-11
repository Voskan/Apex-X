from __future__ import annotations

import math

from apex_x.routing import BudgetDualController


def _simulate_closed_loop(
    controller: BudgetDualController,
    *,
    base_cost: float,
    slope: float,
    steps: int,
) -> tuple[list[float], list[float]]:
    costs: list[float] = []
    mus: list[float] = []
    for _ in range(steps):
        expected_cost = max(0.0, base_cost - slope * controller.mu)
        costs.append(expected_cost)
        controller.update_mu(expected_cost=expected_cost)
        mus.append(controller.mu)
    return costs, mus


def test_dual_controller_converges_from_over_budget_trace() -> None:
    controller = BudgetDualController(
        budget=10.0,
        mu_init=0.0,
        mu_lr=0.25,
        mu_min=0.0,
        mu_max=10.0,
        adaptive_lr=True,
        lr_decay=0.01,
        delta_clip=0.75,
        deadband_ratio=0.0,
        error_ema_beta=0.8,
        adaptive_lr_min_scale=0.75,
        adaptive_lr_max_scale=2.5,
    )

    costs, mus = _simulate_closed_loop(controller, base_cost=14.0, slope=2.0, steps=120)
    tail_mean = sum(costs[-20:]) / 20.0
    assert abs(tail_mean - 10.0) < 0.2
    assert all(0.0 <= mu <= 10.0 for mu in mus)
    assert mus[-1] > mus[0]


def test_dual_controller_converges_from_under_budget_trace() -> None:
    controller = BudgetDualController(
        budget=10.0,
        mu_init=4.0,
        mu_lr=0.25,
        mu_min=0.0,
        mu_max=10.0,
        adaptive_lr=True,
        lr_decay=0.01,
        delta_clip=0.75,
        deadband_ratio=0.0,
        error_ema_beta=0.8,
        adaptive_lr_min_scale=0.75,
        adaptive_lr_max_scale=2.5,
    )

    costs, mus = _simulate_closed_loop(controller, base_cost=14.0, slope=2.0, steps=120)
    tail_mean = sum(costs[-20:]) / 20.0
    assert abs(tail_mean - 10.0) < 0.2
    assert all(0.0 <= mu <= 10.0 for mu in mus)
    assert mus[-1] < mus[0]


def test_dual_controller_applies_delta_clip_limit() -> None:
    controller = BudgetDualController(
        budget=5.0,
        mu_init=0.0,
        mu_lr=1.0,
        mu_min=0.0,
        mu_max=10.0,
        adaptive_lr=True,
        delta_clip=0.1,
        error_ema_beta=0.5,
    )

    mu_next = controller.update_mu(expected_cost=100.0)
    assert math.isclose(mu_next, 0.1, rel_tol=1e-9)
    assert math.isclose(controller.last_applied_delta, 0.1, rel_tol=1e-9)
    assert controller.last_raw_delta > controller.last_applied_delta


def test_dual_controller_deadband_prevents_mu_churn_near_target() -> None:
    controller = BudgetDualController(
        budget=10.0,
        mu_init=1.0,
        mu_lr=0.5,
        mu_min=0.0,
        mu_max=10.0,
        deadband_ratio=0.02,
    )

    mu_before = controller.mu
    mu_after = controller.update_mu(expected_cost=10.1)  # 1% error, inside deadband
    assert math.isclose(mu_after, mu_before, rel_tol=1e-12)
    assert math.isclose(controller.last_applied_delta, 0.0, rel_tol=1e-12)
