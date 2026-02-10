import math

from apex_x.routing import (
    delta_loss_oracle,
    dual_update,
    expected_cost,
    greedy_utility_per_cost,
    hysteresis_update,
    sigmoid,
    stable_rank_tile_ids,
    ste_gate,
)


def test_sigmoid_and_ste():
    p = sigmoid(0.0)
    assert math.isclose(p, 0.5, rel_tol=1e-6)
    assert ste_gate(p) == 1
    assert ste_gate(0.49) == 0


def test_delta_loss_oracle():
    assert math.isclose(delta_loss_oracle(0.8, 0.5), 0.3, rel_tol=1e-9)


def test_expected_cost_and_dual_update():
    utilities = [0.0, 1.0]
    ec = expected_cost(utilities, c_heavy=1.0, c_cheap=0.0)
    assert 1.2 < ec < 1.3

    mu = dual_update(mu=0.1, exp_cost=2.0, budget=1.0, dual_lr=0.5)
    assert math.isclose(mu, 0.6)


def test_greedy_budgeting():
    selected, spent = greedy_utility_per_cost(
        utilities=[0.1, 0.9, 0.8],
        costs=[1.0, 1.0, 2.0],
        budget=2.0,
        kmax=2,
    )
    assert selected == [1, 0]
    assert math.isclose(spent, 2.0)


def test_hysteresis():
    out = hysteresis_update([0.7, 0.5, 0.3], [0, 1, 1], theta_on=0.6, theta_off=0.4)
    assert out == [1, 1, 0]


def test_stable_rank_tile_ids_tie_breaks_by_tile_id() -> None:
    scores = [0.8, 1.2, 1.2, 0.8, 1.2]
    order = stable_rank_tile_ids(scores)
    assert order == [1, 2, 4, 0, 3]


def test_stable_rank_tile_ids_is_identical_across_runs() -> None:
    scores = [1.0, 2.0, 2.0, 0.5, 2.0, 1.0]
    first = stable_rank_tile_ids(scores)
    for _ in range(100):
        assert stable_rank_tile_ids(scores) == first


def test_greedy_budgeting_uses_stable_tie_breaking() -> None:
    selected, spent = greedy_utility_per_cost(
        utilities=[3.0, 6.0, 2.0],  # all ratios are 2.0
        costs=[1.5, 3.0, 1.0],
        budget=4.0,
        kmax=2,
    )
    assert selected == [0, 2]
    assert math.isclose(spent, 2.5)
