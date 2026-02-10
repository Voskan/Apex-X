from __future__ import annotations

import torch

from apex_x.utils import StableStateSpaceScan


def test_stable_scan_no_nans_with_extreme_inputs() -> None:
    torch.manual_seed(23)
    scan = StableStateSpaceScan(channels=8, min_decay=1e-3, max_decay=0.995)
    tokens = torch.randn(2, 64, 8) * 1e4

    outputs, state, stats = scan(tokens, return_stats=True)

    assert outputs.shape == (2, 64, 8)
    assert state.shape == (2, 8)
    assert torch.isfinite(outputs).all()
    assert torch.isfinite(state).all()
    assert stats.steps == 64
    assert stats.recurrent_updates == 2 * 8 * 64
    assert stats.pairwise_updates == 0

    decay = scan.constrained_decay().detach()
    assert torch.all(decay > 0.0)
    assert torch.all(decay < 1.0)


def test_stable_scan_gradients_flow() -> None:
    torch.manual_seed(3)
    scan = StableStateSpaceScan(channels=4)
    tokens = torch.randn(3, 16, 4, requires_grad=True)

    outputs, state = scan(tokens)
    loss = outputs.square().mean() + state.square().mean()
    loss.backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert float(tokens.grad.abs().sum().item()) > 0.0

    param_grads = [parameter.grad for parameter in scan.parameters() if parameter.grad is not None]
    assert param_grads
    assert all(torch.isfinite(grad).all() for grad in param_grads)
    assert any(float(grad.abs().sum().item()) > 0.0 for grad in param_grads)


def test_stable_scan_reports_linear_work_in_k() -> None:
    scan = StableStateSpaceScan(channels=6)
    updates: list[int] = []
    for steps in (8, 16, 32):
        tokens = torch.randn(1, steps, 6)
        _, _, stats = scan(tokens, return_stats=True)
        assert stats.steps == steps
        assert stats.pairwise_updates == 0
        updates.append(stats.recurrent_updates)

    assert updates[1] == 2 * updates[0]
    assert updates[2] == 2 * updates[1]
