from __future__ import annotations

import torch

from apex_x.utils import StableBidirectionalStateSpaceScan


def test_bidirectional_scan_no_nans_and_gate_range() -> None:
    torch.manual_seed(101)
    scan = StableBidirectionalStateSpaceScan(channels=8, min_decay=1e-3, max_decay=0.995)
    tokens = torch.randn(2, 64, 8) * 1e4

    merged, state_f, state_b, stats = scan(tokens, return_stats=True)

    assert merged.shape == (2, 64, 8)
    assert state_f.shape == (2, 8)
    assert state_b.shape == (2, 8)
    assert torch.isfinite(merged).all()
    assert torch.isfinite(state_f).all()
    assert torch.isfinite(state_b).all()
    gate = scan.merge_gate().detach()
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)
    assert stats.steps == 64
    assert stats.pairwise_updates == 0
    assert stats.recurrent_updates == 2 * 2 * 8 * 64


def test_bidirectional_scan_merge_formula_matches_gate() -> None:
    torch.manual_seed(13)
    scan = StableBidirectionalStateSpaceScan(channels=4)
    tokens = torch.randn(1, 12, 4)

    forward_out, forward_state = scan.forward_scan(tokens, return_stats=False)
    backward_rev, backward_state = scan.backward_scan(
        torch.flip(tokens, dims=(1,)),
        return_stats=False,
    )
    backward_out = torch.flip(backward_rev, dims=(1,))

    merged, state_f, state_b = scan(tokens, return_stats=False)
    gate = scan.merge_gate().reshape(1, 1, -1)
    expected = gate * forward_out + (1.0 - gate) * backward_out

    assert torch.allclose(merged, expected)
    assert torch.allclose(state_f, forward_state)
    assert torch.allclose(state_b, backward_state)


def test_bidirectional_scan_gradients_flow() -> None:
    torch.manual_seed(7)
    scan = StableBidirectionalStateSpaceScan(channels=6)
    tokens = torch.randn(3, 20, 6, requires_grad=True)

    merged, state_f, state_b = scan(tokens)
    loss = merged.square().mean() + state_f.square().mean() + state_b.square().mean()
    loss.backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert float(tokens.grad.abs().sum().item()) > 0.0

    param_grads = [parameter.grad for parameter in scan.parameters() if parameter.grad is not None]
    assert param_grads
    assert all(torch.isfinite(grad).all() for grad in param_grads)
    assert any(float(grad.abs().sum().item()) > 0.0 for grad in param_grads)


def test_bidirectional_scan_reports_linear_work_in_k() -> None:
    scan = StableBidirectionalStateSpaceScan(channels=5)
    updates: list[int] = []
    for steps in (8, 16, 32):
        tokens = torch.randn(1, steps, 5)
        _, _, _, stats = scan(tokens, return_stats=True)
        assert stats.steps == steps
        assert stats.pairwise_updates == 0
        updates.append(stats.recurrent_updates)

    assert updates[1] == 2 * updates[0]
    assert updates[2] == 2 * updates[1]
