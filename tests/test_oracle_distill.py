from __future__ import annotations

import torch

from apex_x.routing import (
    compute_oracle_delta_targets,
    summarize_oracle_delta_targets,
    utility_oracle_loss,
    utility_ranking_loss,
    utility_regression_loss,
)


def test_oracle_delta_sign_sanity() -> None:
    cheap = torch.tensor([[1.0, 0.5, 0.6]], dtype=torch.float32)
    heavy = torch.tensor([[0.8, 0.7, 0.4]], dtype=torch.float32)
    targets = compute_oracle_delta_targets(cheap, heavy, [0, 1, 2])

    expected = torch.tensor([[0.2, -0.2, 0.2]], dtype=torch.float32)
    assert torch.allclose(targets.delta_targets, expected, atol=1e-6, rtol=1e-6)
    assert float(targets.delta_targets[0, 0].item()) > 0.0
    assert float(targets.delta_targets[0, 1].item()) < 0.0


def test_stop_grad_targets_in_oracle_utility_losses() -> None:
    cheap = torch.tensor([[1.2, 0.9, 0.7]], dtype=torch.float32, requires_grad=True)
    heavy = torch.tensor([[0.8, 1.0, 0.6]], dtype=torch.float32, requires_grad=True)
    targets = compute_oracle_delta_targets(cheap, heavy, [0, 1, 2])

    assert not targets.delta_targets.requires_grad

    utility_logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, requires_grad=True)
    out = utility_oracle_loss(
        utility_logits,
        targets,
        regression_weight=1.0,
        ranking_weight=1.0,
        ranking_margin=0.2,
    )
    out.total_loss.backward()

    assert utility_logits.grad is not None
    assert torch.isfinite(utility_logits.grad).all()
    assert cheap.grad is None
    assert heavy.grad is None


def test_ranking_loss_sign_sanity() -> None:
    cheap = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    heavy = torch.zeros_like(cheap)
    targets = compute_oracle_delta_targets(cheap, heavy, [0, 1, 2])

    good_logits = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    good_loss, good_pairs = utility_ranking_loss(good_logits, targets, margin=0.3)
    bad_loss, bad_pairs = utility_ranking_loss(bad_logits, targets, margin=0.3)

    assert good_pairs == bad_pairs
    assert good_pairs > 0
    assert float(good_loss.item()) < float(bad_loss.item())


def test_regression_loss_uses_sampled_indices_only() -> None:
    cheap = torch.tensor([[2.0, 10.0, 0.0, -1.0]], dtype=torch.float32)
    heavy = torch.tensor([[1.0, 0.0, 0.0, -3.0]], dtype=torch.float32)
    targets = compute_oracle_delta_targets(cheap, heavy, [0, 3])

    utilities = torch.tensor([[1.0, 100.0, 100.0, 1.5]], dtype=torch.float32)
    reg = utility_regression_loss(utilities, targets, loss_type="mse")

    # Targets for selected tiles are [1.0, 2.0]; predictions are [1.0, 1.5]
    assert torch.allclose(reg, torch.tensor(0.125, dtype=torch.float32), atol=1e-7, rtol=1e-7)


def test_oracle_delta_summary_reports_clipping_ratio() -> None:
    raw = torch.tensor([[3.0, -0.5, -2.6, 0.0]], dtype=torch.float32)
    clipped = raw.clamp(min=-2.0, max=2.0)

    summary = summarize_oracle_delta_targets(
        clipped,
        raw_delta_targets=raw,
        clamp_abs=2.0,
    )

    assert summary.count == 4
    assert summary.clipped_count == 2
    assert summary.clipped_ratio == 0.5
    assert summary.abs_p95 >= 2.0
