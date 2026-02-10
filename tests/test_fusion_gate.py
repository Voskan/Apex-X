from __future__ import annotations

import torch

from apex_x.model import FusionGate


def test_fusion_gate_alpha_shape_range_and_fused_formula() -> None:
    gate = FusionGate()
    base = torch.randn(2, 4, 8, 8)
    heavy = torch.randn(2, 4, 8, 8)
    boundary = torch.rand(2, 1, 8, 8)
    uncertainty = torch.rand(2, 1, 8, 8)

    fused, alpha = gate(base, heavy, boundary, uncertainty)

    assert alpha.shape == (2, 1, 8, 8)
    assert fused.shape == base.shape
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)
    assert torch.allclose(fused, base + alpha * (heavy - base))


def test_fusion_gate_alpha_is_sensitive_to_boundary_and_uncertainty() -> None:
    gate = FusionGate(init_boundary_weight=1.2, init_uncertainty_weight=0.8, init_bias=-0.5)
    base = torch.zeros(1, 2, 4, 4)
    heavy = torch.ones(1, 2, 4, 4)

    low_boundary = torch.zeros(1, 1, 4, 4)
    high_boundary = torch.ones(1, 1, 4, 4)
    low_uncertainty = torch.zeros(1, 1, 4, 4)
    high_uncertainty = torch.ones(1, 1, 4, 4)

    _, alpha_low = gate(base, heavy, low_boundary, low_uncertainty)
    _, alpha_high_boundary = gate(base, heavy, high_boundary, low_uncertainty)
    _, alpha_high_uncertainty = gate(base, heavy, low_boundary, high_uncertainty)
    _, alpha_high_both = gate(base, heavy, high_boundary, high_uncertainty)

    alpha_low_mean = float(alpha_low.mean().detach().item())
    alpha_high_boundary_mean = float(alpha_high_boundary.mean().detach().item())
    alpha_high_uncertainty_mean = float(alpha_high_uncertainty.mean().detach().item())
    alpha_high_both_mean = float(alpha_high_both.mean().detach().item())

    assert alpha_high_boundary_mean > alpha_low_mean
    assert alpha_high_uncertainty_mean > alpha_low_mean
    assert alpha_high_both_mean > alpha_high_boundary_mean
    assert alpha_high_both_mean > alpha_high_uncertainty_mean
