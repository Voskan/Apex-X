from __future__ import annotations

import torch

from apex_x.kernels.triton.fusiongate import (
    fusiongate_alpha_reference,
    fusiongate_dispatch,
    get_triton_fusiongate_availability,
)
from apex_x.model import FusionGate
from apex_x.runtime import evaluate_parity_outputs
from apex_x.utils.repro import seed_all


def test_fusiongate_alpha_reference_matches_model_compute_alpha() -> None:
    seed_all(7, deterministic=True)
    gate = FusionGate(init_boundary_weight=1.2, init_uncertainty_weight=0.8, init_bias=-0.2)
    boundary = torch.rand((2, 1, 16, 16), dtype=torch.float32)
    uncertainty = torch.rand((2, 1, 16, 16), dtype=torch.float32)

    expected = gate.compute_alpha(boundary, uncertainty, like=boundary)
    actual = fusiongate_alpha_reference(
        boundary,
        uncertainty,
        boundary_log_weight=float(gate.boundary_log_weight.item()),
        uncertainty_log_weight=float(gate.uncertainty_log_weight.item()),
        bias=float(gate.bias.item()),
    )
    report = evaluate_parity_outputs(
        case_name="fusiongate-alpha-reference",
        reference_backend="model_fusiongate",
        candidate_backend="triton_reference_impl",
        reference_output=expected,
        candidate_output=actual,
    )
    assert report.passed is True
    assert torch.all(actual >= 0.0)
    assert torch.all(actual <= 1.0)


def test_fusiongate_dispatch_cpu_fallback_alpha_and_fusion_parity() -> None:
    seed_all(9, deterministic=True)
    gate = FusionGate(init_boundary_weight=0.7, init_uncertainty_weight=1.1, init_bias=0.05)
    base = torch.randn((1, 8, 32, 32), dtype=torch.float32)
    detail = torch.randn((1, 8, 32, 32), dtype=torch.float32)
    boundary = torch.rand((1, 1, 32, 32), dtype=torch.float32)
    uncertainty = torch.rand((1, 1, 32, 32), dtype=torch.float32)

    expected_fused, expected_alpha = gate(base, detail, boundary, uncertainty)
    out = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        base_features=base,
        detail_features=detail,
        boundary_log_weight=float(gate.boundary_log_weight.item()),
        uncertainty_log_weight=float(gate.uncertainty_log_weight.item()),
        bias=float(gate.bias.item()),
        apply_fusion=True,
        prefer_triton=True,
        allow_fallback=True,
    )
    assert out.backend == "reference"
    assert out.fused is not None
    report = evaluate_parity_outputs(
        case_name="fusiongate-dispatch-cpu",
        reference_backend="fusiongate_model",
        candidate_backend="fusiongate_dispatch",
        reference_output={"alpha": expected_alpha, "fused": expected_fused},
        candidate_output={"alpha": out.alpha, "fused": out.fused},
    )
    assert report.passed is True
    assert torch.all(out.alpha >= 0.0)
    assert torch.all(out.alpha <= 1.0)


def test_fusiongate_dispatch_autograd_fallback() -> None:
    seed_all(15, deterministic=True)
    boundary = torch.rand((1, 1, 8, 8), dtype=torch.float32, requires_grad=True)
    uncertainty = torch.rand((1, 1, 8, 8), dtype=torch.float32, requires_grad=True)
    out = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        apply_fusion=False,
        prefer_triton=True,
        allow_fallback=True,
        inference_only=True,
    )
    assert out.backend == "reference"
    assert out.fallback_reason == "autograd_not_supported_for_triton_fusiongate"
    loss = out.alpha.sum()
    loss.backward()
    assert boundary.grad is not None
    assert uncertainty.grad is not None
    assert torch.isfinite(boundary.grad).all()
    assert torch.isfinite(uncertainty.grad).all()


def test_fusiongate_availability_object_cpu_safe() -> None:
    availability = get_triton_fusiongate_availability()
    assert isinstance(availability.available, bool)
    if not availability.available:
        assert availability.reason is not None
