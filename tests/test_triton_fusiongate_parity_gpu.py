from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.fusiongate import fusiongate_dispatch, get_triton_fusiongate_availability
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
from apex_x.utils.repro import seed_all


@pytest.mark.parametrize(
    ("shape", "weights"),
    [
        ((1, 16, 64, 64), (1.0, 1.0, 0.0)),
        ((2, 24, 32, 32), (0.8, 1.2, -0.1)),
    ],
)
def test_triton_fusiongate_alpha_parity_fp16(
    shape: tuple[int, int, int, int],
    weights: tuple[float, float, float],
) -> None:
    availability = get_triton_fusiongate_availability()
    if not availability.available:
        pytest.skip(f"Triton fusiongate unavailable: {availability.reason}")

    b, _, h, w = shape
    seed_all(111, deterministic=True)
    boundary = torch.rand((b, 1, h, w), dtype=torch.float16, device="cuda")
    uncertainty = torch.rand((b, 1, h, w), dtype=torch.float16, device="cuda")
    bw, uw, bias = weights

    ref = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        boundary_log_weight=bw,
        uncertainty_log_weight=uw,
        bias=bias,
        apply_fusion=False,
        prefer_triton=False,
    )
    tri = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        boundary_log_weight=bw,
        uncertainty_log_weight=uw,
        bias=bias,
        apply_fusion=False,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert tri.backend == "triton"
    report = evaluate_parity_outputs(
        case_name="triton-fusiongate-alpha-fp16",
        reference_backend="reference",
        candidate_backend="triton",
        reference_output=ref.alpha,
        candidate_output=tri.alpha,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True
    assert torch.all(tri.alpha >= 0.0)
    assert torch.all(tri.alpha <= 1.0)


def test_triton_fusiongate_fuse_parity_fp16() -> None:
    availability = get_triton_fusiongate_availability()
    if not availability.available:
        pytest.skip(f"Triton fusiongate unavailable: {availability.reason}")

    seed_all(211, deterministic=True)
    base = torch.randn((1, 16, 64, 64), dtype=torch.float16, device="cuda")
    detail = torch.randn((1, 16, 64, 64), dtype=torch.float16, device="cuda")
    boundary = torch.rand((1, 1, 64, 64), dtype=torch.float16, device="cuda")
    uncertainty = torch.rand((1, 1, 64, 64), dtype=torch.float16, device="cuda")

    ref = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        base_features=base,
        detail_features=detail,
        boundary_log_weight=1.0,
        uncertainty_log_weight=1.0,
        bias=0.0,
        apply_fusion=True,
        prefer_triton=False,
    )
    tri = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        base_features=base,
        detail_features=detail,
        boundary_log_weight=1.0,
        uncertainty_log_weight=1.0,
        bias=0.0,
        apply_fusion=True,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert tri.backend == "triton"
    assert tri.fused is not None
    assert ref.fused is not None
    report = evaluate_parity_outputs(
        case_name="triton-fusiongate-fuse-fp16",
        reference_backend="reference",
        candidate_backend="triton",
        reference_output={"alpha": ref.alpha, "fused": ref.fused},
        candidate_output={"alpha": tri.alpha, "fused": tri.fused},
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True

