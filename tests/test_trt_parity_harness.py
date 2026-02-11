from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from apex_x.runtime import (
    ParityMatrixCase,
    get_parity_tolerance_profile,
    run_parity_matrix_case,
    run_parity_sweep,
)


def _make_backend_fns(
    *,
    quantized_dtype: torch.dtype,
) -> dict[str, Callable[[torch.Tensor], dict[str, torch.Tensor]]]:
    def _ref(x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"y": x * 2.0 + 0.5}

    def _triton(x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = x * 2.0 + 0.5
        return {"y": y.to(dtype=quantized_dtype).to(dtype=torch.float32)}

    def _tensorrt(x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = x * 2.0 + 0.5
        y = y.to(dtype=quantized_dtype).to(dtype=torch.float32)
        return {"y": y + 1e-5}

    return {
        "reference": _ref,
        "triton": _triton,
        "tensorrt": _tensorrt,
    }


def test_run_parity_matrix_case_covers_reference_triton_tensorrt_pairs() -> None:
    profile = get_parity_tolerance_profile("balanced")
    case = ParityMatrixCase(
        name="matrix-basic",
        input_factory=lambda: torch.randn(1, 3, 8, 8, dtype=torch.float32),
        backend_fns=_make_backend_fns(quantized_dtype=torch.float16),
        backend_pairs=(
            ("reference", "triton"),
            ("reference", "tensorrt"),
            ("triton", "tensorrt"),
        ),
    )

    reports = run_parity_matrix_case(
        case,
        seed=11,
        deterministic=True,
        tolerances=profile.e2e_tolerances,
        mismatch_ratio_limit=profile.mismatch_ratio_limit,
    )

    assert len(reports) == 3
    assert all(report.passed for report in reports)
    assert {(report.reference_backend, report.candidate_backend) for report in reports} == {
        ("reference", "triton"),
        ("reference", "tensorrt"),
        ("triton", "tensorrt"),
    }


def test_run_parity_matrix_case_rejects_unknown_backend_pairs() -> None:
    case = ParityMatrixCase(
        name="matrix-invalid-pairs",
        input_factory=lambda: torch.randn(1, 3, 8, 8, dtype=torch.float32),
        backend_fns=_make_backend_fns(quantized_dtype=torch.float16),
        backend_pairs=(("reference", "unknown"),),
    )

    with pytest.raises(ValueError, match="Unknown backend in pair"):
        run_parity_matrix_case(case)


def test_run_parity_sweep_supports_shape_and_precision_matrix() -> None:
    cases = (
        ParityMatrixCase(
            name="shape_8_fp32",
            input_factory=lambda: torch.randn(1, 3, 8, 8, dtype=torch.float32),
            backend_fns=_make_backend_fns(quantized_dtype=torch.float32),
        ),
        ParityMatrixCase(
            name="shape_8_fp16",
            input_factory=lambda: torch.randn(1, 3, 8, 8, dtype=torch.float32),
            backend_fns=_make_backend_fns(quantized_dtype=torch.float16),
        ),
        ParityMatrixCase(
            name="shape_16_fp32",
            input_factory=lambda: torch.randn(1, 3, 16, 16, dtype=torch.float32),
            backend_fns=_make_backend_fns(quantized_dtype=torch.float32),
        ),
        ParityMatrixCase(
            name="shape_16_fp16",
            input_factory=lambda: torch.randn(1, 3, 16, 16, dtype=torch.float32),
            backend_fns=_make_backend_fns(quantized_dtype=torch.float16),
        ),
    )

    sweep = run_parity_sweep(
        sweep_name="trt-parity-e2e",
        cases=cases,
        profile_name="balanced",
        seed=101,
        deterministic=True,
    )

    assert sweep.profile_name == "balanced"
    assert sweep.case_count == 4
    assert len(sweep.reports) == 12
    assert sweep.passed is True
