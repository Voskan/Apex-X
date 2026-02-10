from __future__ import annotations

import pytest
import torch

from apex_x.runtime import (
    ParityCase,
    ToleranceConfig,
    evaluate_parity_outputs,
    format_parity_report,
    run_parity_case,
)


def test_evaluate_parity_outputs_identical_passes() -> None:
    reference = {"a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
    candidate = {"a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}

    report = evaluate_parity_outputs(
        case_name="identical",
        reference_backend="torch_ref",
        candidate_backend="torch_candidate",
        reference_output=reference,
        candidate_output=candidate,
    )

    assert report.passed is True
    assert report.mismatch_count == 0
    assert report.total_count == 3
    assert report.outputs[0].max_abs_err == 0.0


def test_evaluate_parity_outputs_detects_mismatch_and_reports_metrics() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 2.2, 3.0, 4.0], dtype=torch.float32)

    report = evaluate_parity_outputs(
        case_name="mismatch",
        reference_backend="torch_ref",
        candidate_backend="torch_candidate",
        reference_output=reference,
        candidate_output=candidate,
        mismatch_ratio_limit=0.0,
    )

    assert report.passed is False
    assert report.mismatch_count == 1
    assert report.total_count == 4
    assert report.outputs[0].max_abs_err == pytest.approx(0.2, rel=1e-6, abs=1e-6)
    text = format_parity_report(report)
    assert "max_abs=" in text
    assert "mismatch=1/4" in text


def test_run_parity_case_uses_seed_for_determinism() -> None:
    case = ParityCase(
        name="seeded-random",
        input_factory=lambda: torch.rand((2, 3), dtype=torch.float32),
        reference_fn=lambda x: x * 2.0,
        candidate_fn=lambda x: x * 2.0,
        reference_backend="torch_ref",
        candidate_backend="torch_candidate",
    )

    report_a = run_parity_case(case, seed=77, deterministic=True)
    report_b = run_parity_case(case, seed=77, deterministic=True)
    report_c = run_parity_case(case, seed=78, deterministic=True)

    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.seed != report_c.seed
    assert report_c.passed is True


def test_tolerance_config_uses_int8_profile() -> None:
    tolerances = ToleranceConfig()
    reference = torch.tensor([1, 2, 3], dtype=torch.int8)
    candidate = torch.tensor([1, 2, 4], dtype=torch.int8)

    report = evaluate_parity_outputs(
        case_name="int8-check",
        reference_backend="torch_ref",
        candidate_backend="trt_int8",
        reference_output=reference,
        candidate_output=candidate,
        tolerances=tolerances,
    )

    assert report.passed is False
    assert report.outputs[0].atol == 0.0
    assert report.outputs[0].rtol == 0.0
    assert report.outputs[0].mismatch_count == 1
