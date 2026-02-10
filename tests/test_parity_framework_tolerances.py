from __future__ import annotations

import pytest
import torch

from apex_x.runtime import NumericTolerance, ToleranceConfig, evaluate_parity_outputs


def test_fp16_tolerance_allows_small_error() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    candidate = torch.tensor([1.001, 2.0, 3.0], dtype=torch.float16)

    report = evaluate_parity_outputs(
        case_name="fp16-small-delta",
        reference_backend="torch_ref",
        candidate_backend="triton",
        reference_output=reference,
        candidate_output=candidate,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True
    assert report.outputs[0].mismatch_count == 0


def test_bf16_tolerance_can_be_overridden() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    candidate = torch.tensor([1.2, 2.0, 3.0], dtype=torch.bfloat16)
    defaults = ToleranceConfig()
    strict = ToleranceConfig(
        default=defaults.default,
        fp16=defaults.fp16,
        bf16=NumericTolerance(atol=1e-3, rtol=1e-3),
        int8=defaults.int8,
    )

    report = evaluate_parity_outputs(
        case_name="bf16-strict",
        reference_backend="torch_ref",
        candidate_backend="triton",
        reference_output=reference,
        candidate_output=candidate,
        tolerances=strict,
    )
    assert report.passed is False
    assert report.outputs[0].mismatch_count == 1


def test_mismatch_ratio_limit_allows_partial_failures() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float32)

    report = evaluate_parity_outputs(
        case_name="ratio-limit",
        reference_backend="torch_ref",
        candidate_backend="candidate",
        reference_output=reference,
        candidate_output=candidate,
        mismatch_ratio_limit=0.30,
    )
    assert report.passed is True
    assert report.mismatch_ratio == 0.25


def test_output_structure_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Output tensor count mismatch"):
        evaluate_parity_outputs(
            case_name="bad-structure",
            reference_backend="torch_ref",
            candidate_backend="candidate",
            reference_output={"a": torch.ones((2,))},
            candidate_output={"a": torch.ones((2,)), "b": torch.ones((2,))},
        )
