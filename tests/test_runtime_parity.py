from __future__ import annotations

import pytest
import torch

from apex_x.runtime import (
    NumericTolerance,
    ParityToleranceProfile,
    ToleranceConfig,
    get_parity_tolerance_profile,
    list_parity_tolerance_profiles,
)


def test_parity_profile_registry_contains_required_profiles() -> None:
    profiles = list_parity_tolerance_profiles()
    assert profiles == ("balanced", "edge", "quality")


def test_get_parity_profile_returns_expected_contract() -> None:
    profile = get_parity_tolerance_profile("quality")
    assert profile.name == "quality"
    assert isinstance(profile.op_tolerances, ToleranceConfig)
    assert isinstance(profile.e2e_tolerances, ToleranceConfig)
    assert profile.mismatch_ratio_limit == 0.0


def test_get_parity_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported parity tolerance profile"):
        get_parity_tolerance_profile("ultra")


def test_parity_profile_validates_mismatch_limit() -> None:
    with pytest.raises(ValueError, match="mismatch_ratio_limit must be in \\[0, 1\\]"):
        ParityToleranceProfile(
            name="invalid",
            op_tolerances=ToleranceConfig(),
            e2e_tolerances=ToleranceConfig(),
            mismatch_ratio_limit=1.1,
        )


def test_tolerance_config_uses_fp8_profile_when_dtype_is_fp8() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if not isinstance(fp8_dtype, torch.dtype):
        pytest.skip("FP8 dtype is unavailable in current torch build.")

    custom_fp8 = NumericTolerance(atol=9e-3, rtol=9e-2)
    cfg = ToleranceConfig(fp8=custom_fp8)
    resolved = cfg.for_dtype(fp8_dtype)
    assert resolved == custom_fp8
