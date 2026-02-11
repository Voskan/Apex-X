from __future__ import annotations

import pytest
import torch

from apex_x.config import ApexXConfig
from apex_x.runtime import resolve_precision_policy, runtime_reason_catalog
from apex_x.train import ApexXTrainer


def test_balanced_profile_falls_back_to_fp16_on_cpu() -> None:
    cfg = ApexXConfig()
    cfg.runtime.precision_profile = "balanced"
    cfg.validate()

    policy = resolve_precision_policy(cfg, device="cpu")
    assert policy.fp8_requested is True
    assert policy.fp8_enabled is False
    assert policy.heavy_ops_dtype is torch.float16
    assert policy.fallback_reason == "fp8_requires_cuda"
    assert policy.router_dtype is torch.float16
    assert policy.kan_dtype is torch.float16


def test_balanced_profile_enables_fp8_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build does not expose FP8 dtypes")

    from apex_x.runtime import precision as precision_module

    monkeypatch.setattr(precision_module, "_detect_cuda_fp8_support", lambda device: (True, None))

    cfg = ApexXConfig()
    cfg.runtime.precision_profile = "balanced"
    cfg.validate()

    policy = resolve_precision_policy(cfg, device="cuda:0")
    assert policy.fp8_requested is True
    assert policy.fp8_enabled is True
    assert policy.heavy_ops_dtype is torch.float8_e4m3fn
    assert policy.fallback_reason is None
    assert policy.router_dtype is torch.float16
    assert policy.kan_dtype is torch.float16


def test_trainer_reports_precision_fallback_in_summary() -> None:
    cfg = ApexXConfig()
    cfg.runtime.precision_profile = "balanced"
    cfg.train.qat_enable = False
    cfg.train.qat_int8 = False
    cfg.train.qat_fp8 = False
    cfg.validate()

    result = ApexXTrainer(config=cfg, num_classes=2).run(steps_per_stage=1, seed=5)
    precision = result.train_summary["precision"]

    assert precision["profile"] == "balanced"
    assert precision["heavy_ops_dtype"] == "float16"
    assert precision["fp8_requested"] is True
    assert precision["fp8_enabled"] is False
    assert precision["fallback_reason"] == "fp8_requires_cuda"
    assert precision["router_dtype"] == "float16"
    assert precision["kan_dtype"] == "float16"


def test_fp8_fallback_reason_uses_canonical_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    from apex_x.runtime import precision as precision_module

    monkeypatch.setattr(
        precision_module,
        "_detect_cuda_fp8_support",
        lambda device: (False, "compute_capability_below_sm90"),
    )

    cfg = ApexXConfig()
    cfg.runtime.precision_profile = "balanced"
    cfg.validate()

    policy = resolve_precision_policy(cfg, device="cuda:0")
    assert policy.fp8_requested is True
    assert policy.fp8_enabled is False
    assert policy.fallback_reason in set(runtime_reason_catalog()["fp8"])
