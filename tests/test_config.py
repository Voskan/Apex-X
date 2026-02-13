from __future__ import annotations

from pathlib import Path

import pytest

from apex_x.config import ApexXConfig, apply_overrides, load_yaml_config


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_yaml_config() -> None:
    cfg = load_yaml_config(
        path=Path(__file__).parent / "fixtures" / "apex_x_config.yaml",
    )

    assert cfg.model.profile == "base"
    assert cfg.model.kmax_l2 == 64
    assert cfg.routing.budget_b3 == 5.0
    assert cfg.train.qat_enable is True
    assert cfg.runtime.precision_profile == "balanced"


def test_apply_cli_overrides() -> None:
    base = ApexXConfig()
    updated = apply_overrides(
        base,
        [
            "model.profile=large",
            "routing.budget_b1=20",
            "routing.theta_on=0.8",
            "train.qat_enable=true",
            "train.qat_fp8=true",
            "runtime.precision_profile='quality'",
        ],
    )

    assert updated.model.profile == "large"
    assert updated.routing.budget_b1 == 20
    assert updated.routing.theta_on == 0.8
    assert updated.train.qat_enable is True
    assert updated.train.qat_fp8 is True
    assert updated.runtime.precision_profile == "quality"


def test_invalid_divisibility_validation() -> None:
    with pytest.raises(ValueError, match="input_height"):
        ApexXConfig.from_dict(
            {
                "model": {
                    "input_height": 130,
                    "input_width": 128,
                }
            }
        )


def test_invalid_budget_validation() -> None:
    with pytest.raises(ValueError, match=r"b1\+b2\+b3"):
        ApexXConfig.from_dict(
            {
                "routing": {
                    "budget_total": 10,
                    "budget_b1": 8,
                    "budget_b2": 4,
                    "budget_b3": 0,
                }
            }
        )


def test_invalid_kmax_validation() -> None:
    with pytest.raises(ValueError, match="kmax_l2"):
        ApexXConfig.from_dict(
            {
                "model": {
                    "nesting_depth": 2,
                    "kmax_l2": 0,
                    "input_height": 128,
                    "input_width": 128,
                },
                "routing": {"budget_b3": 1},
            }
        )


def test_unknown_override_path_raises() -> None:
    with pytest.raises(KeyError, match="Unknown override path"):
        apply_overrides(ApexXConfig(), ["model.nope=1"])


def test_invalid_train_loss_stability_config_raises() -> None:
    with pytest.raises(ValueError, match="loss_boundary_max_scale"):
        ApexXConfig.from_dict({"train": {"loss_boundary_max_scale": 0.9}})

    with pytest.raises(ValueError, match="loss_box_scale_end"):
        ApexXConfig.from_dict(
            {
                "train": {
                    "loss_box_scale_start": 1.5,
                    "loss_box_scale_end": 1.0,
                }
            }
        )
