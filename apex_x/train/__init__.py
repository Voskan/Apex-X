"""Training loop scaffolding for Apex-X."""

from __future__ import annotations

from typing import Any

from apex_x.config import ApexXConfig
from apex_x.utils import get_logger, log_event

from .ablation import (
    AblationAggregateRecord,
    AblationRunRecord,
    AblationToggleSet,
    ToggleMode,
    build_ablation_grid,
    run_ablation_grid,
    write_ablation_reports,
)
from .pcgrad import (
    DEFAULT_LOSS_GROUP_ORDER,
    LossGroup,
    PCGradDiagnostics,
    apply_pcgradpp,
    diagnostics_to_dict,
    group_loss_terms,
)
from .qat import (
    ActivationFakeQuant,
    ActivationObserver,
    FakeQuantConv2d,
    FakeQuantLinear,
    QuantizationSummary,
    QuantMode,
    WeightPerChannelFakeQuant,
    calibrate_ptq,
    iter_qat_wrappers,
    prepare_int8_ptq,
    prepare_int8_qat,
    set_qat_state,
)
from .trainer import ApexXTrainer, StagedTrainResult, StageResult

LOGGER = get_logger(__name__)


def train_step_placeholder(
    routing_diagnostics: dict[str, Any] | None = None,
    config: ApexXConfig | None = None,
) -> dict[str, Any]:
    diagnostics = routing_diagnostics or {}

    selected_ratios = diagnostics.get("selected_ratios", {})
    budget_usage = diagnostics.get("budget_usage", {})
    mu_history = diagnostics.get("mu_history", [])
    distill_enabled = config.train.distill_enabled() if config is not None else True
    pcgradpp_enabled = config.train.pcgradpp_enabled() if config is not None else True

    log_event(
        LOGGER,
        "train_step_placeholder",
        level="DEBUG",
        fields={
            "l0_selected_ratio": float(selected_ratios.get("l0", 0.0)),
            "b1_budget_ratio": (
                float(budget_usage.get("b1", {}).get("ratio", 0.0))
                if isinstance(budget_usage.get("b1", {}), dict)
                else 0.0
            ),
            "mu_steps": len(mu_history) if isinstance(mu_history, list) else 0,
            "mu_last": (
                float(mu_history[-1]) if isinstance(mu_history, list) and mu_history else 0.0
            ),
            "distill_enabled": distill_enabled,
            "pcgradpp_enabled": pcgradpp_enabled,
        },
    )

    return {
        "routing_diagnostics": diagnostics,
        "training_toggles": {
            "distill_enabled": distill_enabled,
            "pcgradpp_enabled": pcgradpp_enabled,
        },
    }


__all__ = [
    "train_step_placeholder",
    "ApexXTrainer",
    "StageResult",
    "StagedTrainResult",
    "ToggleMode",
    "AblationToggleSet",
    "AblationRunRecord",
    "AblationAggregateRecord",
    "build_ablation_grid",
    "run_ablation_grid",
    "write_ablation_reports",
    "DEFAULT_LOSS_GROUP_ORDER",
    "LossGroup",
    "PCGradDiagnostics",
    "group_loss_terms",
    "apply_pcgradpp",
    "diagnostics_to_dict",
    "QuantMode",
    "QuantizationSummary",
    "ActivationObserver",
    "ActivationFakeQuant",
    "WeightPerChannelFakeQuant",
    "FakeQuantConv2d",
    "FakeQuantLinear",
    "prepare_int8_qat",
    "prepare_int8_ptq",
    "calibrate_ptq",
    "iter_qat_wrappers",
    "set_qat_state",
]
