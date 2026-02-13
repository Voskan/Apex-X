"""Training loop scaffolding for Apex-X."""

from __future__ import annotations

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

__all__ = [
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
