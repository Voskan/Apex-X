from __future__ import annotations

import numpy as np

from apex_x import ApexXModel
from apex_x.infer import extract_routing_diagnostics


def test_infer_output_contains_routing_diagnostics() -> None:
    model = ApexXModel()
    image = np.random.RandomState(123).rand(1, 3, 128, 128).astype(np.float32)
    out = model.forward(image)

    assert "routing_diagnostics" in out
    diag = out["routing_diagnostics"]
    assert "selected_ratios" in diag
    assert "utility_histograms" in diag
    assert "budget_usage" in diag
    assert "mu_history" in diag
    assert "l0" in diag["selected_ratios"]
    assert "l0" in diag["utility_histograms"]
    assert "b1" in diag["budget_usage"]
    assert isinstance(diag["mu_history"], list)
    assert len(diag["mu_history"]) >= 1


def test_extract_routing_diagnostics_surfaces_model_diagnostics() -> None:
    model = ApexXModel()
    image = np.random.RandomState(9).rand(1, 3, 128, 128).astype(np.float32)
    out = model.forward(image, update_dual=True)

    infer_diag = extract_routing_diagnostics(out)

    assert "selected_ratios" in infer_diag
    assert "budget_usage" in infer_diag
    assert "mu_history" in infer_diag
    assert len(infer_diag["mu_history"]) >= 2
