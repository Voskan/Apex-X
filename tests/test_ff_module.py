from __future__ import annotations

import torch

from apex_x import ApexXConfig
from apex_x.model import FFModule


def _build_cfg(*, nesting: bool) -> ApexXConfig:
    cfg = ApexXConfig()
    cfg.model.tile_size_l0 = 4
    cfg.model.tile_size_l1 = 2
    cfg.model.tile_size_l2 = 1
    cfg.model.kmax_l0 = 4
    cfg.routing.cost_heavy = 1.0
    cfg.routing.cost_cheap = 0.2
    cfg.routing.budget_b1 = 2.4
    cfg.routing.budget_b2 = 1.0
    cfg.routing.budget_total = 8.0

    if nesting:
        cfg.model.nesting_depth = 1
        cfg.model.disable_nesting = False
        cfg.model.kmax_l1 = 4
        cfg.model.kmax_l2 = 0
        cfg.routing.budget_b1 = 1.6
    else:
        cfg.model.nesting_depth = 0
        cfg.model.disable_nesting = True
        cfg.model.kmax_l1 = 0
        cfg.model.kmax_l2 = 0

    cfg.validate()
    return cfg


def test_ff_module_train_ste_and_expected_cost_cpu() -> None:
    torch.manual_seed(10)
    cfg = _build_cfg(nesting=False)
    module = FFModule(channels=8, config=cfg).cpu()

    dense = torch.randn(1, 8, 16, 16, dtype=torch.float32)
    logits = torch.full((1, 16), -2.0, dtype=torch.float32)
    logits[:, :5] = 2.0

    out = module.forward_train(dense, logits, update_mu=True)

    assert out.heavy_features.shape == dense.shape
    assert out.detail_map.shape == dense.shape
    assert out.alpha.shape == (1, 1, 16, 16)
    assert out.probabilities.shape == (1, 16)
    assert out.gates.shape == (1, 16)
    assert out.expected_cost.ndim == 0
    assert out.budget_loss.ndim == 0
    assert float(out.expected_cost.detach().item()) > 0.0
    assert torch.isfinite(out.heavy_features).all()
    assert torch.isfinite(out.detail_map).all()
    assert "selected_ratios" in out.diagnostics
    assert "budget_usage" in out.diagnostics
    assert len(out.diagnostics["mu_history"]) >= 2


def test_ff_module_infer_deterministic_budgeted_selection_cpu() -> None:
    cfg = _build_cfg(nesting=False)
    module = FFModule(channels=8, config=cfg).cpu()

    dense = torch.randn(1, 8, 16, 16, dtype=torch.float32)
    utilities = torch.arange(16, 0, -1, dtype=torch.float32).reshape(1, 16)

    out_a = module.forward_infer(dense, utilities)
    out_b = module.forward_infer(dense, utilities)

    assert out_a.selected_l0 == out_b.selected_l0
    assert out_a.l0_kmax_buffers == out_b.l0_kmax_buffers
    assert out_a.selected_l0[0] == [0, 1, 2]
    assert out_a.spent_budget_b1 <= cfg.routing.budget_b1 + 1e-9
    assert out_a.heavy_features.shape == dense.shape
    assert out_a.detail_map.shape == dense.shape
    assert "selected_counts" in out_a.diagnostics
    assert "budget_usage" in out_a.diagnostics


def test_ff_module_infer_with_optional_nesting_cpu() -> None:
    cfg = _build_cfg(nesting=True)
    module = FFModule(channels=8, config=cfg).cpu()

    dense = torch.randn(1, 8, 16, 16, dtype=torch.float32)
    utilities = torch.zeros((1, 16), dtype=torch.float32)
    split_utilities = torch.zeros((1, 16), dtype=torch.float32)
    utilities[0, 0] = 5.0
    utilities[0, 1] = 4.0
    split_utilities[0, 0] = 3.0
    split_utilities[0, 1] = 1.0

    out = module.forward_infer(
        dense,
        utilities,
        split_utilities=split_utilities,
        enable_nesting=True,
    )

    assert out.selected_l0[0] == [0, 1]
    assert len(out.selected_l1[0]) == 4
    assert out.spent_budget_b2 <= cfg.routing.budget_b2 + 1e-9
    assert out.heavy_features.shape == dense.shape
    assert out.detail_map.shape == dense.shape
    assert out.diagnostics["selected_counts"]["l1"] == 4
    assert out.diagnostics["nesting_enabled"] is True
