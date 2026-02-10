from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from apex_x import ApexXConfig, ApexXModel
from apex_x.config import apply_overrides
from apex_x.tiles import tile_grid_shape
from apex_x.train import train_step_placeholder


def test_disable_nesting_override_forces_depth_zero() -> None:
    cfg = apply_overrides(
        ApexXConfig(),
        [
            "model.nesting_depth=2",
            "model.disable_nesting=true",
            "routing.budget_b3=5.0",
        ],
    )

    assert cfg.model.effective_nesting_depth() == 0
    assert cfg.routing.budget_b3 == 5.0


@pytest.mark.parametrize(
    ("router_off", "no_nesting", "no_ssm", "no_distill", "no_pcgradpp"),
    list(product([False, True], repeat=5)),
)
def test_all_toggle_combinations_smoke(
    router_off: bool,
    no_nesting: bool,
    no_ssm: bool,
    no_distill: bool,
    no_pcgradpp: bool,
) -> None:
    cfg = ApexXConfig()
    cfg.model.input_height = 256
    cfg.model.input_width = 256
    cfg.model.force_dense_routing = router_off
    cfg.model.disable_nesting = no_nesting
    cfg.model.disable_ssm = no_ssm
    cfg.train.disable_distill = no_distill
    cfg.train.disable_pcgradpp = no_pcgradpp
    cfg.validate()

    model = ApexXModel(config=cfg)
    image = np.random.RandomState(7).rand(1, 3, 256, 256).astype(np.float32)
    out = model.forward(image, update_dual=True)

    assert out["merged"].shape == out["ff8"].shape
    assert "routing_diagnostics" in out
    assert "feature_toggles" in out

    feature_toggles = out["feature_toggles"]
    assert feature_toggles["router_off"] is router_off
    assert feature_toggles["no_nesting"] is no_nesting
    assert feature_toggles["no_ssm"] is no_ssm
    assert feature_toggles["no_distill"] is no_distill
    assert feature_toggles["no_pcgradpp"] is (not cfg.train.pcgradpp_enabled())

    diagnostics = out["routing_diagnostics"]
    if no_nesting:
        assert diagnostics["total_counts"]["l1"] == 0
        assert diagnostics["total_counts"]["l2"] == 0

    if router_off:
        grid_h, grid_w = tile_grid_shape(
            out["ff8"].shape[2],
            out["ff8"].shape[3],
            cfg.model.tile_size_l0,
        )
        assert len(out["selected_tiles"]) == grid_h * grid_w

    if no_ssm:
        assert np.allclose(out["ssm_state"], 0.0)

    train_summary = train_step_placeholder(
        routing_diagnostics=diagnostics,
        config=cfg,
    )
    assert train_summary["training_toggles"]["distill_enabled"] is (not no_distill)
    assert train_summary["training_toggles"]["pcgradpp_enabled"] is (not no_pcgradpp)
