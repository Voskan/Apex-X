from __future__ import annotations

from pathlib import Path

import pytest

from apex_x.routing import CostModelProtocol, LevelCost, StaticCostModel


def test_cost_model_computations() -> None:
    model = StaticCostModel(
        levels={
            "l0": LevelCost(0.2, 1.0, pack_overhead=0.1, unpack_overhead=0.2, split_overhead=0.3),
            "l1": LevelCost(0.3, 1.5, pack_overhead=0.1, unpack_overhead=0.2, split_overhead=0.4),
            "l2": LevelCost(0.4, 2.0, pack_overhead=0.2, unpack_overhead=0.3, split_overhead=0.5),
        }
    )
    assert isinstance(model, CostModelProtocol)

    assert model.cheap_cost("l0", num_tiles=5) == pytest.approx(1.0)
    assert model.heavy_cost("l0", num_tiles=5, include_pack_unpack=True) == pytest.approx(6.5)
    assert model.heavy_cost("l0", num_tiles=5, include_pack_unpack=False) == pytest.approx(5.0)
    assert model.delta_cost("l0", num_tiles=5, include_pack_unpack=True) == pytest.approx(5.5)
    assert model.split_overhead("l0", num_splits=2) == pytest.approx(0.6)

    expected = model.expected_level_cost("l0", probabilities=[0.0, 0.5, 1.0])
    assert expected == pytest.approx(2.25)

    total = model.total_cost(
        heavy_tiles_by_level={"l0": 2, "l1": 1},
        cheap_tiles_by_level={"l0": 3},
        splits_by_level={"l0": 1, "l1": 2},
    )
    assert total == pytest.approx(6.1)


def test_cost_model_calibration_hook_and_history() -> None:
    model = StaticCostModel()
    before_heavy = model.levels["l1"].c_heavy
    before_pack = model.levels["l1"].pack_overhead

    rec1 = model.apply_empirical_calibration(
        "L1",
        measured_timings={"c_heavy": 2.0, "pack_overhead": 0.4},
        blend=0.5,
        apply=True,
    )
    assert rec1["applied"] is True
    assert model.levels["l1"].c_heavy == pytest.approx(0.5 * before_heavy + 0.5 * 2.0)
    assert model.levels["l1"].pack_overhead == pytest.approx(0.5 * before_pack + 0.5 * 0.4)

    heavy_after_apply = model.levels["l1"].c_heavy
    rec2 = model.apply_empirical_calibration(
        "l1",
        measured_timings={"c_heavy": 5.0},
        blend=1.0,
        apply=False,
    )
    assert rec2["applied"] is False
    assert model.levels["l1"].c_heavy == pytest.approx(heavy_after_apply)
    assert len(model.calibration_history) == 2


def test_cost_model_serialization_roundtrip(tmp_path: Path) -> None:
    model = StaticCostModel()
    model.apply_empirical_calibration(
        "l0",
        measured_timings={"c_heavy": 1.25},
        blend=1.0,
        apply=True,
    )
    out_path = tmp_path / "cost_model.json"
    model.save_json(out_path)

    loaded = StaticCostModel.load_json(out_path)
    assert loaded.to_dict() == model.to_dict()


def test_cost_model_validation_errors() -> None:
    with pytest.raises(ValueError, match="l0, l1, and l2"):
        StaticCostModel(levels={"l0": LevelCost(0.2, 1.0), "l1": LevelCost(0.2, 1.0)})
    with pytest.raises(ValueError, match="unsupported calibration key"):
        StaticCostModel().apply_empirical_calibration("l0", measured_timings={"unknown": 1.0})
    with pytest.raises(ValueError, match="blend"):
        StaticCostModel().apply_empirical_calibration(
            "l0",
            measured_timings={"c_heavy": 1.0},
            blend=1.2,
        )
    with pytest.raises(ValueError, match="must be >= 0"):
        StaticCostModel().cheap_cost("l0", num_tiles=-1)
