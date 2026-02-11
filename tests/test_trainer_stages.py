from __future__ import annotations

from apex_x.config import ApexXConfig
from apex_x.train import ApexXTrainer


def test_staged_trainer_runs_all_required_stages() -> None:
    cfg = ApexXConfig()
    trainer = ApexXTrainer(config=cfg, num_classes=3)
    result = trainer.run(steps_per_stage=1, seed=7)

    stage_ids = [stage.stage_id for stage in result.stage_results]
    stage_names = [stage.name for stage in result.stage_results]

    assert stage_ids == [0, 1, 2, 3, 4]
    assert stage_names == [
        "baseline_warmup",
        "teacher_full_compute",
        "oracle_bootstrap",
        "continuous_budgeting",
        "deterministic_emulation",
    ]
    assert result.loss_proxy >= 0.0
    assert result.final_mu >= 0.0
    assert "selected_ratios" in result.routing_diagnostics
    assert "l0" in result.routing_diagnostics["selected_ratios"]
    assert "pcgrad" in result.train_summary
    assert result.train_summary["pcgrad"]["enabled"] is True
    assert (
        result.train_summary["pcgrad"]["conflict_rate_after"]
        <= result.train_summary["pcgrad"]["conflict_rate_before"]
    )
    stage2_metrics = result.stage_results[2].metrics
    assert stage2_metrics["sampled_long_tail_last"] >= 0
    assert 0.0 <= stage2_metrics["oracle_delta_clipped_ratio_last"] <= 1.0


def test_staged_trainer_is_repeatable_with_fixed_seed() -> None:
    cfg = ApexXConfig()

    first = ApexXTrainer(config=cfg, num_classes=3).run(steps_per_stage=1, seed=11)
    second = ApexXTrainer(config=cfg, num_classes=3).run(steps_per_stage=1, seed=11)

    first_l0 = first.stage_results[4].metrics["selected_l0"]
    second_l0 = second.stage_results[4].metrics["selected_l0"]
    assert first_l0 == second_l0
    assert (
        first.routing_diagnostics["selected_counts"]
        == second.routing_diagnostics["selected_counts"]
    )
    assert first.final_mu == second.final_mu


def test_staged_trainer_pcgrad_snapshot_disabled_when_toggle_off() -> None:
    cfg = ApexXConfig()
    cfg.train.disable_pcgradpp = True

    result = ApexXTrainer(config=cfg, num_classes=3).run(steps_per_stage=1, seed=3)

    snapshot = result.train_summary["pcgrad"]
    assert snapshot["enabled"] is False
    assert snapshot["conflict_rate_before"] == 0.0
    assert snapshot["conflict_rate_after"] == 0.0
