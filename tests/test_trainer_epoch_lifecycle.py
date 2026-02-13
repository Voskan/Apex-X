from __future__ import annotations

import json
from pathlib import Path

from apex_x.config import ApexXConfig
from apex_x.train import ApexXTrainer


def test_trainer_run_multi_epoch_saves_checkpoint_lifecycle(tmp_path: Path) -> None:
    cfg = ApexXConfig()
    cfg.train.output_dir = str(tmp_path / "train_out")
    cfg.train.epochs = 3
    cfg.train.save_interval = 1
    cfg.train.val_interval = 10  # Disable validation path for this lifecycle check.
    cfg.train.allow_synthetic_fallback = True

    trainer = ApexXTrainer(config=cfg, num_classes=3)
    result = trainer.run(steps_per_stage=1, seed=123, enable_budgeting=False)

    ckpt_dir = Path(cfg.train.output_dir) / "checkpoints"
    assert (ckpt_dir / "epoch_0000.pt").exists()
    assert (ckpt_dir / "epoch_0001.pt").exists()
    assert (ckpt_dir / "epoch_0002.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    assert (Path(cfg.train.output_dir) / "train_report.json").exists()
    assert (Path(cfg.train.output_dir) / "train_report.md").exists()
    report_payload = json.loads((Path(cfg.train.output_dir) / "train_report.json").read_text())
    assert "stages" in report_payload
    assert isinstance(report_payload["stages"], list)
    assert "final" in report_payload
    assert "loss_diagnostics" in report_payload["final"]
    assert result.train_summary["epochs"] == 3
    assert trainer.current_epoch == 2


def test_trainer_primary_metric_controls_best_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = ApexXConfig()
    cfg.train.output_dir = str(tmp_path / "train_out")
    cfg.train.epochs = 2
    cfg.train.save_interval = 1
    cfg.train.val_interval = 1
    cfg.train.primary_metric = "val_loss"
    cfg.train.allow_synthetic_fallback = True

    trainer = ApexXTrainer(config=cfg, num_classes=3)

    monkeypatch.setattr(
        trainer,
        "_build_validation_dataloader",
        lambda **kwargs: object(),
    )
    val_metrics = iter([{"val_loss": 2.0}, {"val_loss": 1.0}])
    monkeypatch.setattr(trainer, "validate", lambda **kwargs: next(val_metrics))

    trainer.run(steps_per_stage=1, seed=5, enable_budgeting=False)

    ckpt_dir = Path(cfg.train.output_dir) / "checkpoints"
    assert trainer.best_metric_name == "val_loss"
    assert trainer.best_metric == 1.0
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
