from __future__ import annotations

from pathlib import Path

import pytest

from apex_x import ApexXConfig
from apex_x.train import ApexXTrainer


def _build_config(*, allow_synthetic_fallback: bool, dataset_root: Path) -> ApexXConfig:
    cfg = ApexXConfig()
    cfg.train.allow_synthetic_fallback = allow_synthetic_fallback
    cfg.train.dataloader_num_workers = 0
    cfg.data.dataset_type = "yolo"
    cfg.data.dataset_root = str(dataset_root)
    cfg.validate()
    return cfg


def test_training_dataloader_fails_fast_when_synthetic_fallback_disabled(tmp_path: Path) -> None:
    cfg = _build_config(
        allow_synthetic_fallback=False,
        dataset_root=tmp_path / "missing_dataset",
    )
    trainer = ApexXTrainer(config=cfg)

    with pytest.raises(RuntimeError, match="allow_synthetic_fallback=false"):
        trainer._build_training_dataloader(train_h=128, train_w=128, dataset_path=None)


def test_training_dataloader_uses_synthetic_when_explicitly_enabled(tmp_path: Path) -> None:
    cfg = _build_config(
        allow_synthetic_fallback=True,
        dataset_root=tmp_path / "missing_dataset",
    )
    trainer = ApexXTrainer(config=cfg)

    dataloader, steps_per_epoch, dataset_type = trainer._build_training_dataloader(
        train_h=128,
        train_w=128,
        dataset_path=None,
    )

    assert dataloader is None
    assert steps_per_epoch == 1000
    assert dataset_type == "synthetic"
