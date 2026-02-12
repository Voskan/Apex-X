from __future__ import annotations

from apex_x.train.checkpoint import CheckpointMetadata
from apex_x.train.trainer_utils import add_train_epoch_method


def test_checkpoint_metadata_to_dict_includes_dynamic_ema_state() -> None:
    metadata = CheckpointMetadata(
        epoch=3,
        step=42,
        best_metric=0.77,
        best_metric_name="mAP_segm",
        timestamp="2026-02-12T00:00:00",
        config={"train": {"epochs": 10}},
        train_metrics={"loss": 1.23},
    )
    metadata.__dict__["ema_state_dict"] = {"ema_det_head": {"weight": [1, 2, 3]}}

    payload = metadata.to_dict()

    assert payload["epoch"] == 3
    assert payload["best_metric_name"] == "mAP_segm"
    assert payload["train_metrics"]["loss"] == 1.23
    assert "ema_state_dict" in payload
    assert "ema_det_head" in payload["ema_state_dict"]


def test_trainer_utils_load_checkpoint_wrapper_returns_metadata_payload() -> None:
    metadata = CheckpointMetadata(
        epoch=1,
        step=5,
        best_metric=0.25,
        best_metric_name="loss_proxy",
        timestamp="2026-02-12T00:00:00",
        config={"runtime": {"backend": "cpu"}},
    )

    class _DummyTrainer:
        def load_training_checkpoint(self, path: str, device: str = "cpu") -> CheckpointMetadata:
            assert path == "dummy.pt"
            assert device == "cpu"
            return metadata

    trainer_cls = add_train_epoch_method(_DummyTrainer)
    trainer = trainer_cls()
    payload = trainer.load_checkpoint("dummy.pt")

    assert payload["epoch"] == 1
    assert payload["step"] == 5
    assert payload["best_metric"] == 0.25
    assert payload["best_metric_name"] == "loss_proxy"
