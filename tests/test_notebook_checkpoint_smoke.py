from __future__ import annotations

import pytest
import torch

from scripts.notebook_checkpoint_smoke import (
    _infer_model_family,
    _infer_num_classes,
    _parse_devices,
)


def test_parse_devices_deduplicates_and_preserves_order() -> None:
    assert _parse_devices("cpu,cuda,cpu") == ["cpu", "cuda"]


def test_parse_devices_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        _parse_devices("cpu,tpu")


def test_infer_model_family_from_markers() -> None:
    state_dict: dict[str, torch.Tensor] = {"backbone.conv.weight": torch.zeros((1, 1, 1, 1))}
    assert _infer_model_family(state_dict, "auto") == "teacher_v3"


def test_infer_num_classes_teacher_and_teacher_v3() -> None:
    teacher_sd = {"det_head.cls_pred.weight": torch.zeros((7, 16, 1, 1))}
    v3_sd = {"det_head.stages.0.cls_head.4.weight": torch.zeros((24, 32))}
    assert _infer_num_classes(teacher_sd, "teacher") == 7
    assert _infer_num_classes(v3_sd, "teacher_v3") == 24
