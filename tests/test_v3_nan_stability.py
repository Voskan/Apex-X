from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from apex_x.losses.auxiliary_losses import auxiliary_mask_loss
from apex_x.model.cascade_head import CascadeDetHead
from apex_x.train.train_losses_v3 import compute_v3_training_losses


def test_cascade_apply_deltas_is_finite_with_extreme_inputs() -> None:
    head = CascadeDetHead(in_channels=8, num_classes=3, num_stages=1, iou_thresholds=[0.5])
    boxes = torch.tensor(
        [
            [10.0, 12.0, 50.0, 80.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    deltas = torch.tensor(
        [
            [0.1, -0.1, 120.0, -120.0],
            [float("nan"), float("inf"), float("-inf"), 0.0],
        ],
        dtype=torch.float32,
    )

    refined = head._apply_deltas(boxes, deltas)

    assert torch.isfinite(refined).all()
    assert (refined[:, 2] >= refined[:, 0]).all()
    assert (refined[:, 3] >= refined[:, 1]).all()


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def test_v3_training_losses_sanitize_non_finite_components() -> None:
    model = _DummyModel()
    cfg = SimpleNamespace(
        loss=SimpleNamespace(
            multi_scale_supervision=False,
            boundary_weight=0.5,
            quality_weight=1.0,
            lovasz_weight=0.5,
        ),
    )

    outputs = {
        "scores": torch.tensor(
            [[float("nan"), float("inf"), float("-inf")]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "boxes": torch.tensor(
            [[float("inf"), 3.0, -5.0, float("nan")]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "masks": torch.full((1, 1, 8, 8), float("nan"), dtype=torch.float32, requires_grad=True),
        "predicted_quality": torch.tensor([float("nan")], dtype=torch.float32, requires_grad=True),
        "all_boxes": [
            [torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)],
            [torch.tensor([[float("inf"), 0.0, -1.0, float("nan")]], dtype=torch.float32)],
        ],
    }
    targets = {
        "labels": torch.tensor([1], dtype=torch.long),
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32),
        "masks": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
    }

    total, loss_dict = compute_v3_training_losses(outputs, targets, model, cfg)

    assert torch.isfinite(total)
    assert all(torch.isfinite(v) for v in loss_dict.values())
    total.backward()
    assert outputs["scores"].grad is not None
    assert outputs["boxes"].grad is not None
    assert outputs["masks"].grad is not None


def test_auxiliary_mask_loss_sanitizes_non_finite() -> None:
    aux = [torch.full((2, 8, 8), float("nan"), dtype=torch.float32)]
    target = torch.zeros((2, 8, 8), dtype=torch.float32)
    loss = auxiliary_mask_loss(aux_mask_outputs=aux, target_masks=target, loss_type="dice")
    assert torch.isfinite(loss)
