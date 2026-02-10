from __future__ import annotations

import torch

from apex_x.model import DetHead


def test_det_head_outputs_p3_to_p7_shapes_from_p3_to_p5_input() -> None:
    torch.manual_seed(1)
    head = DetHead(in_channels=128, num_classes=20, hidden_channels=96, depth=2).cpu()
    features = {
        "P3": torch.randn(2, 128, 16, 16),
        "P4": torch.randn(2, 128, 8, 8),
        "P5": torch.randn(2, 128, 4, 4),
    }

    out = head(features)
    assert set(out.cls_logits.keys()) == {"P3", "P4", "P5", "P6", "P7"}
    assert set(out.box_reg.keys()) == {"P3", "P4", "P5", "P6", "P7"}
    assert set(out.quality.keys()) == {"P3", "P4", "P5", "P6", "P7"}

    expected_shapes = {
        "P3": (2, 16, 16),
        "P4": (2, 8, 8),
        "P5": (2, 4, 4),
        "P6": (2, 2, 2),
        "P7": (2, 1, 1),
    }
    for level, (_, h, w) in expected_shapes.items():
        assert out.cls_logits[level].shape == (2, 20, h, w)
        assert out.box_reg[level].shape == (2, 4, h, w)
        assert out.quality[level].shape == (2, 1, h, w)
        assert out.features[level].shape[0] == 2
        assert out.features[level].shape[2:] == (h, w)


def test_det_head_uses_provided_p6_p7_if_available() -> None:
    head = DetHead(in_channels=64, num_classes=5, hidden_channels=64, depth=1).cpu()
    p6 = torch.randn(1, 64, 2, 2)
    p7 = torch.randn(1, 64, 1, 1)
    features = {
        "P3": torch.randn(1, 64, 16, 16),
        "P4": torch.randn(1, 64, 8, 8),
        "P5": torch.randn(1, 64, 4, 4),
        "P6": p6,
        "P7": p7,
    }
    out = head(features)
    assert torch.allclose(out.features["P6"], p6)
    assert torch.allclose(out.features["P7"], p7)


def test_det_head_gradient_flow_cpu() -> None:
    torch.manual_seed(9)
    head = DetHead(in_channels=96, num_classes=7, hidden_channels=96, depth=2).cpu()
    features = {
        "P3": torch.randn(2, 96, 16, 16, requires_grad=True),
        "P4": torch.randn(2, 96, 8, 8, requires_grad=True),
        "P5": torch.randn(2, 96, 4, 4, requires_grad=True),
    }

    out = head(features)
    loss = 0.0
    for level in ("P3", "P4", "P5", "P6", "P7"):
        loss = (
            loss
            + out.cls_logits[level].square().mean()
            + out.box_reg[level].square().mean()
            + out.quality[level].square().mean()
        )
    loss.backward()

    assert features["P3"].grad is not None
    assert features["P4"].grad is not None
    assert features["P5"].grad is not None
    assert torch.isfinite(features["P3"].grad).all()
    assert torch.isfinite(features["P4"].grad).all()
    assert torch.isfinite(features["P5"].grad).all()

    param_grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in head.parameters()
        if parameter.grad is not None
    )
    assert param_grad_sum > 0.0
