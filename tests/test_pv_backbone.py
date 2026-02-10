from __future__ import annotations

import pytest
import torch

from apex_x.model import PVBackbone


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (128, 128),
        (160, 192),
        (256, 320),
    ],
)
def test_pv_backbone_feature_shapes_across_sizes(height: int, width: int) -> None:
    model = PVBackbone(
        in_channels=3,
        p3_channels=32,
        p4_channels=64,
        p5_channels=96,
        norm_groups=1,
    )
    x = torch.randn(2, 3, height, width)
    features = model(x)

    assert set(features.keys()) == {"P3", "P4", "P5"}
    assert features["P3"].shape == (2, 32, height // 8, width // 8)
    assert features["P4"].shape == (2, 64, height // 16, width // 16)
    assert features["P5"].shape == (2, 96, height // 32, width // 32)


def test_pv_backbone_gradient_flow() -> None:
    torch.manual_seed(17)
    model = PVBackbone(p3_channels=24, p4_channels=48, p5_channels=96, norm_groups=1)
    x = torch.randn(1, 3, 128, 128, requires_grad=True)

    features = model(x)
    loss = (
        features["P3"].square().mean()
        + features["P4"].square().mean()
        + features["P5"].square().mean()
    )
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum().item()) > 0.0


def test_pv_backbone_input_validation() -> None:
    model = PVBackbone()
    with pytest.raises(ValueError, match="input must be \\[B,C,H,W\\]"):
        _ = model(torch.randn(3, 64, 64))
    with pytest.raises(ValueError, match="input channel dimension"):
        _ = model(torch.randn(1, 1, 64, 64))
    with pytest.raises(ValueError, match="at least 32x32"):
        _ = model(torch.randn(1, 3, 16, 16))
