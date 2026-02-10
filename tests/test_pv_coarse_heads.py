from __future__ import annotations

import pytest
import torch

from apex_x.model import PVBackbone, PVCoarseHeads


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (128, 128),
        (160, 192),
        (256, 320),
    ],
)
def test_pv_coarse_heads_output_shapes_and_ranges(height: int, width: int) -> None:
    backbone = PVBackbone(p3_channels=32, p4_channels=64, p5_channels=96)
    heads = PVCoarseHeads(in_channels=64, hidden_channels=32)

    x = torch.randn(2, 3, height, width)
    features = backbone(x)
    out = heads(features, level="P4")

    h4 = height // 16
    w4 = width // 16
    assert out.objectness_logits.shape == (2, 1, h4, w4)
    assert out.objectness.shape == (2, 1, h4, w4)
    assert out.boundary_proxy.shape == (2, 1, h4, w4)
    assert out.variance_proxy.shape == (2, 1, h4, w4)
    assert out.uncertainty_proxy.shape == (2, 1, h4, w4)

    assert torch.all(out.objectness >= 0.0) and torch.all(out.objectness <= 1.0)
    assert torch.all(out.boundary_proxy >= 0.0) and torch.all(out.boundary_proxy <= 1.0)
    assert torch.all(out.uncertainty_proxy >= 0.0) and torch.all(out.uncertainty_proxy <= 1.0)
    assert torch.all(out.variance_proxy >= 0.0)


def test_pv_coarse_heads_uncertainty_proxy_definition_is_clear() -> None:
    heads = PVCoarseHeads(in_channels=64, hidden_channels=32)
    features = {"P4": torch.randn(1, 64, 8, 8)}

    with torch.no_grad():
        heads.objectness_head.weight.zero_()
        heads.objectness_head.bias.zero_()

    out_mid = heads(features, level="P4")
    mean_mid = float(out_mid.uncertainty_proxy.mean().detach().item())
    assert mean_mid == pytest.approx(1.0, abs=1e-6)

    with torch.no_grad():
        heads.objectness_head.bias.fill_(8.0)
    out_high = heads(features, level="P4")
    mean_high = float(out_high.uncertainty_proxy.mean().detach().item())
    assert mean_high < 0.01

    with torch.no_grad():
        heads.objectness_head.bias.fill_(-8.0)
    out_low = heads(features, level="P4")
    mean_low = float(out_low.uncertainty_proxy.mean().detach().item())
    assert mean_low < 0.01


def test_pv_coarse_heads_uncertainty_formula_matches_definition() -> None:
    p = torch.tensor([[[[0.1, 0.5, 0.9]]]], dtype=torch.float32)
    expected = 4.0 * p * (1.0 - p)
    got = PVCoarseHeads.uncertainty_from_objectness(p)
    assert torch.allclose(got, expected)


def test_pv_coarse_heads_gradient_flow() -> None:
    backbone = PVBackbone(p3_channels=16, p4_channels=32, p5_channels=64)
    heads = PVCoarseHeads(in_channels=32, hidden_channels=16)
    x = torch.randn(1, 3, 128, 128, requires_grad=True)

    features = backbone(x)
    out = heads(features, level="P4")
    loss = (
        out.objectness.square().mean()
        + out.boundary_proxy.square().mean()
        + out.variance_proxy.square().mean()
        + out.uncertainty_proxy.square().mean()
    )
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum().item()) > 0.0
