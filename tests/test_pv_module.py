from __future__ import annotations

import pytest
import torch

from apex_x.model import PVModule


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (128, 128),
        (192, 160),
        (256, 320),
    ],
)
def test_pv_module_forward_shapes_across_sizes(height: int, width: int) -> None:
    module = PVModule(
        in_channels=3,
        p3_channels=32,
        p4_channels=64,
        p5_channels=96,
        coarse_level="P4",
        coarse_hidden_channels=32,
    )
    x = torch.randn(2, 3, height, width)
    out = module(x)

    assert set(out.features.keys()) == {"P3", "P4", "P5"}
    assert out.features["P3"].shape == (2, 32, height // 8, width // 8)
    assert out.features["P4"].shape == (2, 64, height // 16, width // 16)
    assert out.features["P5"].shape == (2, 96, height // 32, width // 32)

    assert out.coarse.objectness.shape == (2, 1, height // 16, width // 16)
    assert out.coarse.boundary_proxy.shape == (2, 1, height // 16, width // 16)
    assert out.coarse.variance_proxy.shape == (2, 1, height // 16, width // 16)
    assert out.coarse.uncertainty_proxy.shape == (2, 1, height // 16, width // 16)
    assert set(out.proxy_maps.keys()) == {"objectness", "uncertainty", "boundary", "variance"}
    assert out.proxy_maps["objectness"].shape == out.coarse.objectness.shape
    assert out.proxy_maps["uncertainty"].shape == out.coarse.uncertainty_proxy.shape
    assert out.proxy_maps["boundary"].shape == out.coarse.boundary_proxy.shape
    assert out.proxy_maps["variance"].shape == out.coarse.variance_proxy.shape


def test_pv_module_supports_selecting_p3_and_p5_levels() -> None:
    x = torch.randn(1, 3, 160, 160)

    out_p3 = PVModule(
        p3_channels=24,
        p4_channels=48,
        p5_channels=96,
        coarse_level="P3",
    )(x)
    out_p5 = PVModule(
        p3_channels=24,
        p4_channels=48,
        p5_channels=96,
        coarse_level="P5",
    )(x)

    assert out_p3.coarse.objectness.shape[-2:] == (20, 20)
    assert out_p5.coarse.objectness.shape[-2:] == (5, 5)


def test_pv_module_cpu_smoke() -> None:
    torch.manual_seed(11)
    module = PVModule(
        p3_channels=16,
        p4_channels=32,
        p5_channels=64,
        coarse_level="P4",
    )
    x = torch.randn(1, 3, 128, 128)
    out = module(x)

    assert torch.isfinite(out.coarse.objectness).all()
    assert torch.isfinite(out.coarse.boundary_proxy).all()
    assert torch.isfinite(out.coarse.variance_proxy).all()
    assert torch.isfinite(out.coarse.uncertainty_proxy).all()
    assert torch.isfinite(out.proxy_maps["objectness"]).all()
    assert torch.isfinite(out.proxy_maps["uncertainty"]).all()
