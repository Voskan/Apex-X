from __future__ import annotations

import torch

from apex_x.model import PVModule


def test_pv_module_cpu_smoke_forward() -> None:
    torch.manual_seed(21)
    module = PVModule(
        in_channels=3,
        p3_channels=16,
        p4_channels=32,
        p5_channels=64,
        coarse_level="P4",
    ).cpu()
    image = torch.randn(1, 3, 128, 128, device="cpu")

    out = module(image)

    assert out.features["P3"].shape == (1, 16, 16, 16)
    assert out.features["P4"].shape == (1, 32, 8, 8)
    assert out.features["P5"].shape == (1, 64, 4, 4)
    assert out.proxy_maps["objectness"].shape == (1, 1, 8, 8)
    assert out.proxy_maps["uncertainty"].shape == (1, 1, 8, 8)
    assert out.proxy_maps["boundary"].shape == (1, 1, 8, 8)
    assert out.proxy_maps["variance"].shape == (1, 1, 8, 8)
    assert torch.isfinite(out.proxy_maps["objectness"]).all()
    assert torch.isfinite(out.proxy_maps["uncertainty"]).all()
    assert torch.isfinite(out.proxy_maps["boundary"]).all()
    assert torch.isfinite(out.proxy_maps["variance"]).all()
