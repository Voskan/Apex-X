from __future__ import annotations

import torch

from apex_x.model import DualPathFPN, PVBackbone


def test_dual_path_fpn_shapes_cpu() -> None:
    torch.manual_seed(3)
    backbone = PVBackbone(
        in_channels=3,
        p3_channels=80,
        p4_channels=160,
        p5_channels=256,
    ).cpu()
    fpn = DualPathFPN(
        pv_p3_channels=80,
        pv_p4_channels=160,
        pv_p5_channels=256,
        ff_channels=160,
        out_channels=128,
    ).cpu()

    x = torch.randn(2, 3, 128, 128)
    pv = backbone(x)
    ff_high = torch.randn(2, 160, 32, 32)

    out = fpn(pv, ff_high)

    assert set(out.pyramid.keys()) == {"P3", "P4", "P5"}
    assert out.pyramid["P3"].shape == (2, 128, 16, 16)
    assert out.pyramid["P4"].shape == (2, 128, 8, 8)
    assert out.pyramid["P5"].shape == (2, 128, 4, 4)
    assert out.ff_aligned.shape == (2, 128, 16, 16)
    assert torch.isfinite(out.pyramid["P3"]).all()
    assert torch.isfinite(out.pyramid["P4"]).all()
    assert torch.isfinite(out.pyramid["P5"]).all()


def test_dual_path_fpn_ff_branch_changes_fused_p3() -> None:
    torch.manual_seed(7)
    backbone = PVBackbone(
        in_channels=3,
        p3_channels=80,
        p4_channels=160,
        p5_channels=256,
    ).cpu()
    fpn = DualPathFPN(
        pv_p3_channels=80,
        pv_p4_channels=160,
        pv_p5_channels=256,
        ff_channels=64,
        out_channels=96,
    ).cpu()
    x = torch.randn(1, 3, 128, 128)
    pv = backbone(x)

    ff_zero = torch.zeros(1, 64, 16, 16)
    ff_one = torch.ones(1, 64, 16, 16)

    out_zero = fpn(pv, ff_zero)
    out_one = fpn(pv, ff_one)

    diff = (out_zero.pyramid["P3"] - out_one.pyramid["P3"]).abs().mean()
    assert float(diff.item()) > 0.0


def test_dual_path_fpn_gradient_flow_cpu() -> None:
    torch.manual_seed(11)
    fpn = DualPathFPN(
        pv_p3_channels=32,
        pv_p4_channels=64,
        pv_p5_channels=96,
        ff_channels=32,
        out_channels=64,
    ).cpu()
    pv = {
        "P3": torch.randn(2, 32, 16, 16, requires_grad=True),
        "P4": torch.randn(2, 64, 8, 8, requires_grad=True),
        "P5": torch.randn(2, 96, 4, 4, requires_grad=True),
    }
    ff_high = torch.randn(2, 32, 32, 32, requires_grad=True)

    out = fpn(pv, ff_high)
    loss = (
        out.pyramid["P3"].square().mean()
        + out.pyramid["P4"].square().mean()
        + out.pyramid["P5"].square().mean()
        + out.ff_aligned.square().mean()
    )
    loss.backward()

    assert pv["P3"].grad is not None
    assert pv["P4"].grad is not None
    assert pv["P5"].grad is not None
    assert ff_high.grad is not None
    assert torch.isfinite(pv["P3"].grad).all()
    assert torch.isfinite(pv["P4"].grad).all()
    assert torch.isfinite(pv["P5"].grad).all()
    assert torch.isfinite(ff_high.grad).all()

    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in fpn.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0.0
