from __future__ import annotations

import math

import torch
from torch import nn

from apex_x.config import ApexXConfig
from apex_x.train import (
    ApexXTrainer,
    FakeQuantConv2d,
    FakeQuantLinear,
    iter_qat_wrappers,
    prepare_int8_ptq,
    prepare_int8_qat,
)


class _TinyQATNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.router = nn.Conv2d(8, 8, kernel_size=1)
        self.gating = nn.Conv2d(8, 8, kernel_size=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.stem(x))
        y = y + 0.05 * self.router(y)
        y = y * torch.sigmoid(self.gating(y))
        return self.fc(y.mean(dim=(2, 3)))


def test_prepare_int8_qat_skips_router_and_gating_modules() -> None:
    torch.manual_seed(0)
    model = _TinyQATNet()
    summary = prepare_int8_qat(model)

    assert summary.mode == "qat_int8"
    assert summary.router_gating_fp16 is True
    assert summary.wrapped_modules >= 2
    assert isinstance(model.stem, FakeQuantConv2d)
    assert isinstance(model.fc, FakeQuantLinear)
    assert isinstance(model.router, nn.Conv2d)
    assert isinstance(model.gating, nn.Conv2d)

    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(loss)


def test_prepare_int8_ptq_calibration_disables_observers_after_calibration() -> None:
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(6, 6, kernel_size=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(6, 3),
    )
    calibration_inputs = [torch.randn(1, 3, 16, 16) for _ in range(3)]
    summary = prepare_int8_ptq(
        model,
        calibration_inputs=calibration_inputs,
        forward_fn=lambda module, batch: module(batch),
    )

    assert summary.mode == "ptq_int8"
    assert summary.calibration_batches == 3
    assert summary.wrapped_modules == 3
    wrappers = list(iter_qat_wrappers(model))
    assert len(wrappers) == 3
    for wrapped in wrappers:
        assert wrapped.observer_enabled is False
        assert wrapped.fake_quant_enabled is True

    out = model(torch.randn(1, 3, 16, 16))
    assert torch.isfinite(out).all()


def test_trainer_qat_and_ptq_modes_are_finite() -> None:
    qat_cfg = ApexXConfig()
    qat_cfg.train.qat_enable = True
    qat_cfg.train.qat_int8 = True
    qat_cfg.runtime.precision_profile = "balanced"
    qat_cfg.validate()

    qat_result = ApexXTrainer(config=qat_cfg, num_classes=2).run(steps_per_stage=1, seed=6)
    qat_quant = qat_result.train_summary["quantization"]
    assert qat_quant["mode"] == "qat_int8"
    assert int(qat_quant["wrapped_modules"]) > 0
    assert bool(qat_quant["router_gating_fp16"]) is True
    assert math.isfinite(qat_result.loss_proxy)

    ptq_cfg = ApexXConfig()
    ptq_cfg.train.qat_enable = False
    ptq_cfg.train.qat_int8 = False
    ptq_cfg.runtime.precision_profile = "edge"
    ptq_cfg.validate()

    ptq_result = ApexXTrainer(config=ptq_cfg, num_classes=2).run(steps_per_stage=1, seed=6)
    ptq_quant = ptq_result.train_summary["quantization"]
    assert ptq_quant["mode"] == "ptq_int8"
    assert int(ptq_quant["calibration_batches"]) > 0
    assert bool(ptq_quant["router_gating_fp16"]) is True
    assert math.isfinite(ptq_result.loss_proxy)
