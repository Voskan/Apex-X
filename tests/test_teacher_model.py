from __future__ import annotations

import torch

from apex_x.model import (
    DetHead,
    DualPathFPN,
    PVModule,
    TeacherModel,
    flatten_logits_for_distill,
)


def _build_small_teacher(*, use_ema: bool, use_ema_for_forward: bool = True) -> TeacherModel:
    pv_module = PVModule(
        in_channels=3,
        p3_channels=16,
        p4_channels=24,
        p5_channels=32,
        coarse_level="P4",
    )
    fpn = DualPathFPN(
        pv_p3_channels=16,
        pv_p4_channels=24,
        pv_p5_channels=32,
        ff_channels=16,
        out_channels=16,
    )
    det_head = DetHead(
        in_channels=16,
        num_classes=5,
        hidden_channels=16,
        depth=1,
    )
    return TeacherModel(
        num_classes=5,
        pv_module=pv_module,
        fpn=fpn,
        det_head=det_head,
        feature_layers=("P3", "P4"),
        use_ema=use_ema,
        ema_decay=0.9,
        use_ema_for_forward=use_ema_for_forward,
    )


def test_teacher_model_outputs_standardized_distill_contract() -> None:
    torch.manual_seed(1)
    teacher = _build_small_teacher(use_ema=False, use_ema_for_forward=False)
    image = torch.randn((2, 3, 64, 64), dtype=torch.float32)

    out = teacher(image)

    assert out.logits.ndim == 2
    assert out.logits.shape[0] == 2
    assert out.logits.shape[1] > 0
    assert set(out.features.keys()) == {"P3", "P4"}
    assert out.boundaries.shape == (2, 1, 64, 64)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.boundaries).all()
    assert float(out.boundaries.min().item()) >= 0.0
    assert float(out.boundaries.max().item()) <= 1.0


def test_flatten_logits_for_distill_uses_deterministic_level_order() -> None:
    p3 = torch.full((1, 2, 2, 2), 3.0, dtype=torch.float32)
    p5 = torch.full((1, 2, 1, 1), 5.0, dtype=torch.float32)
    extra = torch.full((1, 1, 1, 2), 7.0, dtype=torch.float32)
    logits_by_level = {"P5": p5, "X1": extra, "P3": p3}

    out = flatten_logits_for_distill(logits_by_level)
    expected = torch.cat(
        [p3.reshape(1, -1), p5.reshape(1, -1), extra.reshape(1, -1)],
        dim=1,
    )
    assert torch.allclose(out, expected)


def test_teacher_model_ema_updates_and_forward_switch() -> None:
    torch.manual_seed(2)
    teacher = _build_small_teacher(use_ema=True, use_ema_for_forward=True)
    image = torch.randn((1, 3, 64, 64), dtype=torch.float32)

    out_ema_before = teacher(image, use_ema=True).logits.detach()
    out_online_before = teacher(image, use_ema=False).logits.detach()
    assert torch.allclose(out_ema_before, out_online_before, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        for param in teacher.det_head.parameters():
            param.add_(0.2)
            break

    out_online_after = teacher(image, use_ema=False).logits.detach()
    out_ema_stale = teacher(image, use_ema=True).logits.detach()
    assert not torch.allclose(out_online_after, out_online_before, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_ema_stale, out_ema_before, atol=1e-6, rtol=1e-6)

    teacher.update_ema(decay=0.5)
    out_ema_after = teacher(image, use_ema=True).logits.detach()

    dist_before = torch.mean(torch.abs(out_ema_stale - out_online_after))
    dist_after = torch.mean(torch.abs(out_ema_after - out_online_after))
    assert float(dist_after.item()) < float(dist_before.item())

    assert teacher.ema_det_head is not None
    for parameter in teacher.ema_det_head.parameters():
        assert parameter.requires_grad is False
