from __future__ import annotations

import torch

from apex_x.losses import (
    boundary_distill_loss,
    distillation_losses,
    feature_l2_distill,
    logits_kl_distill,
)


def test_logits_kl_distill_is_near_zero_when_logits_match() -> None:
    teacher = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [0.1, -0.2, 1.2],
        ],
        dtype=torch.float32,
    )
    student = teacher.clone().requires_grad_(True)

    loss = logits_kl_distill(student, teacher, temperature=2.0)
    loss.backward()

    assert float(loss.item()) < 1e-7
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_feature_l2_distill_selected_layers_and_weights() -> None:
    student = {
        "P3": torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        "P4": torch.ones((1, 8, 2, 2), dtype=torch.float32),
    }
    teacher = {
        "P3": torch.ones((1, 8, 4, 4), dtype=torch.float32),
        "P4": torch.ones((1, 8, 2, 2), dtype=torch.float32),
    }
    loss, used_layers = feature_l2_distill(
        student,
        teacher,
        selected_layers=("P3", "P4"),
        layer_weights={"P3": 2.0, "P4": 0.0},
    )
    assert used_layers == ("P3",)
    assert torch.allclose(loss, torch.tensor(1.0, dtype=torch.float32))


def test_boundary_distill_penalizes_shifted_boundaries_more() -> None:
    target = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    target[:, :, 8:24, 8:24] = 1.0
    teacher_logits = torch.where(target > 0.5, torch.tensor(8.0), torch.tensor(-8.0))

    aligned_student = teacher_logits.clone()

    shifted = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    shifted[:, :, 8:24, 11:27] = 1.0
    shifted_student = torch.where(shifted > 0.5, torch.tensor(8.0), torch.tensor(-8.0))

    good = boundary_distill_loss(aligned_student, teacher_logits)
    bad = boundary_distill_loss(shifted_student, teacher_logits)

    assert float(good.item()) < float(bad.item())


def test_combined_distillation_loss_decreases_in_toy_optimization() -> None:
    torch.manual_seed(12)
    teacher_logits = torch.tensor(
        [[1.0, -0.5, 0.2], [-0.7, 0.3, 1.1]],
        dtype=torch.float32,
    )
    student_logits = torch.nn.Parameter(torch.zeros_like(teacher_logits))

    teacher_features = {"P3": torch.randn((1, 8, 8, 8), dtype=torch.float32)}
    student_features_p3 = torch.nn.Parameter(torch.zeros((1, 8, 8, 8), dtype=torch.float32))

    teacher_mask = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
    teacher_mask[:, :, 6:18, 8:16] = 1.0
    teacher_mask_logits = torch.where(teacher_mask > 0.5, torch.tensor(7.0), torch.tensor(-7.0))
    student_mask_logits = torch.nn.Parameter(torch.zeros_like(teacher_mask_logits))

    optimizer = torch.optim.Adam(
        [student_logits, student_features_p3, student_mask_logits],
        lr=0.2,
    )
    history: list[float] = []

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        out = distillation_losses(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features={"P3": student_features_p3},
            teacher_features=teacher_features,
            student_mask_logits=student_mask_logits,
            teacher_mask_logits=teacher_mask_logits,
            temperature=2.0,
            selected_feature_layers=("P3",),
            logits_weight=1.0,
            feature_weight=1.0,
            boundary_weight=1.0,
            dt_iterations=6,
            dt_temperature=0.25,
        )
        history.append(float(out.total_loss.detach().item()))
        out.total_loss.backward()
        optimizer.step()

    assert history[-1] < history[0]
    assert history[-1] <= 0.9 * history[0]
