from __future__ import annotations

import torch

from apex_x.losses import (
    boundary_distance_transform_surrogate_loss,
    instance_segmentation_losses,
    mask_bce_loss,
    mask_dice_loss,
    soft_boundary_distance_transform,
)


def _square_mask(height: int, width: int, *, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def test_bce_and_dice_are_small_for_perfect_logits() -> None:
    target = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    target[0, 0, 2:10, 3:11] = 1.0
    target[0, 1, 4:12, 6:14] = 1.0
    logits = torch.where(target > 0.5, torch.tensor(12.0), torch.tensor(-12.0))

    bce = mask_bce_loss(logits, target)
    dice = mask_dice_loss(logits, target)

    assert float(bce.item()) < 1e-4
    assert float(dice.item()) < 1e-4


def test_soft_boundary_distance_transform_has_larger_center_distance() -> None:
    boundary = torch.zeros((1, 1, 17, 17), dtype=torch.float32)
    boundary[:, :, 8, :] = 1.0
    boundary[:, :, :, 8] = 1.0

    dist = soft_boundary_distance_transform(boundary, iterations=8, temperature=0.25)
    center = float(dist[0, 0, 8, 8].item())
    corner = float(dist[0, 0, 0, 0].item())
    near = float(dist[0, 0, 7, 7].item())

    assert center < near
    assert near < corner


def test_boundary_surrogate_penalizes_shifted_edges_more() -> None:
    target = _square_mask(32, 32, x1=8, y1=8, x2=24, y2=24)[None, None]
    good_logits = torch.where(target > 0.5, torch.tensor(8.0), torch.tensor(-8.0))

    shifted_target = _square_mask(32, 32, x1=11, y1=8, x2=27, y2=24)[None, None]
    bad_logits = torch.where(shifted_target > 0.5, torch.tensor(8.0), torch.tensor(-8.0))

    good = boundary_distance_transform_surrogate_loss(
        good_logits,
        target,
        dt_iterations=8,
        dt_temperature=0.25,
    )
    bad = boundary_distance_transform_surrogate_loss(
        bad_logits,
        target,
        dt_iterations=8,
        dt_temperature=0.25,
    )

    assert float(bad.item()) > float(good.item())


def test_combined_seg_loss_is_finite_and_decreases_on_toy_optimization() -> None:
    torch.manual_seed(0)
    target = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
    target[:, :, 6:18, 7:17] = 1.0

    logits = torch.nn.Parameter(torch.zeros_like(target))
    optimizer = torch.optim.Adam([logits], lr=0.25)
    history: list[float] = []

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        out = instance_segmentation_losses(
            logits,
            target,
            bce_weight=1.0,
            dice_weight=1.0,
            boundary_weight=1.0,
            dt_iterations=6,
            dt_temperature=0.25,
        )
        loss = out.total_loss
        history.append(float(loss.detach().item()))
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        optimizer.step()

    assert torch.isfinite(torch.tensor(history)).all()
    assert history[-1] < history[0]


def test_instance_weights_are_supported() -> None:
    target = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    target[0, 0, 1:5, 1:5] = 1.0
    target[0, 1, 3:7, 3:7] = 1.0
    logits = torch.zeros_like(target, requires_grad=True)
    weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    out = instance_segmentation_losses(logits, target, instance_weights=weights)
    out.total_loss.backward()

    assert torch.isfinite(out.total_loss)
    assert torch.isfinite(out.bce_loss)
    assert torch.isfinite(out.dice_loss)
    assert torch.isfinite(out.boundary_loss)
    assert logits.grad is not None
