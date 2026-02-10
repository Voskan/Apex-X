from __future__ import annotations

import torch

from apex_x.losses import build_simota_targets_for_anchors, det_loss_with_simota


def _decode_boxes_from_deltas(
    anchor_centers_xy: torch.Tensor,
    deltas_raw: torch.Tensor,
) -> torch.Tensor:
    deltas = torch.nn.functional.softplus(deltas_raw)
    x1 = anchor_centers_xy[:, 0] - deltas[:, 0]
    y1 = anchor_centers_xy[:, 1] - deltas[:, 1]
    x2 = anchor_centers_xy[:, 0] + deltas[:, 2]
    y2 = anchor_centers_xy[:, 1] + deltas[:, 3]
    return torch.stack((x1, y1, x2, y2), dim=1)


def test_build_simota_targets_produces_anchor_targets() -> None:
    anchor_centers = torch.tensor(
        [
            [1.0, 1.0],
            [1.4, 1.1],
            [1.8, 1.2],
            [3.0, 1.0],
            [3.2, 1.1],
            [3.6, 1.1],
        ],
        dtype=torch.float32,
    )
    pred_boxes = torch.tensor(
        [
            [0.6, 0.6, 1.4, 1.4],
            [0.8, 0.7, 1.8, 1.5],
            [1.2, 0.8, 2.0, 1.6],
            [2.6, 0.6, 3.4, 1.4],
            [2.8, 0.7, 3.6, 1.5],
            [3.1, 0.7, 3.9, 1.5],
        ],
        dtype=torch.float32,
    )
    pred_cls_logits = torch.tensor(
        [
            [3.0, -2.0],
            [2.0, -1.0],
            [1.0, -0.5],
            [-2.0, 3.0],
            [-1.0, 2.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    gt_boxes = torch.tensor(
        [
            [0.7, 0.7, 1.7, 1.5],
            [2.7, 0.7, 3.7, 1.5],
        ],
        dtype=torch.float32,
    )
    gt_classes = torch.tensor([0, 1], dtype=torch.int64)

    targets = build_simota_targets_for_anchors(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes,
        anchor_centers_xy=anchor_centers,
        gt_boxes_xyxy=gt_boxes,
        gt_classes=gt_classes,
        topk_center=3,
        dynamic_topk=3,
        min_dynamic_k=1,
    )

    assert targets.num_foreground > 0
    assert targets.foreground_mask.shape == (6,)
    assert targets.cls_target.shape == (6, 2)
    assert targets.box_target.shape == (6, 4)
    assert targets.quality_target.shape == (6,)
    assert targets.positive_weights.shape == (6,)
    assert torch.all(targets.matching.matching_matrix.sum(dim=0) <= 1)

    fg_idx = torch.nonzero(targets.foreground_mask, as_tuple=False).flatten()
    bg_idx = torch.nonzero(~targets.foreground_mask, as_tuple=False).flatten()
    assert torch.all(targets.matched_gt_indices[fg_idx] >= 0)
    assert torch.allclose(
        targets.cls_target[fg_idx].sum(dim=1),
        torch.ones_like(fg_idx, dtype=torch.float32),
    )
    if bg_idx.numel() > 0:
        assert torch.allclose(
            targets.cls_target[bg_idx],
            torch.zeros_like(targets.cls_target[bg_idx]),
        )


def test_det_loss_simota_is_stable_for_small_objects_and_crowded_scene() -> None:
    anchor_centers = torch.tensor(
        [
            [5.0, 5.0],
            [5.2, 5.0],
            [5.4, 5.0],
            [5.6, 5.0],
            [5.8, 5.0],
            [6.0, 5.0],
            [6.2, 5.0],
        ],
        dtype=torch.float32,
    )
    pred_cls_logits = torch.zeros((7, 2), dtype=torch.float32)
    pred_boxes = torch.tensor(
        [
            [4.8, 4.7, 5.2, 5.3],
            [5.0, 4.7, 5.4, 5.3],
            [5.2, 4.7, 5.6, 5.3],
            [5.4, 4.7, 5.8, 5.3],
            [5.6, 4.7, 6.0, 5.3],
            [5.8, 4.7, 6.2, 5.3],
            [6.0, 4.7, 6.4, 5.3],
        ],
        dtype=torch.float32,
    )
    pred_quality_logits = torch.zeros((7,), dtype=torch.float32)
    # Two tiny/crowded GTs.
    gt_boxes = torch.tensor(
        [
            [5.05, 4.85, 5.35, 5.15],
            [5.40, 4.85, 5.70, 5.15],
        ],
        dtype=torch.float32,
    )
    gt_classes = torch.tensor([0, 1], dtype=torch.int64)

    out = det_loss_with_simota(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes,
        pred_quality_logits=pred_quality_logits,
        anchor_centers_xy=anchor_centers,
        gt_boxes_xyxy=gt_boxes,
        gt_classes=gt_classes,
        topk_center=4,
        dynamic_topk=4,
        min_dynamic_k=1,
        small_object_boost=2.5,
    )

    assert torch.isfinite(out.total_loss)
    assert torch.isfinite(out.cls_loss)
    assert torch.isfinite(out.box_loss)
    assert torch.isfinite(out.quality_loss)
    assert out.targets.num_foreground > 0
    assert torch.all(out.targets.matching.matching_matrix.sum(dim=0) <= 1)
    fg = out.targets.foreground_mask
    assert torch.all(out.targets.positive_weights[fg] >= 1.0)


def test_toy_training_det_loss_with_simota_decreases() -> None:
    torch.manual_seed(17)
    grid_x = torch.linspace(2.0, 6.0, steps=5)
    grid_y = torch.linspace(2.0, 6.0, steps=5)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    anchor_centers = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    num_anchors = anchor_centers.shape[0]

    gt_boxes = torch.tensor(
        [
            [2.4, 2.4, 3.2, 3.2],
            [2.9, 2.5, 3.7, 3.3],
        ],
        dtype=torch.float32,
    )
    gt_classes = torch.tensor([0, 1], dtype=torch.int64)

    pred_cls_logits = torch.nn.Parameter(torch.full((num_anchors, 2), -2.0, dtype=torch.float32))
    pred_quality_logits = torch.nn.Parameter(torch.full((num_anchors,), -2.0, dtype=torch.float32))
    pred_box_deltas = torch.nn.Parameter(torch.full((num_anchors, 4), 1.0, dtype=torch.float32))

    optimizer = torch.optim.Adam(
        [pred_cls_logits, pred_quality_logits, pred_box_deltas],
        lr=0.08,
    )

    history: list[float] = []
    for _step in range(50):
        optimizer.zero_grad(set_to_none=True)
        pred_boxes = _decode_boxes_from_deltas(anchor_centers, pred_box_deltas)
        out = det_loss_with_simota(
            pred_cls_logits=pred_cls_logits,
            pred_boxes_xyxy=pred_boxes,
            pred_quality_logits=pred_quality_logits,
            anchor_centers_xy=anchor_centers,
            gt_boxes_xyxy=gt_boxes,
            gt_classes=gt_classes,
            topk_center=8,
            dynamic_topk=8,
            min_dynamic_k=1,
            small_object_boost=2.0,
            cls_loss_type="focal",
            quality_loss_type="qfl",
        )
        loss = out.total_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    assert history[-1] < history[0]
    assert history[-1] <= 0.85 * history[0]


def test_det_losses_are_finite_with_extreme_logits_and_tiny_boxes() -> None:
    anchor_centers = torch.tensor(
        [
            [4.0, 4.0],
            [4.2, 4.0],
            [4.4, 4.0],
            [4.6, 4.0],
        ],
        dtype=torch.float32,
    )
    pred_cls_logits = torch.tensor(
        [
            [120.0, -120.0],
            [80.0, -90.0],
            [-70.0, 75.0],
            [-110.0, 130.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_quality_logits = torch.tensor(
        [100.0, -100.0, 90.0, -95.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [
            [3.95, 3.95, 4.05, 4.05],
            [4.15, 3.95, 4.25, 4.05],
            [4.35, 3.95, 4.45, 4.05],
            [4.55, 3.95, 4.65, 4.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    gt_boxes = torch.tensor(
        [
            [4.00, 3.98, 4.08, 4.06],
            [4.38, 3.98, 4.46, 4.06],
        ],
        dtype=torch.float32,
    )
    gt_classes = torch.tensor([0, 1], dtype=torch.int64)

    out = det_loss_with_simota(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes,
        pred_quality_logits=pred_quality_logits,
        anchor_centers_xy=anchor_centers,
        gt_boxes_xyxy=gt_boxes,
        gt_classes=gt_classes,
        topk_center=4,
        dynamic_topk=4,
        min_dynamic_k=1,
        small_object_boost=2.5,
        cls_loss_type="focal",
        quality_loss_type="qfl",
    )

    assert torch.isfinite(out.total_loss)
    assert torch.isfinite(out.cls_loss)
    assert torch.isfinite(out.box_loss)
    assert torch.isfinite(out.quality_loss)

    out.total_loss.backward()
    assert pred_cls_logits.grad is not None
    assert pred_quality_logits.grad is not None
    assert pred_boxes.grad is not None
    assert torch.isfinite(pred_cls_logits.grad).all()
    assert torch.isfinite(pred_quality_logits.grad).all()
    assert torch.isfinite(pred_boxes.grad).all()
