from __future__ import annotations

import torch

from apex_x.losses import (
    center_prior_cost,
    classification_cost,
    compute_simota_cost,
    dynamic_k_from_top_ious,
    dynamic_k_matching,
    iou_cost,
    topk_center_candidates,
)


def test_topk_center_candidates_per_gt() -> None:
    anchor_centers = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gt_boxes = torch.tensor(
        [
            [0.6, -0.5, 1.6, 0.5],  # center ~ (1.1, 0.0)
            [2.7, -0.5, 3.7, 0.5],  # center ~ (3.2, 0.0)
        ],
        dtype=torch.float32,
    )

    idx = topk_center_candidates(anchor_centers, gt_boxes, topk=2)

    assert idx.shape == (2, 2)
    assert idx.tolist()[0] == [1, 2]
    assert idx.tolist()[1] == [3, 4]


def test_classification_cost_prefers_higher_positive_logit() -> None:
    logits = torch.tensor(
        [
            [-1.0, -2.0],
            [-1.0, 0.0],
            [-1.0, 3.0],
        ],
        dtype=torch.float32,
    )  # [N=3, C=2]
    gt_classes = torch.tensor([1], dtype=torch.int64)

    bce_cost = classification_cost(logits, gt_classes, mode="bce")[0]
    focal_cost = classification_cost(logits, gt_classes, mode="focal")[0]

    assert bce_cost[2] < bce_cost[1] < bce_cost[0]
    assert focal_cost[2] < focal_cost[1] < focal_cost[0]


def test_iou_cost_is_one_minus_iou() -> None:
    pred_boxes = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],  # perfect overlap
            [0.5, 0.5, 2.5, 2.5],  # partial overlap
        ],
        dtype=torch.float32,
    )
    gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)

    cost = iou_cost(pred_boxes, gt_boxes)
    assert cost.shape == (1, 2)
    assert torch.isclose(cost[0, 0], torch.tensor(0.0))
    assert cost[0, 1] > 0.0


def test_compute_simota_cost_ranks_reasonable_anchor_on_synthetic_setup() -> None:
    pred_cls_logits = torch.tensor(
        [
            [4.0],  # anchor 0: high cls, should win with best IoU/center
            [1.0],  # anchor 1: medium cls and IoU
            [4.0],  # anchor 2: high cls but far center and bad IoU
        ],
        dtype=torch.float32,
    )
    pred_boxes = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],  # IoU=1
            [0.4, 0.0, 2.4, 2.0],  # moderate IoU
            [3.0, 3.0, 5.0, 5.0],  # IoU~0
        ],
        dtype=torch.float32,
    )
    anchor_centers = torch.tensor(
        [
            [1.0, 1.0],
            [1.2, 1.0],
            [4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
    gt_classes = torch.tensor([0], dtype=torch.int64)

    out = compute_simota_cost(
        pred_cls_logits=pred_cls_logits,
        pred_boxes_xyxy=pred_boxes,
        anchor_centers_xy=anchor_centers,
        gt_boxes_xyxy=gt_boxes,
        gt_classes=gt_classes,
        topk_center=2,
        classification_mode="focal",
        cls_weight=1.0,
        iou_weight=3.0,
        center_weight=1.0,
        non_candidate_penalty=1e5,
    )

    assert out.total_cost.shape == (1, 3)
    assert out.candidate_indices.shape == (1, 2)
    assert out.candidate_mask.shape == (1, 3)
    assert out.candidate_indices.tolist()[0] == [0, 1]
    assert out.total_cost[0, 0] < out.total_cost[0, 1]
    assert out.total_cost[0, 2] >= torch.tensor(1e5)


def test_center_prior_cost_prefers_nearby_anchor_centers() -> None:
    anchor_centers = torch.tensor([[1.0, 1.0], [1.2, 1.0], [3.5, 3.5]], dtype=torch.float32)
    gt_boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
    center_cost = center_prior_cost(anchor_centers, gt_boxes)

    assert center_cost.shape == (1, 3)
    assert center_cost[0, 0] < center_cost[0, 2]
    assert center_cost[0, 1] < center_cost[0, 2]


def test_dynamic_k_from_top_ious_computation() -> None:
    ious = torch.tensor(
        [
            [0.90, 0.80, 0.70, 0.10, 0.00],
            [0.85, 0.82, 0.78, 0.10, 0.00],
        ],
        dtype=torch.float32,
    )
    dynamic_ks = dynamic_k_from_top_ious(ious, topk=4, min_k=1)

    assert dynamic_ks.shape == (2,)
    assert dynamic_ks.tolist() == [2, 2]


def test_dynamic_k_matching_resolves_conflicts_by_min_cost_in_crowded_case() -> None:
    ious = torch.tensor(
        [
            [0.90, 0.80, 0.70, 0.10, 0.00],
            [0.85, 0.82, 0.78, 0.10, 0.00],
        ],
        dtype=torch.float32,
    )
    total_cost = torch.tensor(
        [
            [0.05, 0.10, 0.20, 2.0, 3.0],  # gt0
            [0.04, 0.12, 0.13, 2.0, 3.0],  # gt1
        ],
        dtype=torch.float32,
    )

    out = dynamic_k_matching(total_cost, ious, dynamic_topk=4, min_k=1)

    assert out.dynamic_ks.tolist() == [2, 2]
    assert out.matching_matrix.shape == (2, 5)
    assert out.foreground_mask.shape == (5,)
    assert out.matched_gt_indices.shape == (5,)
    assert out.num_foreground == 2
    assert out.matched_gt_indices[0].item() == 1  # anchor 0 -> gt1 (lower cost)
    assert out.matched_gt_indices[1].item() == 0  # anchor 1 -> gt0 (lower cost)
    assert out.matched_gt_indices[2].item() == -1
    assert torch.all(out.matching_matrix.sum(dim=0) <= 1)


def test_dynamic_k_matching_respects_candidate_mask_in_crowded_case() -> None:
    ious = torch.tensor(
        [
            [0.90, 0.80, 0.70, 0.10],
            [0.90, 0.80, 0.70, 0.10],
        ],
        dtype=torch.float32,
    )
    total_cost = torch.tensor(
        [
            [0.05, 0.10, 0.15, 9.0],  # gt0 prefers 0,1
            [9.00, 0.08, 0.09, 0.30],  # gt1 cannot take 0 by mask, prefers 1,2
        ],
        dtype=torch.float32,
    )
    candidate_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )

    out = dynamic_k_matching(
        total_cost,
        ious,
        candidate_mask=candidate_mask,
        dynamic_topk=4,
        min_k=1,
    )

    # No match should appear outside per-GT candidate sets.
    assert torch.all(~out.matching_matrix[0, ~candidate_mask[0]])
    assert torch.all(~out.matching_matrix[1, ~candidate_mask[1]])
    assert out.matched_gt_indices[0].item() == 0
