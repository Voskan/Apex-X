from __future__ import annotations

import torch

from apex_x.infer import decode_and_nms, deterministic_nms
from apex_x.model import DetHeadOutput


def _build_single_level_det_output() -> DetHeadOutput:
    cls_logits = torch.full((1, 2, 1, 3), -8.0, dtype=torch.float32)
    quality = torch.full((1, 1, 1, 3), 8.0, dtype=torch.float32)
    box_reg = torch.full((1, 4, 1, 3), 2.0, dtype=torch.float32)

    # Anchor 0, class 0: highest score.
    cls_logits[0, 0, 0, 0] = 8.0
    # Anchor 1, class 0: slightly lower; should be suppressed by NMS.
    cls_logits[0, 0, 0, 1] = 7.5
    # Anchor 2, class 1: different class, should survive class-wise NMS.
    cls_logits[0, 1, 0, 2] = 7.0

    level = "P3"
    return DetHeadOutput(
        cls_logits={level: cls_logits},
        box_reg={level: box_reg},
        quality={level: quality},
        features={level: torch.zeros((1, 8, 1, 3), dtype=torch.float32)},
    )


def test_decode_and_nms_is_deterministic_and_classwise() -> None:
    det_output = _build_single_level_det_output()

    first = decode_and_nms(
        det_output,
        image_size=(32, 32),
        strides={"P3": 8},
        score_threshold=0.05,
        pre_nms_topk=100,
        iou_threshold=0.5,
        max_detections=10,
    )
    second = decode_and_nms(
        det_output,
        image_size=(32, 32),
        strides={"P3": 8},
        score_threshold=0.05,
        pre_nms_topk=100,
        iou_threshold=0.5,
        max_detections=10,
    )

    assert first.boxes.shape == (1, 10, 4)
    assert first.scores.shape == (1, 10)
    assert first.class_ids.shape == (1, 10)
    assert first.valid_counts.tolist() == [2]

    valid = int(first.valid_counts[0].item())
    kept_classes = first.class_ids[0, :valid].tolist()
    assert kept_classes == [0, 1]
    assert float(first.scores[0, 0].item()) >= float(first.scores[0, 1].item())

    assert torch.allclose(first.boxes, second.boxes)
    assert torch.allclose(first.scores, second.scores)
    assert torch.equal(first.class_ids, second.class_ids)
    assert torch.equal(first.valid_counts, second.valid_counts)


def test_deterministic_nms_tie_break_prefers_lower_index() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32)
    class_ids = torch.tensor([0, 0, 0], dtype=torch.int64)

    keep = deterministic_nms(
        boxes,
        scores,
        class_ids,
        iou_threshold=0.5,
        max_detections=3,
    )

    assert keep.tolist() == [0, 1, 2]


def test_deterministic_nms_does_not_cross_suppress_classes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    class_ids = torch.tensor([0, 1], dtype=torch.int64)

    keep = deterministic_nms(
        boxes,
        scores,
        class_ids,
        iou_threshold=0.1,
        max_detections=10,
    )
    assert keep.tolist() == [0, 1]
