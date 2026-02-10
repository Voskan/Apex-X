from __future__ import annotations

import torch

from apex_x.infer import HungarianAssociator, hungarian_assignment


def test_hungarian_assignment_finds_global_optimum() -> None:
    cost = torch.tensor(
        [
            [0.10, 0.20],
            [0.11, 0.90],
        ],
        dtype=torch.float32,
    )
    rows, cols = hungarian_assignment(cost)
    pairs = sorted(zip(rows.tolist(), cols.tolist(), strict=True))
    assert pairs == [(0, 1), (1, 0)]


def test_iou_and_embedding_distance_gating_blocks_invalid_matches() -> None:
    associator = HungarianAssociator(
        iou_gate=0.1,
        embedding_distance_gate=0.3,
        iou_weight=0.5,
        embedding_weight=0.5,
        max_age=3,
        memory_bank_size=4,
    )

    boxes0 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    emb0 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    scores0 = torch.tensor([0.9], dtype=torch.float32)
    out0 = associator.associate(boxes0, emb0, scores0, None, frame_index=0)

    boxes1 = torch.tensor([[30.0, 30.0, 40.0, 40.0]], dtype=torch.float32)
    emb1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    scores1 = torch.tensor([0.8], dtype=torch.float32)
    out1 = associator.associate(boxes1, emb1, scores1, out0.next_state, frame_index=1)

    assert out1.matched_detection_indices.numel() == 0
    assert out1.matched_track_indices.numel() == 0
    assert out1.new_detection_indices.tolist() == [0]
    assert out1.created_track_ids.tolist() == [1]
    assert out1.next_state.track_ids.tolist() == [0, 1]
    assert out1.next_state.ages.tolist() == [1, 0]


def test_track_lifecycle_terminates_after_max_age() -> None:
    associator = HungarianAssociator(
        iou_gate=0.0,
        embedding_distance_gate=2.0,
        iou_weight=0.4,
        embedding_weight=0.6,
        max_age=1,
        memory_bank_size=3,
    )
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    emb = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    scores = torch.tensor([0.9], dtype=torch.float32)
    out0 = associator.associate(boxes, emb, scores, None, frame_index=0)

    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    empty_emb = torch.zeros((0, 2), dtype=torch.float32)
    empty_scores = torch.zeros((0,), dtype=torch.float32)

    out1 = associator.associate(
        empty_boxes, empty_emb, empty_scores, out0.next_state, frame_index=1
    )
    assert out1.next_state.track_ids.tolist() == [0]
    assert out1.next_state.ages.tolist() == [1]
    assert out1.terminated_track_ids.numel() == 0

    out2 = associator.associate(
        empty_boxes, empty_emb, empty_scores, out1.next_state, frame_index=2
    )
    assert out2.next_state.num_tracks == 0
    assert out2.terminated_track_ids.tolist() == [0]


def test_embedding_memory_bank_updates_and_caps_size() -> None:
    associator = HungarianAssociator(
        iou_gate=0.0,
        embedding_distance_gate=2.0,
        iou_weight=0.2,
        embedding_weight=0.8,
        max_age=3,
        memory_bank_size=2,
    )

    boxes0 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    emb0 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    scores0 = torch.tensor([0.9], dtype=torch.float32)
    out0 = associator.associate(boxes0, emb0, scores0, None, frame_index=0)

    boxes1 = torch.tensor([[1.0, 0.0, 11.0, 10.0]], dtype=torch.float32)
    emb1 = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    scores1 = torch.tensor([0.9], dtype=torch.float32)
    out1 = associator.associate(boxes1, emb1, scores1, out0.next_state, frame_index=1)

    assert out1.next_state.memory_counts is not None
    assert out1.next_state.memory_counts.tolist() == [2]
    assert out1.next_state.hit_counts is not None
    assert out1.next_state.hit_counts.tolist() == [2]

    boxes2 = torch.tensor([[2.0, 0.0, 12.0, 10.0]], dtype=torch.float32)
    emb2 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    scores2 = torch.tensor([0.9], dtype=torch.float32)
    out2 = associator.associate(boxes2, emb2, scores2, out1.next_state, frame_index=2)

    assert out2.next_state.memory_counts is not None
    assert out2.next_state.memory_counts.tolist() == [2]
    assert out2.next_state.hit_counts is not None
    assert out2.next_state.hit_counts.tolist() == [3]
    assert out2.next_state.memory_bank is not None
    latest = torch.nn.functional.normalize(emb2, p=2.0, dim=1, eps=1e-6)[0]
    sims = out2.next_state.memory_bank[0] @ latest
    assert float(sims.max().item()) > 0.999


def test_synthetic_moving_objects_keep_consistent_ids() -> None:
    associator = HungarianAssociator(
        iou_gate=0.15,
        embedding_distance_gate=0.4,
        iou_weight=0.4,
        embedding_weight=0.6,
        max_age=2,
        memory_bank_size=4,
    )

    boxes_t0 = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 0.0, 30.0, 10.0],
        ],
        dtype=torch.float32,
    )
    emb_t0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.9], dtype=torch.float32)
    out0 = associator.associate(boxes_t0, emb_t0, scores, None, frame_index=0)
    assert out0.created_track_ids.tolist() == [0, 1]

    boxes_t1 = torch.tensor(
        [
            [16.0, 0.0, 26.0, 10.0],
            [4.0, 0.0, 14.0, 10.0],
        ],
        dtype=torch.float32,
    )
    emb_t1 = torch.tensor([[0.02, 0.98], [0.98, 0.02]], dtype=torch.float32)
    out1 = associator.associate(boxes_t1, emb_t1, scores, out0.next_state, frame_index=1)

    assigned_t1: dict[int, int] = {}
    for det_idx, track_idx in zip(
        out1.matched_detection_indices.tolist(),
        out1.matched_track_indices.tolist(),
        strict=True,
    ):
        assigned_t1[det_idx] = int(out0.next_state.track_ids[track_idx].item())
    for det_idx, track_id in zip(
        out1.new_detection_indices.tolist(),
        out1.created_track_ids.tolist(),
        strict=True,
    ):
        assigned_t1[det_idx] = track_id
    assert assigned_t1[0] == 1
    assert assigned_t1[1] == 0

    boxes_t2 = torch.tensor(
        [
            [12.0, 0.0, 22.0, 10.0],
            [8.0, 0.0, 18.0, 10.0],
        ],
        dtype=torch.float32,
    )
    emb_t2 = torch.tensor([[0.05, 0.95], [0.95, 0.05]], dtype=torch.float32)
    out2 = associator.associate(boxes_t2, emb_t2, scores, out1.next_state, frame_index=2)

    assigned_t2: dict[int, int] = {}
    for det_idx, track_idx in zip(
        out2.matched_detection_indices.tolist(),
        out2.matched_track_indices.tolist(),
        strict=True,
    ):
        assigned_t2[det_idx] = int(out1.next_state.track_ids[track_idx].item())
    for det_idx, track_id in zip(
        out2.new_detection_indices.tolist(),
        out2.created_track_ids.tolist(),
        strict=True,
    ):
        assigned_t2[det_idx] = track_id
    assert assigned_t2[0] == 1
    assert assigned_t2[1] == 0
