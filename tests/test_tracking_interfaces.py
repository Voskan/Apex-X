from __future__ import annotations

import torch

from apex_x.infer import (
    AssociationProtocol,
    AssociationResult,
    GreedyCosineAssociator,
    TrackAssociatorProtocol,
    TrackState,
)


def _use_associator(
    associator: AssociationProtocol,
    state: TrackState | None,
) -> AssociationResult:
    det_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=torch.float32,
    )
    det_emb = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    det_scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    return associator.associate(det_boxes, det_emb, det_scores, state, frame_index=3)


def test_track_state_empty_contract() -> None:
    state = TrackState.empty(embedding_dim=4, frame_index=2)
    assert state.num_tracks == 0
    assert state.embedding_dim == 4
    assert state.next_track_id == 0
    assert state.frame_index == 2


def test_greedy_associator_protocol_conformance() -> None:
    associator = GreedyCosineAssociator(match_threshold=0.6)
    assert isinstance(associator, AssociationProtocol)
    assert isinstance(associator, TrackAssociatorProtocol)


def test_greedy_associator_matches_and_creates_new_tracks_deterministically() -> None:
    associator = GreedyCosineAssociator(match_threshold=0.7, max_age=5)
    state = TrackState(
        track_ids=torch.tensor([10, 11], dtype=torch.int64),
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        boxes_xyxy=torch.tensor(
            [[0.0, 0.0, 5.0, 5.0], [5.0, 5.0, 10.0, 10.0]], dtype=torch.float32
        ),
        scores=torch.tensor([0.5, 0.6], dtype=torch.float32),
        ages=torch.tensor([0, 1], dtype=torch.int64),
        frame_index=2,
    )

    out1 = _use_associator(associator, state)
    out2 = _use_associator(associator, state)

    assert torch.equal(out1.matched_detection_indices, torch.tensor([0, 1], dtype=torch.int64))
    assert torch.equal(out1.matched_track_indices, torch.tensor([0, 1], dtype=torch.int64))
    assert torch.equal(out1.new_detection_indices, torch.tensor([2], dtype=torch.int64))

    next_state = out1.next_state
    assert next_state.num_tracks == 3
    assert next_state.track_ids.tolist() == [10, 11, 12]
    assert next_state.ages.tolist() == [0, 0, 0]
    assert next_state.frame_index == 3

    assert torch.equal(out1.matched_detection_indices, out2.matched_detection_indices)
    assert torch.equal(out1.matched_track_indices, out2.matched_track_indices)
    assert torch.equal(out1.new_detection_indices, out2.new_detection_indices)
    assert torch.equal(out1.next_state.track_ids, out2.next_state.track_ids)


def test_greedy_associator_ages_out_tracks_when_no_detections() -> None:
    associator = GreedyCosineAssociator(match_threshold=0.1, max_age=1)
    state = TrackState(
        track_ids=torch.tensor([1, 2], dtype=torch.int64),
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        boxes_xyxy=torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]], dtype=torch.float32),
        scores=torch.tensor([0.4, 0.4], dtype=torch.float32),
        ages=torch.tensor([0, 1], dtype=torch.int64),
        frame_index=0,
    )
    det_boxes = torch.zeros((0, 4), dtype=torch.float32)
    det_emb = torch.zeros((0, 2), dtype=torch.float32)
    det_scores = torch.zeros((0,), dtype=torch.float32)

    out = associator.associate(det_boxes, det_emb, det_scores, state, frame_index=1)
    assert out.next_state.track_ids.tolist() == [1]
    assert out.next_state.ages.tolist() == [1]
