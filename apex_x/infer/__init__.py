"""Inference orchestration scaffolding for Apex-X."""

from __future__ import annotations

from typing import Any

from .detection import (
    DetectionBatch,
    DetectionCandidates,
    batched_deterministic_nms,
    decode_anchor_free_candidates,
    decode_and_nms,
    deterministic_nms,
)
from .eval_metrics import (
    EvalSummary,
    evaluate_fixture_file,
    evaluate_fixture_payload,
    tiny_eval_fixture_payload,
    write_eval_reports,
)
from .panoptic import PanopticOutput, PanopticSegmentInfo, generate_panoptic_output
from .pq_eval import OfficialPQPaths, PQClassMetrics, PQMetrics, evaluate_panoptic_quality
from .tracking import (
    AssociationProtocol,
    AssociationResult,
    GreedyCosineAssociator,
    HungarianAssociator,
    TrackAssociator,
    TrackAssociatorProtocol,
    TrackState,
    hungarian_assignment,
)


def infer_placeholder(model_output: dict[str, Any] | None = None) -> dict[str, Any]:
    if model_output is None:
        return {}
    diagnostics = model_output.get("routing_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return {}
    return diagnostics


__all__ = [
    "DetectionBatch",
    "DetectionCandidates",
    "decode_anchor_free_candidates",
    "deterministic_nms",
    "batched_deterministic_nms",
    "decode_and_nms",
    "EvalSummary",
    "evaluate_fixture_file",
    "evaluate_fixture_payload",
    "tiny_eval_fixture_payload",
    "write_eval_reports",
    "PanopticSegmentInfo",
    "PanopticOutput",
    "generate_panoptic_output",
    "PQClassMetrics",
    "PQMetrics",
    "OfficialPQPaths",
    "evaluate_panoptic_quality",
    "TrackState",
    "AssociationResult",
    "AssociationProtocol",
    "TrackAssociatorProtocol",
    "TrackAssociator",
    "GreedyCosineAssociator",
    "HungarianAssociator",
    "hungarian_assignment",
    "infer_placeholder",
]
