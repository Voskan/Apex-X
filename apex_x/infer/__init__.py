"""Inference orchestration scaffolding for Apex-X."""

from __future__ import annotations

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
from .runner import (
    EvalDataset,
    InferenceRunResult,
    ModelDatasetEvalSummary,
    RuntimeMetadata,
    evaluate_model_dataset,
    extract_routing_diagnostics,
    load_eval_dataset_npz,
    load_eval_images_npz,
    run_model_inference,
)
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
    "RuntimeMetadata",
    "InferenceRunResult",
    "ModelDatasetEvalSummary",
    "EvalDataset",
    "extract_routing_diagnostics",
    "run_model_inference",
    "load_eval_dataset_npz",
    "load_eval_images_npz",
    "evaluate_model_dataset",
    "TrackState",
    "AssociationResult",
    "AssociationProtocol",
    "TrackAssociatorProtocol",
    "TrackAssociator",
    "GreedyCosineAssociator",
    "HungarianAssociator",
    "hungarian_assignment",
]
