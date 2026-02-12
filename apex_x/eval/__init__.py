"""Evaluation utilities for Apex-X models."""

from __future__ import annotations

from .coco_evaluator import PYCOCOTOOLS_AVAILABLE, COCOEvaluator, convert_predictions_to_coco_format

__all__ = [
    "COCOEvaluator",
    "convert_predictions_to_coco_format",
    "PYCOCOTOOLS_AVAILABLE",
]

