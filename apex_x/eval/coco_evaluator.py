"""COCO evaluation wrapper for computing mAP metrics.

Provides a simplified interface to pycocotools for evaluating detection and
segmentation predictions in COCO format.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

# Lazy import pycocotools (optional dependency)
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    LOGGER.warning("pycocotools not installed - COCO evaluation will not be available")


class COCOEvaluator:
    """Wrapper around pycocotools for mAP computation.
    
    Accumulates predictions over multiple batches and computes mAP metrics
    for detection (bbox) and/or segmentation (segm).
    """
    
    def __init__(
        self,
        coco_gt: COCO | None = None,
        iou_types: list[str] | None = None,
        ann_file: str | None = None,
    ) -> None:
        """Initialize COCO evaluator.
        
        Args:
            coco_gt: COCO ground truth API object (if already loaded)
            iou_types: List of evaluation types ['bbox', 'segm']
            ann_file: Path to COCO annotation JSON (if coco_gt not provided)
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise RuntimeError(
                "pycocotools is required for COCO evaluation. "
                "Install with: pip install pycocotools"
            )
        
        if coco_gt is None and ann_file is None:
            raise ValueError("Must provide either coco_gt or ann_file")
        
        self.coco_gt = coco_gt if coco_gt is not None else COCO(ann_file)
        self.iou_types = iou_types or ["bbox"]
        self.predictions: list[dict[str, Any]] = []
        
        LOGGER.info(f"Initialized COCO evaluator for {self.iou_types}")
    
    def update(self, predictions: list[dict[str, Any]]) -> None:
        """Accumulate predictions for one batch.
        
        Args:
            predictions: List of dicts in COCO results format:
                {
                    'image_id': int,
                    'category_id': int,
                    'bbox': [x, y, w, h],
                    'score': float,
                    'segmentation': RLE or polygon (optional),
                }
        """
        self.predictions.extend(predictions)
    
    def reset(self) -> None:
        """Clear accumulated predictions."""
        self.predictions = []
    
    def compute(self) -> dict[str, float]:
        """Run COCO evaluation and return metrics.
        
        Returns:
            Dict of metrics:
                - mAP_bbox: Average Precision @ IoU=0.50:0.95 (detection)
                - mAP_50_bbox: Average Precision @ IoU=0.50
                - mAP_75_bbox: Average Precision @ IoU=0.75
                - mAP_segm: Average Precision for segmentation (if applicable)
                - etc.
        """
        if not self.predictions:
            LOGGER.warning("No predictions to evaluate")
            return {}
        
        results: dict[str, float] = {}
        
        for iou_type in self.iou_types:
            # Create COCO detection results
            coco_dt = self.coco_gt.loadRes(self.predictions)
            
            # Run evaluation
            coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            # stats[0] = AP @ IoU=0.50:0.95
            # stats[1] = AP @ IoU=0.50
            # stats[2] = AP @ IoU=0.75
            # stats[3] = AP small
            # stats[4] = AP medium
            # stats[5] = AP large
            results[f"mAP_{iou_type}"] = float(coco_eval.stats[0])
            results[f"mAP_50_{iou_type}"] = float(coco_eval.stats[1])
            results[f"mAP_75_{iou_type}"] = float(coco_eval.stats[2])
            results[f"mAP_small_{iou_type}"] = float(coco_eval.stats[3])
            results[f"mAP_medium_{iou_type}"] = float(coco_eval.stats[4])
            results[f"mAP_large_{iou_type}"] = float(coco_eval.stats[5])
            
            LOGGER.info(f"{iou_type} mAP: {results[f'mAP_{iou_type}']:.4f}")
        
        return results


def convert_predictions_to_coco_format(
    boxes: Tensor,  # [N, 4] in xyxy format
    scores: Tensor,  # [N]
    labels: Tensor,  # [N]
    image_id: int,
    masks: Tensor | None = None,  # [N, H, W] optional segmentation masks
) -> list[dict[str, Any]]:
    """Convert model predictions to COCO results format.
    
    Args:
        boxes: Predicted boxes in xyxy format
        scores: Confidence scores
        labels: Class labels (1-indexed for COCO)
        image_id: COCO image ID
        masks: Optional segmentation masks
    
    Returns:
        List of prediction dicts in COCO format
    """
    predictions = []
    
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    for i in range(len(boxes_np)):
        x1, y1, x2, y2 = boxes_np[i]
        w = x2 - x1
        h = y2 - y1
        
        pred = {
            "image_id": int(image_id),
            "category_id": int(labels_np[i]),
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(scores_np[i]),
        }
        
        # Add segmentation if available
        if masks is not None and i < len(masks):
            mask = masks[i]
            
            # Convert mask to RLE format using pycocotools
            try:
                from pycocotools import mask as mask_utils
                import numpy as np
                
                # Ensure binary mask
                if isinstance(mask, torch.Tensor):
                    mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                else:
                    mask_np = (mask > 0.5).astype(np.uint8)
                
                # Fortran order for pycocotools
                mask_fortran = np.asfortranarray(mask_np)
                
                # Encode to RLE
                rle = mask_utils.encode(mask_fortran)
                rle['counts'] = rle['counts'].decode('utf-8')
                
                pred["segmentation"] = rle
                
            except ImportError:
                # Fallback: polygon format
                try:
                    import cv2
                    mask_uint8 = (mask_np * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        polygon = contours[0].reshape(-1).tolist()
                        if len(polygon) >= 6:
                            pred["segmentation"] = [polygon]
                except Exception:
                    pass  # Skip segmentation if conversion fails
        
        predictions.append(pred)

    
    return predictions


__all__ = [
    "COCOEvaluator",
    "convert_predictions_to_coco_format",
    "PYCOCOTOOLS_AVAILABLE",
]
