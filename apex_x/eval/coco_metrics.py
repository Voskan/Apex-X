"""Standard COCO evaluation wrapper for Apex-X.

Calculates mAP 50-95 using pycocotools, replacing ad-hoc metric proxies.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


class COCOEvaluator:
    """Wrapper for pycocotools evaluation."""
    
    def __init__(self, coco_gt: COCO, iou_types: list[str] = ["bbox", "segm"]) -> None:
        """Initialize evaluator.
        
        Args:
            coco_gt: Loaded COCO ground truth object
            iou_types: List of tasks to evaluate ("bbox", "segm")
        """
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.predictions: list[dict[str, Any]] = []
        
    def update(
        self,
        image_id: int,
        boxes: torch.Tensor | np.ndarray,      # [N, 4] xyxy
        scores: torch.Tensor | np.ndarray,     # [N]
        classes: torch.Tensor | np.ndarray,    # [N]
        masks: torch.Tensor | np.ndarray | None = None, # [N, H, W] binary or RLE
    ) -> None:
        """Add predictions for a single image."""
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            # Encode binary masks to RLE for efficiency
            # pycocotools expects RLE for segmentation eval usually, or polygon
            # For strict RLE:
            # from pycocotools import mask as mask_utils
            # rle = mask_utils.encode(np.asfortranarray(masks.cpu().numpy().astype(np.uint8)))
            # But converting [N, H, W] to RLE list is heavy. 
            pass # TODO: Implement mask encoding if efficient RLE needed, otherwise let's stick to boxes for now or simple RLE

        # Convert xyxy to xywh for COCO
        # w = x2 - x1, h = y2 - y1
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        
        for i in range(len(scores)):
            pred = {
                "image_id": int(image_id),
                "category_id": int(classes[i]), # Assuming mapped 1-1 or remapped
                "bbox": boxes_xywh[i].tolist(),
                "score": float(scores[i]),
            }
            
            # Add segmentation if available
            # if masks is not None:
            #     pred["segmentation"] = rle[i]
            
            self.predictions.append(pred)
            
    def synchronize_between_processes(self) -> None:
        pass # TODO for DDP
        
    def evaluate(self) -> dict[str, float]:
        """Run COCO evaluation."""
        if not self.predictions:
            LOGGER.warning("No predictions to evaluate!")
            return {}
            
        # Save to json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(self.predictions, f)
            f.flush()
            
            coco_dt = self.coco_gt.loadRes(f.name)
            
        results = {}
        for iou_type in self.iou_types:
            coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            results[f"{iou_type}_mAP"] = float(coco_eval.stats[0]) # mAP 50-95
            results[f"{iou_type}_mAP50"] = float(coco_eval.stats[1]) # mAP 50
            results[f"{iou_type}_mAP75"] = float(coco_eval.stats[2]) # mAP 75
            
        return results
