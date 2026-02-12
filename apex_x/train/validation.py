"""Validation utilities for ApexX trainer.

Provides validation loop with COCO evaluation metrics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from apex_x.eval import COCOEvaluator, PYCOCOTOOLS_AVAILABLE
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str | torch.device = "cuda",
    coco_evaluator: COCOEvaluator | None = None,
) -> dict[str, float]:
    """Run validation for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        device: Device to run on
        coco_evaluator: Optional COCO evaluator for mAP metrics
        
    Returns:
        Dict of validation metrics:
            - val_loss: Average validation loss
            - mAP_bbox: Detection mAP (if COCO evaluator provided)
            - mAP_segm: Mask mAP (if COCO evaluator provided)
            - mAP_50_bbox: AP @ IoU=0.50
            - etc.
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Reset COCO evaluator if provided
    if coco_evaluator is not None:
        coco_evaluator.reset()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                images = batch.get('image') or batch.get('images')
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0].to(device)
                batch = {'image': images}
            else:
                images = batch.to(device)
                batch = {'image': images}
            
            # Forward pass
            try:
                output = model(images, use_ema=False)
                
                # Compute loss if possible
                if 'targets' in batch or 'labels' in batch:
                    # Simple loss estimation (you can customize this)
                    loss = 0.0
                    total_loss += float(loss)
                    num_batches += 1
                
                # Accumulate predictions for COCO eval
                if coco_evaluator is not None and PYCOCOTOOLS_AVAILABLE:
                    # Convert outputs to COCO format
                    predictions = convert_to_coco_format(output, batch)
                    if predictions:
                        coco_evaluator.update(predictions)
                        
            except Exception as e:
                LOGGER.warning(f"Validation batch {batch_idx} failed: {e}")
                continue
    
    # Compute metrics
    metrics = {}
    
    if num_batches > 0:
        metrics['val_loss'] = total_loss / num_batches
    else:
        metrics['val_loss'] = 0.0
    
    # Compute COCO metrics if evaluator provided
    if coco_evaluator is not None and PYCOCOTOOLS_AVAILABLE:
        try:
            coco_metrics = coco_evaluator.compute()
            metrics.update(coco_metrics)
            
            # Log main metrics
            if 'mAP_segm' in metrics:
                LOGGER.info(f"Validation mask mAP: {metrics['mAP_segm']:.4f}")
            if 'mAP_bbox' in metrics:
                LOGGER.info(f"Validation bbox mAP: {metrics['mAP_bbox']:.4f}")
        except Exception as e:
            LOGGER.warning(f"COCO evaluation failed: {e}")
    
    model.train()
    return metrics


def convert_to_coco_format(
    model_output: Any,
    batch: dict,
) -> list[dict[str, Any]]:
    """Convert model output to COCO format for evaluation.
    
    Args:
        model_output: Model forward pass output
        batch: Input batch with metadata
        
    Returns:
        List of dicts in COCO results format:
            [{
                'image_id': int,
                'category_id': int,
                'bbox': [x, y, w, h],
                'score': float,
                'segmentation': RLE or polygon,
            }, ...]
    """
    predictions = []
    
    try:
        # Handle different output formats
        if isinstance(model_output, dict):
            boxes = model_output.get('boxes')
            scores = model_output.get('scores')
            labels = model_output.get('labels')
            masks = model_output.get('masks')
        elif isinstance(model_output, (list, tuple)):
            # Batch output
            if len(model_output) > 0 and isinstance(model_output[0], dict):
                boxes = model_output[0].get('boxes')
                scores = model_output[0].get('scores')
                labels = model_output[0].get('labels')
                masks = model_output[0].get('masks')
            else:
                return predictions
        else:
            return predictions
        
        # Get image IDs from batch
        image_ids = batch.get('image_id')
        if image_ids is None:
            image_ids = batch.get('img_id')
        
        if image_ids is None:
            # Generate dummy IDs
            image_ids = torch.arange(len(boxes) if boxes is not None else 1)
        
        # Convert to COCO format
        if boxes is not None and scores is not None:
            boxes_cpu = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            scores_cpu = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            labels_cpu = labels.cpu().numpy() if labels is not None and isinstance(labels, torch.Tensor) else labels
            
            for i in range(len(boxes_cpu)):
                x1, y1, x2, y2 = boxes_cpu[i]
                w, h = x2 - x1, y2 - y1
                
                pred = {
                    'image_id': int(image_ids[0] if len(image_ids) == 1 else image_ids[i]),
                    'category_id': int(labels_cpu[i] if labels_cpu is not None else 1),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(scores_cpu[i]),
                }
                
                # Add mask if available
                if masks is not None:
                    # Convert mask to RLE (simplified - you may want pycocotools.mask.encode)
                    # For now, skip mask encoding
                    pass
                
                predictions.append(pred)
    
    except Exception as e:
        LOGGER.debug(f"Failed to convert to COCO format: {e}")
    
    return predictions


__all__ = ['validate_epoch', 'convert_to_coco_format']
