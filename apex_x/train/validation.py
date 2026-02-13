"""Validation utilities for ApexX trainer.

Provides validation loop with COCO evaluation metrics and proper loss computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from apex_x.eval import COCOEvaluator, PYCOCOTOOLS_AVAILABLE
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

# Try to import pycocotools for mask encoding
try:
    from pycocotools import mask as mask_util
    MASK_UTIL_AVAILABLE = True
except ImportError:
    MASK_UTIL_AVAILABLE = False


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str | torch.device = "cuda",
    loss_fn: Any = None,
    config: Any = None,
    coco_evaluator: COCOEvaluator | None = None,
) -> dict[str, float]:
    """Run validation for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        device: Device to run on
        loss_fn: Loss function (e.g. compute_v3_training_losses)
        config: Training config (passed to loss_fn)
        coco_evaluator: Optional COCO evaluator for mAP metrics
        
    Returns:
        Dict of validation metrics including val_loss and optionally mAP
    """
    model.eval()
    device_obj = torch.device(device)
    non_blocking = device_obj.type == "cuda"
    
    total_loss = 0.0
    num_batches = 0
    loss_components: dict[str, float] = {}
    
    # Reset COCO evaluator if provided
    if coco_evaluator is not None:
        coco_evaluator.reset()
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, dict):
                images = batch.get('images')
                if images is None:
                    continue
                images = images.to(device_obj, non_blocking=non_blocking)
                
                # The StandardCollate format maps directly to compute_v3_training_losses expectations
                targets = {
                    'boxes': batch.get('boxes').to(device_obj, non_blocking=non_blocking) if batch.get('boxes') is not None else None,
                    'labels': batch.get('labels').to(device_obj, non_blocking=non_blocking) if batch.get('labels') is not None else None,
                    'masks': batch.get('masks').to(device_obj, non_blocking=non_blocking) if batch.get('masks') is not None else None,
                    'batch_idx': batch.get('batch_idx').to(device_obj, non_blocking=non_blocking) if batch.get('batch_idx') is not None else None,
                }
            else:
                continue
            
            try:
                output = model(images)
                
                # Compute loss using the real loss function
                if loss_fn is not None and config is not None:
                    loss, loss_dict = loss_fn(output, targets, model, config)
                    total_loss += float(loss.item())
                    num_batches += 1
                    
                    # Accumulate component losses
                    for k, v in loss_dict.items():
                        value = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                        loss_components[k] = loss_components.get(k, 0.0) + value
                
                # Accumulate predictions for COCO eval
                if coco_evaluator is not None and PYCOCOTOOLS_AVAILABLE:
                    predictions = convert_to_coco_format(output, batch)
                    if predictions:
                        coco_evaluator.update(predictions)
                        
            except Exception as e:
                LOGGER.warning(f"Validation batch {batch_idx} failed: {e}")
                continue
            finally:
                del images, targets
                if 'output' in locals():
                    del output
    
    # Compute metrics
    metrics: dict[str, float] = {}
    
    if num_batches > 0:
        metrics['val_loss'] = total_loss / num_batches
        for k, v in loss_components.items():
            metrics[f'val_{k}'] = v / num_batches
    else:
        metrics['val_loss'] = float('inf')
    
    # Compute COCO metrics if evaluator provided
    if coco_evaluator is not None and PYCOCOTOOLS_AVAILABLE:
        try:
            coco_metrics = coco_evaluator.compute()
            metrics.update(coco_metrics)
            
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
    batch: Any,
) -> list[dict[str, Any]]:
    """Convert model output to COCO format for evaluation.
    
    Args:
        model_output: Model forward pass output
        batch: Input batch with metadata
        
    Returns:
        List of dicts in COCO results format with proper mask RLE encoding.
    """
    predictions = []
    
    try:
        if isinstance(model_output, dict):
            boxes = model_output.get('boxes')
            scores = model_output.get('scores')
            labels = model_output.get('labels')
            masks = model_output.get('masks')
        else:
            return predictions
        
        if boxes is None or scores is None:
            return predictions
        
        # Get image IDs
        if isinstance(batch, (list, tuple)) and hasattr(batch[0], 'image'):
            image_ids = list(range(len(batch)))
        elif isinstance(batch, dict):
            image_ids = batch.get('image_id', batch.get('img_id', torch.arange(1)))
        else:
            image_ids = [0]
        
        boxes_cpu = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
        scores_cpu = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        
        # Handle multi-class scores: take max score and argmax label
        if scores_cpu.ndim == 2:
            labels_cpu = scores_cpu.argmax(axis=1)
            scores_cpu = scores_cpu.max(axis=1)
        elif labels is not None:
            labels_cpu = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        else:
            labels_cpu = np.zeros(len(boxes_cpu), dtype=np.int64)
        
        for i in range(len(boxes_cpu)):
            x1, y1, x2, y2 = boxes_cpu[i]
            w, h = x2 - x1, y2 - y1
            
            # Score threshold
            score = float(scores_cpu[i])
            if score < 0.001:
                continue
            
            pred: dict[str, Any] = {
                'image_id': int(image_ids[0] if len(image_ids) == 1 else image_ids[min(i, len(image_ids)-1)]),
                'category_id': int(labels_cpu[i]),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': score,
            }
            
            # Encode mask to RLE for proper mAP computation
            if masks is not None and MASK_UTIL_AVAILABLE and i < masks.shape[0]:
                mask_i = masks[i]
                if isinstance(mask_i, torch.Tensor):
                    mask_i = mask_i.cpu().numpy()
                # Squeeze channel dim if present
                if mask_i.ndim == 3 and mask_i.shape[0] == 1:
                    mask_i = mask_i[0]
                # Binarize
                binary_mask = (mask_i > 0.5).astype(np.uint8)
                # Encode to RLE
                rle = mask_util.encode(np.asfortranarray(binary_mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                pred['segmentation'] = rle
            
            predictions.append(pred)
    
    except Exception as e:
        LOGGER.debug(f"Failed to convert to COCO format: {e}")
    
    return predictions


__all__ = ['validate_epoch', 'convert_to_coco_format']
