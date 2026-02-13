"""Standard collation for Apex-X datasets.

Provides a unified batch format that combines images into stacks
and annotations into concatenated tensors with batch indexing.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Any
from apex_x.data.transforms import TransformSample

def standard_collate_fn(batch: list[TransformSample]) -> dict[str, Any]:
    """Collate TransformSamples into a unified tensor dictionary.
    
    Args:
        batch: List of samples from a Dataset.
        
    Returns:
        Dictionary with:
            - images: [B, 3, H, W] tensor
            - boxes: [N_total, 4] tensor
            - labels: [N_total] tensor
            - masks: [N_total, H, W] tensor or None
            - batch_idx: [N_total] tensor (batch index for each instance)
            - image_ids: list of str/int
            - metadata: list of dicts
    """
    images = []
    all_boxes = []
    all_labels = []
    all_masks = []
    batch_indices = []
    image_ids = []
    metadata = []
    
    for i, sample in enumerate(batch):
        # Image
        if isinstance(sample.image, np.ndarray):
            img_t = torch.from_numpy(sample.image).permute(2, 0, 1).float()
            if img_t.max() > 1.0:
                img_t /= 255.0
            images.append(img_t)
        else:
            images.append(sample.image)
            
        # Instances
        n_inst = sample.boxes_xyxy.shape[0]
        if n_inst > 0:
            all_boxes.append(torch.from_numpy(sample.boxes_xyxy).float())
            all_labels.append(torch.from_numpy(sample.class_ids).long())
            batch_indices.append(torch.full((n_inst,), i, dtype=torch.long))
            
            if sample.masks is not None:
                all_masks.append(torch.from_numpy(sample.masks).float())
        
        # Meta
        image_ids.append(getattr(sample, "image_id", i))
        metadata.append(getattr(sample, "metadata", {}))
        
    stacked_images = torch.stack(images, dim=0)
    
    res = {
        "images": stacked_images,
        "image_ids": image_ids,
        "metadata": metadata,
    }
    
    if all_boxes:
        res["boxes"] = torch.cat(all_boxes, dim=0)
        res["labels"] = torch.cat(all_labels, dim=0)
        res["batch_idx"] = torch.cat(batch_indices, dim=0)
    else:
        res["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        res["labels"] = torch.zeros((0,), dtype=torch.long)
        res["batch_idx"] = torch.zeros((0,), dtype=torch.long)
        
    if all_masks:
        res["masks"] = torch.cat(all_masks, dim=0)
    else:
        res["masks"] = None
        
    return res

__all__ = ["standard_collate_fn"]
