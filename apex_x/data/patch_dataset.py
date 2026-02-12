"""Patch-based dataset for training on high-resolution satellite imagery.

Enables training on 1024x1024 images by extracting random 512x512 patches.
This allows using larger batch sizes and better GPU utilization.

Key features:
- Random patch extraction from large images
- Maintains instance annotations
- Overlap handling for edge cases
- Full-resolution inference support

Expected benefit: 4x larger batch size, better convergence
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from apex_x.data.transforms import TransformSample


class PatchDataset(Dataset):
    """Extract random patches from large satellite images during training.
    
    For 1024x1024 images, extracts 512x512 patches at random locations.
    This reduces memory usage and allows larger batch sizes.
    
    Args:
        base_dataset: Underlying dataset with full-resolution images
        patch_size: Size of patches to extract (default 512)
        num_patches_per_image: How many patches to sample per epoch
        min_objects_per_patch: Minimum objects required in patch
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        patch_size: int = 512,
        num_patches_per_image: int = 4,
        min_objects_per_patch: int = 1,
    ) -> None:
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.min_objects_per_patch = min_objects_per_patch
        
        # Create virtual dataset of patches
        self.total_patches = len(base_dataset) * num_patches_per_image
        
    def __len__(self) -> int:
        return self.total_patches
    
    def __getitem__(self, idx: int) -> TransformSample:
        """Get a random patch from the base dataset.
        
        Args:
            idx: Patch index
            
        Returns:
            TransformSample with patch
        """
        # Map patch index to image index
        img_idx = idx // self.num_patches_per_image
        
        # Get full image
        sample = self.base_dataset[img_idx]
        
        #Extract random patch with objects
        for attempt in range(10):  # Try up to 10 times to find good patch
            patch = self._extract_random_patch(sample)
            
            # Check if patch has enough objects
            if patch.boxes is None or len(patch.boxes) >= self.min_objects_per_patch:
                return patch
                
        # Fallback: return patch even if not enough objects
        return patch
    
    def _extract_random_patch(self, sample: TransformSample) -> TransformSample:
        """Extract a random patch from the sample.
        
        Args:
            sample: Full-resolution sample
            
        Returns:
            Cropped patch as TransformSample
        """
        img = sample.image
        if isinstance(img, Tensor):
            _, h, w = img.shape
        else:
            h, w = img.shape[:2]
            
        # Random top-left corner
        max_y = max(0, h - self.patch_size)
        max_x = max(0, w - self.patch_size)
        
        if max_y == 0:
            y1 = 0
        else:
            y1 = random.randint(0, max_y)
            
        if max_x == 0:
            x1 = 0
        else:
            x1 = random.randint(0, max_x)
            
        y2 = min(y1 + self.patch_size, h)
        x2 = min(x1 + self.patch_size, w)
        
        # Crop image
        if isinstance(img, Tensor):
            patch_img = img[:, y1:y2, x1:x2]
        else:
            patch_img = img[y1:y2, x1:x2]
            
        # Crop and filter boxes
        patch_boxes = None
        patch_masks = None
        patch_classes = None
        
        if sample.boxes is not None and len(sample.boxes) > 0:
            boxes = sample.boxes.clone() if isinstance(sample.boxes, Tensor) else torch.tensor(sample.boxes)
            
            # Translate boxes to patch coordinates
            boxes[:, [0, 2]] -= x1
            boxes[:, [1, 3]] -= y1
            
            # Clip boxes to patch boundaries
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, x2 - x1)
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, y2 - y1)
            
            # Filter boxes that are too small or outside patch
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            valid = (box_w > 4) & (box_h > 4)
            
            if valid.sum() > 0:
                patch_boxes = boxes[valid]
                
                # Filter masks
                if sample.masks is not None:
                    masks = sample.masks
                    if isinstance(masks, Tensor):
                        masks_np = masks.numpy()
                    else:
                        masks_np = masks
                        
                    if masks_np.ndim == 3:  # [N, H, W]
                        patch_masks_np = masks_np[valid.numpy(), y1:y2, x1:x2]
                        patch_masks = torch.from_numpy(patch_masks_np)
                    else:
                        patch_masks_np = masks_np[y1:y2, x1:x2]
                        patch_masks = torch.from_numpy(patch_masks_np)
                        
                # Filter class IDs
                if sample.class_ids is not None:
                    if isinstance(sample.class_ids, Tensor):
                        patch_classes = sample.class_ids[valid]
                    else:
                        patch_classes = torch.tensor(sample.class_ids)[valid]
                        
        return TransformSample(
            image=patch_img,
            boxes=patch_boxes,
            masks=patch_masks,
            class_ids=patch_classes,
            width=x2 - x1,
            height=y2 - y1,
        )


class SlidingWindowInference:
    """Sliding window inference for large images.
    
    Processes large images in overlapping patches and merges predictions.
    Used for full-resolution 1024x1024 inference after training on 512x512 patches.
    
    Args:
        model: Trained model
        patch_size: Patch size (should match training)
        stride: Stride for sliding window (overlap = patch_size - stride)
        device: Device for inference
    """
    
    def __init__(
        self,
        model: Any,
        patch_size: int = 512,
        stride: int = 384,  # 25% overlap
        device: str = 'cuda',
    ) -> None:
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        
    def __call__(self, image: Tensor) -> dict[str, Tensor]:
        """Run inference on large image using sliding window.
        
        Args:
            image: Input image [C, H, W]
            
        Returns:
            Dict with merged predictions
        """
        _, h, w = image.shape
        
        # Generate patch coordinates
        y_coords = list(range(0, h - self.patch_size + 1, self.stride))
        x_coords = list(range(0, w - self.patch_size + 1, self.stride))
        
        # Ensure we cover the entire image
        if y_coords[-1] + self.patch_size < h:
            y_coords.append(h - self.patch_size)
        if x_coords[-1] + self.patch_size < w:
            x_coords.append(w - self.patch_size)
            
        all_boxes = []
        all_scores = []
        all_classes = []
        all_masks = []
        
        # Run model on each patch
        for y in y_coords:
            for x in x_coords:
                patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
                patch_batch = patch.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred = self.model(patch_batch)
                    
                # Translate boxes back to full image coordinates
                if 'boxes' in pred[0] and pred[0]['boxes'] is not None:
                    boxes = pred[0]['boxes']  # First batch item
                    boxes[:, [0, 2]] += x
                    boxes[:, [1, 3]] += y
                    
                    all_boxes.append(boxes)
                    if 'scores' in pred[0]:
                        all_scores.append(pred[0]['scores'])
                    if 'labels' in pred[0]:
                        all_classes.append(pred[0]['labels'])
                    if 'masks' in pred[0] and pred[0]['masks'] is not None:
                        # Masks are typically [N, 1, H_patch, W_patch] or [N, H_patch, W_patch]
                        # Squeeze the channel dimension if present
                        patch_masks = pred[0]['masks'].squeeze(1) # [N, H_patch, W_patch]
                        num_instances = patch_masks.shape[0]
                        
                        # Create a zero-filled tensor for masks in full image coordinates
                        full_masks = torch.zeros(
                            (num_instances, h, w), 
                            device=self.device, 
                            dtype=patch_masks.dtype
                        )
                        # Place the patch masks into the full image canvas
                        full_masks[:, y:y+self.patch_size, x:x+self.patch_size] = patch_masks
                        all_masks.append(full_masks)
                        
        # Merge all predictions
        if len(all_boxes) > 0:
            merged_boxes = torch.cat(all_boxes, dim=0)
            merged_scores = torch.cat(all_scores, dim=0) if all_scores else None
            merged_classes = torch.cat(all_classes, dim=0) if all_classes else None
            merged_masks = torch.cat(all_masks, dim=0) if all_masks else None
            
            # Apply NMS to remove duplicates from overlapping patches
            if merged_scores is not None:
                keep = self._nms(merged_boxes, merged_scores, iou_threshold=0.5)
                merged_boxes = merged_boxes[keep]
                merged_scores = merged_scores[keep]
                merged_classes = merged_classes[keep] if merged_classes is not None else None
                merged_masks = merged_masks[keep] if merged_masks is not None else None
                
            return {
                'boxes': merged_boxes,
                'scores': merged_scores,
                'labels': merged_classes,
                'masks': merged_masks,
            }
        else:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty((0,)),
                'labels': torch.empty((0,), dtype=torch.long),
                'masks': torch.empty((0, h, w), dtype=torch.float32),
            }
            
    def _nms(self, boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
        """Non-maximum suppression.
        
        Args:
            boxes: Boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)


__all__ = [
    "PatchDataset",
    "SlidingWindowInference",
]
