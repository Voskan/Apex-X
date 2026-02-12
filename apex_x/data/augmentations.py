"""Advanced data augmentation strategies for Apex-X.

Implements state-of-the-art augmentation techniques used in YOLO26, YOLOv11,
and other modern detectors: Mosaic, MixUp, and CopyPaste.
"""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from apex_x.data import TransformSample
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


class MosaicAugmentation:
    """Mosaic augmentation - combine 4 images into 2x2 grid.
    
    This is one of the most effective augmentations for detection, used in
    YOLO series and many SOTA detectors. It:
    - Increases batch diversity
    - Improves learning of objects at different scales
    - Helps detect small objects by placing them in different contexts
    
    Expected impact: +3-5% mAP
    """
    
    def __init__(
        self,
        dataset,
        output_size: int = 640,
        mosaic_prob: float = 0.5,
        min_offset: float = 0.3,
        max_offset: float = 0.7,
    ) -> None:
        """Initialize Mosaic augmentation.
        
        Args:
            dataset: Dataset to sample additional images from
            output_size: Target output size (both height and width)
            mosaic_prob: Probability of applying mosaic
            min_offset: Minimum offset for center point (relative to output_size)
            max_offset: Maximum offset for center point
        """
        self.dataset = dataset
        self.output_size = output_size
        self.mosaic_prob = mosaic_prob
        self.min_offset = min_offset
        self.max_offset = max_offset
    
    def __call__(self, sample: TransformSample) -> TransformSample:
        """Apply mosaic augmentation to sample.
        
        Args:
            sample: Input sample
        
        Returns:
            Augmented sample with 4 images combined
        """
        if random.random() > self.mosaic_prob:
            return sample
        
        # Sample 3 additional images
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        samples = [sample] + [self.dataset[i] for i in indices]
        
        # Random center point for the mosaic
        center_x = int(random.uniform(self.min_offset, self.max_offset) * self.output_size)
        center_y = int(random.uniform(self.min_offset, self.max_offset) * self.output_size)
        
        # Create output mosaic
        mosaic_img = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        mosaic_boxes = []
        mosaic_classes = []
        mosaic_masks = [] if samples[0].masks is not None else None
        
        # Place each image in a quadrant
        placements = [
            (0, 0, center_x, center_y),  # Top-left
            (center_x, 0, self.output_size, center_y),  # Top-right
            (0, center_y, center_x, self.output_size),  # Bottom-left
            (center_x, center_y, self.output_size, self.output_size),  # Bottom-right
        ]
        
        for sample_idx, (s, (x1, y1, x2, y2)) in enumerate(zip(samples, placements)):
            # Get image
            if isinstance(s.image, torch.Tensor):
                img = s.image.cpu().numpy()
                if img.shape[0] == 3:  # CHW → HWC
                    img = np.transpose(img, (1, 2, 0))
            else:
                img = s.image
            
            # Scale image to fit quadrant
            quad_h, quad_w = y2 - y1, x2 - x1
            h, w = img.shape[:2]
            scale = min(quad_w / w, quad_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize
            import cv2
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Paste into mosaic
            paste_x1 = x1
            paste_y1 = y1
            paste_x2 = min(x1 + new_w, x2)
            paste_y2 = min(y1 + new_h, y2)
            
            img_x2 = paste_x2 - paste_x1
            img_y2 = paste_y2 - paste_y1
            
            mosaic_img[paste_y1:paste_y2, paste_x1:paste_x2] = img_resized[:img_y2, :img_x2]
            
            # Adjust boxes
            if s.boxes is not None and len(s.boxes) > 0:
                boxes = s.boxes.clone() if isinstance(s.boxes, torch.Tensor) else torch.tensor(s.boxes)
                
                # Scale boxes
                boxes[:, [0, 2]] *= scale
                boxes[:, [1, 3]] *= scale
                
                # Translate to quadrant position
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                
                # Clip to mosaic bounds
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, self.output_size)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, self.output_size)
                
                # Filter out boxes that are too small after clipping
                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                valid = (box_w > 2) & (box_h > 2)
                
                if valid.sum() > 0:
                    mosaic_boxes.append(boxes[valid])
                    if s.class_ids is not None:
                        class_ids = s.class_ids if isinstance(s.class_ids, torch.Tensor) else torch.tensor(s.class_ids)
                        mosaic_classes.append(class_ids[valid])
            
            # Handle masks if present
            if mosaic_masks is not None and s.masks is not None:
                # Resize masks to match scaled image
                source_masks = s.masks if isinstance(s.masks, torch.Tensor) else torch.from_numpy(s.masks)
                
                # Resize masks using bilinear interpolation
                if source_masks.ndim == 3:
                    # [N, H, W] format
                    num_masks = source_masks.shape[0]
                    if num_masks > 0:
                        # Resize to scaled dimensions
                        resized_masks = torch.nn.functional.interpolate(
                            source_masks.unsqueeze(0).float(),  # [1, N, H, W]
                            size=(scaled_h, scaled_w),
                            mode='bilinear',
                            align_corners=False
                        )[0]  # [N, H, W]
                        
                        # Apply same validity filter as boxes
                        if valid.sum() > 0:
                            resized_masks = resized_masks[valid]
                            
                            # Place masks in mosaic at correct position
                            for mask_idx in range(resized_masks.shape[0]):
                                mask = (resized_masks[mask_idx] > 0.5).byte()
                                # Copy into mosaic canvas
                                h_end = min(y_off + scaled_h, self.output_size)
                                w_end = min(x_off + scaled_w, self.output_size)
                                h_slice = slice(y_off, h_end)
                                w_slice = slice(x_off, w_end)
                                
                                # Create or expand mosaic_masks if needed
                                if mask_idx >= mosaic_masks.shape[0]:
                                    # Need more mask channels
                                    extra = torch.zeros(
                                        (mask_idx - mosaic_masks.shape[0] + 1, self.output_size, self.output_size),
                                        dtype=torch.uint8
                                    )
                                    mosaic_masks = torch.cat([mosaic_masks, extra], dim=0)
                                
                                # Place mask
                                mask_h = h_end - y_off
                                mask_w = w_end - x_off
                                mosaic_masks[mask_idx, h_slice, w_slice] = torch.maximum(
                                    mosaic_masks[mask_idx, h_slice, w_slice],
                                    mask[:mask_h, :mask_w]
                                )

        
        # Concatenate all boxes and classes
        if len(mosaic_boxes) > 0:
            final_boxes = torch.cat(mosaic_boxes, dim=0)
            final_classes = torch.cat(mosaic_classes, dim=0) if len(mosaic_classes) > 0 else None
        else:
            final_boxes = torch.zeros((0, 4))
            final_classes = torch.zeros((0,), dtype=torch.int64)
        
        # Create output sample
        # Convert back to CHW if needed
        mosaic_img_tensor = torch.from_numpy(mosaic_img).permute(2, 0, 1).float() / 255.0
        
        return TransformSample(
            image=mosaic_img_tensor,
            boxes=final_boxes,
            class_ids=final_classes,
            masks=None,  # Masks not implemented yet
            width=self.output_size,
            height=self.output_size,
        )


class MixUpAugmentation:
    """MixUp augmentation - blend two images with alpha.
    
    MixUp creates convex combinations of two images and their labels.
    This acts as a strong regularizer.
    
    Expected impact: +1-2% mAP
    """
    
    def __init__(
        self,
        dataset,
        alpha: float = 0.5,
        mixup_prob: float = 0.15,
    ) -> None:
        """Initialize MixUp augmentation.
        
        Args:
            dataset: Dataset to sample second image from
            alpha: Mixing coefficient (higher = more blending)
            mixup_prob: Probability of applying mixup
        """
        self.dataset = dataset
        self.alpha = alpha
        self.mixup_prob = mixup_prob
    
    def __call__(self, sample: TransformSample) -> TransformSample:
        """Apply MixUp augmentation.
        
        Args:
            sample: Input sample
        
        Returns:
            Mixed sample
        """
        if random.random() > self.mixup_prob:
            return sample
        
        # Sample second image
        idx = random.randint(0, len(self.dataset) - 1)
        sample2 = self.dataset[idx]
        
        # Random mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        img1 = sample.image if isinstance(sample.image, torch.Tensor) else torch.tensor(sample.image)
        img2 = sample2.image if isinstance(sample2.image, torch.Tensor) else torch.tensor(sample2.image)
        
        # Ensure same size
        if img1.shape != img2.shape:
            import torch.nn.functional as F
            img2 = F.interpolate(
                img2.unsqueeze(0),
                size=img1.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # Merge boxes and classes from both images
        boxes1 = sample.boxes if sample.boxes is not None else torch.zeros((0, 4))
        boxes2 = sample2.boxes if sample2.boxes is not None else torch.zeros((0, 4))
        
        classes1 = sample.class_ids if sample.class_ids is not None else torch.zeros((0,), dtype=torch.int64)
        classes2 = sample2.class_ids if sample2.class_ids is not None else torch.zeros((0,), dtype=torch.int64)
        
        # Scale boxes2 if needed
        if img1.shape != img2.shape:
            scale_y = img1.shape[-2] / sample2.height
            scale_x = img1.shape[-1] / sample2.width
            boxes2 = boxes2.clone()
            boxes2[:, [0, 2]] *= scale_x
            boxes2[:, [1, 3]] *= scale_y
        
        mixed_boxes = torch.cat([boxes1, boxes2], dim=0)
        mixed_classes = torch.cat([classes1, classes2], dim=0)
        
        return TransformSample(
            image=mixed_img,
            boxes=mixed_boxes,
            class_ids=mixed_classes,
            masks=None,
            width=sample.width,
            height=sample.height,
        )


class CopyPasteAugmentation:
    """CopyPaste augmentation - paste object instances from other images.
    
    Extracts objects (with masks) from other images and pastes them onto
    the current image, updating boxes and masks accordingly.
    
    Expected impact: +2-4% instance segmentation AP
    """
    
    def __init__(
        self,
        dataset,
        paste_prob: float = 0.5,
        max_paste: int = 10,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """Initialize CopyPaste augmentation.
        
        Args:
            dataset: Dataset to sample objects from
            paste_prob: Probability of applying copy-paste
            max_paste: Maximum number of objects to paste
            scale_range: Range for random scaling of pasted objects
        """
        self.dataset = dataset
        self.paste_prob = paste_prob
        self.max_paste = max_paste
        self.scale_range = scale_range
    
    def __call__(self, sample: TransformSample) -> TransformSample:
        """Apply CopyPaste augmentation.
        
        Args:
            sample: Input sample
        
        Returns:
            Augmented sample with pasted objects
        """
        if random.random() > self.paste_prob:
            return sample
            
        if sample.masks is None or len(sample.boxes) == 0:
            return sample
        
        import numpy as np
        import cv2
        
        # Convert to numpy for easier manipulation
        img = sample.image
        if isinstance(img, torch.Tensor):
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img = img.copy()
            
        boxes = sample.boxes.numpy() if isinstance(sample.boxes, torch.Tensor) else sample.boxes.copy()
        masks = sample.masks.numpy() if isinstance(sample.masks, torch.Tensor) else sample.masks.copy()
        class_ids = sample.class_ids.numpy() if isinstance(sample.class_ids, torch.Tensor) else sample.class_ids.copy() if sample.class_ids is not None else None
        
        # Sample N objects from other images
        num_paste = random.randint(1, self.max_paste)
        
        for _ in range(num_paste):
            if len(self.dataset) == 0:
                break
                
            # Get random source image
            src_idx = random.randint(0, len(self.dataset) - 1)
            try:
                src_sample = self.dataset[src_idx]
            except:
                continue
                
            if src_sample.masks is None or len(src_sample.boxes) == 0:
                continue
            
            # Pick random object from source
            obj_idx = random.randint(0, len(src_sample.boxes) - 1)
            src_box = src_sample.boxes[obj_idx]
            
            # Extract masks
            src_masks = src_sample.masks.numpy() if isinstance(src_sample.masks, torch.Tensor) else src_sample.masks
            if src_masks.ndim == 3:
                src_mask = src_masks[obj_idx]
            else:
                src_mask = src_masks
            
            # Extract source image
            src_img = src_sample.image
            if isinstance(src_img, torch.Tensor):
                src_img = (src_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Extract object region
            x1, y1, x2, y2 = map(int, src_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(src_img.shape[1], x2), min(src_img.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            obj_img = src_img[y1:y2, x1:x2]
            obj_mask = src_mask[y1:y2, x1:x2]
            
            # Random scale
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_h = max(1, int(obj_img.shape[0] * scale))
            new_w = max(1, int(obj_img.shape[1] * scale))
            
            # Resize object and mask
            obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(
                obj_mask.astype(np.uint8), 
                (new_w, new_h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Random paste position (ensure object fits)
            max_y = max(0, img.shape[0] - new_h)
            max_x = max(0, img.shape[1] - new_w)
            if max_y == 0 or max_x == 0:
                continue
                
            paste_y = random.randint(0, max_y)
            paste_x = random.randint(0, max_x)
            
            # Blend using mask with smooth edges
            mask_3d = np.expand_dims(obj_mask, -1).astype(np.float32)
            
            # Apply Gaussian blur to mask for smoother blending
            if self.blend_alpha < 1.0:
                mask_3d = cv2.GaussianBlur(mask_3d, (5, 5), 0) * self.blend_alpha + mask_3d * (1 - self.blend_alpha)
            
            # Paste object into image
            img[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = (
                img[paste_y:paste_y+new_h, paste_x:paste_x+new_w] * (1 - mask_3d) +
                obj_img * mask_3d
            ).astype(np.uint8)
            
            # Create new box for pasted object
            new_box = np.array([paste_x, paste_y, paste_x + new_w, paste_y + new_h], dtype=np.float32)
            boxes = np.vstack([boxes, new_box])
            
            # Update masks
            new_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            new_mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = obj_mask
            masks = np.concatenate([masks, new_mask[np.newaxis, :, :]], axis=0)
            
            # Update class IDs
            if class_ids is not None and src_sample.class_ids is not None:
                src_class_ids = src_sample.class_ids.numpy() if isinstance(src_sample.class_ids, torch.Tensor) else src_sample.class_ids
                src_class = src_class_ids[obj_idx] if hasattr(src_class_ids, '__getitem__') else src_class_ids
                class_ids = np.append(class_ids, src_class)
        
        # Convert back to tensor format
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.from_numpy(boxes).float()
        masks_tensor = torch.from_numpy(masks).bool()
        class_ids_tensor = torch.from_numpy(class_ids).long() if class_ids is not None else None
        
        return TransformSample(
            image=img_tensor,
            boxes=boxes_tensor,
            masks=masks_tensor,
            class_ids=class_ids_tensor,
            width=sample.width,
            height=sample.height,
        )


__all__ = [
    "MosaicAugmentation",
    "MixUpAugmentation",
    "CopyPasteAugmentation",
]
