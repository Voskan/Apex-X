"""Large Scale Jittering (LSJ) augmentation.

Extreme scale variation during training for improved multi-scale robustness.
Used in Mask2Former and modern detectors for +1-2% mAP.

Reference: https://arxiv.org/abs/2103.12340
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from apex_x.data.transforms import TransformSample


class LargeScaleJitter:
    """Large Scale Jittering augmentation.
    
    Applies extreme scale variations (0.1x to 2.0x) followed by random
    cropping/padding to original size. Improves multi-scale robustness.
    
    Args:
        output_size: Target output size (height, width)
        min_scale: Minimum scale factor (default: 0.1)
        max_scale: Maximum scale factor (default: 2.0)
        lsj_prob: Probability of applying LSJ (default: 0.5)
    
    Expected impact: +1-2% mAP, especially on objects at extreme scales
    """
    
    def __init__(
        self,
        output_size: int | tuple[int, int] = 640,
        min_scale: float = 0.1,
        max_scale: float = 2.0,
        lsj_prob: float = 0.5,
    ) -> None:
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.lsj_prob = lsj_prob
    
    def __call__(
        self,
        sample: TransformSample,
        rng: np.random.RandomState | None = None,
    ) -> TransformSample:
        """Apply LSJ augmentation.
        
        Args:
            sample: Input sample with image and annotations
            rng: Random number generator
        
        Returns:
            Augmented sample with LSJ applied
        """
        if rng is None:
            rng = np.random.RandomState()

        if float(rng.rand()) > self.lsj_prob:
            return sample  # No augmentation
        
        image = sample.image
        boxes = sample.boxes_xyxy
        class_ids = sample.class_ids
        masks = sample.masks
        
        h, w = image.shape[:2]
        target_h, target_w = self.output_size
        
        # Sample random scale
        scale = rng.uniform(self.min_scale, self.max_scale)
        
        # Calculate new size
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        image_pil = Image.fromarray(image)
        image_resized = np.array(
            image_pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        )
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes_scaled = boxes * scale
        else:
            boxes_scaled = boxes
        
        # Resize masks if present
        if masks is not None:
            masks_resized = []
            for mask in masks:
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
                masks_resized.append(np.array(mask_resized) / 255.0)
            masks_resized = np.stack(masks_resized, axis=0)
        else:
            masks_resized = None
        
        # Crop or pad to target size
        if scale > 1.0:
            # Scale up: random crop
            image_final, boxes_final, masks_final = self._random_crop(
                image_resized,
                boxes_scaled,
                masks_resized,
                target_h,
                target_w,
                rng,
            )
        else:
            # Scale down: pad to center
            image_final, boxes_final, masks_final = self._pad_to_size(
                image_resized,
                boxes_scaled,
                masks_resized,
                target_h,
                target_w,
            )
        
        return TransformSample(
            image=image_final,
            boxes_xyxy=boxes_final,
            class_ids=class_ids,
            masks=masks_final,
        )
    
    def _random_crop(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray | None,
        target_h: int,
        target_w: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Random crop from larger image."""
        h, w = image.shape[:2]
        
        # Random crop position
        y_offset = rng.randint(0, max(1, h - target_h))
        x_offset = rng.randint(0, max(1, w - target_w))
        
        # Crop image
        image_cropped = image[
            y_offset:y_offset + target_h,
            x_offset:x_offset + target_w
        ]
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes_adjusted = boxes.copy()
            boxes_adjusted[:, [0, 2]] -= x_offset
            boxes_adjusted[:, [1, 3]] -= y_offset
            
            # Clip to image bounds
            boxes_adjusted[:, [0, 2]] = np.clip(boxes_adjusted[:, [0, 2]], 0, target_w)
            boxes_adjusted[:, [1, 3]] = np.clip(boxes_adjusted[:, [1, 3]], 0, target_h)
            
            # Filter out boxes that became too small
            widths = boxes_adjusted[:, 2] - boxes_adjusted[:, 0]
            heights = boxes_adjusted[:, 3] - boxes_adjusted[:, 1]
            valid = (widths > 1) & (heights > 1)
            
            boxes_adjusted = boxes_adjusted[valid]
        else:
            boxes_adjusted = boxes
        
        # Crop masks
        if masks is not None:
            masks_cropped = masks[
                :,
                y_offset:y_offset + target_h,
                x_offset:x_offset + target_w
            ]
            if len(boxes_adjusted) < len(masks):
                # Filter masks to match boxes
                masks_cropped = masks_cropped[:len(boxes_adjusted)]
        else:
            masks_cropped = None
        
        return image_cropped, boxes_adjusted, masks_cropped
    
    def _pad_to_size(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray | None,
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Pad smaller image to target size (centered)."""
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Center padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad image
        image_padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes_adjusted = boxes.copy()
            boxes_adjusted[:, [0, 2]] += pad_left
            boxes_adjusted[:, [1, 3]] += pad_top
        else:
            boxes_adjusted = boxes
        
        # Pad masks
        if masks is not None:
            masks_padded = np.pad(
                masks,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0,
            )
        else:
            masks_padded = None
        
        return image_padded, boxes_adjusted, masks_padded


__all__ = ['LargeScaleJitter']
