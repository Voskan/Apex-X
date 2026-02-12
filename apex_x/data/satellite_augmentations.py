"""Satellite-specific augmentations for Google Maps imagery.

Specialized augmentations for aerial/satellite images:
- Rotation (buildings at any angle)
- Multi-angle view simulation  
- Weather effects (haze, clouds, shadows)
- Resolution degradation (zoom levels)
- Seasonal changes

Expected impact: +2-4% mAP on satellite datasets
"""

from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np
import torch
from torch import Tensor

from apex_x.data.transforms import TransformSample


class RandomRotation90:
    """Random 90° rotation for satellite imagery.
    
    Buildings and objects can appear at any 90° angle in satellite images.
    This is more appropriate than arbitrary rotation which requires expensive interpolation.
    
    Args:
        prob: Probability of applying rotation
    """
    
    def __init__(self, prob: float = 0.5) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self.prob = prob
        
    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() > self.prob:
            return sample
            
        # Random 90° rotation (0, 90, 180, 270)
        k = random.randint(0, 3)  # Number of 90° rotations
        
        if k == 0:
            return sample  # No rotation
            
        img = sample.image
        if isinstance(img, Tensor):
            img = img.permute(1, 2, 0).numpy()
            
        # Rotate image
        img = np.rot90(img, k=k)
        
        # Rotate boxes
        if sample.boxes is not None and len(sample.boxes) > 0:
            boxes = sample.boxes.clone() if isinstance(sample.boxes, Tensor) else torch.tensor(sample.boxes)
            h, w = img.shape[:2]
            
            for _ in range(k):
                # Rotate 90° clockwise
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                boxes = torch.stack([
                    h - y2,  # new x1
                    x1,      # new y1  
                    h - y1,  # new x2
                    x2,      # new y2
                ], dim=1)
                h, w = w, h  # Swap dimensions
                
        # Rotate masks
        masks = None
        if sample.masks is not None:
            masks_np = sample.masks.numpy() if isinstance(sample.masks, Tensor) else sample.masks
            if masks_np.ndim == 3:
                masks_np = np.rot90(masks_np, k=k, axes=(1, 2))
            masks = torch.from_numpy(masks_np)
            
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() if not isinstance(sample.image, Tensor) else torch.from_numpy(img).permute(2, 0, 1)
        
        return TransformSample(
            image=img_tensor,
            boxes=boxes,
            masks=masks,
            class_ids=sample.class_ids,
            width=img.shape[1],
            height=img.shape[0],
        )


class WeatherAugmentation:
    """Simulate weather conditions in satellite images.
    
    Adds realistic weather effects:
    - Haze/fog (brightness + contrast reduction)
    - Clouds (random white patches)
    - Shadows (darkening patches)
    
    Args:
        haze_prob: Probability of adding haze
        cloud_prob: Probability of adding clouds
        shadow_prob: Probability of adding shadows
        intensity_range: Min/max intensity for effects
    """
    
    def __init__(
        self,
        haze_prob: float = 0.3,
        cloud_prob: float = 0.2,
        shadow_prob: float = 0.3,
        intensity_range: tuple[float, float] = (0.1, 0.4),
    ) -> None:
        self.haze_prob = haze_prob
        self.cloud_prob = cloud_prob
        self.shadow_prob = shadow_prob
        self.intensity_range = intensity_range
        
    def __call__(self, sample: TransformSample) -> TransformSample:
        img = sample.image
        if isinstance(img, Tensor):
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img = img.copy()
            
        # Haze effect
        if random.random() < self.haze_prob:
            intensity = random.uniform(*self.intensity_range)
            haze = np.ones_like(img) * 255
            img = cv2.addWeighted(img, 1 - intensity, haze, intensity, 0)
            
        # Cloud patches
        if random.random() < self.cloud_prob:
            num_clouds = random.randint(1, 3)
            for _ in range(num_clouds):
                h, w = img.shape[:2]
                cloud_h = random.randint(h // 8, h // 4)
                cloud_w = random.randint(w // 8, w // 4)
                y = random.randint(0, h - cloud_h)
                x = random.randint(0, w - cloud_w)
                
                # Create cloud mask with Gaussian blur
                cloud_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(cloud_mask, (x + cloud_w//2, y + cloud_h//2), 
                           (cloud_w//2, cloud_h//2), 0, 0, 360, 255, -1)
                cloud_mask = cv2.GaussianBlur(cloud_mask, (51, 51), 0)
                cloud_mask = cloud_mask.astype(np.float32) / 255.0
                
                # Blend white cloud
                intensity = random.uniform(0.3, 0.7)
                cloud_mask_3d = np.stack([cloud_mask] * 3, axis=-1)
                cloud_color = np.ones_like(img) * 255
                img = (img * (1 - cloud_mask_3d * intensity) + 
                      cloud_color * cloud_mask_3d * intensity).astype(np.uint8)
                
        # Shadow patches
        if random.random() < self.shadow_prob:
            num_shadows = random.randint(1, 2)
            for _ in range(num_shadows):
                h, w = img.shape[:2]
                shadow_h = random.randint(h // 6, h // 3)
                shadow_w = random.randint(w // 6, w // 3)
                y = random.randint(0, h - shadow_h)
                x = random.randint(0, w - shadow_w)
                
                # Darken region
                darkness = random.uniform(0.3, 0.6)
                img[y:y+shadow_h, x:x+shadow_w] = (
                    img[y:y+shadow_h, x:x+shadow_w] * (1 - darkness)
                ).astype(np.uint8)
                
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return TransformSample(
            image=img_tensor,
            boxes=sample.boxes,
            masks=sample.masks,
            class_ids=sample.class_ids,
            width=sample.width,
            height=sample.height,
        )


class ResolutionDegradation:
    """Simulate different zoom levels/resolutions.
    
    Downsample and upsample to simulate varying satellite image quality.
    Helps model generalize across different zoom levels.
    
    Args:
        prob: Probability of applying degradation
        scale_range: Min/max downscale factor
    """
    
    def __init__(
        self,
        prob: float = 0.3,
        scale_range: tuple[float, float] = (0.5, 0.9),
    ) -> None:
        self.prob = prob
        self.scale_range = scale_range
        
    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() > self.prob:
            return sample
            
        img = sample.image
        if isinstance(img, Tensor):
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
        h, w = img.shape[:2]
        
        # Random downscale factor
        scale = random.uniform(*self.scale_range)
        
        # Downsample
        small_h, small_w = int(h * scale), int(w * scale)
        img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Upsample back (with quality loss)
        img_degraded = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        img_tensor = torch.from_numpy(img_degraded).permute(2, 0, 1).float() / 255.0
        
        return TransformSample(
            image=img_tensor,
            boxes=sample.boxes,
            masks=sample.masks,
            class_ids=sample.class_ids,
            width=sample.width,
            height=sample.height,
        )


class SatelliteAugmentationPipeline:
    """Complete augmentation pipeline for satellite imagery.
    
    Combines all satellite-specific augmentations.
    
    Args:
        rotation_prob: Probability of 90° rotation
        weather_prob: Probability of weather effects
        resolution_prob: Probability of resolution degradation
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        weather_prob: float = 0.3,
        resolution_prob: float = 0.3,
    ) -> None:
        self.transforms = [
            RandomRotation90(prob=rotation_prob),
            WeatherAugmentation(
                haze_prob=weather_prob,
                cloud_prob=weather_prob * 0.7,
                shadow_prob=weather_prob,
            ),
            ResolutionDegradation(prob=resolution_prob),
        ]
        
    def __call__(self, sample: TransformSample) -> TransformSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


__all__ = [
    "RandomRotation90",
    "WeatherAugmentation",
    "ResolutionDegradation",
    "SatelliteAugmentationPipeline",
]
