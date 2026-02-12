"""Data quality filtering for cleaner training.

Filters low-quality images based on:
- Entropy (information content)
- Sharpness (blur detection)
- Cloud coverage (for satellite imagery)
- Minimum object count
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from PIL import Image
import cv2


class ImageQualityFilter:
    """Filter images based on quality metrics.
    
    Args:
        min_entropy: Minimum entropy threshold (0-8, higher = more info)
        min_sharpness: Minimum sharpness score (0-1000+)
        max_cloud_coverage: Maximum cloud coverage ratio (0-1)
        min_objects: Minimum number of objects required
        enable_entropy: Enable entropy filtering
        enable_sharpness: Enable sharpness filtering
        enable_cloud: Enable cloud coverage filtering (satellite only)
        enable_min_objects: Enable minimum object count filtering
    """
    
    def __init__(
        self,
        min_entropy: float = 4.0,
        min_sharpness: float = 100.0,
        max_cloud_coverage: float = 0.3,
        min_objects: int = 1,
        enable_entropy: bool = True,
        enable_sharpness: bool = True,
        enable_cloud: bool = False,
        enable_min_objects: bool = True,
    ):
        self.min_entropy = min_entropy
        self.min_sharpness = min_sharpness
        self.max_cloud_coverage = max_cloud_coverage
        self.min_objects = min_objects
        
        self.enable_entropy = enable_entropy
        self.enable_sharpness = enable_sharpness
        self.enable_cloud = enable_cloud
        self.enable_min_objects = enable_min_objects
    
    def filter_image(
        self,
        image: np.ndarray | Image.Image,
        annotations: list[dict] | None = None,
    ) -> tuple[bool, dict[str, float]]:
        """Check if image passes quality filters.
        
        Args:
            image: Input image (numpy array or PIL Image)
            annotations: Optional list of annotations (for object count)
            
        Returns:
            Tuple of (passes_filter, metrics_dict)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        metrics = {}
        passes = True
        
        # Entropy check
        if self.enable_entropy:
            entropy = self._compute_entropy(image)
            metrics['entropy'] = entropy
            if entropy < self.min_entropy:
                passes = False
        
        # Sharpness check
        if self.enable_sharpness:
            sharpness = self._compute_sharpness(image)
            metrics['sharpness'] = sharpness
            if sharpness < self.min_sharpness:
                passes = False
        
        # Cloud coverage check (satellite)
        if self.enable_cloud:
            cloud_coverage = self._estimate_cloud_coverage(image)
            metrics['cloud_coverage'] = cloud_coverage
            if cloud_coverage > self.max_cloud_coverage:
                passes = False
        
        # Object count check
        if self.enable_min_objects and annotations is not None:
            num_objects = len(annotations)
            metrics['num_objects'] = num_objects
            if num_objects < self.min_objects:
                passes = False
        
        return passes, metrics
    
    def _compute_entropy(self, image: np.ndarray) -> float:
        """Compute image entropy (information content).
        
        Higher entropy = more information/detail
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        
        # Normalize
        hist = hist / hist.sum()
        
        # Compute entropy
        hist = hist[hist > 0]  # Avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    def _compute_sharpness(self, image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance.
        
        Higher variance = sharper image
        Lower variance = blurry image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Variance of Laplacian
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    def _estimate_cloud_coverage(self, image: np.ndarray) -> float:
        """Estimate cloud coverage ratio for satellite imagery.
        
        Simple heuristic: bright pixels in satellite images often indicate clouds.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold for bright pixels (clouds)
        # Typically clouds are > 200 in 0-255 range
        cloud_mask = gray > 200
        
        # Cloud coverage ratio
        coverage = cloud_mask.sum() / gray.size
        
        return float(coverage)


class DatasetQualityFilter:
    """Filter entire dataset based on quality metrics.
    
    Wraps a dataset and filters out low-quality samples.
    """
    
    def __init__(
        self,
        dataset,
        quality_filter: ImageQualityFilter,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.quality_filter = quality_filter
        self.verbose = verbose
        
        # Build valid indices
        self.valid_indices = self._filter_dataset()
    
    def _filter_dataset(self) -> list[int]:
        """Filter dataset and return valid indices."""
        valid_indices = []
        
        total = len(self.dataset)
        filtered_count = 0
        
        for idx in range(total):
            try:
                sample = self.dataset[idx]
                
                # Extract image and annotations
                if isinstance(sample, dict):
                    image = sample.get('image')
                    annotations = sample.get('annotations')
                elif isinstance(sample, (tuple, list)):
                    image = sample[0]
                    annotations = sample[1] if len(sample) > 1 else None
                else:
                    image = sample
                    annotations = None
                
                # Check quality
                passes, metrics = self.quality_filter.filter_image(image, annotations)
                
                if passes:
                    valid_indices.append(idx)
                else:
                    filtered_count += 1
                    if self.verbose and filtered_count <= 10:
                        print(f"Filtered sample {idx}: {metrics}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error filtering sample {idx}: {e}")
                filtered_count += 1
        
        if self.verbose:
            print(f"\nFiltered {filtered_count}/{total} samples ({100*filtered_count/total:.1f}%)")
            print(f"Remaining: {len(valid_indices)} samples")
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        return self.dataset[real_idx]


__all__ = ['ImageQualityFilter', 'DatasetQualityFilter']
