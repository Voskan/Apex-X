"""COCO dataset wrapper for detection and segmentation.

Provides a PyTorch Dataset interface for COCO-format annotations with support
for detection boxes, instance segmentation masks, and integration with
Apex-X's TransformSample format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from apex_x.data.transforms import TransformSample
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

# Lazy import pycocotools (optional dependency)
try:
    from pycocotools.coco import COCO
    
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    LOGGER.warning("pycocotools not installed - COCO dataset will not be available")


class CocoDetectionDataset:
    """COCO detection dataset with instance segmentation support.
    
    Loads images and annotations from COCO-format JSON file and converts
    them to TransformSample format for use with Apex-X training pipeline.
    """
    
    def __init__(
        self,
        root: Path | str,
        ann_file: Path | str,
        transforms: Any | None = None,
        filter_crowd: bool = True,
        remap_categories: bool = True,
    ) -> None:
        """Initialize COCO dataset.
        
        Args:
            root: Root directory containing images
            ann_file: Path to COCO annotation JSON file
            transforms: Optional transform function (receives TransformSample)
            filter_crowd: If True, filter out crowd annotations (iscrowd=1)
            remap_categories: If True, remap category IDs to contiguous [0, N-1]
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise RuntimeError(
                "pycocotools is required for COCO dataset. "
                "Install with: pip install pycocotools"
            )
        
        self.root = Path(root)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        self.filter_crowd = filter_crowd
        
        # Load COCO annotations
        LOGGER.info(f"Loading COCO annotations from {self.ann_file}")
        self.coco = COCO(str(self.ann_file))
        
        # Get all image IDs (filter out images with no annotations if desired)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Build category ID mapping
        self.cat_ids = sorted(self.coco.getCatIds())
        if remap_categories:
            # Remap to contiguous 0-indexed IDs
            self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}
            LOGGER.info(f"Remapped {len(self.cat_ids)} categories to [0, {len(self.cat_ids)-1}]")
        else:
            # Keep original category IDs
            self.cat_id_to_label = {cat_id: cat_id for cat_id in self.cat_ids}
            self.label_to_cat_id = self.cat_id_to_label
        
        self.num_classes = len(self.cat_ids)
        LOGGER.info(f"Loaded {len(self.image_ids)} images with {self.num_classes} categories")
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> TransformSample:
        """Load image and annotations at index.
        
        Args:
            idx: Index of image to load
        
        Returns:
            TransformSample with image, boxes, class IDs, and optional masks
        """
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = self.root / img_info["file_name"]
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Filter crowd annotations if requested
        if self.filter_crowd:
            anns = [ann for ann in anns if ann.get("iscrowd", 0) == 0]
        
        if len(anns) == 0:
            # Empty annotation - return empty arrays
            sample = TransformSample(
                image=image_np,
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                class_ids=np.zeros((0,), dtype=np.int64),
                masks=None,
            )
        else:
            # Extract boxes (COCO format: [x, y, w, h] -> xyxy)
            boxes = []
            class_ids = []
            masks = []
            
            for ann in anns:
                # Convert box format: [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = ann["bbox"]
                x2 = x + w
                y2 = y + h
                boxes.append([x, y, x2, y2])
                
                # Remap category ID
                cat_id = ann["category_id"]
                label = self.cat_id_to_label.get(cat_id, 0)
                class_ids.append(label)
                
                # Extract mask if available
                if "segmentation" in ann and ann["segmentation"]:
                    # Convert segmentation to binary mask
                    mask = self.coco.annToMask(ann)
                    masks.append(mask)
                else:
                    masks.append(None)
            
            # Convert to numpy arrays
            boxes_np = np.array(boxes, dtype=np.float32)
            class_ids_np = np.array(class_ids, dtype=np.int64)
            
            # Stack masks if all are present, else None
            if all(m is not None for m in masks):
                masks_np = np.stack(masks, axis=0).astype(np.float32)
            else:
                masks_np = None
            
            sample = TransformSample(
                image=image_np,
                boxes_xyxy=boxes_np,
                class_ids=class_ids_np,
                masks=masks_np,
            )
        
        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample, rng=np.random.RandomState())
        
        return sample
    
    def get_category_name(self, label: int) -> str:
        """Get category name for a label.
        
        Args:
            label: Remapped label ID (0-indexed)
        
        Returns:
            Category name string
        """
        cat_id = self.label_to_cat_id.get(label, -1)
        if cat_id == -1:
            return "unknown"
        
        cats = self.coco.loadCats(cat_id)
        return cats[0]["name"] if cats else "unknown"


def coco_collate_fn(batch: list[TransformSample]) -> dict[str, Any]:
    """Collate COCO samples into batch tensors.
    
    Handles variable-size annotations by returning lists of tensors
    instead of stacked tensors.
    
    Args:
        batch: List of TransformSample objects
    
    Returns:
        Dict with:
            - images: [B, 3, H, W] tensor (if all same size) or list
            - boxes: list of [N_i, 4] tensors
            - class_ids: list of [N_i] tensors  
            - masks: list of [N_i, H, W] tensors or None
            - image_ids: list of int (if available)
    """
    import torch
    
    images = []
    boxes = []
    class_ids = []
    masks = []
    
    for sample in batch:
        # Convert image to tensor [3, H, W]
        img_tensor = torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
        
        # Keep boxes and class_ids as tensors (variable size)
        boxes.append(torch.from_numpy(sample.boxes_xyxy).float())
        class_ids.append(torch.from_numpy(sample.class_ids).long())
        
        # Masks (if present)
        if sample.masks is not None:
            masks.append(torch.from_numpy(sample.masks).float())
        else:
            masks.append(None)
    
    # Try to stack images if all same size, else return list
    try:
        images_tensor = torch.stack(images, dim=0)
    except RuntimeError:
        images_tensor = images  # Keep as list if sizes differ
    
    return {
        "images": images_tensor,
        "boxes": boxes,
        "class_ids": class_ids,
        "masks": masks if any(m is not None for m in masks) else None,
    }


__all__ = [
    "CocoDetectionDataset",
    "coco_collate_fn",
    "PYCOCOTOOLS_AVAILABLE",
]
