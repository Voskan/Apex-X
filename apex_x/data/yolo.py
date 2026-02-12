"""YOLO segmentation dataset support for Apex-X.

Reads YOLO-format .txt labels with polygon coordinates and data.yaml config.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from apex_x.data.transforms import TransformSample
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


class YOLOSegmentationDataset(Dataset):
    """Dataset for YOLO segmentation format.
    
    Expected directory structure:
    path/to/dataset/
        data.yaml
        train/
            images/
            labels/
        val/
            images/
            labels/
            
    Labels are .txt files where each line is:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    (normalized coordinates)
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: Any = None,
        image_size: int = 640,
    ) -> None:
        """Initialize YOLO Segmentation Dataset.
        
        Args:
            root: Root path of the dataset
            split: Split name ('train', 'val', 'test')
            transforms: Augmentations/transforms
            image_size: Target image size
        """
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        
        # Load data.yaml
        yaml_path = self.root / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
            
        self.classes = self.data_config.get("names", [])
        self.num_classes = len(self.classes)
        
        # Determine image and label paths
        # data.yaml typically has: train: train/images
        split_path = self.data_config.get(split)
        if split_path is None:
            # Fallback to standard structure
            images_dir = self.root / split / "images"
        else:
            images_dir = self.root / split_path
            
        if not images_dir.is_absolute():
            images_dir = self.root / images_dir
            
        labels_dir = images_dir.parent / "labels"
        
        self.image_files = sorted([
            f for f in images_dir.iterdir() 
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        ])
        
        self.label_files = []
        for img_file in self.image_files:
            lbl_file = labels_dir / f"{img_file.stem}.txt"
            self.label_files.append(lbl_file if lbl_file.exists() else None)
            
        LOGGER.info(f"Loaded {len(self.image_files)} images for split '{split}'")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> TransformSample:
        """Get image and labels for given index."""
        img_path = self.image_files[idx]
        lbl_path = self.label_files[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        boxes = []
        class_ids = []
        polygons = []
        
        if lbl_path and lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5:
                        continue
                        
                    cls_id = int(parts[0])
                    coords = np.array(parts[1:]).reshape(-1, 2)
                    
                    # Denormalize coordinates
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    
                    # Compute bbox from polygon
                    x1, y1 = coords.min(axis=0)
                    x2, y2 = coords.max(axis=0)
                    
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(cls_id)
                    polygons.append(coords)
                    
        # Convert to tensors (using numpy for TransformSample as per its definition)
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        class_ids_np = np.array(class_ids, dtype=np.int64) if class_ids else np.zeros((0,), dtype=np.int64)
        
        # TransformSample expects np.ndarray for image, boxes_xyxy, and class_ids
        sample = TransformSample(
            image=image,
            boxes_xyxy=boxes_np,
            class_ids=class_ids_np,
        )
        
        if self.transforms:
            # Note: Transform classes in apex_x typically expect a 'rng' argument.
            # TransformPipeline handles this, but individual calls might need it.
            try:
                sample = self.transforms(sample)
            except TypeError:
                # If transforms is a function that doesn't take rng
                sample = self.transforms(sample)
            
        return sample

def yolo_collate_fn(batch: list[TransformSample]) -> list[TransformSample]:
    """Collate function for YOLO dataset."""
    return batch
