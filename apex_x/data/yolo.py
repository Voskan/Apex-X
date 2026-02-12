"""YOLO segmentation dataset support for Apex-X.

Reads YOLO-format .txt labels with polygon coordinates and data.yaml config.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
                    # CLIP polygons to [0, 1] range first to be safe
                    coords[:, 0] = np.clip(coords[:, 0], 0.0, 1.0)
                    coords[:, 1] = np.clip(coords[:, 1], 0.0, 1.0)
                    
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    
                    # Compute bbox from polygon
                    x1, y1 = coords.min(axis=0)
                    x2, y2 = coords.max(axis=0)
                    
                    # Strict clipping to image boundaries
                    x1 = np.clip(x1, 0.0, float(w))
                    x2 = np.clip(x2, 0.0, float(w))
                    y1 = np.clip(y1, 0.0, float(h))
                    y2 = np.clip(y2, 0.0, float(h))
                    
                    # Filter degenerate boxes (width or height <= 0.1 pixels)
                    if (x2 - x1) > 0.1 and (y2 - y1) > 0.1:
                        boxes.append([x1, y1, x2, y2])
                        class_ids.append(cls_id)
                        polygons.append(coords)
                    else:
                        LOGGER.debug(f"Skipping degenerate polygon in {lbl_path.name} (width: {x2-x1:.4f}, height: {y2-y1:.4f})")
                    
        # Convert to tensors (using numpy for TransformSample as per its definition)
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        class_ids_np = np.array(class_ids, dtype=np.int64) if class_ids else np.zeros((0,), dtype=np.int64)
        
        # Generate compact binary masks from polygons.
        # Keep uint8 on CPU to avoid 4x RAM overhead from float32 masks.
        masks_np = None
        if polygons:
            masks_list = []
            for poly in polygons:
                mask = np.zeros((h, w), dtype=np.uint8)
                # Convert polygon to integer coordinates for cv2.fillPoly
                poly_int = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly_int], 1)
                masks_list.append(mask)
            masks_np = np.stack(masks_list, axis=0)  # [N, H, W], uint8
        
        # TransformSample expects np.ndarray for image, boxes_xyxy, and class_ids
        sample = TransformSample(
            image=image,
            boxes_xyxy=boxes_np,
            class_ids=class_ids_np,
            masks=masks_np,
        )
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

def yolo_collate_fn(batch: list[TransformSample]) -> list[TransformSample]:
    """Collate function for YOLO dataset."""
    return batch
