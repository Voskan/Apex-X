"""SAM-2 Label Refinement Tool.

Uses Segment Anything 2 to refine or generate masks from bounding box annotations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def refine_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    model_cfg: str = "sam2_hiera_l.yaml",
    checkpoint: str = "checkpoints/sam2_hiera_large.pt",
    device: str = "cuda",
) -> None:
    """Refine dataset labels using SAM-2."""
    if not SAM2_AVAILABLE:
        LOGGER.error("SAM-2 is not installed. Please install 'sam2' package.")
        return

    LOGGER.info(f"Loading SAM-2 from {checkpoint}...")
    # Assume user has downloaded checkpoints. 
    # If not, we might fail, but that's expected for this tool.
    if not os.path.exists(checkpoint):
        LOGGER.warning(f"Checkpoint {checkpoint} not found. Ensure you have downloaded SAM-2 weights.")
        
    try:
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        LOGGER.error(f"Failed to load SAM-2: {e}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    
    LOGGER.info(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        # Load image
        image = load_image(str(img_path))
        predictor.set_image(image)
        
        # Load corresponding label (YOLO format expected: class, x, y, w, h normalized)
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        h, w = image.shape[:2]
        
        boxes = []
        class_ids = []
        
        for line in lines:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            cx, cy, bw, bh = parts[1:5]
            
            # Convert to xyxy pixel coords
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            
            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls_id)
            
        if not boxes:
            continue
            
        boxes_np = np.array(boxes)
        
        # Query SAM-2
        # Predict masks for all boxes
        # SAM-2 predictor can take batch of boxes? 
        # API: predict(point_coords=None, point_labels=None, box=None, multimask_output=True)
        # box should be [xyxy] or [N, 4]
        
        # iterate boxes (batching might be supported depending on version, safe loop)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False, # We want best mask per box
        )
        # masks: [N, 1, H, W] -> Squeeze to [N, H, W]
        masks = masks.squeeze(1)
        
        # Save new labels (YOLO Segmentation format: class x1 y1 x2 y2 ... polygon)
        # Or standard YOLO mask format
        
        target_file = output_path / f"{img_path.stem}.txt"
        with open(target_file, "w") as f:
            for i, mask in enumerate(masks):
                cls_id = class_ids[i]
                
                # Convert binary mask to polygon
                # cv2.findContours
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Find largest contour
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                
                # Normalize points
                points = c.reshape(-1, 2) / np.array([w, h])
                
                # Write: class p1x p1y p2x p2y ...
                line_str = f"{cls_id}"
                for p in points:
                    line_str += f" {p[0]:.6f} {p[1]:.6f}"
                f.write(line_str + "\n")
                
    LOGGER.info(f"Refinement complete. Saved to {output_dir}")

