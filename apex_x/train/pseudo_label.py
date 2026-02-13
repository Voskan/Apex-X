"""Curriculum Pseudo-Label Generator for Semi-Supervised Learning.

1. Loads a trained TeacherModelV3.
2. Infers on unlabeled images.
3. Filters predictions by:
   - Box Confidence > 0.8
   - Mask Quality > 0.85
4. Saves high-quality instances as COCO annotations.

Usage:
    python -m apex_x.train.pseudo_label \
        --model_path outputs/best/model.pt \
        --data_dir /path/to/unlabeled \
        --output_path /path/to/pseudo.json
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from apex_x.model.teacher_v3 import TeacherModelV3
from apex_x.train.checkpoint import safe_torch_load
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


def generate_pseudo_labels(
    model_path: str,
    data_dir: str,
    output_path: str,
    conf_threshold: float = 0.8,
    quality_threshold: float = 0.85,
    device: str = "cuda",
):
    """Generate pseudo-labels from unlabeled data."""
    
    # 1. Load Model
    LOGGER.info(f"Loading model from {model_path}...")
    ckpt = safe_torch_load(model_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    
    model = TeacherModelV3(
        num_classes=80, # Configurable
        backbone_model="facebook/dinov2-large" # Configurable
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Scan Data
    image_paths = sorted(list(Path(data_dir).glob("*.jpg")) + list(Path(data_dir).glob("*.png")))
    LOGGER.info(f"Found {len(image_paths)} images.")
    
    annotations = []
    images_info = []
    ann_id = 1
    
    # 3. Inference Loop
    LOGGER.info("Starting inference...")
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_paths)):
            # Load Image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(device).unsqueeze(0)
            
            # Predict
            outputs = model(img_tensor)
            
            # Filter
            boxes = outputs["boxes"]
            scores = outputs["scores"]
            masks = outputs["masks"]
            quality = outputs["predicted_quality"]
            classes = torch.zeros_like(scores) # Placeholder if not present
            
            # Apply thresholds
            keep = (scores > conf_threshold)
            if quality is not None:
                keep = keep & (quality > quality_threshold)
            
            if not keep.any():
                continue
            
            # Save Image Info
            images_info.append({
                "id": i + 1,
                "file_name": img_path.name,
                "height": h,
                "width": w,
            })
            
            # Save Annotations
            valid_idx = torch.where(keep)[0]
            for idx in valid_idx:
                box = boxes[idx].cpu().numpy()
                score = float(scores[idx])
                
                # Mask RLE
                if masks is not None:
                    mask_prob = masks[idx, 0].cpu().numpy()
                    mask_bin = (mask_prob > 0.5).astype("uint8")
                    rle = mask_utils.encode(np.asfortranarray(mask_bin))
                    rle["counts"] = rle["counts"].decode("utf-8")
                else:
                    rle = None
                
                annotations.append({
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": 1, # Default class
                    "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                    "score": score,
                    "segmentation": rle,
                    "iscrowd": 0,
                    "area": float(mask_utils.area(rle)) if rle else 0.0
                })
                ann_id += 1
                
    # 4. Save
    coco_output = {
        "images": images_info,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "roof"}]
    }
    
    with open(output_path, "w") as f:
        json.dump(coco_output, f)
        
    LOGGER.info(f"Saved {len(annotations)} pseudo-labels to {output_path}")


if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    
    generate_pseudo_labels(args.model_path, args.data_dir, args.output_path)
