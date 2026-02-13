#!/usr/bin/env python3
"""CLI tool to mine hard examples from imagery."""

import argparse
import sys
from pathlib import Path

import cv2
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from apex_x.data.miner import DataMiner
from apex_x.model import TeacherModelV3
from apex_x.train.checkpoint import extract_model_state_dict, safe_torch_load
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Mine hard examples from imagery')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-path', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--output-dir', type=str, default='data/mined_hard_examples')
    parser.add_argument('--num-examples', type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    LOGGER.info(f"Loading model from {args.model_path}")
    checkpoint_payload = safe_torch_load(args.model_path, map_location="cpu")
    state_dict, checkpoint_format = extract_model_state_dict(checkpoint_payload)

    cls_weight = state_dict.get("det_head.stages.0.cls_head.4.weight")
    num_classes = int(cls_weight.shape[0]) if isinstance(cls_weight, torch.Tensor) else 80

    # Initialize Model V3
    model = TeacherModelV3(num_classes=num_classes)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    LOGGER.info(
        "checkpoint_loaded",
        checkpoint_format=checkpoint_format,
        classes=num_classes,
        missing_keys=len(getattr(incompatible, "missing_keys", [])),
        unexpected_keys=len(getattr(incompatible, "unexpected_keys", [])),
    )
    
    miner = DataMiner()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_root = Path(args.image_path)
    if image_root.is_file():
        image_paths = [image_root]
    else:
        image_paths = []
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            image_paths.extend(image_root.glob(suffix))
    
    LOGGER.info(f"Processing {len(image_paths)} images...")
    
    total_mined = 0
    for img_path in image_paths:
        if total_mined >= args.num_examples:
            break
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Simple preprocess
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            output = model(img_t)
            
        hard_indices = miner.find_hard_tiles(output, num_tiles=5)
        
        if hard_indices:
            # Save metadata or export tiles
            # In a real pipeline, we'd crop the tiles and save them for labeling
            LOGGER.info(f"Image {img_path.name}: Found {len(hard_indices)} hard tiles.")
            total_mined += len(hard_indices)
            
    LOGGER.info(f"Mining complete. Identified {total_mined} potential hard examples.")

if __name__ == '__main__':
    main()
