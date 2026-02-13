
import torch
import numpy as np
import cv2
import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apex_x.data.augmentations import CopyPasteAugmentation, MosaicAugmentation, MixUpAugmentation
from apex_x.data.transforms import TransformSample, build_robust_transforms
from apex_x.data.dataset_wrappers import AugmentedDataset

class MockDataset:
    def __init__(self, length=10):
        self.length = length
        
    def __len__(self):
        return int(self.length)
        
    def __getitem__(self, idx):
        # Create a 256x256 image with a random color square
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        
        # Draw a square based on index to identify source
        x1 = 50 + (idx % 3) * 20
        y1 = 50 + (idx % 3) * 20
        w, h = 100, 100
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color.tolist(), -1)
        
        boxes = np.array([[x1, y1, x1+w, y1+h]], dtype=np.float32)
        classes = np.array([1], dtype=np.int64)
        
        # Mask matching the box
        mask = np.zeros((256, 256), dtype=bool)
        mask[y1:y1+h, x1:x1+w] = True
        masks = mask[None, ...] # [1, H, W]
        
        return TransformSample(
            image=img,
            boxes_xyxy=boxes,
            class_ids=classes,
            masks=masks
        )

def verify_augmentations():
    print("--- Verifying Augmentations ---")
    
    # Setup Output Dir
    output_dir = "artifacts/verify_aug"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Mock Dataset
    raw_dataset = MockDataset(length=20)
    
    # 2. Build Pipeline
    # Mosaic
    mosaic = MosaicAugmentation(
        dataset=raw_dataset,
        output_size=512,
        mosaic_prob=1.0 # Force Mosaic
    )
    
    # CopyPaste
    copypaste = CopyPasteAugmentation(
        dataset=raw_dataset,
        paste_prob=1.0, # Force CopyPaste
        max_paste=3
    )
    
    # MixUp
    mixup = MixUpAugmentation(
        dataset=raw_dataset,
        mixup_prob=1.0 # Force MixUp
    )
    
    # Base Transform (Resize)
    base = build_robust_transforms(height=512, width=512)
    
    class CompositePipeline:
        def __init__(self, augs, base):
            self.augs = augs
            self.base = base
            
        def __call__(self, sample, rng=None):
            for aug in self.augs:
                sample = aug(sample)
            return self.base(sample, rng=rng)

    # Test 1: Mosaic Only
    print("Generating Mosaic samples...")
    pipeline_mosaic = CompositePipeline([mosaic], base)
    ds_mosaic = AugmentedDataset(raw_dataset, pipeline_mosaic)
    for i in range(3):
        sample = ds_mosaic[i]
        path = os.path.join(output_dir, f"mosaic_{i}.jpg")
        cv2.imwrite(path, cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        print(f"Saved {path} (Boxes: {len(sample.boxes_xyxy)})")
        
    # Test 2: CopyPaste Only
    print("Generating CopyPaste samples...")
    pipeline_cp = CompositePipeline([copypaste], base)
    ds_cp = AugmentedDataset(raw_dataset, pipeline_cp)
    for i in range(3):
        sample = ds_cp[i]
        path = os.path.join(output_dir, f"copypaste_{i}.jpg")
        cv2.imwrite(path, cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        print(f"Saved {path} (Boxes: {len(sample.boxes_xyxy)})")
        
    # Test 3: MixUp Only
    print("Generating MixUp samples...")
    pipeline_mix = CompositePipeline([mixup], base)
    ds_mix = AugmentedDataset(raw_dataset, pipeline_mix)
    for i in range(3):
        sample = ds_mix[i]
        path = os.path.join(output_dir, f"mixup_{i}.jpg")
        cv2.imwrite(path, cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        print(f"Saved {path} (Boxes: {len(sample.boxes_xyxy)})")

    # Test 4: All Combined
    print("Generating Combined samples...")
    pipeline_all = CompositePipeline([mosaic, copypaste, mixup], base)
    ds_all = AugmentedDataset(raw_dataset, pipeline_all)
    for i in range(3):
        sample = ds_all[i]
        path = os.path.join(output_dir, f"combined_{i}.jpg")
        cv2.imwrite(path, cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
        print(f"Saved {path} (Boxes: {len(sample.boxes_xyxy)})")

if __name__ == "__main__":
    verify_augmentations()
