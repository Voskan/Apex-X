import os
import torch
from apex_x.data import YOLOSegmentationDataset

def test_yolo_dataset():
    root = "/media/voskan/New Volume/2TB HDD/YOLO26_SUPER_MERGED"
    
    try:
        dataset = YOLOSegmentationDataset(root=root, split="train")
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Number of images: {len(dataset)}")
        print(f"✓ Classes: {dataset.classes}")
        print(f"✓ Number of classes: {dataset.num_classes}")
        
        sample = dataset[0]
        print(f"✓ Sample 0 loaded")
        print(f"✓ Image shape: {sample.image.shape}")
        print(f"✓ Boxes (boxes_xyxy): {sample.boxes_xyxy.shape}")
        print(f"✓ Class IDs: {sample.class_ids.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

if __name__ == "__main__":
    test_yolo_dataset()
