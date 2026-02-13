
import torch
import os
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apex_x.config import ApexXConfig
from apex_x.train.trainer import ApexXTrainer
from apex_x.model.teacher_v3 import TeacherModelV3
from apex_x.data import TransformSample

def verify_trainer_v3():
    print("--- Verifying ApexXTrainer with V3 Profile ---")
    
    # 1. Load Config and Set Profile
    cfg = ApexXConfig()
    cfg.model.profile = "worldclass"
    cfg.model.backbone_model = "facebook/dinov2-small" # Use small backbone for verification speed
    cfg.model.lora_rank = 4
    cfg.train.allow_synthetic_fallback = True # Allow synthetic data for "training"
    
    # 2. Initialize Trainer
    print("Initializing ApexXTrainer...")
    try:
        trainer = ApexXTrainer(config=cfg, num_classes=3)
    except Exception as e:
        print(f"FAILED to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Check Model Type
    if isinstance(trainer.teacher, TeacherModelV3):
        print("SUCCESS: Trainer initialized TeacherModelV3.")
    else:
        print(f"FAILURE: Trainer initialized {type(trainer.teacher)}, expected TeacherModelV3.")
        return

    # 4. Mock Validation Call
    print("Running Mock Validation Step...")
    
    # Create fake batch
    B, C, H, W = 2, 3, 320, 320
    images = torch.rand(B, C, H, W).to("cpu") # Check if CUDA available?
    
    # Mock DataLoader batch
    # trainer.validate expects: dict {"images": Tensor, "image_ids": list, "boxes": list of tensors, ...}
    # But compute_v3_training_losses expects "samples" to be list of TransformSample OR dict.
    # trainer.validate passes `batch_data` as `samples`.
    
    # Let's create a specific batch that satisfies compute_v3_training_losses
    # compute_v3_training_losses supports unified batch dict keys: boxes, labels, masks, batch_idx
    
    # Create dummy targets
    batch_data = {
        "images": images,
        "image_ids": [101, 102],
        "boxes": torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "masks": torch.zeros((2, 320, 320), dtype=torch.uint8),
        "batch_idx": torch.tensor([0, 1], dtype=torch.int64) # One target per image
    }
    
    # Mock COCO Evaluator to avoid dependency issues if pycocotools missing or setup complex
    # Trainer imports COCOEvaluator specifically.
    # We can mock `trainer.validate` dependencies or just run it and see if it crashes before COCO step.
    # Current trainer code:
    # 1. Forward
    # 2. Loss computation (V3)
    # 3. COCO update
    
    # Trainer validate might use eval(), but to compute PointRend loss we need training outputs (point_logits).
    # TeacherModelV3 uses self.training to decide whether to run PointRend Inference or Sampling.
    trainer.teacher.train() 
    
    # Reduce size to avoid CPU bottleneck
    B, C, H, W = 2, 3, 64, 64
    images = torch.rand(B, C, H, W).to("cpu")
    batch_data["images"] = images
    
    # Create valid masks matching boxes approximately
    masks = torch.zeros((2, 64, 64), dtype=torch.uint8)
    # Box 1: 10,10,50,50
    masks[0, 10:50, 10:50] = 1
    # Box 2: 20,20,60,60 - Clip to 64
    masks[1, 20:60, 20:60] = 1
    batch_data["masks"] = masks
    
    try:
        # We call the loss function directly first to verify it works
        # This mocks the inside of validate() loop
        print("Starting forward pass...")
        out = trainer.teacher(images)
        print("Forward pass successful.")
        
        from apex_x.train.train_losses_v3 import compute_v3_training_losses
        
        print("Starting loss computation...")
        loss, loss_dict = compute_v3_training_losses(
            outputs=out,
            targets=batch_data,
            model=trainer.teacher,
            config=cfg
        )
        print(f"Loss computation successful. Total Loss: {loss.item():.4f}")
        print("-" * 30)
        print("Loss Components:")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
        print("-" * 30)
        if "point_rend" in loss_dict:
             print("SUCCESS: PointRend loss detected.")
        else:
             print("WARNING: PointRend loss NOT detected (maybe no points sampled?).")

        # Check Output Structure for Batch Indices (My Fix)
        if "batch_indices" in out:
             print("SUCCESS: batch_indices found in output.")
        else:
             print("FAILURE: batch_indices MISSING in output.")

    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_trainer_v3()
