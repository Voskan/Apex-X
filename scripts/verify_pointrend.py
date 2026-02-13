
import sys
import os
sys.path.append(os.getcwd())

import torch
from apex_x.model.teacher_v3 import TeacherModelV3
from apex_x.train.train_losses_v3 import compute_v3_training_losses

def test_pointrend_integration():
    print("Initializing TeacherModelV3...")
    model = TeacherModelV3(
        num_classes=3,
        backbone_model="facebook/dinov2-small", # Use small for speed
        lora_rank=4,
        fpn_channels=64, # Small for speed
        num_cascade_stages=3 # Default
    )
    model.train()
    
    # Dummy Input
    B, C, H, W = 2, 3, 320, 320 # Must be divisible by 32? (320/14 = 22.8)
    image = torch.randn(B, C, H, W)
    
    # Dummy Targets
    # Needs "boxes", "labels", "masks"
    targets = {
        "boxes": torch.tensor([
            [10, 10, 100, 100],
            [150, 150, 250, 250]
        ], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "masks": torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)
    }
    
    # Monkey-patch det_head to return dummy boxes so MaskHead and PointRend run
    def dummy_det_forward(*args, **kwargs):
        print("DEBUG: dummy_det_forward called")
        # Return valid boxes for batch size B
        # List of lists [Stage][Batch]
        B = 2
        # Use simple boxes that are definitely inside the image
        # Image is 320x320
        boxes = [torch.tensor([[10., 10., 100., 100.]], device=args[0].device) for _ in range(B)]
        scores = [torch.tensor([[0.9, 0.1, 0.0]], device=args[0].device) for _ in range(B)] # [N, C]
        
        # Return 4 stages to be safe (Proposals + 3 Refinements)
        # TeacherV3 uses all_boxes[1:], so we need at least 2 to have non-empty input
        all_boxes = [boxes, boxes, boxes, boxes]
        all_scores = [torch.cat(scores, dim=0)] * 4
        
        return {
            "boxes": all_boxes,
            "scores": all_scores
        }
    
    model.det_head.forward = dummy_det_forward
    
    # Forward Pass
    print("Running Forward Pass...")
    try:
        outputs = model(image, targets)
    except Exception as e:
        print(f"Forward failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Keys in output:", outputs.keys())
    
    if "point_logits" in outputs:
        print("SUCCESS: point_logits found.")
        print("Shape:", outputs["point_logits"].shape)
    else:
        print("FAILURE: point_logits NOT found.")
        
    if "point_coords" in outputs:
        print("SUCCESS: point_coords found.")
        print("Shape:", outputs["point_coords"].shape)
    else:
        print("FAILURE: point_coords NOT found.")

    # Loss Computation
    print("Computing Losses...")
    try:
        class Config:
            pass
        cfg = Config()
        total_loss, loss_dict = compute_v3_training_losses(outputs, targets, model, cfg)
        print("Total Loss:", total_loss.item())
        print("Loss Dict:", loss_dict)
        
        if "point_rend" in loss_dict:
            print("SUCCESS: point_rend loss computed:", loss_dict["point_rend"].item())
        else:
            print("FAILURE: point_rend loss NOT computed.")
            
    except Exception as e:
        print(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pointrend_integration()
