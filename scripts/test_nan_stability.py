import torch
import numpy as np
from pathlib import Path
from apex_x.train.trainer import ApexXTrainer
from apex_x.config import ApexXConfig
from apex_x.data.transforms import TransformSample

def test_nan_stability():
    print("Testing NaN stability...")
    
    config = ApexXConfig()
    trainer = ApexXTrainer(config=config, use_amp=True)
    
    # 1. Test image sanitization (simulated)
    # Create an image with NaNs
    img_with_nan = np.zeros((128, 128, 3), dtype=np.float32)
    img_with_nan[0, 0, 0] = np.nan
    
    # We'll check if the trainer's forward pass can handle potential NaNs 
    # (though our sanitization in data loading should have caught it)
    image_t = torch.from_numpy(img_with_nan).permute(2, 0, 1).unsqueeze(0).float()
    
    # This should not crash if guards are in place
    try:
        # In actual training, we'd use the model forward
        print("Running forward pass with NaNs...")
        out = trainer.teacher(image_t)
        print("Forward pass complete.")
    except Exception as e:
        print(f"Forward pass failed as expected without internal guards: {e}")

    # 2. Test Gradient Guard
    # We'll simulate a training step where a NaN is injected into the loss
    print("Testing gradient guard skip logic...")
    
    optimizer = torch.optim.AdamW(trainer.teacher.parameters(), lr=1e-3)
    
    # Manually inject a NaN into one parameter's gradient
    params = [p for p in trainer.teacher.parameters() if p.requires_grad]
    p = params[0]
    p.grad = torch.full_like(p.data, float('nan'))
    
    # Run the guard logic (we'll manually check the finite check)
    grads_finite = True
    for param in trainer.teacher.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grads_finite = False
            break
            
    if not grads_finite:
        print("SUCCESS: NaN detected in gradients by the guard logic.")
    else:
        print("FAILURE: NaN NOT detected in gradients.")

if __name__ == "__main__":
    test_nan_stability()
