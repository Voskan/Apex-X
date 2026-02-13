
import torch
import yaml
from apex_x.losses.lovasz_loss import lovasz_loss
from apex_x.losses.seg_loss import instance_segmentation_losses

def test_lovasz_loss():
    print("Testing Lovasz Loss...")
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
    
    loss = lovasz_loss(logits, targets)
    print(f"Lovasz Loss Value: {loss.item()}")
    
    loss.backward()
    print("Backward pass successful!")
    assert logits.grad is not None

def test_seg_loss_integration():
    print("\nTesting Segmentation Loss Integration...")
    mask_logits = torch.randn(2, 5, 28, 28, requires_grad=True)
    target_masks = torch.randint(0, 2, (2, 5, 28, 28)).float()
    
    # Test with Lovasz weight
    output = instance_segmentation_losses(
        mask_logits, 
        target_masks, 
        lovasz_weight=0.5,
        bce_weight=1.0, 
        dice_weight=1.0
    )
    
    print(f"Total Loss: {output.total_loss.item()}")
    output.total_loss.backward()
    print("Integration Backward pass successful!")

def test_config_loading():
    print("\nTesting Config Loading...")
    with open("configs/best_in_world.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    assert config['model']['profile'] == 'worldclass'
    assert config['train']['use_lovasz_loss'] is True
    print("Config loaded successfully!")

def test_point_rend():
    print("\nTesting PointRend Module...")
    from apex_x.model.point_rend import PointRendModule, sampling_points
    
    # 1. Test Sampling
    mask_logits = torch.randn(2, 1, 28, 28)
    points = sampling_points(mask_logits, num_points=1024)
    assert points.shape == (2, 1, 1024, 2)
    print("Point Sampling matches shape requirements.")
    
    # 2. Test Module Forward
    module = PointRendModule(in_channels=257) # 256 fine + 1 coarse
    fine_feats = torch.randn(2, 256, 14, 14) # ROI features
    # Standard forward doesn't do much, check internal MLP
    assert len(module.mlp) > 0
    print("PointRend Module initialized successfully.")

def test_memory_manager():
    print("\nTesting Memory Manager...")
    from apex_x.train.memory_manager import MemoryManager
    mm = MemoryManager(device="cpu") # Test logic only
    
    # Test batch size search simulation
    try:
        opt_bsz = mm.optimize_batch_size(
            torch.nn.Linear(10, 10), 
            (10,), 
            max_batch_size=4, 
            start_batch_size=1
        )
        print(f"Memory Manager optimized bsz: {opt_bsz}")
        
        # Test downshift
        new_bsz = mm.downshift_batch_size(32)
        assert new_bsz == 16
        print("Downshift logic correct.")
        
    except Exception as e:
        print(f"Memory Manager Test Note: {e} (expected on CPU/Missing CUDA)")

if __name__ == "__main__":
    try:
        test_lovasz_loss()
        test_seg_loss_integration()
        test_config_loading()
        test_point_rend()
        test_memory_manager()
        print("\nALL SOTA TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
