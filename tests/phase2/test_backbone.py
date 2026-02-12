import pytest
import torch

from apex_x.model.timm_backbone import TimmBackboneAdapter


def test_timm_backbone_shapes():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        # Run on CPU if needed, assuming timm works on CPU
        device = "cpu"
    else:
        device = "cpu" # Force CPU for simple shape test to avoid VRAM issues during test

    # Test with a small efficientnet
    try:
        import timm
    except ImportError:
        pytest.skip("timm not installed")

    # Use a small model for testing
    model_name = "efficientnet_b0"
    adapter = TimmBackboneAdapter(model_name=model_name, pretrained=False)
    
    # Check channels
    # EfficientNet-B0 features:
    # 0: stride 2, 16 ch
    # 1: stride 4, 24 ch
    # 2: stride 8, 40 ch (P3)
    # 3: stride 16, 112 ch (P4)
    # 4: stride 32, 320 ch (P5) -- actually 1280 or 320 depending on include_top
    # Let's verify what `features_only` returns for indices (2,3,4)
    
    # Create dummy input
    x = torch.randn(1, 3, 256, 256)
    
    outputs = adapter(x)
    
    assert "P3" in outputs
    assert "P4" in outputs
    assert "P5" in outputs
    
    p3 = outputs["P3"]
    p4 = outputs["P4"]
    p5 = outputs["P5"]
    
    # Check strides
    assert p3.shape[2] == 256 // 8
    assert p3.shape[3] == 256 // 8
    
    assert p4.shape[2] == 256 // 16
    assert p4.shape[3] == 256 // 16
    
    assert p5.shape[2] == 256 // 32
    assert p5.shape[3] == 256 // 32
    
    # Check channels match reported properties
    assert p3.shape[1] == adapter.p3_channels
    assert p4.shape[1] == adapter.p4_channels
    assert p5.shape[1] == adapter.p5_channels
    
    print(f"P3 channels: {p3.shape[1]}")
    print(f"P4 channels: {p4.shape[1]}")
    print(f"P5 channels: {p5.shape[1]}")

def test_timm_backbone_resnet():
    try:
        import timm
    except ImportError:
        pytest.skip("timm not installed")

    # Test with resnet18
    adapter = TimmBackboneAdapter(model_name="resnet18", pretrained=False)
    
    # ResNet18:
    # P3: 64, P4: 128, P5: 256? Or 128/256/512?
    # Usually: layer1(64), layer2(128, s8), layer3(256, s16), layer4(512, s32)
    # indices=(2,3,4) -> layer 2, 3, 4
    
    x = torch.randn(1, 3, 256, 256)
    outputs = adapter(x)
    
    assert outputs["P3"].shape[1] == 128
    assert outputs["P4"].shape[1] == 256
    assert outputs["P5"].shape[1] == 512
    
    assert adapter.p3_channels == 128
    assert adapter.p4_channels == 256
    assert adapter.p5_channels == 512
