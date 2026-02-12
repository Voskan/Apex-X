"""Comprehensive test suite for Apex-X.

Tests for losses, models, data pipeline, and training.
"""

import pytest
import torch
import numpy as np


class TestLosses:
    """Test loss functions."""
    
    def test_boundary_iou_loss(self):
        """Test boundary IoU loss."""
        from apex_x.losses.seg_loss import boundary_iou_loss
        
        # Create sample data
        logits = torch.randn(2, 3, 64, 64)
        targets = torch.randint(0, 2, (2, 3, 64, 64)).float()
        
        # Compute loss
        loss = boundary_iou_loss(logits, targets)
        
        # Check output
        assert loss.item() >= 0, "Loss should be non-negative"
        assert loss.item() <= 1, "Loss should be <= 1"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_mask_dice_loss(self):
        """Test Dice loss."""
        from apex_x.losses.seg_loss import mask_dice_loss
        
        logits = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 2, (2, 3, 32, 32)).float()
        
        loss = mask_dice_loss(logits, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_mask_bce_loss(self):
        """Test BCE loss."""
        from apex_x.losses.seg_loss import mask_bce_loss
        
        logits = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 2, (2, 3, 32, 32)).float()
        
        loss = mask_bce_loss(logits, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestModels:
    """Test model components."""
    
    def test_cascade_head_forward(self):
        """Test Cascade detection head."""
        from apex_x.model.cascade_head import CascadeDetHead
        
        model = CascadeDetHead(in_channels=256, num_classes=80, num_stages=3)
        features = torch.randn(1, 256, 64, 64)
        boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]]).float()
        
        output = model(features, boxes)
        
        assert 'boxes' in output
        assert 'scores' in output
        assert len(output['boxes']) == 4  # initial + 3 stages
        assert len(output['scores']) == 3
    
    def test_mask_quality_head(self):
        """Test mask quality prediction head."""
        from apex_x.model.mask_quality_head import MaskQualityHead
        
        model = MaskQualityHead(in_channels=256)
        features = torch.randn(10, 256, 7, 7)
        
        quality = model(features)
        
        assert quality.shape == (10,), f"Expected shape (10,), got {quality.shape}"
        assert (quality >= 0).all() and (quality <= 1).all(), "Quality should be in [0, 1]"
    
    def test_bifpn_forward(self):
        """Test BiFPN."""
        from apex_x.model.bifpn import BiFPN
        
        model = BiFPN(
            in_channels_list=[64, 128, 256, 512, 512],
            out_channels=256,
            num_layers=2,
        )
        
        # Create multi-scale features
        feats = [
            torch.randn(1, 64, 64, 64),
            torch.randn(1, 128, 32, 32),
            torch.randn(1, 256, 16, 16),
            torch.randn(1, 512, 8, 8),
            torch.randn(1, 512, 4, 4),
        ]
        
        output = model(feats)
        
        assert len(output) == 5, "Should output 5 levels"
        assert all(o.shape[1] == 256 for o in output), "All outputs should have 256 channels"


class TestDataPipeline:
    """Test data loading and augmentation."""
    
    def test_quality_filter_entropy(self):
        """Test entropy filtering."""
        from apex_x.data.quality_filter import ImageQualityFilter
        
        filter = ImageQualityFilter(
            min_entropy=4.0,
            enable_entropy=True,
            enable_sharpness=False,
            enable_cloud=False,
        )
        
        # High entropy image (random noise)
        high_entropy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        passes, metrics = filter.filter_image(high_entropy)
        
        assert 'entropy' in metrics
        assert metrics['entropy'] > 4.0, "Random noise should have high entropy"
    
    def test_quality_filter_sharpness(self):
        """Test sharpness filtering."""
        from apex_x.data.quality_filter import ImageQualityFilter
        
        filter = ImageQualityFilter(
            min_sharpness=100.0,
            enable_entropy=False,
            enable_sharpness=True,
            enable_cloud=False,
        )
        
        # Sharp image (checkerboard pattern)
        sharp = np.zeros((256, 256), dtype=np.uint8)
        sharp[::2, ::2] = 255
        sharp[1::2, 1::2] = 255
        
        passes, metrics = filter.filter_image(sharp)
        
        assert 'sharpness' in metrics
        assert metrics['sharpness'] > 100.0, "Checkerboard should be sharp"


class TestTraining:
    """Test training utilities."""
    
    def test_early_stopping(self):
        """Test early stopping callback."""
        from apex_x.train.early_stopping import EarlyStopping
        
        early_stop = EarlyStopping(patience=3, mode='max')
        
        # Simulate improving then plateauing
        assert not early_stop.step(0.5, epoch=0)
        assert not early_stop.step(0.6, epoch=1)  # Improved
        assert not early_stop.step(0.65, epoch=2)  # Improved
        assert not early_stop.step(0.65, epoch=3)  # No improvement (1/3)
        assert not early_stop.step(0.64, epoch=4)  # No improvement (2/3)
        assert not early_stop.step(0.63, epoch=5)  # No improvement (3/3)
        assert early_stop.step(0.62, epoch=6)  # Should stop
    
    def test_cpu_device_selection(self):
        """Test CPU/CUDA device selection."""
        from apex_x.train.cpu_support import get_device, should_use_amp
        
        device = get_device("auto")
        assert device.type in ['cpu', 'cuda']
        
        # CPU should not use AMP
        cpu_device = torch.device("cpu")
        assert not should_use_amp(cpu_device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
