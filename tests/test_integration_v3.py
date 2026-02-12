"""Integration tests for TeacherModelV3 and v2.0 features."""

import pytest
import torch
import numpy as np

from apex_x.model import TeacherModelV3
from apex_x.train.train_losses_v3 import compute_v3_training_losses
from apex_x.data.quality_filter import ImageQualityFilter
from apex_x.losses.seg_loss import boundary_iou_loss


class TestTeacherModelV3Integration:
    """Test TeacherModelV3 forward pass and outputs."""
    
    def test_model_forward(self):
        """Test forward pass produces expected outputs."""
        model = TeacherModelV3(num_classes=80)
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 3, 512, 512)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x)
        
        # Check outputs
        assert 'boxes' in outputs, "Missing boxes output"
        assert 'scores' in outputs, "Missing scores output"
        assert 'predicted_quality' in outputs, "Missing quality prediction"
        assert 'fpn_features' in outputs, "Missing FPN features"
        
        # Check shapes
        assert outputs['boxes'].dim() == 2, "Boxes should be 2D"
        assert outputs['boxes'].shape[1] == 4, "Boxes should have 4 coordinates"
        assert outputs['predicted_quality'].dim() == 1, "Quality should be 1D"
    
    def test_cascade_outputs(self):
        """Test cascade intermediate outputs."""
        model = TeacherModelV3(num_classes=80, num_cascade_stages=3)
        model.eval()
        
        x = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            outputs = model(x)
        
        # Check cascade outputs
        assert 'all_boxes' in outputs, "Missing cascade boxes"
        assert 'all_masks' in outputs, "Missing cascade masks"  
        assert 'all_scores' in outputs, "Missing cascade scores"
        
        # Should have: initial + 3 stages = 4 boxes, 3 stages masks/scores
        assert len(outputs['all_boxes']) == 4, f"Expected 4 box sets, got {len(outputs['all_boxes'])}"
        assert len(outputs['all_masks']) == 3, f"Expected 3 mask sets, got {len(outputs['all_masks'])}"


class TestV3Losses:
    """Test v2.0 loss functions."""
    
    def test_boundary_iou_loss(self):
        """Test boundary IoU loss computation."""
        # Create sample masks
        logits = torch.randn(2, 3, 64, 64)
        targets = torch.randint(0, 2, (2, 3, 64, 64)).float()
        
        loss = boundary_iou_loss(logits, targets, boundary_width=3)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_v3_losses_computation(self):
        """Test compute_v3_training_losses function."""
        from types import SimpleNamespace
        
        # Create dummy model
        model = TeacherModelV3(num_classes=3)
        
        # Create dummy outputs and targets
        outputs = {
            'scores': torch.randn(10, 3),
            'masks': torch.randn(10, 1, 28, 28),
            'predicted_quality': torch.sigmoid(torch.randn(10)),
        }
        
        targets = {
            'labels': torch.randint(0, 3, (10,)),
            'masks': torch.randint(0, 2, (10, 1, 28, 28)).float(),
        }
        
        # Create dummy config
        config = SimpleNamespace(
            loss=SimpleNamespace(
                multi_scale_supervision=False,
                boundary_weight=0.5,
                quality_weight=1.0,
            )
        )
        
        # Compute losses
        total_loss, loss_dict = compute_v3_training_losses(outputs, targets, model, config)
        
        # Check outputs
        assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
        assert total_loss.item() > 0, "Total loss should be positive"
        assert isinstance(loss_dict, dict), "Loss dict should be dict"
        assert len(loss_dict) > 0, "Loss dict should not be empty"


class TestDataQualityPipeline:
    """Test data quality filtering."""
    
    def test_quality_filter_entropy(self):
        """Test entropy-based filtering."""
        quality_filter = ImageQualityFilter(
            min_entropy=4.0,
            enable_entropy=True,
            enable_sharpness=False,
            enable_cloud=False,
        )
        
        # High entropy image (random)
        high_entropy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        passes, metrics = quality_filter.filter_image(high_entropy_img)
        
        assert 'entropy' in metrics
        assert metrics['entropy'] > 4.0, "Random image should have high entropy"
        assert passes, "High entropy image should pass"
    
    def test_quality_filter_low_quality(self):
        """Test that low quality images are filtered."""
        quality_filter = ImageQualityFilter(
            min_entropy=6.0,  # Very high threshold
            enable_entropy=True,
        )
        
        # Low entropy image (uniform)
        low_entropy_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        passes, metrics = quality_filter.filter_image(low_entropy_img)
        
        assert not passes, "Low entropy image should not pass"


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_training_iteration(self):
        """Test one training iteration works."""
        # Create model
        model = TeacherModelV3(num_classes=3)
        model.train()
        
        # Create dummy batch
        images = torch.randn(2, 3, 512, 512)
        targets = {
            'labels': torch.randint(0, 3, (10,)),
            'boxes': torch.rand(10, 4) * 512,
            'masks': torch.randint(0, 2, (10, 1, 64, 64)).float(),
        }
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        from types import SimpleNamespace
        config = SimpleNamespace(
            loss=SimpleNamespace(
                multi_scale_supervision=False,
                boundary_weight=0.5,
            )
        )
        
        total_loss, _ = compute_v3_training_losses(outputs, targets, model, config)
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "Model should have gradients after backward pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
