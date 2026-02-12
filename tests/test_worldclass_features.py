"""Unit tests for world-class features.

Tests all advanced modules:
- DINOv2 backbone
- LSJ augmentation
- TTA
- SE attention
- Auxiliary losses
- Focal frequency loss
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestDINOv2Backbone:
    """Test DINOv2 integration."""
    
    def test_dinov2_import(self):
        """Test DINOv2 module imports."""
        from apex_x.model import PVModuleDINOv2, DINOV2_AVAILABLE, LoRAAdapter
        
        assert PVModuleDINOv2 is not None
        assert LoRAAdapter is not None
        # DINOV2_AVAILABLE may be False if transformers not installed
    
    @pytest.mark.skipif(not hasattr(torch.cuda, 'is_available') or not torch.cuda.is_available(),
                        reason="Requires GPU")
    def test_dinov2_forward(self):
        """Test DINOv2 forward pass."""
        try:
            from apex_x.model import PVModuleDINOv2, DINOV2_AVAILABLE
            
            if not DINOV2_AVAILABLE:
                pytest.skip("transformers not installed")
            
            pv = PVModuleDINOv2(
                model_name='facebook/dinov2-base',  # Smaller for testing
                lora_rank=4,
            )
            
            # Test forward
            x = torch.randn(1, 3, 224, 224)
            out = pv(x)
            
            assert 'P3' in out
            assert 'P4' in out
            assert 'P5' in out
            
        except ImportError:
            pytest.skip("DINOv2 dependencies not available")


class TestLSJAugmentation:
    """Test Large Scale Jittering."""
    
    def test_lsj_import(self):
        """Test LSJ imports."""
        from apex_x.data import LargeScaleJitter
        assert LargeScaleJitter is not None
    
    def test_lsj_augmentation(self):
        """Test LSJ augmentation."""
        from apex_x.data import LargeScaleJitter, TransformSample
        
        lsj = LargeScaleJitter(output_size=640, min_scale=0.5, max_scale=1.5)
        
        # Create sample
        sample = TransformSample(
            image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            boxes_xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            class_ids=np.array([1], dtype=np.int64),
            masks=None,
        )
        
        # Apply LSJ
        rng = np.random.RandomState(42)
        augmented = lsj(sample, rng)
        
        assert augmented.image is not None
        assert augmented.image.shape[:2] == (640, 640)


class TestTTA:
    """Test Test-Time Augmentation."""
    
    def test_tta_import(self):
        """Test TTA imports."""
        from apex_x.infer.tta import TestTimeAugmentation
        assert TestTimeAugmentation is not None
    
    def test_tta_creation(self):
        """Test TTA initialization."""
        from apex_x.infer.tta import TestTimeAugmentation
        
        tta = TestTimeAugmentation(
            scales=[0.8, 1.0, 1.2],
            use_flip=True,
        )
        
        assert tta.scales == [0.8, 1.0, 1.2]
        assert tta.use_flip == True


class TestSEAttention:
    """Test Squeeze-and-Excitation attention."""
    
    def test_se_block_import(self):
        """Test SE block imports."""
        from apex_x.model.se_attention import SEBlock, SEDualPathFPN
        assert SEBlock is not None
        assert SEDualPathFPN is not None
    
    def test_se_block_forward(self):
        """Test SE block forward pass."""
        from apex_x.model.se_attention import SEBlock
        
        se = SEBlock(channels=256, reduction=16)
        
        x = torch.randn(2, 256, 32, 32)
        out = se(x)
        
        assert out.shape == x.shape  # Same shape
        assert not torch.allclose(out, x)  # Different values (attention applied)


class TestAuxiliaryLosses:
    """Test auxiliary decoder losses."""
    
    def test_aux_loss_import(self):
        """Test auxiliary loss imports."""
        from apex_x.losses.auxiliary_losses import (
            auxiliary_mask_loss,
            dice_loss,
            focal_loss,
        )
        assert auxiliary_mask_loss is not None
        assert dice_loss is not None
        assert focal_loss is not None
    
    def test_dice_loss(self):
        """Test Dice loss computation."""
        from apex_x.losses.auxiliary_losses import dice_loss
        
        pred = torch.randn(4, 64, 64)
        target = torch.randint(0, 2, (4, 64, 64)).float()
        
        loss = dice_loss(pred, target)
        
        assert loss.item() >= 0.0
        assert loss.item() <= 1.0
    
    def test_auxiliary_loss(self):
        """Test auxiliary loss with multiple outputs."""
        from apex_x.losses.auxiliary_losses import auxiliary_mask_loss
        
        aux_outputs = [
            torch.randn(2, 32, 32),
            torch.randn(2, 32, 32),
            torch.randn(2, 32, 32),
        ]
        target = torch.randint(0, 2, (2, 32, 32)).float()
        
        loss = auxiliary_mask_loss(
            aux_mask_outputs=aux_outputs,
            target_masks=target,
            loss_type='dice',
        )
        
        assert loss.item() >= 0.0


class TestFocalFrequencyLoss:
    """Test Focal Frequency Loss."""
    
    def test_ffl_import(self):
        """Test FFL imports."""
        from apex_x.losses.focal_freq_loss import (
            FocalFrequencyLoss,
            focal_frequency_loss,
        )
        assert FocalFrequencyLoss is not None
        assert focal_frequency_loss is not None
    
    def test_ffl_computation(self):
        """Test FFL computation."""
        from apex_x.losses.focal_freq_loss import focal_frequency_loss
        
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        
        loss = focal_frequency_loss(pred, target, alpha=1.0)
        
        assert loss.item() >= 0.0
    
    def test_ffl_module(self):
        """Test FFL as nn.Module."""
        from apex_x.losses.focal_freq_loss import FocalFrequencyLoss
        
        ffl = FocalFrequencyLoss(alpha=1.0, loss_weight=0.1)
        
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        loss = ffl(pred, target)
        
        assert loss.item() >= 0.0


class TestAdditionalAugmentations:
    """Test RandomErasing and GridMask."""
    
    def test_random_erasing_import(self):
        """Test RandomErasing import."""
        from apex_x.data.additional_augmentations import RandomErasing
        assert RandomErasing is not None
    
    def test_grid_mask_import(self):
        """Test GridMask import."""
        from apex_x.data.additional_augmentations import GridMask
        assert GridMask is not None
    
    def test_random_erasing(self):
        """Test RandomErasing augmentation."""
        from apex_x.data.additional_augmentations import RandomErasing
        from apex_x.data import TransformSample
        
        re = RandomErasing(prob=1.0)  # Always apply for testing
        
        sample = TransformSample(
            image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            boxes_xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            class_ids=np.array([1], dtype=np.int64),
            masks=None,
        )
        
        rng = np.random.RandomState(42)
        augmented = re(sample, rng)
        
        assert augmented.image is not None
        # Image should be modified
        assert not np.array_equal(augmented.image, sample.image)


class TestDDPWrapper:
    """Test Multi-GPU DDP wrapper."""
    
    def test_ddp_import(self):
        """Test DDP imports."""
        from apex_x.train.ddp import DDPWrapper, is_main_process, reduce_dict
        assert DDPWrapper is not None
        assert is_main_process is not None
        assert reduce_dict is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_ddp_creation(self):
        """Test DDP wrapper creation."""
        # This would require actual distributed setup
        # Just test import for now
        from apex_x.train.ddp import DDPWrapper
        
        # Can't actually initialize without distributed environment
        # Just verify class exists
        assert DDPWrapper is object().__class__.__bases__[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
