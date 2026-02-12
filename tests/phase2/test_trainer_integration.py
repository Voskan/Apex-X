import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from apex_x.train.trainer import ApexXTrainer


@pytest.fixture
def dummy_dataset_path():
    # Create temp dir
    d = tempfile.mkdtemp()
    path = Path(d)
    
    # Create 5 dummy images
    for i in range(5):
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        # Add a dummy object
        cv2.rectangle(mask, (100, 100), (200, 200), 255, -1)
        
        cv2.imwrite(str(path / f"image_{i:02d}.tif"), img)
        cv2.imwrite(str(path / f"image_{i:02d}_mask.tif"), mask)
        
    yield d
    shutil.rmtree(d)

def test_trainer_integration(dummy_dataset_path):
    # Test simple run with dataset
    trainer = ApexXTrainer(num_classes=2, backbone_type="pv") # Use default PV for speed
    
    # Run for minimal steps
    result = trainer.run(
        steps_per_stage=2,
        dataset_path=dummy_dataset_path,
        enable_budgeting=False
    )
    
    assert result.loss_proxy > 0
    assert len(result.stage_results) == 5

def test_trainer_timm_backbone(dummy_dataset_path):
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
         pass # OK on CPU

    try:
        import timm
    except ImportError:
        pytest.skip("timm not installed")
        
    # Test with timm backbone (efficientnet_b0 for speed)
    trainer = ApexXTrainer(
        num_classes=2, 
        backbone_type="timm",
        backbone_name="efficientnet_b0",
        pretrained_backbone=False
    )
    
    result = trainer.run(
        steps_per_stage=1,
        dataset_path=dummy_dataset_path,
        enable_budgeting=False
    )
    
    assert result.loss_proxy > 0
