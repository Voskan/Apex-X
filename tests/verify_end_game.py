
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock

from apex_x.model.post_process import paste_masks_in_image, post_process_roi_outputs
from apex_x.infer.tta import TestTimeAugmentation
from apex_x.eval.coco_metrics import COCOEvaluator

class TestEndGameComponents(unittest.TestCase):
    
    def test_paste_masks(self):
        # 1. Test mask pasting
        N, M = 2, 28
        masks = torch.randn(N, 1, M, M)
        boxes = torch.tensor([[0, 0, 50, 50], [50, 50, 100, 100]], dtype=torch.float32)
        image_shape = (100, 100)
        
        pasted = paste_masks_in_image(masks, boxes, image_shape)
        self.assertEqual(pasted.shape, (N, 100, 100))
        self.assertEqual(pasted.dtype, torch.bool)
        print("Paste Masks: OK")
        
    def test_tta_structure(self):
        # 2. Test TTA wrapper structure
        model = MagicMock()
        # Mock model return
        model.return_value = {
            "boxes": torch.tensor([[10, 10, 20, 20]]),
            "scores": torch.tensor([[0.9]]), # [N, C] or similar
            "masks": torch.randn(1, 1, 28, 28)
        }
        
        tta = TestTimeAugmentation(model, scales=[1.0], flips=False)
        
        # Mock input
        img = torch.zeros(1, 3, 100, 100)
        output = tta(img)
        
        self.assertTrue(isinstance(output, list))
        self.assertTrue("boxes" in output[0])
        print("TTA Structure: OK")

if __name__ == '__main__':
    unittest.main()
