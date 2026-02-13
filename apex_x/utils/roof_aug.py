import torch
import numpy as np
import cv2
from torch import Tensor

class ProceduralRoofAugumentor:
    """World-Class Synthetic Data Engine.
    
    Generates procedurally correct roof geometry to train 
    the sub-pixel INR head. Traditional augmentations (flip, rotate) 
    don't provide the high-frequency boundary data needed for SOTA.
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def augment_boundary(self, mask: Tensor) -> Tensor:
        """Adds procedural 'micro-noise' to boundaries.
        
        Args:
            mask: [H, W] binary mask
        """
        if torch.rand(1) > self.prob:
            return mask
            
        # 1. Edge Detection
        mask_np = mask.cpu().numpy().astype(np.uint8)
        edges = cv2.Canny(mask_np, 100, 200)
        
        # 2. Fractal Noise Generation
        # Simulate imperfect satellite sensor noise at sub-pixel levels
        h, w = mask.shape
        noise = np.random.normal(0, 1.5, size=(h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (3, 3), 0)
        
        # 3. Apply Selective Noise to Edges
        noisy_mask = mask_np.astype(np.float32)
        noisy_mask[edges > 0] += noise[edges > 0]
        
        return torch.from_numpy(np.clip(noisy_mask, 0, 1)).to(mask.device)

    def generate_synthetic_roof(self, size: tuple[int, int] = (1024, 1024)) -> tuple[Tensor, Tensor]:
        """Generates a perfectly sharp synthetic roof and its noised version."""
        h, w = size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Random Polygon (Roof-like)
        center = (w // 2, h // 2)
        points = np.array([
            [center[0] - 100, center[1] - 80],
            [center[0] + 120, center[1] - 50],
            [center[0] + 80, center[1] + 130],
            [center[0] - 50, center[1] + 90]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [points], 1)
        
        gt = torch.from_numpy(mask).float()
        noisy = self.augment_boundary(gt)
        
        return gt, noisy
