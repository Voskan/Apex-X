"""Test Time Augmentation (TTA) module for Apex-X inference.

Implements multi-scale and flip augmentation strategies with result aggregation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from apex_x.infer.ensemble import weighted_boxes_fusion
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TTAParams:
    """Parameters for TTA."""
    scales: List[float] = (0.8, 1.0, 1.2)
    flips: List[str] = ("none", "horizontal")  # "none", "horizontal", "vertical"
    conf_threshold: float = 0.01
    iou_threshold: float = 0.55  # For WBF
    wbf_skip_box_thr: float = 0.001


class TestTimeAugmentation:
    """Wrapper for model to perform TTA."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        params: TTAParams | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.params = params or TTAParams()

    def _apply_aug(
        self,
        image_np: np.ndarray,
        scale: float,
        flip: str,
    ) -> np.ndarray:
        """Apply augmentation to image."""
        img = image_np.copy()
        
        # Scaling
        if scale != 1.0:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Flipping
        if flip == "horizontal":
            img = cv2.flip(img, 1)
        elif flip == "vertical":
            img = cv2.flip(img, 0)
            
        return img

    def _inverse_aug_boxes(
        self,
        boxes: np.ndarray,
        orig_shape: Tuple[int, int],
        aug_shape: Tuple[int, int],
        scale: float,
        flip: str,
    ) -> np.ndarray:
        """Inverse transform boxes to original image coordinates."""
        if boxes.shape[0] == 0:
            return boxes
        
        inv_boxes = boxes.copy()
        h, w = aug_shape
        orig_h, orig_w = orig_shape
        
        # Inverse Flip
        if flip == "horizontal":
            inv_boxes[:, [0, 2]] = w - inv_boxes[:, [2, 0]]
        elif flip == "vertical":
            inv_boxes[:, [1, 3]] = h - inv_boxes[:, [3, 1]]
            
        # Inverse Scale
        if scale != 1.0:
            inv_boxes[:, [0, 2]] /= (float(w) / float(orig_w))
            inv_boxes[:, [1, 3]] /= (float(h) / float(orig_h))
            
        return inv_boxes
    
    def _inverse_aug_masks(
        self,
        masks_np: np.ndarray,
        orig_shape: Tuple[int, int],
        aug_shape: Tuple[int, int],
        scale: float,
        flip: str,
    ) -> np.ndarray:
        """Inverse transform masks.
        
        masks_np: [N, H_aug, W_aug] (numpy)
        """
        if masks_np.size == 0:
            return masks_np
            
        N, h, w = masks_np.shape
        orig_h, orig_w = orig_shape
        
        # We process manually or via cv2/torch
        # Using cv2 loop is safest for numpy
        
        inv_masks = []
        for i in range(N):
            m = masks_np[i] # [H, W]
            
            # Inverse Flip
            if flip == "horizontal":
                m = cv2.flip(m, 1)
            elif flip == "vertical":
                m = cv2.flip(m, 0)
            
            # Inverse Scale
            if scale != 1.0:
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Ensure size match
                if m.shape != (orig_h, orig_w):
                     m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            inv_masks.append(m)
            
        return np.stack(inv_masks, axis=0)

    def predict(
        self,
        image_tensor: Tensor,  # [1, 3, H, W] normalized - UNUSED in loop, we build from np
        orig_image_np: np.ndarray, # [H, W, 3] uint8 for resizing reference
    ) -> dict[str, Any]:
        """Run TTA inference."""
        all_boxes_list = []
        all_scores_list = []
        all_labels_list = []
        all_masks_list = []
        
        # Base image dimensions
        orig_h, orig_w = orig_image_np.shape[:2]
        
        for scale in self.params.scales:
            for flip in self.params.flips:
                # 1. Augment
                aug_img = self._apply_aug(orig_image_np, scale, flip)
                aug_h, aug_w = aug_img.shape[:2]
                
                # Prepare tensor
                input_tensor = torch.from_numpy(aug_img).float() / 255.0
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # 2. Inference
                with torch.no_grad():
                    # Call model and post-process
                    outputs = self.model(input_tensor)
                    
                    from apex_x.model.post_process import post_process_detections
                    dets = post_process_detections(
                        cls_logits_by_level=outputs.logits_by_level,
                        box_reg_by_level=outputs.boxes_by_level,
                        quality_by_level=outputs.quality_by_level,
                        conf_threshold=self.params.wbf_skip_box_thr,
                        nms_threshold=1.0, # Weak/No NMS
                        masks=outputs.masks, # Pass masks!
                    )[0] # Batch size 1
                
                # 3. Inverse Transform for this view
                boxes = dets['boxes'].cpu().numpy()
                scores = dets['scores'].cpu().numpy()
                labels = dets['classes'].cpu().numpy()
                masks = None
                if 'masks' in dets:
                     masks = dets['masks'].cpu().numpy() # [N, H_aug, W_aug] (probs)
                
                if len(boxes) > 0:
                    inv_boxes = self._inverse_aug_boxes(
                        boxes,
                        orig_shape=(orig_h, orig_w),
                        aug_shape=(aug_h, aug_w),
                        scale=scale,
                        flip=flip
                    )
                    
                    # Normalize boxes to 0..1 for WBF
                    inv_boxes_norm = inv_boxes.copy()
                    inv_boxes_norm[:, [0, 2]] /= float(orig_w)
                    inv_boxes_norm[:, [1, 3]] /= float(orig_h)
                    inv_boxes_norm = np.clip(inv_boxes_norm, 0.0, 1.0)
                    
                    all_boxes_list.append(inv_boxes_norm)
                    all_scores_list.append(scores)
                    all_labels_list.append(labels)
                    
                    if masks is not None:
                         scale_ratio = scale # Already handled
                         inv_masks = self._inverse_aug_masks(
                             masks, 
                             orig_shape=(orig_h, orig_w),
                             aug_shape=(aug_h, aug_w),
                             scale=scale,
                             flip=flip
                         )
                         all_masks_list.append(inv_masks)
                    
        # 4. Aggregate (WBF)
        if not all_boxes_list:
             return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'classes': torch.tensor([])}

        wbf_args = {
             "boxes_list": all_boxes_list,
             "scores_list": all_scores_list,
             "labels_list": all_labels_list,
             "iou_thr": self.params.iou_threshold,
             "skip_box_thr": self.params.conf_threshold,
             "conf_type": 'avg',
        }
        if all_masks_list:
             wbf_args["masks_list"] = all_masks_list

        wbf_boxes, wbf_scores, wbf_labels, wbf_masks = weighted_boxes_fusion(**wbf_args)
        
        # Convert back to absolute coordinates
        final_boxes = wbf_boxes.copy()
        final_boxes[:, [0, 2]] *= float(orig_w)
        final_boxes[:, [1, 3]] *= float(orig_h)
        
        result = {
            'boxes': torch.from_numpy(final_boxes).to(self.device).float(),
            'scores': torch.from_numpy(wbf_scores).to(self.device).float(),
            'classes': torch.from_numpy(wbf_labels).to(self.device).long(),
        }
        
        if wbf_masks is not None:
             # Masks are numpy float [N, H, W]
             result['masks'] = torch.from_numpy(wbf_masks).to(self.device).float()
             
        return result
