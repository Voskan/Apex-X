import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

def weighted_boxes_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    masks_list: list[np.ndarray] | None = None,
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    conf_type: str = 'avg'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Functional wrapper for WBF compatible with common object detection ensembling.
    
    Includes support for soft-mask blending.
    """
    ensemble = BespokeEnsemble(iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    # Prepare inputs as predictions list
    predictions = []
    num_views = len(boxes_list)
    for i in range(num_views):
        p = {
            'boxes': torch.from_numpy(boxes_list[i]),
            'scores': torch.from_numpy(scores_list[i]),
            'labels': torch.from_numpy(labels_list[i]).long()
        }
        if masks_list is not None and len(masks_list) > i:
             p['masks'] = torch.from_numpy(masks_list[i])
        predictions.append(p)
        
    fused = ensemble.fuse(predictions)
    
    res_masks = fused['masks'].cpu().numpy() if 'masks' in fused else None
    
    return (
        fused['boxes'].cpu().numpy(),
        fused['scores'].cpu().numpy(),
        fused['labels'].cpu().numpy(),
        res_masks
    )

class BespokeEnsemble:
    """World-Class Instance Ensemble Engine.
    
    Implements Weighted Boxes Fusion (WBF) + Soft-Mask Meta-Blending.
    Ensures that when multiple models or test-time augmentations (TTA) 
    predict the same roof, the boundaries are fused optimally.
    """
    
    def __init__(self, iou_thr: float = 0.55, skip_box_thr: float = 0.05):
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr

    def fuse(self, predictions: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """
        Args:
            predictions: List of dicts with 'boxes', 'scores', 'masks', 'labels'
        """
        if not predictions:
            return {}
            
        # 1. Flatten all predictions
        all_boxes = torch.cat([p['boxes'] for p in predictions], dim=0)
        all_scores = torch.cat([p['scores'] for p in predictions], dim=0)
        all_masks = torch.cat([p['masks'] for p in predictions], dim=0) if 'masks' in predictions[0] else None
        all_labels = torch.cat([p['labels'] for p in predictions], dim=0)
        
        # 2. Weighted Boxes Fusion (Logic)
        B = all_boxes.shape[0]
        if B == 0:
            return predictions[0] if predictions else {}

        # Calculate IoU Matrix
        from apex_x.losses.iou_loss import bbox_iou
        iou_mat = bbox_iou(all_boxes.unsqueeze(1), all_boxes.unsqueeze(0), xywh=False)
        
        # Group by overlap
        groups = (iou_mat > self.iou_thr) & (all_labels.unsqueeze(1) == all_labels.unsqueeze(0))
        
        fused_boxes = []
        fused_scores = []
        fused_masks = []
        fused_labels = []
        
        visited = torch.zeros(B, dtype=torch.bool, device=all_boxes.device)
        
        for i in range(B):
            if visited[i] or all_scores[i] < self.skip_box_thr:
                continue
                
            group_idx = torch.where(groups[i])[0]
            visited[group_idx] = True
            
            # Weighted average of boxes and masks based on scores
            weights = all_scores[group_idx].unsqueeze(1) # [G, 1]
            total_weight = weights.sum()
            
            # Box Fusion
            avg_box = (all_boxes[group_idx] * weights).sum(dim=0) / total_weight
            
            # Mask Fusion (The SOTA part)
            avg_mask = None
            if all_masks is not None:
                mask_weights = weights.unsqueeze(2).to(all_masks.device) # [G, 1, 1]
                avg_mask = (all_masks[group_idx] * mask_weights).sum(dim=0) / total_weight
            
            # Top-1 Score for result
            res_score = all_scores[group_idx].max()
            
            fused_boxes.append(avg_box)
            fused_scores.append(res_score)
            if avg_mask is not None:
                fused_masks.append(avg_mask)
            fused_labels.append(all_labels[i])
            
        res = {
            'boxes': torch.stack(fused_boxes) if fused_boxes else torch.zeros((0, 4)),
            'scores': torch.stack(fused_scores) if fused_scores else torch.zeros(0),
            'labels': torch.stack(fused_labels) if fused_labels else torch.zeros(0, dtype=torch.long)
        }
        if fused_masks:
            res['masks'] = torch.stack(fused_masks)
            
        return res
