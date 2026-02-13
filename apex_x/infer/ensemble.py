import torch
from torch import Tensor
import torch.nn.functional as F

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
        all_masks = torch.cat([p['masks'] for p in predictions], dim=0)
        all_labels = torch.cat([p['labels'] for p in predictions], dim=0)
        
        # 2. Weighted Boxes Fusion (Logic)
        # For simplicity in this implementation, we use a vectorized soft-grouping
        # based on IoU overlap.
        
        B = all_boxes.shape[0]
        if B == 0:
            return predictions[0]

        # Calculate IoU Matrix
        # [B, 4], [B, 4] -> [B, B]
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
            # [G, H, W] * [G, 1, 1] -> weighted mean
            mask_weights = weights.unsqueeze(2) # [G, 1, 1]
            avg_mask = (all_masks[group_idx] * mask_weights).sum(dim=0) / total_weight
            
            # Top-1 Score for result
            res_score = all_scores[group_idx].max()
            
            fused_boxes.append(avg_box)
            fused_scores.append(res_score)
            fused_masks.append(avg_mask)
            fused_labels.append(all_labels[i])
            
        return {
            'boxes': torch.stack(fused_boxes) if fused_boxes else torch.zeros((0, 4)),
            'scores': torch.stack(fused_scores) if fused_scores else torch.zeros(0),
            'masks': torch.stack(fused_masks) if fused_masks else torch.zeros((0, all_masks.shape[1], all_masks.shape[2])),
            'labels': torch.stack(fused_labels) if fused_labels else torch.zeros(0, dtype=torch.long)
        }
