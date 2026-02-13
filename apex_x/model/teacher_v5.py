import torch
import torch.nn as nn
from .hybrid_backbone_v5 import HybridBackboneV5
from apex_x.routing import RouterKANLike
from .inr_head import ImplicitNeuralHead
from .cascade_head import CascadeDetHead
from .cascade_mask_head import CascadeMaskHead
from .point_rend import PointRendModule
from .worldclass_deps import ensure_worldclass_dependencies

class TeacherModelV5(nn.Module):
    """World-Class Ascension V5 Architecture.
    
    The absolute flagship of Apex-X.
    """
    def __init__(self, num_classes: int = 80, ssm_dim: int = 768):
        super().__init__()
        ensure_worldclass_dependencies(context="TeacherModelV5")
        
        # 1. SOTA Backbone: Hybrid DINOv2 + SSM
        self.backbone = HybridBackboneV5(ssm_dim=ssm_dim)
        
        # 2. Adaptive KAN Router (Decides routing to INR or PointRend)
        self.router = RouterKANLike(in_channels=ssm_dim, num_outputs=2)
        
        # 3. Detection Head (Standard Cascade for initial seeds)
        self.det_head = CascadeDetHead(in_channels=ssm_dim, num_classes=num_classes)
        
        # 4. Standard Mask Head (Coarse seeds)
        self.mask_head = CascadeMaskHead(in_channels=ssm_dim)
        
        # 5. SOTA Implicit Neural Head (Infinite Precision)
        self.inr_head = ImplicitNeuralHead(in_channels=ssm_dim, hidden_dim=256)
        
        # 6. PointRend (Standard SOTA fallback)
        self.point_rend = PointRendModule(in_channels=ssm_dim, out_channels=1)
        
        self._init_weights()

    def _init_weights(self):
        """God-tier initialization for stability."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Ensure the router starts balanced
        if hasattr(self.router, 'gate'):
            nn.init.constant_(self.router.gate.weight, 0)

    def forward(self, images: torch.Tensor):
        # Backbone Features [B, C, H_feat, W_feat]
        feats = self.backbone(images)[0] 
        B, C, H_f, W_f = feats.shape
        
        # Global Feature for Router [B, C]
        global_feat = torch.mean(feats, dim=[2, 3])
        routing = self.router(global_feat) # [B, 2]
        
        # Detection & Coarse Masks
        # In Ascension V5, we use 100 seeds per image as standard for multi-instance detection
        initial_boxes = [torch.zeros((100, 4), device=images.device) for _ in range(B)]
        det_out = self.det_head(feats, initial_boxes, image_size=images.shape[2:])
        final_boxes = det_out["boxes"][-1]
        
        coarse_masks = self.mask_head(feats, det_out["boxes"], image_size=images.shape[2:])
        
        # 5. INR Head Refinement (Instance-aware Infinite resolution)
        # We query the INR head for each detected instance box (100 per image)
        num_boxes = final_boxes[0].shape[0]
        grid_h, grid_w = 28, 28
        W_im, H_im = images.shape[-1], images.shape[-2]
        
        # Create base local grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, grid_h, device=images.device),
            torch.linspace(-1, 1, grid_w, device=images.device),
            indexing="ij"
        )
        local_grid = torch.stack([xx, yy], dim=-1) # [28, 28, 2]
        
        inst_coords = []
        for b in range(B):
            boxes = final_boxes[b] # [100, 4] (x1, y1, x2, y2)
            x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
            
            # Map local [-1, 1] to box coordinates, then to global [-1, 1]
            scaled_grid_x = x1.view(-1, 1, 1, 1) + (local_grid[..., 0] + 1) / 2 * (x2 - x1).view(-1, 1, 1, 1)
            scaled_grid_y = y1.view(-1, 1, 1, 1) + (local_grid[..., 1] + 1) / 2 * (y2 - y1).view(-1, 1, 1, 1)
            
            global_grid_x = (scaled_grid_x / W_im) * 2 - 1
            global_grid_y = (scaled_grid_y / H_im) * 2 - 1
            
            inst_coords.append(torch.stack([global_grid_x, global_grid_y], dim=-1))
            
        all_coords = torch.cat(inst_coords, dim=0).reshape(B * num_boxes, -1, 2) # [B*100, 28*28, 2]
        
        # Expand features for instance query
        feats_expanded = feats.repeat_interleave(num_boxes, dim=0)
        
        # [B*100, P, 1]
        inr_refined = self.inr_head(feats_expanded, all_coords)
        final_masks = inr_refined.reshape(B * num_boxes, 1, grid_h, grid_w)
        
        # ⚠️ SOTA Alignment: Returning keys exactly as expected by train_losses_v3
        return {
            "boxes": torch.cat(final_boxes, dim=0) if final_boxes else torch.zeros(0, 4, device=images.device),
            "scores": det_out["scores"][-1],
            "masks": final_masks, # TotalInst masks
            "coarse_mask": coarse_masks[-1] if isinstance(coarse_masks, list) else coarse_masks,
            "routing": routing,
            "features": feats
        }
