"""Cascade instance segmentation head with iterative mask refinement.

Extends cascade detection to segmentation with progressive mask refinement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CascadeMaskStage(nn.Module):
    """Single stage of cascade mask prediction.
    
    Args:
        in_channels: Input feature channels
        mask_size: Output mask resolution
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        mask_size: int = 28,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.mask_size = mask_size
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),  # Final mask logits
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_features: Tensor) -> Tensor:
        """Predict masks from RoI features.
        
        Args:
            roi_features: RoI features [N, C, H, W]
            
        Returns:
            Mask logits [N, 1, mask_size, mask_size]
        """
        return self.mask_head(roi_features)


class CascadeMaskHead(nn.Module):
    """Cascade mask head with multi-stage refinement.
    
    Progressively refines masks through multiple stages,
    similar to cascade box detection.
    
    Args:
        in_channels: Input feature channels
        num_stages: Number of refinement stages
        mask_sizes: Mask resolution for each stage
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_stages: int = 3,
        mask_sizes: list[int] | None = None,
    ):
        super().__init__()
        
        self.num_stages = num_stages
        
        # Default mask sizes: [14, 28, 28] (low to high res)
        if mask_sizes is None:
            mask_sizes = [14, 28, 28]
        
        if len(mask_sizes) != num_stages:
            raise ValueError(f"mask_sizes length must equal num_stages")
        
        self.mask_sizes = mask_sizes
        
        # Create mask stages
        self.stages = nn.ModuleList([
            CascadeMaskStage(
                in_channels=in_channels,
                mask_size=size,
            )
            for size in mask_sizes
        ])
        
        self.point_head = None # Lazy init

    
    def forward(
        self,
        features: Tensor,
        boxes: list[list[Tensor]],
        *,
        image_size: tuple[int, int] | None = None,
    ) -> list[Tensor]:
        """Progressive mask refinement for batches.
        
        Args:
            features: Feature maps [B, C, H, W]
            boxes: List of stages, each containing a list of boxes per batch element [S, B, N_i, 4]
            image_size: Optional ``(H, W)`` of input image used to map
                image-space boxes to feature-space for RoIAlign.
            
        Returns:
            List of mask logits from each stage [S, N_total, 1, H_mask, W_mask]
        """
        all_masks = []
        
        # Iterate over stages
        # self.stages is ModuleList of CascadeMaskStage
        # boxes is list of [B, N, 4] per stage? No, logic says boxes is list of stages.
        # Each stage is list of boxes per batch?
        # TeacherModelV3 passes `all_boxes[1:]`.
        # `all_boxes` structure in TeacherModelV3: "list[list[Tensor]] -> [Stage][Batch]"
        
        for i, stage_module in enumerate(self.stages):
            if i >= len(boxes):
                break
                
            stage_boxes_list = boxes[i] # list[Tensor] of len B
            
            # Flatten boxes [N_total, 5] (batch_idx, x1, y1, x2, y2)
            flat_boxes, counts = self.flatten_boxes_for_roi(stage_boxes_list, features.device)
            
            if flat_boxes.numel() == 0:
                # No boxes in this stage
                # We should append None or empty tensor?
                # TeacherModelV3 expects list of tensors.
                # Let's append None or handle it.
                # If we return None, `final_masks_flat` logic will fail.
                # Return empty tensor [0, 1, mask_size, mask_size]
                sz = stage_module.mask_size
                all_masks.append(torch.zeros((0, 1, sz, sz), device=features.device, dtype=features.dtype))
                continue

            # Extract RoI features
            # RoIAlign expects [K, 5] (batch_idx, x1, y1, x2, y2)
            # Output size: 14x14 usually?
            # CascadeMaskStage head starts with Conv2d(in, hidden, 3, 1).
            # If we want 14x14 output from 14x14 input?
            # Usually Mask Head takes 14x14 and upsamples to 28x28.
            # `CascadeMaskStage` has `ConvTranspose2d` (stride 2).
            # So if input is 14x14, output is 28x28.
            # If `mask_size` is 28, we need 14x14 input.
            # If `mask_size` is 56, we need 28x28 input? 
            # Or `CascadeMaskStage` structure is fixed?
            # It has `ConvTranspose2d`.
            # Check `CascadeMaskStage.__init__`:
            # 4 convs of 3x3 padding 1 -> size stays same.
            # ConvTranspose2d 2x2 stride 2 -> size doubles.
            # Conv2d 1x1 -> size stays same.
            # So Output size = 2 * Input size.
            
            # We need to determine RoIAlign size based on target mask_size.
            # target = stage_module.mask_size
            # input = target // 2
            
            input_size = stage_module.mask_size // 2
            
            roi_features = self._roi_align(
                features,
                flat_boxes,
                output_size=(input_size, input_size),
                image_size=image_size,
            )
            
            # Predict
            mask_logits = stage_module(roi_features) # [N_total, 1, mask_size, mask_size]
            all_masks.append(mask_logits)
        
        return all_masks

    def forward_with_points(
        self,
        features: Tensor,
        boxes: list[list[Tensor]],
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        """Forward pass extracting point logits for training."""
        all_masks = []
        
        # Use final stage boxes for point selection
        final_boxes_list = boxes[-1] 
        flat_boxes, _ = self.flatten_boxes_for_roi(final_boxes_list, features.device)
        
        # RoI Align for final stage
        roi_features = self._roi_align(
            features,
            flat_boxes,
            output_size=(14, 14),
        )
        
        # Coarse prediction from final stage standard head
        # Note: We should ideally use the cascade logic, but for PointRend training
        # we often just need the final coarse mask and corresponding fine features.
        # Here we simplify: assume last stage head gives coarse mask.
        coarse_logits = self.stages[-1](roi_features) # [N, 1, 28, 28]
        
        if self.point_head is None:
             from apex_x.model.point_rend import PointRendModule, sampling_points
             # Lazy init
             self.point_head = PointRendModule(in_channels=features.shape[1] + 1).to(features.device)
        else:
             from apex_x.model.point_rend import sampling_points
             
        # Sample points
        # For training: sample based on uncertainty of coarse logits
        point_coords = sampling_points(coarse_logits, num_points=2048) # [N, 1, P, 2]
        
        # Get point logits
        # We need fine features. 
        # Strategy: Sample from the ROI feature map (14x14)?? 
        # No, PointRend needs FINE features. 
        # If `features` is P2 (stride 4), we should sample from `features` directly using `flat_boxes`.
        # This requires mapping point_coords (0..1 in ROI) to image coords.
        # But `flatten_boxes_for_roi` gives us boxes in feature coords!
        # box = [batch_idx, x1, y1, x2, y2]
        
        # Map points to feature map space
        # pt_x_feat = x1 + pt_x_roi * (x2 - x1)
        # pt_y_feat = y1 + pt_y_roi * (y2 - y1)
        
        N, _, P, _ = point_coords.shape
        point_coords_flat = point_coords.view(N, P, 2)
        
        x1 = flat_boxes[:, 1:2]
        y1 = flat_boxes[:, 2:3]
        w  = flat_boxes[:, 3:4] - x1
        h  = flat_boxes[:, 4:5] - y1
        
        grid_x = x1 + point_coords_flat[:, :, 0] * w
        grid_y = y1 + point_coords_flat[:, :, 1] * h
        
        # Normalize to [-1, 1] for the whole feature map
        # features shape [B, C, Hf, Wf]
        B, C, Hf, Wf = features.shape
        
        # We need batch index for each point to sample correctly from batch
        # grid_sample handles [B, C, H, W] and [B, H_grid, W_grid, 2]
        # But our points are mixed in N roi boxes from different batch images.
        # This is hard to vectorize purely with grid_sample if N rois come from random images.
        # BUT: flat_boxes has batch_idx in column 0.
        
        # Loop per batch image? Or use fancy indexing.
        # PointRend official usually does per-image logic.
        
        # Optimized approach:
        # Group ROIs by batch index.
        point_logits_list = []
        
        # For each batch index
        for b in range(B):
            mask = flat_boxes[:, 0] == b
            if not mask.any():
                continue
                
            # Points for this image: [N_i, P, 2] (in feature coords)
            batch_grid_x = grid_x[mask] # [N_i, P]
            batch_grid_y = grid_y[mask]
            
            # Normalize to [-1, 1] relative to feature map size
            # x_norm = 2 * x / Wf - 1
            batch_grid_x_norm = 2.0 * batch_grid_x / Wf - 1.0
            batch_grid_y_norm = 2.0 * batch_grid_y / Hf - 1.0
            
            grid = torch.stack([batch_grid_x_norm, batch_grid_y_norm], dim=-1) # [N_i, P, 2]
            
            # We need [1, H_out, W_out, 2] for grid sample to sample from 1 image
            # Treat all points as a "width" of P, height of N_i? 
            # Input: [1, C, Hf, Wf]
            # Grid: [1, N_i, P, 2]
            
            feat_b = features[b:b+1] # [1, C, Hf, Wf]
            grid_b = grid.unsqueeze(0) # [1, N_i, P, 2]
            
            fine_feats = F.grid_sample(feat_b, grid_b, align_corners=False) # [1, C, N_i, P]
            fine_feats = fine_feats.permute(0, 2, 1, 3).squeeze(0) # [N_i, C, P]
            
            # Also sample Coarse Logits at the same relative coords [0, 1]
            # Coarse logits: [N_i, 1, 28, 28]
            # Coords relative: point_coords_flat[mask] -> [N_i, P, 2]
            coords_rel = point_coords_flat[mask].unsqueeze(1) # [N_i, 1, P, 2]
            coarse_sampled = F.grid_sample(
                coarse_logits[mask], 
                2.0 * coords_rel - 1.0, 
                align_corners=False
            ).squeeze(2) # [N_i, 1, P]
            
            # Predict
            # MLP inputs: cat([fine, coarse], dim=1) -> [N_i, C+1, P]
            inp = torch.cat([fine_feats, coarse_sampled], dim=1)
            plogits = self.point_head.mlp(inp) # [N_i, 1, P]
            point_logits_list.append(plogits)
            
        if point_logits_list:
            point_logits = torch.cat(point_logits_list, dim=0)
        else:
            point_logits = torch.zeros((0, 1, 2048), device=features.device)
            
        return all_masks, point_logits, point_coords_flat


    def flatten_boxes_for_roi(self, boxes_list: list[Tensor], device: torch.device) -> tuple[Tensor, list[int]]:
        flat_boxes = []
        counts = []
        for i, boxes in enumerate(boxes_list):
            if boxes.numel() == 0:
                counts.append(0)
                continue
            batch_idx = torch.full((boxes.shape[0], 1), i, dtype=boxes.dtype, device=device)
            flat_boxes.append(torch.cat([batch_idx, boxes], dim=1))
            counts.append(boxes.shape[0])
        
        if not flat_boxes:
            return torch.zeros((0, 5), dtype=torch.float32, device=device), counts
            
        return torch.cat(flat_boxes, dim=0), counts

    def _roi_align(
        self,
        features: Tensor,
        boxes_with_batch: Tensor,
        output_size: tuple[int, int] = (14, 14),
        image_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """RoI Align for mask features."""
        if boxes_with_batch.numel() == 0:
            B, C, _, _ = features.shape
            N = boxes_with_batch.shape[0]
            return torch.zeros((N, C, *output_size), device=features.device, dtype=features.dtype)

        boxes = boxes_with_batch
        if image_size is not None:
            img_h, img_w = image_size
            feat_h, feat_w = features.shape[-2:]
            scale_x = float(feat_w) / max(float(img_w), 1.0)
            scale_y = float(feat_h) / max(float(img_h), 1.0)
            boxes = boxes_with_batch.clone()
            boxes[:, 1] = boxes[:, 1] * scale_x
            boxes[:, 3] = boxes[:, 3] * scale_x
            boxes[:, 2] = boxes[:, 2] * scale_y
            boxes[:, 4] = boxes[:, 4] * scale_y

        try:
            from torchvision.ops import roi_align
            return roi_align(
                features,
                boxes,
                output_size=output_size,
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
        except ImportError:
            # Fallback
            B, C, _, _ = features.shape
            N = boxes_with_batch.shape[0]
            return torch.zeros((N, C, *output_size), device=features.device, dtype=features.dtype)


__all__ = ['CascadeMaskHead', 'CascadeMaskStage']
