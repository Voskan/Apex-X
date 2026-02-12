from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Final, TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as f

from apex_x.config import ApexXConfig

from .det_head import DetHead, DetHeadOutput
from .fpn import DualPathFPN
from .inst_seg_head import PrototypeInstanceSegHead
from .pv_module import PVModule, PVModuleOutput
from .post_process import compute_anchor_centers

_LOGIT_LEVEL_ORDER: Final[tuple[str, ...]] = ("P3", "P4", "P5", "P6", "P7")
ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class TeacherDistillOutput:
    """Standardized teacher outputs for distillation."""

    logits: Tensor  # [B,L] flattened logits across P3..P7
    logits_by_level: dict[str, Tensor]  # [B,C,H,W] per level
    features: dict[str, Tensor]  # selected feature layers for feature distill
    boundaries: Tensor  # [B,1,H,W] boundary proxy aligned to input image size
    
    # Detection outputs for loss computation
    boxes_by_level: dict[str, Tensor]  # [B, 4, H, W] per level box regressions
    quality_by_level: dict[str, Tensor]  # [B, 1, H, W] per level quality scores
    
    # Segmentation outputs (optional, may be None if no seg head)
    masks: Tensor | None = None  # [B, N, H_mask, W_mask] instance masks if available


def flatten_logits_for_distill(logits_by_level: dict[str, Tensor]) -> Tensor:
    """Flatten multi-level logits into a deterministic [B,L] representation."""
    if not logits_by_level:
        raise ValueError("logits_by_level must not be empty")

    first = next(iter(logits_by_level.values()))
    if first.ndim != 4:
        raise ValueError("logits tensors must be [B,C,H,W]")
    batch = int(first.shape[0])

    ordered_levels = [level for level in _LOGIT_LEVEL_ORDER if level in logits_by_level]
    ordered_levels.extend(level for level in sorted(logits_by_level) if level not in ordered_levels)
    if not ordered_levels:
        raise ValueError("no logits levels available")

    flat_parts: list[Tensor] = []
    for level in ordered_levels:
        logits = logits_by_level[level]
        if logits.ndim != 4:
            raise ValueError(f"logits level {level!r} must be [B,C,H,W]")
        if int(logits.shape[0]) != batch:
            raise ValueError("all logits levels must share batch size")
        # if not torch.isfinite(logits).all():
        #     raise ValueError(f"logits level {level!r} must contain finite values")
        flat_parts.append(logits.reshape(batch, -1))

    return torch.cat(flat_parts, dim=1)


class TeacherModel(nn.Module):
    """Full-compute teacher model with standardized distillation outputs and optional EMA."""

    def __init__(
        self,
        *,
        num_classes: int,
        config: ApexXConfig | None = None,
        pv_module: PVModule | None = None,
        fpn: DualPathFPN | None = None,
        det_head: DetHead | None = None,
        feature_layers: tuple[str, ...] = ("P3", "P4", "P5"),
        use_ema: bool = False,
        ema_decay: float = 0.999,
        use_ema_for_forward: bool = True,
        enable_seg_head: bool = True,
        seg_num_instances: int = 16,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if not feature_layers:
            raise ValueError("feature_layers must not be empty")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")

        self.config = config or ApexXConfig()
        self.config.validate()
        self.num_classes = int(num_classes)
        self.feature_layers = tuple(str(layer) for layer in feature_layers)
        self.full_compute_mode = True
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.use_ema_for_forward = bool(use_ema_for_forward)
        self.seg_num_instances = max(1, int(seg_num_instances))

        if pv_module is None or fpn is None or det_head is None:
            model_cfg = self.config.model
            p3_c, p4_c, p5_c = (int(value) for value in model_cfg.pv_channels[:3])
            out_c = int(model_cfg.ff_channels[0])
            self.pv_module = PVModule(
                in_channels=3,
                p3_channels=p3_c,
                p4_channels=p4_c,
                p5_channels=p5_c,
                coarse_level="P4",
            )
            self.fpn = DualPathFPN(
                pv_p3_channels=p3_c,
                pv_p4_channels=p4_c,
                pv_p5_channels=p5_c,
                ff_channels=p3_c,
                out_channels=out_c,
            )
            self.det_head = DetHead(
                in_channels=out_c,
                num_classes=self.num_classes,
                hidden_channels=out_c,
            )
            self.seg_head = (
                PrototypeInstanceSegHead(
                    in_channels=out_c,
                    num_prototypes=16,
                    coeff_hidden_dim=max(64, out_c),
                )
                if enable_seg_head
                else None
            )
        else:
            self.pv_module = pv_module
            self.fpn = fpn
            self.det_head = det_head
            self.seg_head = (
                PrototypeInstanceSegHead(
                    in_channels=self.det_head.in_channels,
                    num_prototypes=16,
                    coeff_hidden_dim=max(64, self.det_head.in_channels),
                )
                if enable_seg_head
                else None
            )

        self.ema_pv_module: PVModule | None = None
        self.ema_fpn: DualPathFPN | None = None
        self.ema_det_head: DetHead | None = None
        if self.use_ema:
            self.ema_pv_module = self._clone_frozen_module(self.pv_module)
            self.ema_fpn = self._clone_frozen_module(self.fpn)
            self.ema_det_head = self._clone_frozen_module(self.det_head)

    def _clone_frozen_module(self, module: ModuleT) -> ModuleT:
        clone = copy.deepcopy(module)
        clone.requires_grad_(False)
        return clone

    @staticmethod
    def _ema_update_module(target: nn.Module, source: nn.Module, decay: float) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(
                target.parameters(),
                source.parameters(),
                strict=True,
            ):
                target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
            for target_buf, source_buf in zip(target.buffers(), source.buffers(), strict=True):
                target_buf.data.copy_(source_buf.data)

    def update_ema(self, *, decay: float | None = None) -> None:
        if not self.use_ema:
            return
        if self.ema_pv_module is None or self.ema_fpn is None or self.ema_det_head is None:
            raise RuntimeError("EMA modules are not initialized")

        d = self.ema_decay if decay is None else float(decay)
        if not (0.0 < d < 1.0):
            raise ValueError("decay must be in (0, 1)")

        self._ema_update_module(self.ema_pv_module, self.pv_module, d)
        self._ema_update_module(self.ema_fpn, self.fpn, d)
        self._ema_update_module(self.ema_det_head, self.det_head, d)

    def _forward_modules(
        self,
        image: Tensor,
        *,
        pv_module: nn.Module,
        fpn: DualPathFPN,
        det_head: DetHead,
    ) -> tuple[PVModuleOutput | dict[str, Tensor], DetHeadOutput, dict[str, Tensor]]:
        pv_out = pv_module(image)
        if isinstance(pv_out, dict):
            pv_features = pv_out
        else:
            pv_features = pv_out.features
        ff_high = pv_features["P3"]
        fpn_out = fpn(pv_features, ff_high)
        det_out = det_head(fpn_out.pyramid)
        return pv_out, det_out, fpn_out.pyramid

    def _select_features(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        selected: dict[str, Tensor] = {}
        for layer in self.feature_layers:
            if layer not in features:
                available = ", ".join(sorted(features.keys()))
                raise ValueError(f"feature layer {layer!r} not available; got: {available}")
            selected[layer] = features[layer]
        return selected

    def _propose_segmentation_boxes(
        self,
        *,
        det_out: DetHeadOutput,
        image: Tensor,
    ) -> Tensor | None:
        if "P3" not in det_out.cls_logits or "P3" not in det_out.box_reg or "P3" not in det_out.quality:
            return None

        cls_logits = det_out.cls_logits["P3"]
        box_reg = det_out.box_reg["P3"]
        quality = det_out.quality["P3"]
        if cls_logits.ndim != 4 or box_reg.ndim != 4 or quality.ndim != 4:
            return None

        bsz, num_classes, feat_h, feat_w = cls_logits.shape
        num_anchors = feat_h * feat_w
        if num_anchors <= 0:
            return None

        topk = min(self.seg_num_instances, num_anchors)
        anchor_centers = compute_anchor_centers((feat_h, feat_w), stride=8, device=cls_logits.device)
        anchor_centers = anchor_centers.to(dtype=cls_logits.dtype)

        image_h = float(image.shape[2])
        image_w = float(image.shape[3])
        batch_boxes: list[Tensor] = []

        for batch_idx in range(bsz):
            cls_flat = cls_logits[batch_idx].permute(1, 2, 0).reshape(num_anchors, num_classes)
            box_flat = box_reg[batch_idx].permute(1, 2, 0).reshape(num_anchors, 4)
            quality_flat = quality[batch_idx].permute(1, 2, 0).reshape(num_anchors).sigmoid()

            score_flat = cls_flat.sigmoid().amax(dim=1) * quality_flat
            topk_indices = torch.topk(score_flat, k=topk, largest=True, sorted=True).indices

            selected_box = box_flat[topk_indices]
            selected_center = anchor_centers[topk_indices]
            distances = selected_box * 8.0
            x1 = selected_center[:, 0] - distances[:, 0]
            y1 = selected_center[:, 1] - distances[:, 1]
            x2 = selected_center[:, 0] + distances[:, 2]
            y2 = selected_center[:, 1] + distances[:, 3]

            boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=1)
            boxes_xyxy[:, 0] = boxes_xyxy[:, 0].clamp(0.0, image_w)
            boxes_xyxy[:, 2] = boxes_xyxy[:, 2].clamp(0.0, image_w)
            boxes_xyxy[:, 1] = boxes_xyxy[:, 1].clamp(0.0, image_h)
            boxes_xyxy[:, 3] = boxes_xyxy[:, 3].clamp(0.0, image_h)
            batch_boxes.append(boxes_xyxy)

        if not batch_boxes:
            return None
        return torch.stack(batch_boxes, dim=0)

    def forward(self, image: Tensor, *, use_ema: bool | None = None) -> TeacherDistillOutput:
        if image.ndim != 4:
            raise ValueError("image must be [B,3,H,W]")
        if image.shape[1] != 3:
            raise ValueError("image channel dim must be 3")
        # if not torch.isfinite(image).all():
        #     raise ValueError("image must contain finite values")

        run_with_ema = self.use_ema_for_forward if use_ema is None else bool(use_ema)
        if run_with_ema and self.use_ema:
            if self.ema_pv_module is None or self.ema_fpn is None or self.ema_det_head is None:
                raise RuntimeError("EMA modules are not initialized")
            pv_out, det_out, fpn_features = self._forward_modules(
                image,
                pv_module=self.ema_pv_module,
                fpn=self.ema_fpn,
                det_head=self.ema_det_head,
            )
        else:
            pv_out, det_out, fpn_features = self._forward_modules(
                image,
                pv_module=self.pv_module,
                fpn=self.fpn,
                det_head=self.det_head,
            )

        if isinstance(pv_out, dict):
            boundary_source = image.new_zeros((image.shape[0], 1, image.shape[2], image.shape[3]))
        else:
            boundary_source = f.interpolate(
                pv_out.proxy_maps["boundary"],
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        boundaries = boundary_source.clamp(0.0, 1.0)
        logits_flat = flatten_logits_for_distill(det_out.cls_logits)
        selected_features = self._select_features(fpn_features)

        masks = None
        if self.seg_head is not None:
            seg_boxes = self._propose_segmentation_boxes(det_out=det_out, image=image)
            if seg_boxes is not None and seg_boxes.shape[1] > 0:
                seg_output = self.seg_head(
                    features=fpn_features,
                    boxes_xyxy=seg_boxes,
                    image_size=(int(image.shape[2]), int(image.shape[3])),
                    output_size=(int(image.shape[2]), int(image.shape[3])),
                    normalized_boxes=False,
                    crop_to_boxes=True,
                )
                masks = seg_output.masks

        return TeacherDistillOutput(
            logits=logits_flat,
            logits_by_level=det_out.cls_logits,
            features=selected_features,
            boundaries=boundaries,
            boxes_by_level=det_out.box_reg,
            quality_by_level=det_out.quality,
            masks=masks,
        )


__all__ = [
    "TeacherDistillOutput",
    "flatten_logits_for_distill",
    "TeacherModel",
]
