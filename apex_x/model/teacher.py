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
from .pv_module import PVModule, PVModuleOutput

_LOGIT_LEVEL_ORDER: Final[tuple[str, ...]] = ("P3", "P4", "P5", "P6", "P7")
ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class TeacherDistillOutput:
    """Standardized teacher outputs for distillation."""

    logits: Tensor  # [B,L] flattened logits across P3..P7
    logits_by_level: dict[str, Tensor]  # [B,C,H,W] per level
    features: dict[str, Tensor]  # selected feature layers for feature distill
    boundaries: Tensor  # [B,1,H,W] boundary proxy aligned to input image size


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
        else:
            self.pv_module = pv_module
            self.fpn = fpn
            self.det_head = det_head

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
        pv_module: PVModule,
        fpn: DualPathFPN,
        det_head: DetHead,
    ) -> tuple[PVModuleOutput, DetHeadOutput, dict[str, Tensor]]:
        pv_out = pv_module(image)
        ff_high = pv_out.features["P3"]
        fpn_out = fpn(pv_out.features, ff_high)
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

        boundaries = f.interpolate(
            pv_out.proxy_maps["boundary"],
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        logits_flat = flatten_logits_for_distill(det_out.cls_logits)
        selected_features = self._select_features(fpn_features)

        return TeacherDistillOutput(
            logits=logits_flat,
            logits_by_level=det_out.cls_logits,
            features=selected_features,
            boundaries=boundaries,
        )


__all__ = [
    "TeacherDistillOutput",
    "flatten_logits_for_distill",
    "TeacherModel",
]
