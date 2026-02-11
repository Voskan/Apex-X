from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as f

from apex_x.tiles import OverlapMode, TilePackTorch, TileUnpackTorch


@dataclass(frozen=True)
class InstanceSegOutput:
    """Prototype-based instance segmentation outputs."""

    prototypes: Tensor  # [B,M,Hp,Wp]
    coefficients: Tensor  # [B,N,M]
    mask_logits_lowres: Tensor  # [B,N,Hp,Wp]
    mask_logits: Tensor  # [B,N,Hout,Wout]
    masks: Tensor  # [B,N,Hout,Wout] in [0,1]
    mask_scores: Tensor  # [B,N]


def assemble_mask_logits_from_prototypes(prototypes: Tensor, coefficients: Tensor) -> Tensor:
    """Assemble per-instance mask logits via prototype linear combination."""
    if prototypes.ndim != 4:
        raise ValueError("prototypes must be [B,M,H,W]")
    if coefficients.ndim != 3:
        raise ValueError("coefficients must be [B,N,M]")
    if prototypes.shape[0] != coefficients.shape[0]:
        raise ValueError("prototypes and coefficients must share batch size")
    if prototypes.shape[1] != coefficients.shape[2]:
        raise ValueError("prototype channels M must match coefficient dimension")

    return torch.einsum("bnm,bmhw->bnhw", coefficients, prototypes)


def _normalize_xyxy(boxes_xyxy: Tensor) -> Tensor:
    x1 = torch.minimum(boxes_xyxy[..., 0], boxes_xyxy[..., 2])
    y1 = torch.minimum(boxes_xyxy[..., 1], boxes_xyxy[..., 3])
    x2 = torch.maximum(boxes_xyxy[..., 0], boxes_xyxy[..., 2])
    y2 = torch.maximum(boxes_xyxy[..., 1], boxes_xyxy[..., 3])
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _project_boxes_to_size(
    boxes_xyxy: Tensor,
    *,
    target_height: int,
    target_width: int,
    normalized_boxes: bool,
    image_size: tuple[int, int] | None,
) -> Tensor:
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_height and target_width must be > 0")

    boxes = _normalize_xyxy(boxes_xyxy)
    if normalized_boxes:
        scale_x = float(target_width)
        scale_y = float(target_height)
        projected = boxes.clone()
        projected[..., 0] = projected[..., 0] * scale_x
        projected[..., 2] = projected[..., 2] * scale_x
        projected[..., 1] = projected[..., 1] * scale_y
        projected[..., 3] = projected[..., 3] * scale_y
        return projected

    if image_size is None:
        return boxes

    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_size must be (H,W) with positive dimensions")

    scale_x = float(target_width) / float(image_w)
    scale_y = float(target_height) / float(image_h)
    projected = boxes.clone()
    projected[..., 0] = projected[..., 0] * scale_x
    projected[..., 2] = projected[..., 2] * scale_x
    projected[..., 1] = projected[..., 1] * scale_y
    projected[..., 3] = projected[..., 3] * scale_y
    return projected


def rasterize_box_masks(
    boxes_xyxy: Tensor,
    *,
    height: int,
    width: int,
    normalized_boxes: bool = False,
    image_size: tuple[int, int] | None = None,
) -> Tensor:
    """Rasterize axis-aligned boxes into binary masks."""
    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must be [B,N,4]")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be > 0")
    if not torch.isfinite(boxes_xyxy).all():
        raise ValueError("boxes_xyxy must contain finite values")

    projected = _project_boxes_to_size(
        boxes_xyxy,
        target_height=height,
        target_width=width,
        normalized_boxes=normalized_boxes,
        image_size=image_size,
    )
    projected = _normalize_xyxy(projected)
    projected[..., 0] = projected[..., 0].clamp(0.0, float(width))
    projected[..., 2] = projected[..., 2].clamp(0.0, float(width))
    projected[..., 1] = projected[..., 1].clamp(0.0, float(height))
    projected[..., 3] = projected[..., 3].clamp(0.0, float(height))

    bsz, num_instances, _ = projected.shape
    masks = torch.zeros(
        (bsz, num_instances, height, width),
        device=boxes_xyxy.device,
        dtype=torch.bool,
    )
    for b in range(bsz):
        for n in range(num_instances):
            box = projected[b, n]
            x1 = int(torch.floor(box[0]).item())
            y1 = int(torch.floor(box[1]).item())
            x2 = int(torch.ceil(box[2]).item())
            y2 = int(torch.ceil(box[3]).item())
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            masks[b, n, y1:y2, x1:x2] = True
    return masks


class FFTileRefinementHook(nn.Module):
    """Refine mask logits using FF high-res features on active tiles only."""

    def __init__(
        self,
        *,
        tile_size: int,
        order_mode: str = "hilbert",
        overlap_mode: OverlapMode = "override",
        blend_alpha: float = 0.5,
        strength_init: float = 0.5,
    ) -> None:
        super().__init__()
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if not (0.0 <= blend_alpha <= 1.0):
            raise ValueError("blend_alpha must be in [0,1]")
        if overlap_mode not in {"override", "blend"}:
            raise ValueError("overlap_mode must be 'override' or 'blend'")

        self.tile_size = int(tile_size)
        self.order_mode = str(order_mode)
        self.overlap_mode: OverlapMode = overlap_mode
        self.blend_alpha = float(blend_alpha)
        self.log_strength = nn.Parameter(torch.tensor(float(strength_init)))

        self.packer = TilePackTorch()
        self.unpacker = TileUnpackTorch()

    def forward(
        self,
        mask_logits: Tensor,
        ff_highres_features: Tensor,
        active_tile_indices: Tensor,
    ) -> Tensor:
        if mask_logits.ndim != 4:
            raise ValueError("mask_logits must be [B,N,H,W]")
        if ff_highres_features.ndim != 4:
            raise ValueError("ff_highres_features must be [B,C,H,W]")
        if active_tile_indices.ndim != 2:
            raise ValueError("active_tile_indices must be [B,K]")
        if mask_logits.shape[0] != ff_highres_features.shape[0]:
            raise ValueError("mask_logits and ff_highres_features must share batch size")
        if mask_logits.shape[0] != active_tile_indices.shape[0]:
            raise ValueError("active_tile_indices batch size must match mask_logits")
        if active_tile_indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("active_tile_indices must be an integer tensor")

        if active_tile_indices.shape[1] == 0:
            return mask_logits

        ff_aligned = ff_highres_features
        if ff_aligned.shape[2:] != mask_logits.shape[2:]:
            ff_aligned = f.interpolate(
                ff_aligned,
                size=mask_logits.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        ff_signal = torch.tanh(ff_aligned.mean(dim=1, keepdim=True))

        mask_tiles, meta = self.packer.pack(
            feature_map=mask_logits,
            indices=active_tile_indices,
            tile_size=self.tile_size,
            order_mode=self.order_mode,
        )
        ff_tiles, _ = self.packer.pack(
            feature_map=ff_signal,
            indices=active_tile_indices,
            tile_size=self.tile_size,
            order_mode=self.order_mode,
        )
        strength = torch.nn.functional.softplus(self.log_strength)
        ff_delta = strength * ff_tiles.expand(-1, -1, mask_logits.shape[1], -1, -1)
        refined_tiles = mask_tiles + ff_delta

        refined_logits, _ = self.unpacker.unpack(
            base_map=mask_logits,
            packed_out=refined_tiles,
            meta=meta,
            level_priority=1,
            overlap_mode=self.overlap_mode,
            blend_alpha=self.blend_alpha,
        )
        return refined_logits


class PrototypeInstanceSegHead(nn.Module):
    """Prototype-based instance segmentation head with mask assembly."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_prototypes: int = 32,
        coeff_input_dim: int | None = None,
        coeff_hidden_dim: int = 128,
        proto_hidden_dim: int | None = None,
        proto_layers: int = 2,
        mask_fill_value: float = -20.0,
        enable_ff_refine: bool = False,
        ff_refine_tile_size: int = 8,
        ff_refine_order_mode: str = "hilbert",
        ff_refine_overlap_mode: OverlapMode = "override",
        ff_refine_blend_alpha: float = 0.5,
        ff_refine_strength_init: float = 0.5,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be > 0")
        if coeff_hidden_dim <= 0:
            raise ValueError("coeff_hidden_dim must be > 0")
        if proto_layers <= 0:
            raise ValueError("proto_layers must be > 0")

        hidden = int(in_channels if proto_hidden_dim is None else proto_hidden_dim)
        if hidden <= 0:
            raise ValueError("proto_hidden_dim must be > 0")
        coeff_dim = int(in_channels if coeff_input_dim is None else coeff_input_dim)
        if coeff_dim <= 0:
            raise ValueError("coeff_input_dim must be > 0")

        self.in_channels = int(in_channels)
        self.num_prototypes = int(num_prototypes)
        self.coeff_input_dim = coeff_dim
        self.mask_fill_value = float(mask_fill_value)
        self.enable_ff_refine = bool(enable_ff_refine)

        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for _ in range(proto_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        hidden,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups=1, num_channels=hidden),
                    nn.SiLU(inplace=False),
                ]
            )
            current_channels = hidden
        self.proto_trunk = nn.Sequential(*layers)
        self.proto_pred = nn.Conv2d(hidden, self.num_prototypes, kernel_size=1, bias=True)
        self.coeff_source = nn.Conv2d(
            self.in_channels,
            self.coeff_input_dim,
            kernel_size=1,
            bias=False,
        )
        self.coeff_mlp = nn.Sequential(
            nn.Linear(self.coeff_input_dim, coeff_hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(coeff_hidden_dim, self.num_prototypes),
        )
        self.ff_refine_hook: FFTileRefinementHook | None
        if self.enable_ff_refine:
            self.ff_refine_hook = FFTileRefinementHook(
                tile_size=ff_refine_tile_size,
                order_mode=ff_refine_order_mode,
                overlap_mode=ff_refine_overlap_mode,
                blend_alpha=ff_refine_blend_alpha,
                strength_init=ff_refine_strength_init,
            )
        else:
            self.ff_refine_hook = None

    def _select_feature(self, features: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(features, Tensor):
            if features.ndim != 4:
                raise ValueError("feature tensor must be [B,C,H,W]")
            if features.shape[1] != self.in_channels:
                raise ValueError("feature channel dim does not match in_channels")
            return features

        preferred_levels = ("P3", "P4", "P5")
        selected: Tensor | None = None
        for level in preferred_levels:
            if level in features:
                selected = features[level]
                break
        if selected is None:
            if not features:
                raise ValueError("features dict must not be empty")
            first_key = sorted(features.keys())[0]
            selected = features[first_key]

        if selected.ndim != 4:
            raise ValueError("selected feature map must be [B,C,H,W]")
        if selected.shape[1] != self.in_channels:
            raise ValueError("selected feature channel dim does not match in_channels")
        return selected

    def _roi_mean_pool(
        self,
        source: Tensor,
        boxes_xyxy: Tensor,
        *,
        normalized_boxes: bool,
        image_size: tuple[int, int] | None,
    ) -> Tensor:
        bsz, channels, height, width = source.shape
        if boxes_xyxy.shape[0] != bsz:
            raise ValueError("boxes batch size must match feature batch size")
        num_instances = boxes_xyxy.shape[1]

        projected = _project_boxes_to_size(
            boxes_xyxy,
            target_height=height,
            target_width=width,
            normalized_boxes=normalized_boxes,
            image_size=image_size,
        )
        projected = _normalize_xyxy(projected)
        projected[..., 0] = projected[..., 0].clamp(0.0, float(width))
        projected[..., 2] = projected[..., 2].clamp(0.0, float(width))
        projected[..., 1] = projected[..., 1].clamp(0.0, float(height))
        projected[..., 3] = projected[..., 3].clamp(0.0, float(height))

        pooled = source.new_zeros((bsz, num_instances, channels))
        for b in range(bsz):
            for n in range(num_instances):
                box = projected[b, n]
                x1 = int(torch.floor(box[0]).item())
                y1 = int(torch.floor(box[1]).item())
                x2 = int(torch.ceil(box[2]).item())
                y2 = int(torch.ceil(box[3]).item())

                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = source[b, :, y1:y2, x1:x2]
                pooled[b, n] = roi.mean(dim=(1, 2))

        return pooled

    def forward(
        self,
        features: Tensor | dict[str, Tensor],
        boxes_xyxy: Tensor,
        *,
        instance_embeddings: Tensor | None = None,
        image_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
        normalized_boxes: bool = False,
        crop_to_boxes: bool = True,
        ff_highres_features: Tensor | None = None,
        active_tile_indices: Tensor | None = None,
    ) -> InstanceSegOutput:
        if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
            raise ValueError("boxes_xyxy must be [B,N,4]")
        if not torch.isfinite(boxes_xyxy).all():
            raise ValueError("boxes_xyxy must contain finite values")

        feature_map = self._select_feature(features)
        batch_size = feature_map.shape[0]
        if boxes_xyxy.shape[0] != batch_size:
            raise ValueError("boxes batch size must match features batch size")

        num_instances = boxes_xyxy.shape[1]
        prototypes = self.proto_pred(self.proto_trunk(feature_map))

        coeff_source = self.coeff_source(feature_map)
        if instance_embeddings is None:
            coeff_input = self._roi_mean_pool(
                coeff_source,
                boxes_xyxy,
                normalized_boxes=normalized_boxes,
                image_size=image_size,
            )
        else:
            if instance_embeddings.ndim != 3:
                raise ValueError("instance_embeddings must be [B,N,D]")
            if (
                instance_embeddings.shape[0] != batch_size
                or instance_embeddings.shape[1] != num_instances
            ):
                raise ValueError("instance_embeddings shape must align with [B,N]")
            if instance_embeddings.shape[2] != self.coeff_input_dim:
                raise ValueError("instance_embeddings last dim must match coeff_input_dim")
            coeff_input = instance_embeddings

        coefficients = self.coeff_mlp(coeff_input)
        mask_logits_lowres = assemble_mask_logits_from_prototypes(prototypes, coefficients)

        if output_size is None:
            if image_size is None:
                output_height = int(mask_logits_lowres.shape[-2])
                output_width = int(mask_logits_lowres.shape[-1])
            else:
                output_height, output_width = image_size
        else:
            output_height, output_width = output_size
        if output_height <= 0 or output_width <= 0:
            raise ValueError("output size must be positive")

        if output_height != int(mask_logits_lowres.shape[-2]) or output_width != int(
            mask_logits_lowres.shape[-1]
        ):
            mask_logits = f.interpolate(
                mask_logits_lowres,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            mask_logits = mask_logits_lowres

        if self.ff_refine_hook is not None:
            if ff_highres_features is None:
                raise ValueError("ff_highres_features must be provided when enable_ff_refine=True")
            if active_tile_indices is None:
                raise ValueError("active_tile_indices must be provided when enable_ff_refine=True")
            mask_logits = self.ff_refine_hook(
                mask_logits=mask_logits,
                ff_highres_features=ff_highres_features,
                active_tile_indices=active_tile_indices,
            )

        if crop_to_boxes and num_instances > 0:
            box_masks = rasterize_box_masks(
                boxes_xyxy,
                height=output_height,
                width=output_width,
                normalized_boxes=normalized_boxes,
                image_size=image_size,
            )
            fill_value = torch.full_like(mask_logits, self.mask_fill_value)
            mask_logits = torch.where(box_masks, mask_logits, fill_value)
        else:
            box_masks = torch.ones_like(mask_logits, dtype=torch.bool)

        masks = torch.sigmoid(mask_logits)
        masked_sum = (masks * box_masks.to(dtype=masks.dtype)).sum(dim=(2, 3))
        masked_area = box_masks.to(dtype=masks.dtype).sum(dim=(2, 3)).clamp(min=1.0)
        mask_scores = masked_sum / masked_area

        return InstanceSegOutput(
            prototypes=prototypes,
            coefficients=coefficients,
            mask_logits_lowres=mask_logits_lowres,
            mask_logits=mask_logits,
            masks=masks,
            mask_scores=mask_scores,
        )


__all__ = [
    "InstanceSegOutput",
    "FFTileRefinementHook",
    "PrototypeInstanceSegHead",
    "assemble_mask_logits_from_prototypes",
    "rasterize_box_masks",
]
