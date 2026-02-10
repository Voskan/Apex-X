from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as f


@dataclass(frozen=True)
class TrackEmbeddingOutput:
    """Track embedding head outputs."""

    embeddings: Tensor  # [B,N,D], L2-normalized
    raw_embeddings: Tensor  # [B,N,D], pre-normalization
    pooled_features: Tensor  # [B,N,C]


def _normalize_xyxy(boxes_xyxy: Tensor) -> Tensor:
    x1 = boxes_xyxy[..., 0].minimum(boxes_xyxy[..., 2])
    y1 = boxes_xyxy[..., 1].minimum(boxes_xyxy[..., 3])
    x2 = boxes_xyxy[..., 0].maximum(boxes_xyxy[..., 2])
    y2 = boxes_xyxy[..., 1].maximum(boxes_xyxy[..., 3])
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _project_boxes_to_feature(
    boxes_xyxy: Tensor,
    *,
    target_h: int,
    target_w: int,
    normalized_boxes: bool,
    image_size: tuple[int, int] | None,
) -> Tensor:
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_h and target_w must be > 0")
    boxes = _normalize_xyxy(boxes_xyxy)

    if normalized_boxes:
        out = boxes.clone()
        out[..., 0] = out[..., 0] * float(target_w)
        out[..., 2] = out[..., 2] * float(target_w)
        out[..., 1] = out[..., 1] * float(target_h)
        out[..., 3] = out[..., 3] * float(target_h)
        return out

    if image_size is None:
        return boxes

    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_size must contain positive dimensions")

    scale_x = float(target_w) / float(image_w)
    scale_y = float(target_h) / float(image_h)
    out = boxes.clone()
    out[..., 0] = out[..., 0] * scale_x
    out[..., 2] = out[..., 2] * scale_x
    out[..., 1] = out[..., 1] * scale_y
    out[..., 3] = out[..., 3] * scale_y
    return out


class TrackEmbeddingHead(nn.Module):
    """ROI pooled tracking embedding head."""

    def __init__(
        self,
        in_channels: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        proj_channels: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        proj_c = int(in_channels if proj_channels is None else proj_channels)
        if proj_c <= 0:
            raise ValueError("proj_channels must be > 0")

        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)
        self.proj_channels = proj_c

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.proj_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.proj_channels),
            nn.SiLU(inplace=False),
        )
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.proj_channels, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

    def _select_feature(self, features: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(features, Tensor):
            if features.ndim != 4:
                raise ValueError("features tensor must be [B,C,H,W]")
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
            raise ValueError("selected feature must be [B,C,H,W]")
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
        if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
            raise ValueError("boxes_xyxy must be [B,N,4]")
        if not source.isfinite().all():
            raise ValueError("source features must be finite")
        if not boxes_xyxy.isfinite().all():
            raise ValueError("boxes_xyxy must be finite")

        bsz, channels, height, width = source.shape
        if boxes_xyxy.shape[0] != bsz:
            raise ValueError("boxes batch size must match feature batch size")
        num_instances = boxes_xyxy.shape[1]
        if num_instances == 0:
            return source.new_zeros((bsz, 0, channels))

        projected = _project_boxes_to_feature(
            boxes_xyxy,
            target_h=height,
            target_w=width,
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
                x1 = int(box[0].floor().item())
                y1 = int(box[1].floor().item())
                x2 = int(box[2].ceil().item())
                y2 = int(box[3].ceil().item())
                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                pooled[b, n] = source[b, :, y1:y2, x1:x2].mean(dim=(1, 2))
        return pooled

    def forward(
        self,
        features: Tensor | dict[str, Tensor],
        boxes_xyxy: Tensor,
        *,
        image_size: tuple[int, int] | None = None,
        normalized_boxes: bool = False,
    ) -> TrackEmbeddingOutput:
        feature_map = self._select_feature(features)
        projected = self.feature_proj(feature_map)
        pooled = self._roi_mean_pool(
            projected,
            boxes_xyxy,
            normalized_boxes=normalized_boxes,
            image_size=image_size,
        )

        raw = self.embedding_mlp(pooled)
        emb = f.normalize(raw, p=2.0, dim=-1, eps=1e-6)
        return TrackEmbeddingOutput(
            embeddings=emb,
            raw_embeddings=raw,
            pooled_features=pooled,
        )


__all__ = ["TrackEmbeddingHead", "TrackEmbeddingOutput"]
