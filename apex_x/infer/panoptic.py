from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as f


@dataclass(frozen=True, slots=True)
class PanopticSegmentInfo:
    """Metadata for one panoptic segment."""

    id: int
    category_id: int
    isthing: bool
    area: int
    score: float | None = None
    instance_index: int | None = None


@dataclass(frozen=True, slots=True)
class PanopticOutput:
    """Panoptic output bundle for a batch."""

    panoptic_map: Tensor  # [B,H,W], segment ids (0 is void)
    segments_info: list[list[PanopticSegmentInfo]]  # per-image segment metadata
    semantic_labels: Tensor  # [B,H,W], argmax class ids


def _validate_panoptic_inputs(
    semantic_logits: Tensor,
    instance_masks: Tensor,
    instance_scores: Tensor,
    instance_class_ids: Tensor,
) -> None:
    if semantic_logits.ndim != 4:
        raise ValueError("semantic_logits must be [B,C,H,W]")
    if instance_masks.ndim != 4:
        raise ValueError("instance_masks must be [B,N,H,W]")
    if instance_scores.ndim != 2:
        raise ValueError("instance_scores must be [B,N]")
    if instance_class_ids.ndim != 2:
        raise ValueError("instance_class_ids must be [B,N]")

    if semantic_logits.shape[0] != instance_masks.shape[0]:
        raise ValueError("semantic_logits and instance_masks must share batch size")
    if instance_masks.shape[0] != instance_scores.shape[0]:
        raise ValueError("instance masks/scores must share batch size")
    if instance_masks.shape[0] != instance_class_ids.shape[0]:
        raise ValueError("instance masks/class ids must share batch size")
    if instance_masks.shape[1] != instance_scores.shape[1]:
        raise ValueError("instance_masks and instance_scores must share N")
    if instance_masks.shape[1] != instance_class_ids.shape[1]:
        raise ValueError("instance_masks and instance_class_ids must share N")

    if not torch.isfinite(semantic_logits).all():
        raise ValueError("semantic_logits must contain finite values")
    if not torch.isfinite(instance_masks).all():
        raise ValueError("instance_masks must contain finite values")
    if not torch.isfinite(instance_scores).all():
        raise ValueError("instance_scores must contain finite values")

    if instance_class_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("instance_class_ids must be an integer tensor")


def _normalize_thing_class_ids(
    thing_class_ids: Iterable[int],
    *,
    num_semantic_classes: int,
) -> set[int]:
    thing_ids = {int(class_id) for class_id in thing_class_ids}
    for class_id in thing_ids:
        if class_id < 0 or class_id >= num_semantic_classes:
            raise ValueError("thing_class_ids entries must be in [0, C_sem)")
    return thing_ids


def generate_panoptic_output(
    semantic_logits: Tensor,
    instance_masks: Tensor,
    instance_scores: Tensor,
    instance_class_ids: Tensor,
    *,
    thing_class_ids: Iterable[int],
    mask_threshold: float = 0.5,
    score_threshold: float = 0.05,
    min_instance_area: int = 1,
    min_stuff_area: int = 1,
    masks_are_logits: bool = False,
) -> PanopticOutput:
    """Fuse semantic logits and instance masks into deterministic panoptic output."""
    if not (0.0 <= mask_threshold <= 1.0):
        raise ValueError("mask_threshold must be in [0,1]")
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be in [0,1]")
    if min_instance_area <= 0:
        raise ValueError("min_instance_area must be > 0")
    if min_stuff_area <= 0:
        raise ValueError("min_stuff_area must be > 0")

    _validate_panoptic_inputs(
        semantic_logits=semantic_logits,
        instance_masks=instance_masks,
        instance_scores=instance_scores,
        instance_class_ids=instance_class_ids,
    )

    batch_size, num_semantic_classes, sem_h, sem_w = semantic_logits.shape
    _, num_instances, mask_h, mask_w = instance_masks.shape
    thing_ids = _normalize_thing_class_ids(
        thing_class_ids,
        num_semantic_classes=num_semantic_classes,
    )
    semantic_labels = torch.argmax(semantic_logits, dim=1).to(dtype=torch.int64)

    if (mask_h, mask_w) != (sem_h, sem_w):
        instance_masks = f.interpolate(
            instance_masks,
            size=(sem_h, sem_w),
            mode="bilinear",
            align_corners=False,
        )

    if masks_are_logits:
        mask_probs = torch.sigmoid(instance_masks)
    else:
        mask_probs = torch.nan_to_num(instance_masks, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0,
            1.0,
        )

    panoptic_map = torch.zeros(
        (batch_size, sem_h, sem_w),
        dtype=torch.int64,
        device=semantic_logits.device,
    )
    segments_info: list[list[PanopticSegmentInfo]] = []

    for batch_idx in range(batch_size):
        occupied = torch.zeros((sem_h, sem_w), dtype=torch.bool, device=semantic_logits.device)
        current_segments: list[PanopticSegmentInfo] = []
        next_segment_id = 1

        ranked_instances = sorted(
            range(num_instances),
            key=lambda idx: (-float(instance_scores[batch_idx, idx].item()), idx),
        )

        # Thing instances are fused first and override stuff deterministically.
        for inst_idx in ranked_instances:
            score = float(instance_scores[batch_idx, inst_idx].item())
            if score < score_threshold:
                continue

            class_id = int(instance_class_ids[batch_idx, inst_idx].item())
            if class_id not in thing_ids:
                continue

            mask = mask_probs[batch_idx, inst_idx] >= mask_threshold
            mask = mask & (~occupied)
            area = int(mask.sum().item())
            if area < min_instance_area:
                continue

            panoptic_map[batch_idx][mask] = next_segment_id
            occupied = occupied | mask
            current_segments.append(
                PanopticSegmentInfo(
                    id=next_segment_id,
                    category_id=class_id,
                    isthing=True,
                    area=area,
                    score=score,
                    instance_index=inst_idx,
                )
            )
            next_segment_id += 1

        # Remaining pixels are filled with semantic stuff in ascending class order.
        sem_pred = semantic_labels[batch_idx]
        for class_id in range(num_semantic_classes):
            if class_id in thing_ids:
                continue
            stuff_mask = (sem_pred == class_id) & (~occupied)
            area = int(stuff_mask.sum().item())
            if area < min_stuff_area:
                continue
            panoptic_map[batch_idx][stuff_mask] = next_segment_id
            occupied = occupied | stuff_mask
            current_segments.append(
                PanopticSegmentInfo(
                    id=next_segment_id,
                    category_id=class_id,
                    isthing=False,
                    area=area,
                    score=None,
                    instance_index=None,
                )
            )
            next_segment_id += 1

        segments_info.append(current_segments)

    return PanopticOutput(
        panoptic_map=panoptic_map,
        segments_info=segments_info,
        semantic_labels=semantic_labels,
    )


__all__ = [
    "PanopticSegmentInfo",
    "PanopticOutput",
    "generate_panoptic_output",
]
