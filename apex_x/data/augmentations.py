"""Advanced data augmentation strategies for Apex-X.

This module provides robust implementations of Mosaic, MixUp and Copy-Paste
that work with the canonical ``TransformSample`` contract:

- ``image``: HWC numpy array
- ``boxes_xyxy``: [N, 4] float32
- ``class_ids``: [N] int64
- ``masks``: optional [N, H, W]
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from apex_x.data.transforms import TransformSample


def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _resize_image_and_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    src_h, src_w = image.shape[:2]
    if src_h == target_h and src_w == target_w:
        return image.copy(), boxes_xyxy.copy()

    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if boxes_xyxy.shape[0] == 0:
        return resized, boxes_xyxy.copy()

    scale_x = float(target_w) / float(max(src_w, 1))
    scale_y = float(target_h) / float(max(src_h, 1))
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return resized, boxes


class MosaicAugmentation:
    """Mosaic augmentation combining 4 images into a single canvas."""

    def __init__(
        self,
        dataset,
        output_size: int = 640,
        mosaic_prob: float = 0.5,
        min_offset: float = 0.3,
        max_offset: float = 0.7,
    ) -> None:
        self.dataset = dataset
        self.output_size = int(output_size)
        self.mosaic_prob = float(mosaic_prob)
        self.min_offset = float(min_offset)
        self.max_offset = float(max_offset)

    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() > self.mosaic_prob:
            return sample

        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        samples = [sample] + [self.dataset[i] for i in indices]

        center_x = int(random.uniform(self.min_offset, self.max_offset) * self.output_size)
        center_y = int(random.uniform(self.min_offset, self.max_offset) * self.output_size)

        mosaic = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        all_boxes: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []

        placements = [
            (0, 0, center_x, center_y),
            (center_x, 0, self.output_size, center_y),
            (0, center_y, center_x, self.output_size),
            (center_x, center_y, self.output_size, self.output_size),
        ]

        for s, (x1, y1, x2, y2) in zip(samples, placements, strict=True):
            quad_w = max(1, x2 - x1)
            quad_h = max(1, y2 - y1)
            img = _as_hwc_uint8(s.image)
            src_h, src_w = img.shape[:2]
            scale = min(float(quad_w) / float(max(src_w, 1)), float(quad_h) / float(max(src_h, 1)))
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            paste_x2 = min(x1 + new_w, x2)
            paste_y2 = min(y1 + new_h, y2)
            mosaic[y1:paste_y2, x1:paste_x2] = resized[: paste_y2 - y1, : paste_x2 - x1]

            if s.boxes_xyxy.shape[0] == 0:
                continue

            boxes = s.boxes_xyxy.copy()
            boxes[:, [0, 2]] *= float(new_w) / float(max(src_w, 1))
            boxes[:, [1, 3]] *= float(new_h) / float(max(src_h, 1))
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(self.output_size))
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(self.output_size))
            keep = ((boxes[:, 2] - boxes[:, 0]) > 1.0) & ((boxes[:, 3] - boxes[:, 1]) > 1.0)
            if np.any(keep):
                all_boxes.append(boxes[keep])
                all_classes.append(s.class_ids[keep])

        out_boxes = (
            np.concatenate(all_boxes, axis=0).astype(np.float32)
            if all_boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        out_classes = (
            np.concatenate(all_classes, axis=0).astype(np.int64)
            if all_classes
            else np.zeros((0,), dtype=np.int64)
        )
        return TransformSample(
            image=mosaic,
            boxes_xyxy=out_boxes,
            class_ids=out_classes,
            masks=None,
        )


class MixUpAugmentation:
    """MixUp augmentation by convexly blending two images."""

    def __init__(
        self,
        dataset,
        alpha: float = 0.5,
        mixup_prob: float = 0.15,
    ) -> None:
        self.dataset = dataset
        self.alpha = float(alpha)
        self.mixup_prob = float(mixup_prob)

    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() > self.mixup_prob:
            return sample

        other = self.dataset[random.randint(0, len(self.dataset) - 1)]
        img1 = _as_hwc_uint8(sample.image)
        img2 = _as_hwc_uint8(other.image)
        tgt_h, tgt_w = img1.shape[:2]
        img2_resized, boxes2_resized = _resize_image_and_boxes(img2, other.boxes_xyxy, tgt_h, tgt_w)

        lam = np.random.beta(self.alpha, self.alpha)
        mixed = (lam * img1.astype(np.float32) + (1.0 - lam) * img2_resized.astype(np.float32))
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        boxes = np.concatenate([sample.boxes_xyxy, boxes2_resized], axis=0).astype(np.float32)
        classes = np.concatenate([sample.class_ids, other.class_ids], axis=0).astype(np.int64)

        return TransformSample(
            image=mixed,
            boxes_xyxy=boxes,
            class_ids=classes,
            masks=None,
        )


class CopyPasteAugmentation:
    """Copy-Paste augmentation for instance segmentation."""

    def __init__(
        self,
        dataset,
        paste_prob: float = 0.5,
        max_paste: int = 10,
        scale_range: tuple[float, float] = (0.5, 2.0),
        blend_alpha: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.paste_prob = float(paste_prob)
        self.max_paste = int(max_paste)
        self.scale_range = scale_range
        self.blend_alpha = float(blend_alpha)

    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() > self.paste_prob:
            return sample
        if sample.masks is None:
            return sample

        img = _as_hwc_uint8(sample.image)
        h, w = img.shape[:2]
        boxes = sample.boxes_xyxy.copy()
        classes = sample.class_ids.copy()
        masks = sample.masks.copy()

        for _ in range(random.randint(1, max(self.max_paste, 1))):
            src = self.dataset[random.randint(0, len(self.dataset) - 1)]
            if src.masks is None or src.boxes_xyxy.shape[0] == 0:
                continue

            src_idx = random.randint(0, src.boxes_xyxy.shape[0] - 1)
            src_img = _as_hwc_uint8(src.image)
            src_mask = src.masks[src_idx].astype(np.uint8)
            x1, y1, x2, y2 = src.boxes_xyxy[src_idx].astype(np.int32).tolist()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(src_img.shape[1], x2)
            y2 = min(src_img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            obj_img = src_img[y1:y2, x1:x2]
            obj_mask = src_mask[y1:y2, x1:x2]
            if obj_img.size == 0 or obj_mask.sum() == 0:
                continue

            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_w = max(1, int(obj_img.shape[1] * scale))
            new_h = max(1, int(obj_img.shape[0] * scale))
            if new_w >= w or new_h >= h:
                continue

            obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)

            paste_x = random.randint(0, w - new_w)
            paste_y = random.randint(0, h - new_h)

            dst = img[paste_y : paste_y + new_h, paste_x : paste_x + new_w]
            m3 = np.repeat(obj_mask[:, :, None], 3, axis=2)
            if self.blend_alpha < 1.0:
                alpha = self.blend_alpha
                dst[m3] = (
                    alpha * obj_img[m3].astype(np.float32)
                    + (1.0 - alpha) * dst[m3].astype(np.float32)
                ).astype(np.uint8)
            else:
                dst[m3] = obj_img[m3]

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = obj_mask.astype(np.uint8)
            new_box = np.array([[paste_x, paste_y, paste_x + new_w, paste_y + new_h]], dtype=np.float32)

            boxes = np.concatenate([boxes, new_box], axis=0)
            cls_value = int(src.class_ids[min(src_idx, src.class_ids.shape[0] - 1)])
            classes = np.concatenate([classes, np.array([cls_value], dtype=np.int64)], axis=0)
            masks = np.concatenate([masks, full_mask[None, ...]], axis=0)

        return TransformSample(
            image=img,
            boxes_xyxy=boxes.astype(np.float32),
            class_ids=classes.astype(np.int64),
            masks=masks.astype(np.uint8),
        )


__all__ = [
    "MosaicAugmentation",
    "MixUpAugmentation",
    "CopyPasteAugmentation",
]

