from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

try:
    import albumentations as A  # type: ignore
except ImportError:
    A = None


@dataclass(frozen=True, slots=True)
class AlbumentationsAdapter:
    """Adapts an Albumentations pipeline to the Transform protocol."""

    pipeline: A.Compose

    def __call__(
        self,
        sample: TransformSample,
        *,
        rng: np.random.RandomState | None = None,
    ) -> TransformSample:
        if A is None:
            # warn or fail? For now, no-op if missing to allow running without deps
            return sample
        
        # Albumentations uses its own RNG, but we should try to seed it if possible
        # or just rely on its internal state. 
        # A.Compose doesn't easily take an external RNG state for single call.
        # We'll rely on global seeding or separate seeding if needed.
        
        # Prepare inputs
        kwargs = {"image": sample.image}
        
        # Boxes: [N, 4] -> [N, 4] (x1, y1, x2, y2)
        # We need to ensure they are valid for Albumentations (x2 > x1, y2 > y1)
        # The sanitizer should have handled this, but A is strict.
        
        if sample.boxes_xyxy.shape[0] > 0:
            kwargs["bboxes"] = sample.boxes_xyxy
            kwargs["class_labels"] = sample.class_ids
            
        if sample.masks is not None:
             # A supports list of masks for 'masks' arg, or single 'mask'. 
             # sample.masks is [N, H, W]. 
             # We can pass list of (H, W) masks.
             kwargs["masks"] = list(sample.masks)  # type: ignore

        # Run pipeline
        # verification: A.Compose must be created with bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
        try:
            res = self.pipeline(**kwargs)
        except ValueError:
            # Fallback for empty boxes sometimes causing issues in older versions
            return sample

        # Unpack outputs
        out_image = res["image"]
        
        out_boxes = np.zeros((0, 4), dtype=np.float32)
        out_classes = np.zeros((0,), dtype=np.int64)
        
        if "bboxes" in res and res["bboxes"]:
            out_boxes = np.array(res["bboxes"], dtype=np.float32)
            if "class_labels" in res:
                out_classes = np.array(res["class_labels"], dtype=np.int64)
            else:
                # Should not happen if configured correctly
                out_classes = np.zeros((len(out_boxes),), dtype=np.int64)
                
        out_masks = None
        if "masks" in res and res["masks"]:
            # List of (H, W) -> [N, H, W]
            out_masks = np.stack(res["masks"], axis=0)

        return TransformSample(
            image=out_image,
            boxes_xyxy=out_boxes,
            class_ids=out_classes,
            masks=out_masks,
        )


def build_robust_transforms(
    height: int,
    width: int,
    *,
    blur_prob: float = 0.5,
    noise_prob: float = 0.5,
    distort_prob: float = 0.5,
) -> Transform:
    """Builds a robust augmentation pipeline for satellite imagery."""
    if A is None:
        return TransformPipeline(transforms=())
        
    transforms = [
        A.RandomCrop(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    
    if blur_prob > 0:
        transforms.append(
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.GaussianBlur(p=1.0),
                A.Defocus(p=1.0),
            ], p=blur_prob)
        )
        
    if noise_prob > 0:
        transforms.append(
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
            ], p=noise_prob)
        )
        
    if distort_prob > 0:
        transforms.append(
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
                A.ImageCompression(quality_range=(50, 90), p=1.0),
            ], p=distort_prob)
        )
        
    pipeline = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.1),
    )
    return AlbumentationsAdapter(pipeline=pipeline)

@dataclass(frozen=True, slots=True)
class TransformSample:
    """Image + instance annotations used by augmentation transforms."""

    image: np.ndarray  # [H,W,3], float32/uint8
    boxes_xyxy: np.ndarray  # [N,4], float32
    class_ids: np.ndarray  # [N], int64
    masks: np.ndarray | None = None  # [N,H,W], bool

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


class Transform(Protocol):
    def __call__(
        self,
        sample: TransformSample,
        *,
        rng: np.random.RandomState | None = None,
    ) -> TransformSample: ...


def _validate_sample_shapes(sample: TransformSample) -> None:
    if sample.image.ndim != 3 or sample.image.shape[2] != 3:
        raise ValueError("sample.image must be [H,W,3]")
    if sample.boxes_xyxy.ndim != 2 or sample.boxes_xyxy.shape[1] != 4:
        raise ValueError("sample.boxes_xyxy must be [N,4]")
    if sample.class_ids.ndim != 1:
        raise ValueError("sample.class_ids must be [N]")
    if sample.class_ids.shape[0] != sample.boxes_xyxy.shape[0]:
        raise ValueError("sample.class_ids length must match box count")
    if sample.masks is not None:
        if sample.masks.ndim != 3:
            raise ValueError("sample.masks must be [N,H,W]")
        if sample.masks.shape[0] != sample.boxes_xyxy.shape[0]:
            raise ValueError("sample.masks N must match box count")
        if sample.masks.shape[1] != sample.height or sample.masks.shape[2] != sample.width:
            raise ValueError("sample.masks spatial shape must match image")


def _box_area_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
    widths = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
    heights = np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return cast(np.ndarray, widths * heights)


def sanitize_sample(
    sample: TransformSample,
    *,
    min_box_area: float = 1.0,
    min_visibility: float = 0.0,
    require_nonempty_mask: bool = True,
) -> TransformSample:
    """Clip and filter boxes/masks so outputs are valid and in-bounds."""
    if min_box_area < 0.0:
        raise ValueError("min_box_area must be >= 0")
    if not (0.0 <= min_visibility <= 1.0):
        raise ValueError("min_visibility must be in [0,1]")

    _validate_sample_shapes(sample)

    boxes = sample.boxes_xyxy.astype(np.float32, copy=True)
    class_ids = sample.class_ids.astype(np.int64, copy=True)
    masks = None if sample.masks is None else sample.masks.astype(bool, copy=True)
    orig_area = _box_area_xyxy(boxes)

    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(sample.width))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(sample.height))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(sample.width))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(sample.height))

    clipped_area = _box_area_xyxy(boxes)
    visibility = clipped_area / np.maximum(orig_area, 1e-6)
    keep = (clipped_area >= float(min_box_area)) & (visibility >= float(min_visibility))

    if masks is not None:
        mask_pixels = masks.reshape(masks.shape[0], -1).sum(axis=1)
        if require_nonempty_mask:
            keep &= mask_pixels > 0

    kept_boxes = boxes[keep]
    kept_classes = class_ids[keep]
    kept_masks = None if masks is None else masks[keep]

    return TransformSample(
        image=sample.image.copy(),
        boxes_xyxy=kept_boxes,
        class_ids=kept_classes,
        masks=kept_masks,
    )


@dataclass(frozen=True, slots=True)
class ClipBoxesAndMasks:
    min_box_area: float = 1.0
    min_visibility: float = 0.0

    def __call__(self, sample: TransformSample, *, rng: np.random.RandomState | None = None) -> TransformSample:
        del rng
        return sanitize_sample(
            sample,
            min_box_area=self.min_box_area,
            min_visibility=self.min_visibility,
            require_nonempty_mask=True,
        )


@dataclass(frozen=True, slots=True)
class RandomHorizontalFlip:
    prob: float = 0.5

    def __call__(self, sample: TransformSample, *, rng: np.random.RandomState | None = None) -> TransformSample:
        if not (0.0 <= self.prob <= 1.0):
            raise ValueError("prob must be in [0,1]")
        _validate_sample_shapes(sample)
        active_rng = np.random.RandomState() if rng is None else rng
        if float(active_rng.rand()) >= self.prob:
            return sample

        image = sample.image[:, ::-1, :].copy()
        width = float(sample.width)
        boxes = sample.boxes_xyxy.astype(np.float32, copy=True)
        old_x1 = boxes[:, 0].copy()
        old_x2 = boxes[:, 2].copy()
        boxes[:, 0] = width - old_x2
        boxes[:, 2] = width - old_x1
        masks = None if sample.masks is None else sample.masks[:, :, ::-1].copy()

        return TransformSample(
            image=image,
            boxes_xyxy=boxes,
            class_ids=sample.class_ids.copy(),
            masks=masks,
        )


@dataclass(slots=True)
class TransformPipeline:
    transforms: tuple[Transform, ...]

    def __call__(
        self,
        sample: TransformSample,
        *,
        rng: np.random.RandomState | None = None,
    ) -> TransformSample:
        _validate_sample_shapes(sample)
        active_rng = np.random.RandomState() if rng is None else rng
        out = sample
        for transform in self.transforms:
            out = transform(out, rng=active_rng)
            _validate_sample_shapes(out)
        return out


@dataclass(frozen=True, slots=True)
class MosaicV2:
    """4-image mosaic with crop heuristic that protects important instances."""

    output_height: int
    output_width: int
    split_jitter: float = 0.1
    important_area_frac: float = 0.03
    min_box_area: float = 1.0
    min_visibility: float = 0.15
    protect_important_instances: bool = True

    def _split_value(self, size: int, *, rng: np.random.RandomState) -> int:
        if size < 2:
            raise ValueError("output size must be >= 2 for mosaic")
        if not (0.0 <= self.split_jitter <= 0.45):
            raise ValueError("split_jitter must be in [0, 0.45]")
        jitter = float(rng.uniform(-self.split_jitter, self.split_jitter))
        split = int(round(float(size) * (0.5 + jitter)))
        split_min = max(1, int(math.floor(size * 0.25)))
        split_max = min(size - 1, int(math.ceil(size * 0.75)))
        return int(np.clip(split, split_min, split_max))

    def _important_indices(
        self,
        boxes_xyxy: np.ndarray,
        *,
        image_h: int,
        image_w: int,
    ) -> np.ndarray:
        if boxes_xyxy.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        areas = _box_area_xyxy(boxes_xyxy)
        area_threshold = float(image_h * image_w) * float(self.important_area_frac)
        important = np.nonzero(areas >= area_threshold)[0].astype(np.int64)
        if important.size > 0:
            return important
        return np.asarray([int(np.argmax(areas))], dtype=np.int64)

    def _choose_crop_origin(
        self,
        *,
        src_h: int,
        src_w: int,
        crop_h: int,
        crop_w: int,
        boxes_xyxy: np.ndarray,
        rng: np.random.RandomState,
    ) -> tuple[int, int]:
        max_y0 = max(0, src_h - crop_h)
        max_x0 = max(0, src_w - crop_w)
        if max_y0 == 0 and max_x0 == 0:
            return 0, 0

        if not self.protect_important_instances or boxes_xyxy.shape[0] == 0:
            y0 = int(rng.randint(0, max_y0 + 1)) if max_y0 > 0 else 0
            x0 = int(rng.randint(0, max_x0 + 1)) if max_x0 > 0 else 0
            return y0, x0

        important = self._important_indices(boxes_xyxy, image_h=src_h, image_w=src_w)
        chosen = int(important[int(rng.randint(0, max(important.size, 1)))])
        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[chosen]]

        min_x0 = int(max(0.0, math.floor(x2 - crop_w)))
        max_x0_allowed = int(min(float(max_x0), math.floor(x1)))
        min_y0 = int(max(0.0, math.floor(y2 - crop_h)))
        max_y0_allowed = int(min(float(max_y0), math.floor(y1)))

        if min_x0 <= max_x0_allowed:
            x0 = int(rng.randint(min_x0, max_x0_allowed + 1))
        else:
            center_x = 0.5 * (x1 + x2)
            x0 = int(np.clip(int(round(center_x - crop_w * 0.5)), 0, max_x0))

        if min_y0 <= max_y0_allowed:
            y0 = int(rng.randint(min_y0, max_y0_allowed + 1))
        else:
            center_y = 0.5 * (y1 + y2)
            y0 = int(np.clip(int(round(center_y - crop_h * 0.5)), 0, max_y0))
        return y0, x0

    def _crop_to_patch(
        self,
        sample: TransformSample,
        *,
        patch_h: int,
        patch_w: int,
        rng: np.random.RandomState,
    ) -> TransformSample:
        _validate_sample_shapes(sample)
        src_h, src_w = sample.height, sample.width
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("patch dimensions must be > 0")

        crop_h = min(src_h, patch_h)
        crop_w = min(src_w, patch_w)
        y0, x0 = self._choose_crop_origin(
            src_h=src_h,
            src_w=src_w,
            crop_h=crop_h,
            crop_w=crop_w,
            boxes_xyxy=sample.boxes_xyxy,
            rng=rng,
        )
        y1 = y0 + crop_h
        x1 = x0 + crop_w

        patch_image = np.zeros((patch_h, patch_w, 3), dtype=sample.image.dtype)
        patch_image[:crop_h, :crop_w] = sample.image[y0:y1, x0:x1]

        boxes = sample.boxes_xyxy.astype(np.float32, copy=True)
        boxes[:, 0::2] -= float(x0)
        boxes[:, 1::2] -= float(y0)
        boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(crop_w))
        boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(crop_w))
        boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(crop_h))
        boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(crop_h))

        masks = None
        if sample.masks is not None:
            masks = np.zeros((sample.masks.shape[0], patch_h, patch_w), dtype=bool)
            masks[:, :crop_h, :crop_w] = sample.masks[:, y0:y1, x0:x1]

        cropped = TransformSample(
            image=patch_image,
            boxes_xyxy=boxes,
            class_ids=sample.class_ids.copy(),
            masks=masks,
        )
        return sanitize_sample(
            cropped,
            min_box_area=self.min_box_area,
            min_visibility=self.min_visibility,
            require_nonempty_mask=True,
        )

    def __call__(
        self,
        samples: list[TransformSample] | tuple[TransformSample, ...],
        *,
        rng: np.random.RandomState | None = None,
    ) -> TransformSample:
        if len(samples) < 4:
            raise ValueError("MosaicV2 requires at least 4 samples")
        active_rng = np.random.RandomState() if rng is None else rng

        for sample in samples[:4]:
            _validate_sample_shapes(sample)

        split_y = self._split_value(self.output_height, rng=active_rng)
        split_x = self._split_value(self.output_width, rng=active_rng)
        quadrants = (
            (0, split_y, 0, split_x),
            (0, split_y, split_x, self.output_width),
            (split_y, self.output_height, 0, split_x),
            (split_y, self.output_height, split_x, self.output_width),
        )

        canvas = np.zeros(
            (self.output_height, self.output_width, 3),
            dtype=samples[0].image.dtype,
        )
        out_boxes: list[np.ndarray] = []
        out_classes: list[np.ndarray] = []
        out_masks: list[np.ndarray] = []
        has_masks = all(sample.masks is not None for sample in samples[:4])

        for sample, (qy0, qy1, qx0, qx1) in zip(samples[:4], quadrants, strict=True):
            qh = qy1 - qy0
            qw = qx1 - qx0
            cropped = self._crop_to_patch(sample, patch_h=qh, patch_w=qw, rng=active_rng)

            canvas[qy0:qy1, qx0:qx1] = cropped.image
            if cropped.boxes_xyxy.shape[0] == 0:
                continue

            shifted = cropped.boxes_xyxy.copy()
            shifted[:, 0::2] += float(qx0)
            shifted[:, 1::2] += float(qy0)
            out_boxes.append(shifted)
            out_classes.append(cropped.class_ids.copy())
            if has_masks and cropped.masks is not None:
                full_mask = np.zeros(
                    (cropped.masks.shape[0], self.output_height, self.output_width),
                    dtype=bool,
                )
                full_mask[:, qy0:qy1, qx0:qx1] = cropped.masks
                out_masks.append(full_mask)

        if out_boxes:
            boxes_xyxy = np.concatenate(out_boxes, axis=0).astype(np.float32, copy=False)
            class_ids = np.concatenate(out_classes, axis=0).astype(np.int64, copy=False)
            masks = (
                np.concatenate(out_masks, axis=0).astype(bool, copy=False)
                if has_masks and out_masks
                else None
            )
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.zeros((0,), dtype=np.int64)
            masks = (
                np.zeros((0, self.output_height, self.output_width), dtype=bool)
                if has_masks
                else None
            )

        mosaic = TransformSample(
            image=canvas,
            boxes_xyxy=boxes_xyxy,
            class_ids=class_ids,
            masks=masks,
        )
        return sanitize_sample(
            mosaic,
            min_box_area=self.min_box_area,
            min_visibility=self.min_visibility,
            require_nonempty_mask=True,
        )


__all__ = [
    "TransformSample",
    "Transform",
    "TransformPipeline",
    "ClipBoxesAndMasks",
    "RandomHorizontalFlip",
    "MosaicV2",
    "sanitize_sample",
]
