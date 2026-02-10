"""Data loading scaffolding for Apex-X."""

import numpy as np

from .coco import (
    CocoAnnotation,
    CocoBBox,
    CocoCategory,
    CocoCategoryMapping,
    CocoDataset,
    CocoImage,
    CocoPolygon,
    CocoRLE,
    CocoSegmentation,
    clear_coco_dataset_cache,
    load_coco_dataset,
    segmentation_to_mask,
)
from .transforms import (
    ClipBoxesAndMasks,
    MosaicV2,
    RandomHorizontalFlip,
    Transform,
    TransformPipeline,
    TransformSample,
    sanitize_sample,
)


def dummy_batch(height: int = 128, width: int = 128) -> np.ndarray:
    return np.zeros((1, 3, height, width), dtype=np.float32)


__all__ = [
    "dummy_batch",
    "CocoBBox",
    "CocoPolygon",
    "CocoRLE",
    "CocoSegmentation",
    "CocoCategory",
    "CocoImage",
    "CocoAnnotation",
    "CocoCategoryMapping",
    "CocoDataset",
    "segmentation_to_mask",
    "load_coco_dataset",
    "clear_coco_dataset_cache",
    "TransformSample",
    "Transform",
    "TransformPipeline",
    "ClipBoxesAndMasks",
    "RandomHorizontalFlip",
    "MosaicV2",
    "sanitize_sample",
]
