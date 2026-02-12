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
from .coco_dataset import PYCOCOTOOLS_AVAILABLE, CocoDetectionDataset, coco_collate_fn
from .satellite import SatelliteDataset, SatelliteTile
from .transforms import (
    AlbumentationsAdapter,
    ClipBoxesAndMasks,
    MosaicV2,
    RandomHorizontalFlip,
    Transform,
    TransformPipeline,
    TransformSample,
    build_robust_transforms,
    sanitize_sample,
)
from .augmentations import (
    MosaicAugmentation,
    MixUpAugmentation,
    CopyPasteAugmentation,
)
from .lsj_augmentation import LargeScaleJitter



from .yolo import YOLOSegmentationDataset, yolo_collate_fn


def dummy_batch(height: int = 128, width: int = 128) -> np.ndarray:
    return np.zeros((1, 3, height, width), dtype=np.float32)


__all__ = [
    "CocoPolygon",
    "CocoRLE",
    "CocoSegmentation",
    "CocoBBox",
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
    "SatelliteDataset",
    "SatelliteTile",
    "AlbumentationsAdapter",
    "build_robust_transforms",
    "sanitize_sample",
    "CocoDetectionDataset",
    "coco_collate_fn",
    "PYCOCOTOOLS_AVAILABLE",
    "MosaicAugmentation",
    "MixUpAugmentation",
    "CopyPasteAugmentation",
    "LargeScaleJitter",
    "YOLOSegmentationDataset",
    "yolo_collate_fn",
]

