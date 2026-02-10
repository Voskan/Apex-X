from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex_x.data import clear_coco_dataset_cache, load_coco_dataset, segmentation_to_mask

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_FIXTURE = FIXTURES_DIR / "coco_valid_mixed.json"
INVALID_MISSING_TOP_KEYS_FIXTURE = FIXTURES_DIR / "coco_invalid_missing_top_keys.json"
INVALID_BAD_RLE_FIXTURE = FIXTURES_DIR / "coco_invalid_bad_rle.json"


def test_coco_loader_parses_valid_fixture_strict() -> None:
    clear_coco_dataset_cache()
    dataset = load_coco_dataset(VALID_FIXTURE, strict=True, use_cache=False)

    assert dataset.image_count == 1
    assert dataset.category_count == 2
    assert dataset.annotation_count == 2

    ann_by_id = {ann.annotation_id: ann for ann in dataset.annotations}
    assert ann_by_id[10].bbox.width == pytest.approx(2.0)
    assert ann_by_id[10].bbox.height == pytest.approx(2.0)
    assert ann_by_id[10].segmentation is not None
    assert ann_by_id[10].segmentation.kind == "polygon"

    assert ann_by_id[11].segmentation is not None
    assert ann_by_id[11].segmentation.kind == "rle"


def test_coco_category_mapping_and_loader_cache() -> None:
    clear_coco_dataset_cache()
    dataset_cached_first = load_coco_dataset(VALID_FIXTURE, strict=True, use_cache=True)
    dataset_cached_second = load_coco_dataset(VALID_FIXTURE, strict=True, use_cache=True)
    dataset_uncached = load_coco_dataset(VALID_FIXTURE, strict=True, use_cache=False)

    assert dataset_cached_first is dataset_cached_second
    assert dataset_uncached is not dataset_cached_first

    mapping_first = dataset_cached_first.category_mapping()
    mapping_second = dataset_cached_first.category_mapping()
    assert mapping_first is mapping_second
    assert mapping_first.original_to_contiguous == {3: 0, 7: 1}
    assert mapping_first.contiguous_to_original == {0: 3, 1: 7}
    assert mapping_first.names_by_original[3] == "cat"
    assert mapping_first.names_by_contiguous[1] == "dog"


def test_polygon_and_rle_segmentation_to_mask() -> None:
    dataset = load_coco_dataset(VALID_FIXTURE, strict=True, use_cache=False)
    image = dataset.images_by_id[1]
    ann_by_id = {ann.annotation_id: ann for ann in dataset.annotations}

    polygon_seg = ann_by_id[10].segmentation
    rle_seg = ann_by_id[11].segmentation
    assert polygon_seg is not None
    assert rle_seg is not None

    polygon_mask = segmentation_to_mask(
        polygon_seg,
        image_height=image.height,
        image_width=image.width,
    )
    rle_mask = segmentation_to_mask(
        rle_seg,
        image_height=image.height,
        image_width=image.width,
    )

    assert polygon_mask.shape == (4, 5)
    assert rle_mask.shape == (4, 5)
    assert polygon_mask.dtype == np.bool_
    assert rle_mask.dtype == np.bool_
    assert int(polygon_mask.sum()) >= 4
    assert int(rle_mask.sum()) == 4


def test_strict_schema_errors_from_fixtures() -> None:
    with pytest.raises(ValueError, match="missing required top-level keys"):
        load_coco_dataset(INVALID_MISSING_TOP_KEYS_FIXTURE, strict=True, use_cache=False)

    with pytest.raises(ValueError, match="RLE counts do not cover full mask size"):
        load_coco_dataset(INVALID_BAD_RLE_FIXTURE, strict=True, use_cache=False)
