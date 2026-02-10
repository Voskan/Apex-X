from __future__ import annotations

import numpy as np

from apex_x.data import (
    ClipBoxesAndMasks,
    MosaicV2,
    RandomHorizontalFlip,
    TransformPipeline,
    TransformSample,
    sanitize_sample,
)


def _make_sample(sample_id: int) -> TransformSample:
    height = 96
    width = 96
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :, 0] = float(sample_id + 1)

    # Important instance: large box near bottom-right so random crops often cut it.
    important = np.asarray([60.0, 60.0, 92.0, 92.0], dtype=np.float32)
    # Small instance: not protected by heuristic.
    small = np.asarray([8.0, 8.0, 18.0, 18.0], dtype=np.float32)
    boxes = np.stack([important, small], axis=0).astype(np.float32)
    class_ids = np.asarray([100 + sample_id, 200 + sample_id], dtype=np.int64)

    masks = np.zeros((2, height, width), dtype=bool)
    masks[0, 60:92, 60:92] = True
    masks[1, 8:18, 8:18] = True

    return TransformSample(
        image=image,
        boxes_xyxy=boxes,
        class_ids=class_ids,
        masks=masks,
    )


def _assert_valid(sample: TransformSample) -> None:
    h, w = sample.height, sample.width
    assert sample.boxes_xyxy.ndim == 2
    assert sample.boxes_xyxy.shape[1] == 4
    assert sample.class_ids.shape[0] == sample.boxes_xyxy.shape[0]
    assert np.all(sample.boxes_xyxy[:, 0] >= 0.0)
    assert np.all(sample.boxes_xyxy[:, 1] >= 0.0)
    assert np.all(sample.boxes_xyxy[:, 2] <= float(w))
    assert np.all(sample.boxes_xyxy[:, 3] <= float(h))
    assert np.all(sample.boxes_xyxy[:, 2] > sample.boxes_xyxy[:, 0])
    assert np.all(sample.boxes_xyxy[:, 3] > sample.boxes_xyxy[:, 1])
    if sample.masks is not None:
        assert sample.masks.shape == (sample.boxes_xyxy.shape[0], h, w)
        per_instance_pixels = sample.masks.reshape(sample.masks.shape[0], -1).sum(axis=1)
        assert np.all(per_instance_pixels > 0)


def test_transform_pipeline_keeps_bbox_and_mask_validity() -> None:
    sample = _make_sample(0)
    # Add one invalid box/mask to ensure clip/filter stage removes it.
    boxes = np.concatenate(
        [sample.boxes_xyxy, np.asarray([[95.0, 95.0, 95.2, 95.2]], dtype=np.float32)],
        axis=0,
    )
    classes = np.concatenate([sample.class_ids, np.asarray([999], dtype=np.int64)], axis=0)
    masks = np.concatenate(
        [sample.masks, np.zeros((1, sample.height, sample.width), dtype=bool)],
        axis=0,
    )
    dirty = TransformSample(
        image=sample.image,
        boxes_xyxy=boxes,
        class_ids=classes,
        masks=masks,
    )

    pipeline = TransformPipeline(
        transforms=(
            RandomHorizontalFlip(prob=1.0),
            ClipBoxesAndMasks(min_box_area=4.0, min_visibility=0.2),
        ),
    )
    out = pipeline(dirty, rng=np.random.RandomState(3))
    _assert_valid(out)
    assert out.boxes_xyxy.shape[0] == 2
    assert 999 not in set(int(v) for v in out.class_ids.tolist())


def test_mosaic_v2_outputs_valid_boxes_and_masks() -> None:
    samples = [_make_sample(i) for i in range(4)]
    mosaic = MosaicV2(
        output_height=128,
        output_width=128,
        split_jitter=0.0,
        protect_important_instances=True,
        min_box_area=4.0,
        min_visibility=0.1,
    )
    out = mosaic(samples, rng=np.random.RandomState(7))
    _assert_valid(out)
    assert out.height == 128
    assert out.width == 128
    assert out.boxes_xyxy.shape[0] >= 4  # at least important objects should survive


def _mean_important_visibility(
    *,
    protect_important_instances: bool,
    seeds: int = 40,
) -> float:
    samples = [_make_sample(i) for i in range(4)]
    mosaic = MosaicV2(
        output_height=128,
        output_width=128,
        split_jitter=0.0,
        protect_important_instances=protect_important_instances,
        min_box_area=4.0,
        min_visibility=0.0,
    )
    important_area = 32.0 * 32.0
    visibilities: list[float] = []

    for seed in range(seeds):
        out = mosaic(samples, rng=np.random.RandomState(seed))
        for sample_id in range(4):
            important_class = 100 + sample_id
            idx = np.nonzero(out.class_ids == important_class)[0]
            if idx.size == 0:
                visibilities.append(0.0)
                continue
            box = out.boxes_xyxy[int(idx[0])]
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            visibilities.append(area / important_area)
    return float(np.mean(np.asarray(visibilities, dtype=np.float64)))


def test_mosaic_v2_heuristic_reduces_important_instance_cutting() -> None:
    protected = _mean_important_visibility(protect_important_instances=True, seeds=40)
    unprotected = _mean_important_visibility(protect_important_instances=False, seeds=40)

    assert protected > 0.95
    assert unprotected < 0.80
    assert protected > unprotected + 0.20


def test_sanitize_sample_clips_out_of_bounds_boxes() -> None:
    sample = _make_sample(0)
    mutated = TransformSample(
        image=sample.image,
        boxes_xyxy=np.asarray([[-10.0, -5.0, 120.0, 99.0]], dtype=np.float32),
        class_ids=np.asarray([1], dtype=np.int64),
        masks=np.ones((1, sample.height, sample.width), dtype=bool),
    )
    out = sanitize_sample(mutated, min_box_area=1.0, min_visibility=0.0)
    _assert_valid(out)
    assert out.boxes_xyxy.shape[0] == 1
    assert out.boxes_xyxy[0, 0] == 0.0
    assert out.boxes_xyxy[0, 1] == 0.0
    assert out.boxes_xyxy[0, 2] == float(sample.width)
    assert out.boxes_xyxy[0, 3] == float(sample.height)
