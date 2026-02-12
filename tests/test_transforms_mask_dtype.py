from __future__ import annotations

import numpy as np

from apex_x.data.transforms import TransformSample, build_robust_transforms


def test_robust_transforms_accept_bool_masks() -> None:
    image = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    boxes = np.array([[4.0, 4.0, 20.0, 20.0]], dtype=np.float32)
    class_ids = np.array([1], dtype=np.int64)
    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 5:18, 5:18] = True
    sample = TransformSample(
        image=image,
        boxes_xyxy=boxes,
        class_ids=class_ids,
        masks=masks,
    )

    transforms = build_robust_transforms(
        height=32,
        width=32,
        blur_prob=0.0,
        noise_prob=0.0,
        distort_prob=0.0,
    )
    out = transforms(sample, rng=np.random.RandomState(0))

    assert out.image.shape == (32, 32, 3)
    if out.masks is not None:
        assert out.masks.dtype == np.bool_
