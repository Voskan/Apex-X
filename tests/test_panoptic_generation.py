from __future__ import annotations

import torch

from apex_x.infer import generate_panoptic_output


def test_panoptic_fusion_is_deterministic_with_overlap_resolution() -> None:
    semantic_logits = torch.full((1, 3, 6, 6), -4.0, dtype=torch.float32)
    semantic_logits[:, 2] = 4.0  # default stuff class

    instance_masks = torch.zeros((1, 2, 6, 6), dtype=torch.float32)
    # Lower-score instance on the left/middle.
    instance_masks[0, 0, 1:5, 1:4] = 1.0
    # Higher-score instance shifted right; overlaps with instance 0.
    instance_masks[0, 1, 1:5, 2:5] = 1.0

    instance_scores = torch.tensor([[0.6, 0.9]], dtype=torch.float32)
    instance_class_ids = torch.tensor([[1, 1]], dtype=torch.int64)

    out_a = generate_panoptic_output(
        semantic_logits,
        instance_masks,
        instance_scores,
        instance_class_ids,
        thing_class_ids={1},
        mask_threshold=0.5,
        score_threshold=0.05,
    )
    out_b = generate_panoptic_output(
        semantic_logits,
        instance_masks,
        instance_scores,
        instance_class_ids,
        thing_class_ids={1},
        mask_threshold=0.5,
        score_threshold=0.05,
    )

    assert torch.equal(out_a.panoptic_map, out_b.panoptic_map)
    assert torch.equal(out_a.semantic_labels, out_b.semantic_labels)
    assert out_a.segments_info == out_b.segments_info

    segments = out_a.segments_info[0]
    assert len(segments) == 3  # 2 things + 1 stuff
    assert segments[0].isthing and segments[0].instance_index == 1  # higher score first
    assert segments[1].isthing and segments[1].instance_index == 0
    assert not segments[2].isthing and segments[2].category_id == 2

    # Overlap pixel belongs to the higher-scored instance segment id.
    overlap_yx = (2, 2)
    assert int(out_a.panoptic_map[0, overlap_yx[0], overlap_yx[1]].item()) == segments[0].id


def test_panoptic_ignores_non_thing_instances_and_preserves_stuff() -> None:
    semantic_logits = torch.full((1, 3, 5, 6), -3.0, dtype=torch.float32)
    semantic_logits[:, 0, :, :3] = 3.0  # stuff class 0 on left half
    semantic_logits[:, 2, :, 3:] = 3.0  # stuff class 2 on right half

    instance_masks = torch.ones((1, 1, 5, 6), dtype=torch.float32)
    instance_scores = torch.tensor([[0.95]], dtype=torch.float32)
    instance_class_ids = torch.tensor([[2]], dtype=torch.int64)  # not a thing class here

    out = generate_panoptic_output(
        semantic_logits,
        instance_masks,
        instance_scores,
        instance_class_ids,
        thing_class_ids={1},
    )

    segments = out.segments_info[0]
    assert len(segments) == 2  # only two stuff segments
    assert all(not seg.isthing for seg in segments)
    category_ids = [seg.category_id for seg in segments]
    assert category_ids == [0, 2]


def test_panoptic_output_shapes_and_ids_on_synthetic_scene() -> None:
    semantic_logits = torch.full((2, 4, 8, 8), -2.0, dtype=torch.float32)
    semantic_logits[:, 0] = 2.0
    semantic_logits[0, 3, 2:6, 2:6] = 4.0
    semantic_logits[1, 2, 1:7, 1:7] = 4.0

    instance_masks = torch.zeros((2, 3, 4, 4), dtype=torch.float32)  # lower-res; will upsample
    instance_masks[0, 0, 1:3, 1:3] = 8.0
    instance_masks[1, 1, 0:2, 0:2] = 8.0
    instance_masks[1, 2, 2:4, 2:4] = 8.0

    instance_scores = torch.tensor(
        [
            [0.8, 0.1, 0.2],
            [0.7, 0.9, 0.85],
        ],
        dtype=torch.float32,
    )
    instance_class_ids = torch.tensor(
        [
            [1, 1, 2],
            [1, 1, 1],
        ],
        dtype=torch.int64,
    )

    out = generate_panoptic_output(
        semantic_logits,
        instance_masks,
        instance_scores,
        instance_class_ids,
        thing_class_ids={1},
        masks_are_logits=True,
        score_threshold=0.2,
        min_instance_area=2,
        min_stuff_area=2,
    )

    assert out.panoptic_map.shape == (2, 8, 8)
    assert out.panoptic_map.dtype == torch.int64
    assert out.semantic_labels.shape == (2, 8, 8)

    for batch_idx, infos in enumerate(out.segments_info):
        ids = [seg.id for seg in infos]
        assert len(ids) == len(set(ids))
        assert all(seg.area > 0 for seg in infos)
        for seg in infos:
            pixel_count = int((out.panoptic_map[batch_idx] == seg.id).sum().item())
            assert pixel_count == seg.area
