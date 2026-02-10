from __future__ import annotations

import torch

from apex_x.model import (
    PrototypeInstanceSegHead,
    assemble_mask_logits_from_prototypes,
    rasterize_box_masks,
)


def test_assemble_mask_logits_from_prototypes_matches_expected() -> None:
    prototypes = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ]
        ],
        dtype=torch.float32,
    )  # [1,2,2,2]
    coefficients = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]], dtype=torch.float32)  # [1,2,2]

    logits = assemble_mask_logits_from_prototypes(prototypes, coefficients)

    assert logits.shape == (1, 2, 2, 2)
    assert torch.allclose(logits[0, 0], prototypes[0, 0])
    expected_second = 0.5 * prototypes[0, 0] + 0.5 * prototypes[0, 1]
    assert torch.allclose(logits[0, 1], expected_second)


def test_prototype_instance_seg_head_forward_shapes_and_ranges() -> None:
    torch.manual_seed(11)
    head = PrototypeInstanceSegHead(
        in_channels=64,
        num_prototypes=16,
        coeff_hidden_dim=64,
    ).cpu()

    features = torch.randn(2, 64, 16, 16)
    boxes = torch.tensor(
        [
            [[8.0, 8.0, 64.0, 64.0], [20.0, 24.0, 100.0, 108.0], [0.0, 0.0, 32.0, 32.0]],
            [[16.0, 16.0, 80.0, 96.0], [48.0, 8.0, 120.0, 56.0], [40.0, 40.0, 44.0, 44.0]],
        ],
        dtype=torch.float32,
    )

    out = head(
        features,
        boxes,
        image_size=(128, 128),
        output_size=(128, 128),
        normalized_boxes=False,
        crop_to_boxes=True,
    )

    assert out.prototypes.shape == (2, 16, 16, 16)
    assert out.coefficients.shape == (2, 3, 16)
    assert out.mask_logits_lowres.shape == (2, 3, 16, 16)
    assert out.mask_logits.shape == (2, 3, 128, 128)
    assert out.masks.shape == (2, 3, 128, 128)
    assert out.mask_scores.shape == (2, 3)
    assert torch.isfinite(out.mask_logits).all()
    assert torch.isfinite(out.masks).all()
    assert torch.isfinite(out.mask_scores).all()
    assert float(out.masks.min().item()) >= 0.0
    assert float(out.masks.max().item()) <= 1.0


def test_prototype_instance_seg_head_is_deterministic_for_same_weights_and_input() -> None:
    torch.manual_seed(123)
    head = PrototypeInstanceSegHead(
        in_channels=32,
        num_prototypes=8,
        coeff_hidden_dim=32,
    ).cpu()
    features = torch.randn(1, 32, 12, 12)
    boxes = torch.tensor([[[1.0, 2.0, 6.0, 8.0], [2.0, 2.0, 10.0, 10.0]]], dtype=torch.float32)

    first = head(features, boxes, output_size=(24, 24), image_size=(24, 24))
    second = head(features, boxes, output_size=(24, 24), image_size=(24, 24))

    assert torch.allclose(first.prototypes, second.prototypes)
    assert torch.allclose(first.coefficients, second.coefficients)
    assert torch.allclose(first.mask_logits, second.mask_logits)
    assert torch.allclose(first.masks, second.masks)
    assert torch.allclose(first.mask_scores, second.mask_scores)


def test_prototype_instance_seg_head_gradient_flow_with_embeddings() -> None:
    torch.manual_seed(7)
    head = PrototypeInstanceSegHead(
        in_channels=48,
        coeff_input_dim=32,
        num_prototypes=12,
        coeff_hidden_dim=64,
    ).cpu()

    features = torch.randn(2, 48, 10, 10, requires_grad=True)
    boxes_norm = torch.tensor(
        [
            [[0.1, 0.1, 0.8, 0.8], [0.2, 0.3, 0.9, 0.9]],
            [[0.0, 0.0, 0.4, 0.5], [0.5, 0.5, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    embeddings = torch.randn(2, 2, 32, requires_grad=True)

    out = head(
        features,
        boxes_norm,
        instance_embeddings=embeddings,
        output_size=(40, 40),
        normalized_boxes=True,
        crop_to_boxes=True,
    )
    loss = out.mask_logits.square().mean() + out.mask_scores.mean()
    loss.backward()

    assert features.grad is not None
    assert embeddings.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(embeddings.grad).all()
    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in head.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0.0


def test_crop_to_boxes_applies_fill_value_outside_boxes() -> None:
    torch.manual_seed(13)
    fill_value = -30.0
    head = PrototypeInstanceSegHead(
        in_channels=16,
        num_prototypes=4,
        coeff_hidden_dim=16,
        mask_fill_value=fill_value,
    ).cpu()
    features = torch.randn(1, 16, 8, 8)
    boxes = torch.tensor([[[2.0, 2.0, 6.0, 6.0]]], dtype=torch.float32)

    out = head(
        features,
        boxes,
        image_size=(8, 8),
        output_size=(8, 8),
        normalized_boxes=False,
        crop_to_boxes=True,
    )
    box_mask = rasterize_box_masks(boxes, height=8, width=8, image_size=(8, 8))
    outside_logits = torch.masked_select(out.mask_logits, ~box_mask)
    assert outside_logits.numel() > 0
    assert torch.allclose(outside_logits, torch.full_like(outside_logits, fill_value))
