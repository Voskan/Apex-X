from __future__ import annotations

import torch

from apex_x.model import FFTileRefinementHook, PrototypeInstanceSegHead


def _tile_mask(height: int, width: int, *, tile_size: int, tile_id: int) -> torch.Tensor:
    grid_w = width // tile_size
    ty = tile_id // grid_w
    tx = tile_id % grid_w
    mask = torch.zeros((height, width), dtype=torch.bool)
    y = ty * tile_size
    x = tx * tile_size
    mask[y : y + tile_size, x : x + tile_size] = True
    return mask


def test_ff_tile_refinement_hook_updates_only_active_tiles() -> None:
    hook = FFTileRefinementHook(tile_size=4, strength_init=1.0).cpu()
    mask_logits = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    ff_highres = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    active = torch.tensor([[3]], dtype=torch.int64)

    refined = hook(mask_logits, ff_highres, active)
    delta = refined - mask_logits

    selected = _tile_mask(8, 8, tile_size=4, tile_id=3)
    selected_delta = delta[:, :, selected]
    non_selected_delta = delta[:, :, ~selected]

    assert torch.all(torch.abs(selected_delta) > 0.0)
    assert torch.allclose(non_selected_delta, torch.zeros_like(non_selected_delta))


def test_ff_tile_refinement_hook_empty_indices_is_noop() -> None:
    hook = FFTileRefinementHook(tile_size=4).cpu()
    mask_logits = torch.randn(1, 3, 8, 8)
    ff_highres = torch.randn(1, 6, 8, 8)
    empty = torch.empty((1, 0), dtype=torch.int64)

    refined = hook(mask_logits, ff_highres, empty)
    assert torch.allclose(refined, mask_logits)


def test_inst_seg_head_refinement_hook_changes_only_active_tiles() -> None:
    torch.manual_seed(9)
    head = PrototypeInstanceSegHead(
        in_channels=8,
        num_prototypes=4,
        coeff_hidden_dim=16,
        proto_layers=1,
        enable_ff_refine=True,
        ff_refine_tile_size=4,
        ff_refine_strength_init=1.0,
    ).cpu()
    features = torch.randn(1, 8, 8, 8)
    boxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]], dtype=torch.float32)
    ff_highres = torch.ones((1, 16, 8, 8), dtype=torch.float32)

    no_refine = head(
        features,
        boxes,
        image_size=(8, 8),
        output_size=(8, 8),
        crop_to_boxes=False,
        ff_highres_features=ff_highres,
        active_tile_indices=torch.empty((1, 0), dtype=torch.int64),
    )
    with_refine = head(
        features,
        boxes,
        image_size=(8, 8),
        output_size=(8, 8),
        crop_to_boxes=False,
        ff_highres_features=ff_highres,
        active_tile_indices=torch.tensor([[1]], dtype=torch.int64),
    )

    delta = with_refine.mask_logits - no_refine.mask_logits
    selected = _tile_mask(8, 8, tile_size=4, tile_id=1)
    selected_delta = delta[:, :, selected]
    non_selected_delta = delta[:, :, ~selected]

    assert torch.all(torch.abs(selected_delta) > 0.0)
    assert torch.allclose(non_selected_delta, torch.zeros_like(non_selected_delta), atol=1e-6)
