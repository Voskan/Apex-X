from __future__ import annotations

import torch

from apex_x.kernels.triton.tileunpack import tileunpack_dispatch, tileunpack_reference
from apex_x.runtime import evaluate_parity_outputs
from apex_x.utils.repro import seed_all


def _overlap_meta(device: torch.device) -> dict[str, torch.Tensor]:
    origins = torch.tensor([[[0, 0], [2, 2]]], dtype=torch.int64, device=device)
    indices = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
    return {
        "indices": indices,
        "origins": origins,
        "tile_size": torch.tensor(4, dtype=torch.int64, device=device),
        "grid": torch.tensor([2, 2], dtype=torch.int64, device=device),
    }


def test_overlap_reference_priority_overwrite_with_levels() -> None:
    base = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    packed = torch.zeros((1, 2, 1, 4, 4), dtype=torch.float32)
    packed[0, 0] = 1.0
    packed[0, 1] = 9.0
    levels = torch.tensor([[0, 2]], dtype=torch.int32)
    meta = _overlap_meta(base.device)

    out = tileunpack_reference(
        base,
        packed,
        meta=meta,
        levels=levels,
        overlap_mode="override",
    )

    expected = torch.zeros_like(base)
    expected[:, :, 0:4, 0:4] = 1.0
    expected[:, :, 2:6, 2:6] = 9.0
    assert torch.equal(out, expected)


def test_overlap_reference_presorted_order_without_levels() -> None:
    base = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    packed = torch.zeros((1, 2, 1, 4, 4), dtype=torch.float32)
    packed[0, 0] = 2.0
    packed[0, 1] = 7.0
    meta = _overlap_meta(base.device)

    out = tileunpack_reference(
        base,
        packed,
        meta=meta,
        levels=None,
        assume_priority_sorted=True,
        overlap_mode="override",
    )
    # Later tile in K-order wins on overlap.
    assert float(out[0, 0, 2, 2].item()) == 7.0
    assert float(out[0, 0, 1, 1].item()) == 2.0


def test_overlap_dispatch_parity_with_reference_override() -> None:
    seed_all(4, deterministic=True)
    base = torch.randn((1, 4, 32, 32), dtype=torch.float32)
    packed = torch.randn((1, 5, 4, 8, 8), dtype=torch.float32)
    origins = torch.tensor(
        [[[0, 0], [4, 4], [8, 8], [12, 8], [16, 12]]],
        dtype=torch.int64,
        device=base.device,
    )
    meta = {
        "indices": torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64, device=base.device),
        "origins": origins,
        "tile_size": torch.tensor(8, dtype=torch.int64, device=base.device),
        "grid": torch.tensor([4, 4], dtype=torch.int64, device=base.device),
    }
    levels = torch.tensor([[0, 1, 1, 2, 3]], dtype=torch.int32, device=base.device)

    expected = tileunpack_reference(
        base,
        packed,
        meta=meta,
        levels=levels,
        overlap_mode="override",
    )
    out = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        overlap_mode="override",
        prefer_triton=True,
        allow_fallback=True,
    )
    report = evaluate_parity_outputs(
        case_name="tileunpack-overlap-dispatch",
        reference_backend="tileunpack_reference",
        candidate_backend="tileunpack_dispatch",
        reference_output=expected,
        candidate_output=out.merged,
    )
    assert report.passed is True


def test_overlap_blend_mode_dispatch_uses_reference_path() -> None:
    base = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    packed = torch.zeros((1, 2, 1, 4, 4), dtype=torch.float32)
    packed[0, 0] = 1.0
    packed[0, 1] = 3.0
    meta = _overlap_meta(base.device)
    levels = torch.tensor([[1, 2]], dtype=torch.int32)

    expected = tileunpack_reference(
        base,
        packed,
        meta=meta,
        levels=levels,
        overlap_mode="blend",
        blend_alpha=0.25,
    )
    out = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        overlap_mode="blend",
        blend_alpha=0.25,
        prefer_triton=True,
        allow_fallback=True,
    )
    assert out.backend == "reference"
    assert out.fallback_reason == "blend_mode_uses_reference_path"
    assert torch.allclose(out.merged, expected)

