from __future__ import annotations

import torch

from apex_x.kernels.triton.tileunpack import (
    get_triton_tileunpack_availability,
    tileunpack_dispatch,
    tileunpack_reference,
)
from apex_x.runtime import evaluate_parity_outputs
from apex_x.tiles import TileUnpackTorch
from apex_x.utils.repro import seed_all


def _unique_indices(batch: int, kmax: int, max_idx: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rows: list[torch.Tensor] = []
    for b in range(batch):
        row = torch.randperm(max_idx, generator=gen, dtype=torch.int64)[:kmax]
        rows.append(torch.sort(row).values)
        gen.manual_seed(seed + b + 1)
    return torch.stack(rows, dim=0).to(dtype=torch.int32).contiguous()


def _make_meta_from_indices(
    indices: torch.Tensor,
    tile_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    grid_h = height // tile_size
    grid_w = width // tile_size
    idx_i64 = indices.to(dtype=torch.int64, device=device)
    origins = torch.stack(
        ((idx_i64 // grid_w) * tile_size, (idx_i64 % grid_w) * tile_size),
        dim=-1,
    )
    return {
        "indices": idx_i64,
        "origins": origins,
        "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=device),
        "grid": torch.tensor([grid_h, grid_w], dtype=torch.int64, device=device),
    }


def test_tileunpack_dispatch_cpu_fallback_parity_no_overlap() -> None:
    seed_all(10, deterministic=True)
    base = torch.randn((2, 8, 32, 32), dtype=torch.float32)
    tile_size = 4
    kmax = 10
    max_idx = (base.shape[2] // tile_size) * (base.shape[3] // tile_size)
    idx = _unique_indices(batch=2, kmax=kmax, max_idx=max_idx, seed=3)
    packed = torch.randn((2, kmax, 8, tile_size, tile_size), dtype=torch.float32)
    meta = _make_meta_from_indices(idx, tile_size, base.shape[2], base.shape[3], device=base.device)

    dispatch = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        prefer_triton=True,
        allow_fallback=True,
    )
    expected, _ = TileUnpackTorch().unpack(
        base_map=base,
        packed_out=packed,
        meta=meta,
        level_priority=1,
        overlap_mode="override",
    )
    assert dispatch.backend == "reference"
    report = evaluate_parity_outputs(
        case_name="tileunpack-cpu-fallback",
        reference_backend="tileunpack_torch",
        candidate_backend="tileunpack_dispatch",
        reference_output=expected,
        candidate_output=dispatch.merged,
    )
    assert report.passed is True


def test_tileunpack_reference_matches_torch_unpack_multiple_shapes() -> None:
    shapes = [
        (1, 4, 16, 16, 4, 4),
        (2, 12, 48, 48, 8, 6),
        (1, 16, 64, 64, 8, 9),
    ]
    for case_idx, (batch, channels, height, width, tile_size, kmax) in enumerate(shapes):
        seed_all(100 + case_idx, deterministic=True)
        base = torch.randn((batch, channels, height, width), dtype=torch.float32)
        packed = torch.randn((batch, kmax, channels, tile_size, tile_size), dtype=torch.float32)
        max_idx = (height // tile_size) * (width // tile_size)
        idx = _unique_indices(batch=batch, kmax=kmax, max_idx=max_idx, seed=200 + case_idx)
        meta = _make_meta_from_indices(idx, tile_size, height, width, device=base.device)

        ref = tileunpack_reference(base, packed, meta=meta)
        expected, _ = TileUnpackTorch().unpack(
            base_map=base,
            packed_out=packed,
            meta=meta,
            level_priority=1,
            overlap_mode="override",
        )
        report = evaluate_parity_outputs(
            case_name=f"tileunpack-reference-{case_idx}",
            reference_backend="tileunpack_torch",
            candidate_backend="tileunpack_reference",
            reference_output=expected,
            candidate_output=ref,
        )
        assert report.passed is True


def test_tileunpack_dispatch_gradient_fallback_when_autograd_requested() -> None:
    seed_all(8, deterministic=True)
    base = torch.randn((1, 2, 16, 16), dtype=torch.float32, requires_grad=True)
    tile_size = 4
    kmax = 3
    max_idx = (base.shape[2] // tile_size) * (base.shape[3] // tile_size)
    idx = _unique_indices(batch=1, kmax=kmax, max_idx=max_idx, seed=19)
    packed = torch.randn(
        (1, kmax, 2, tile_size, tile_size),
        dtype=torch.float32,
        requires_grad=True,
    )
    meta = _make_meta_from_indices(idx, tile_size, base.shape[2], base.shape[3], device=base.device)

    dispatch = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        prefer_triton=True,
        allow_fallback=True,
        inference_only=True,
    )
    loss = dispatch.merged.sum()
    loss.backward()
    assert dispatch.backend == "reference"
    assert dispatch.fallback_reason == "autograd_not_supported_for_triton_tileunpack"
    assert base.grad is not None
    assert packed.grad is not None
    assert torch.isfinite(base.grad).all()
    assert torch.isfinite(packed.grad).all()


def test_tileunpack_availability_object_cpu_safe() -> None:
    availability = get_triton_tileunpack_availability()
    assert isinstance(availability.available, bool)
    if not availability.available:
        assert availability.reason is not None
