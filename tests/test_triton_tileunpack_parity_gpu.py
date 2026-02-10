from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.tileunpack import get_triton_tileunpack_availability, tileunpack_dispatch
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
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


def _meta_from_indices(
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


@pytest.mark.parametrize(
    ("shape", "tile_size", "kmax"),
    [
        ((1, 16, 32, 32), 4, 8),
        ((2, 32, 64, 64), 8, 12),
    ],
)
def test_triton_tileunpack_parity_fp16(
    shape: tuple[int, int, int, int], tile_size: int, kmax: int
) -> None:
    availability = get_triton_tileunpack_availability()
    if not availability.available:
        pytest.skip(f"Triton tileunpack unavailable: {availability.reason}")

    batch, channels, height, width = shape
    seed_all(123, deterministic=True)
    base = torch.randn(shape, dtype=torch.float16, device="cuda").contiguous()
    packed = torch.randn(
        (batch, kmax, channels, tile_size, tile_size),
        dtype=torch.float16,
        device="cuda",
    )
    max_idx = (height // tile_size) * (width // tile_size)
    idx = _unique_indices(batch, kmax, max_idx, seed=77).to(device="cuda")
    meta = _meta_from_indices(idx, tile_size, height, width, device=base.device)

    expected, _ = TileUnpackTorch().unpack(
        base_map=base,
        packed_out=packed,
        meta=meta,
        level_priority=1,
        overlap_mode="override",
    )
    dispatch = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert dispatch.backend == "triton"
    report = evaluate_parity_outputs(
        case_name="triton-tileunpack-fp16",
        reference_backend="tileunpack_torch",
        candidate_backend="triton",
        reference_output=expected,
        candidate_output=dispatch.merged,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True


def test_triton_tileunpack_parity_bf16_if_supported() -> None:
    availability = get_triton_tileunpack_availability()
    if not availability.available:
        pytest.skip(f"Triton tileunpack unavailable: {availability.reason}")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 is not supported on this device")

    seed_all(321, deterministic=True)
    shape = (1, 16, 64, 64)
    tile_size = 8
    kmax = 10
    batch, channels, height, width = shape
    base = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    packed = torch.randn(
        (batch, kmax, channels, tile_size, tile_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    max_idx = (height // tile_size) * (width // tile_size)
    idx = _unique_indices(batch, kmax, max_idx, seed=11).to(device="cuda")
    meta = _meta_from_indices(idx, tile_size, height, width, device=base.device)

    expected, _ = TileUnpackTorch().unpack(
        base_map=base,
        packed_out=packed,
        meta=meta,
        level_priority=1,
        overlap_mode="override",
    )
    dispatch = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert dispatch.backend == "triton"
    report = evaluate_parity_outputs(
        case_name="triton-tileunpack-bf16",
        reference_backend="tileunpack_torch",
        candidate_backend="triton",
        reference_output=expected,
        candidate_output=dispatch.merged,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True
