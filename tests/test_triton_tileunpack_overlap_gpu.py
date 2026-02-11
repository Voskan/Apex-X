from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.tileunpack import get_triton_tileunpack_availability, tileunpack_dispatch
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
from apex_x.utils.repro import seed_all


def _build_overlap_meta(
    *,
    batch: int,
    kmax: int,
    height: int,
    width: int,
    tile_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    max_idx = (height // tile_size) * (width // tile_size)
    idx_cpu = torch.stack(
        [torch.randperm(max_idx, dtype=torch.int64)[:kmax] for _ in range(batch)],
        dim=0,
    )
    idx_cpu = torch.sort(idx_cpu, dim=1).values
    idx = idx_cpu.to(device=device, dtype=torch.int64)

    grid_w = width // tile_size
    origins_y = (idx // grid_w) * tile_size
    origins_x = (idx % grid_w) * tile_size

    shift = max(1, tile_size // 2)
    k_offsets = torch.arange(kmax, device=device).view(1, kmax)
    shift_mask = (k_offsets % 2) == 1
    origins_y = torch.where(
        shift_mask,
        torch.clamp(origins_y - shift, min=0, max=height - tile_size),
        origins_y,
    )
    origins_x = torch.where(
        shift_mask,
        torch.clamp(origins_x - shift, min=0, max=width - tile_size),
        origins_x,
    )
    origins = torch.stack((origins_y, origins_x), dim=-1)

    levels = torch.randint(0, 4, (batch, kmax), dtype=torch.int32, device=device)
    meta = {
        "indices": idx,
        "origins": origins.to(dtype=torch.int64),
        "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=device),
        "grid": torch.tensor(
            [height // tile_size, width // tile_size],
            dtype=torch.int64,
            device=device,
        ),
    }
    return meta, levels


@pytest.mark.parametrize(
    ("shape", "tile_size", "kmax"),
    [
        ((1, 16, 64, 64), 8, 10),
        ((2, 24, 64, 64), 8, 12),
    ],
)
def test_triton_tileunpack_overlap_priority_parity_fp16(
    shape: tuple[int, int, int, int],
    tile_size: int,
    kmax: int,
) -> None:
    availability = get_triton_tileunpack_availability()
    if not availability.available:
        pytest.skip(f"Triton tileunpack unavailable: {availability.reason}")

    seed_all(221, deterministic=True)
    batch, channels, height, width = shape
    base = torch.randn(shape, dtype=torch.float16, device="cuda").contiguous()
    packed = torch.randn(
        (batch, kmax, channels, tile_size, tile_size),
        dtype=torch.float16,
        device="cuda",
    )
    meta, levels = _build_overlap_meta(
        batch=batch,
        kmax=kmax,
        height=height,
        width=width,
        tile_size=tile_size,
        device=base.device,
    )

    ref = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        overlap_mode="override",
        prefer_triton=False,
    )
    tri = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        overlap_mode="override",
        prefer_triton=True,
        allow_fallback=False,
    )
    assert tri.backend == "triton"
    report = evaluate_parity_outputs(
        case_name="tileunpack-overlap-fp16",
        reference_backend="reference",
        candidate_backend="triton",
        reference_output=ref.merged,
        candidate_output=tri.merged,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True


def test_triton_tileunpack_overlap_blend_parity_fp16() -> None:
    availability = get_triton_tileunpack_availability()
    if not availability.available:
        pytest.skip(f"Triton tileunpack unavailable: {availability.reason}")

    seed_all(229, deterministic=True)
    shape = (1, 16, 64, 64)
    batch, channels, height, width = shape
    tile_size = 8
    kmax = 10
    blend_alpha = 0.25

    base = torch.randn(shape, dtype=torch.float16, device="cuda").contiguous()
    packed = torch.randn(
        (batch, kmax, channels, tile_size, tile_size),
        dtype=torch.float16,
        device="cuda",
    )
    meta, levels = _build_overlap_meta(
        batch=batch,
        kmax=kmax,
        height=height,
        width=width,
        tile_size=tile_size,
        device=base.device,
    )

    ref = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        overlap_mode="blend",
        blend_alpha=blend_alpha,
        prefer_triton=False,
    )
    tri = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        overlap_mode="blend",
        blend_alpha=blend_alpha,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert tri.backend == "triton"
    report = evaluate_parity_outputs(
        case_name="tileunpack-overlap-blend-fp16",
        reference_backend="reference",
        candidate_backend="triton",
        reference_output=ref.merged,
        candidate_output=tri.merged,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True
