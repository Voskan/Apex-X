from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.tilepack import get_triton_tilepack_availability, tilepack_dispatch
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
from apex_x.tiles import TilePackTorch
from apex_x.utils.repro import seed_all


def _sorted_l2r_indices(batch: int, kmax: int, grid_h: int, grid_w: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_idx = grid_h * grid_w
    idx = torch.randint(0, max_idx, (batch, kmax), dtype=torch.int32, generator=gen)
    return torch.sort(idx, dim=1).values.contiguous()


@pytest.mark.parametrize(
    ("shape", "tile_size", "kmax"),
    [
        ((1, 16, 32, 32), 4, 8),
        ((2, 32, 64, 64), 8, 12),
    ],
)
def test_triton_tilepack_parity_fp16(
    shape: tuple[int, int, int, int], tile_size: int, kmax: int
) -> None:
    availability = get_triton_tilepack_availability()
    if not availability.available:
        pytest.skip(f"Triton tilepack unavailable: {availability.reason}")

    batch, channels, height, width = shape
    seed_all(123, deterministic=True)
    feature = torch.randn(shape, dtype=torch.float16, device="cuda").contiguous()
    idx = _sorted_l2r_indices(
        batch=batch,
        kmax=kmax,
        grid_h=height // tile_size,
        grid_w=width // tile_size,
        seed=77,
    ).to(device="cuda")

    torch_pack, _ = TilePackTorch().pack(
        feature_map=feature,
        indices=idx.to(dtype=torch.int64),
        tile_size=tile_size,
        order_mode="l2r",
    )
    dispatch = tilepack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=tile_size,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert dispatch.backend == "triton"
    assert dispatch.packed.is_contiguous()

    report = evaluate_parity_outputs(
        case_name="triton-tilepack-fp16",
        reference_backend="tilepack_torch",
        candidate_backend="triton",
        reference_output=torch_pack,
        candidate_output=dispatch.packed,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True


def test_triton_tilepack_parity_bf16_if_supported() -> None:
    availability = get_triton_tilepack_availability()
    if not availability.available:
        pytest.skip(f"Triton tilepack unavailable: {availability.reason}")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 is not supported on this device")

    seed_all(321, deterministic=True)
    shape = (1, 16, 64, 64)
    tile_size = 8
    kmax = 10
    feature = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    idx = _sorted_l2r_indices(
        batch=shape[0],
        kmax=kmax,
        grid_h=shape[2] // tile_size,
        grid_w=shape[3] // tile_size,
        seed=11,
    ).to(device="cuda")

    torch_pack, _ = TilePackTorch().pack(
        feature_map=feature,
        indices=idx.to(dtype=torch.int64),
        tile_size=tile_size,
        order_mode="l2r",
    )
    dispatch = tilepack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=tile_size,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert dispatch.backend == "triton"
    report = evaluate_parity_outputs(
        case_name="triton-tilepack-bf16",
        reference_backend="tilepack_torch",
        candidate_backend="triton",
        reference_output=torch_pack,
        candidate_output=dispatch.packed,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True

