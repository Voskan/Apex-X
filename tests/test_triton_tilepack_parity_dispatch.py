from __future__ import annotations

import torch

from apex_x.kernels.triton.tilepack import (
    get_triton_tilepack_availability,
    tilepack_dispatch,
    tilepack_reference,
)
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
from apex_x.tiles import TilePackTorch
from apex_x.utils.repro import seed_all


def _sorted_l2r_indices(batch: int, kmax: int, grid_h: int, grid_w: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_idx = grid_h * grid_w
    idx = torch.randint(0, max_idx, (batch, kmax), dtype=torch.int32, generator=gen)
    return torch.sort(idx, dim=1).values.contiguous()


def test_tilepack_dispatch_falls_back_on_cpu_and_matches_torch_pack() -> None:
    seed_all(3, deterministic=True)
    feature = torch.randn((2, 8, 32, 32), dtype=torch.float32, device="cpu")
    idx = _sorted_l2r_indices(batch=2, kmax=7, grid_h=8, grid_w=8, seed=9)

    dispatch = tilepack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=4,
        prefer_triton=True,
        allow_fallback=True,
    )
    ref_torch, _ = TilePackTorch().pack(
        feature_map=feature,
        indices=idx.to(dtype=torch.int64),
        tile_size=4,
        order_mode="l2r",
    )

    assert dispatch.backend == "reference"
    assert dispatch.packed.is_contiguous()
    report = evaluate_parity_outputs(
        case_name="tilepack-cpu-fallback",
        reference_backend="tilepack_torch",
        candidate_backend="tilepack_dispatch",
        reference_output=ref_torch,
        candidate_output=dispatch.packed,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True


def test_tilepack_dispatch_gradient_uses_reference_path_when_requires_grad() -> None:
    seed_all(5, deterministic=True)
    feature = torch.randn((1, 4, 16, 16), dtype=torch.float32, requires_grad=True)
    idx = _sorted_l2r_indices(batch=1, kmax=4, grid_h=4, grid_w=4, seed=1)

    dispatch = tilepack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=4,
        prefer_triton=True,
        allow_fallback=True,
        inference_only=True,
    )
    loss = dispatch.packed.sum()
    loss.backward()

    assert dispatch.backend == "reference"
    assert dispatch.fallback_reason == "autograd_not_supported_for_triton_tilepack"
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_tilepack_reference_matches_tilepack_torch_multiple_shapes() -> None:
    packer = TilePackTorch()
    shapes = [
        (1, 8, 32, 32, 4, 5),
        (2, 16, 64, 64, 8, 9),
        (1, 24, 48, 48, 6, 10),
    ]
    for case_idx, (batch, channels, height, width, tile, kmax) in enumerate(shapes):
        seed_all(100 + case_idx, deterministic=True)
        feature = torch.randn((batch, channels, height, width), dtype=torch.float32)
        idx = _sorted_l2r_indices(
            batch=batch,
            kmax=kmax,
            grid_h=height // tile,
            grid_w=width // tile,
            seed=200 + case_idx,
        )

        ref, _ = tilepack_reference(feature, idx, tile)
        torch_pack, _ = packer.pack(
            feature_map=feature,
            indices=idx.to(dtype=torch.int64),
            tile_size=tile,
            order_mode="l2r",
        )
        report = evaluate_parity_outputs(
            case_name=f"tilepack-reference-{case_idx}",
            reference_backend="tilepack_torch",
            candidate_backend="tilepack_reference",
            reference_output=torch_pack,
            candidate_output=ref,
        )
        assert report.passed is True


def test_tilepack_availability_object_cpu_safe() -> None:
    availability = get_triton_tilepack_availability()
    assert isinstance(availability.available, bool)
    if not availability.available:
        assert availability.reason is not None
