from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.fused_pack_op_unpack import (
    apply_pointwise_affine_reglu,
    fused_pack_op_unpack_dispatch,
    get_triton_fused_stage1_availability,
    separate_pack_op_unpack_reference,
)
from apex_x.runtime import evaluate_parity_outputs
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


def test_fused_stage1_dispatch_cpu_fallback_matches_reference_composition() -> None:
    seed_all(3, deterministic=True)
    feature = torch.randn((2, 12, 32, 32), dtype=torch.float32)
    idx = _unique_indices(batch=2, kmax=11, max_idx=64, seed=7)
    params = {
        "value_scale": 0.8,
        "value_bias": -0.05,
        "gate_scale": 1.1,
        "gate_bias": 0.1,
    }

    dispatch = fused_pack_op_unpack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=4,
        prefer_triton=True,
        allow_fallback=True,
        **params,
    )
    expected, _ = separate_pack_op_unpack_reference(
        feature_map=feature,
        indices=idx,
        tile_size=4,
        **params,
    )

    assert dispatch.backend == "reference"
    report = evaluate_parity_outputs(
        case_name="fused-stage1-cpu-fallback",
        reference_backend="separate_reference",
        candidate_backend="fused_dispatch",
        reference_output=expected,
        candidate_output=dispatch.merged,
    )
    assert report.passed is True


def test_fused_stage1_dispatch_gradient_fallback_when_autograd_requested() -> None:
    seed_all(11, deterministic=True)
    feature = torch.randn((1, 8, 16, 16), dtype=torch.float32, requires_grad=True)
    idx = _unique_indices(batch=1, kmax=4, max_idx=16, seed=1)
    dispatch = fused_pack_op_unpack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=4,
        prefer_triton=True,
        allow_fallback=True,
        inference_only=True,
    )
    loss = dispatch.merged.square().mean()
    loss.backward()

    assert dispatch.backend == "reference"
    assert dispatch.fallback_reason == "autograd_not_supported_for_triton_fused_stage1"
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_fused_stage1_rejects_duplicate_indices_for_determinism() -> None:
    seed_all(19, deterministic=True)
    feature = torch.randn((1, 4, 16, 16), dtype=torch.float32)
    idx = torch.tensor([[0, 0, 3, 5]], dtype=torch.int64)

    with pytest.raises(ValueError, match="indices must be unique per batch"):
        fused_pack_op_unpack_dispatch(
            feature_map=feature,
            indices=idx,
            tile_size=4,
            prefer_triton=False,
        )


def test_affine_reglu_stays_finite() -> None:
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float32)
    out = apply_pointwise_affine_reglu(
        x,
        value_scale=0.5,
        value_bias=1.0,
        gate_scale=2.0,
        gate_bias=-0.1,
    )
    assert torch.isfinite(out).all()
    assert out.shape == x.shape


def test_fused_stage1_availability_object_cpu_safe() -> None:
    availability = get_triton_fused_stage1_availability()
    assert isinstance(availability.available, bool)
    if not availability.available:
        assert availability.reason is not None
