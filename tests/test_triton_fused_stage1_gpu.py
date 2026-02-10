from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.fused_pack_op_unpack import (
    fused_pack_op_unpack_dispatch,
    fused_pack_op_unpack_triton,
    get_triton_fused_stage1_availability,
    separate_pack_op_unpack_reference,
)
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
from apex_x.utils.repro import seed_all


def _unique_indices_cuda(
    batch: int, kmax: int, max_idx: int, seed: int, device: torch.device
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rows: list[torch.Tensor] = []
    for b in range(batch):
        row = torch.randperm(max_idx, generator=gen, dtype=torch.int64)[:kmax]
        rows.append(torch.sort(row).values)
        gen.manual_seed(seed + b + 1)
    return torch.stack(rows, dim=0).to(dtype=torch.int32, device=device).contiguous()


@pytest.mark.skipif(
    not get_triton_fused_stage1_availability().available,
    reason="Triton fused stage1 requires CUDA + Triton",
)
def test_fused_stage1_triton_parity_fp16() -> None:
    seed_all(23, deterministic=True)
    device = torch.device("cuda")
    feature = torch.randn((1, 24, 64, 64), dtype=torch.float16, device=device)
    idx = _unique_indices_cuda(batch=1, kmax=20, max_idx=64, seed=4, device=device)
    params = {
        "value_scale": 0.75,
        "value_bias": 0.1,
        "gate_scale": 1.2,
        "gate_bias": -0.15,
    }

    out_triton, _ = fused_pack_op_unpack_triton(
        feature_map=feature,
        indices=idx,
        tile_size=8,
        **params,
    )
    out_ref, _ = separate_pack_op_unpack_reference(
        feature_map=feature,
        indices=idx,
        tile_size=8,
        **params,
    )
    report = evaluate_parity_outputs(
        case_name="fused-stage1-triton-fp16",
        reference_backend="separate_reference",
        candidate_backend="triton_fused",
        reference_output=out_ref,
        candidate_output=out_triton,
        tolerances=ToleranceConfig(),
    )
    assert report.passed is True


@pytest.mark.skipif(
    not get_triton_fused_stage1_availability().available,
    reason="Triton fused stage1 requires CUDA + Triton",
)
def test_fused_stage1_dispatch_uses_triton_when_available() -> None:
    seed_all(29, deterministic=True)
    device = torch.device("cuda")
    feature = torch.randn((1, 16, 32, 32), dtype=torch.float16, device=device)
    idx = _unique_indices_cuda(batch=1, kmax=8, max_idx=16, seed=9, device=device)

    result = fused_pack_op_unpack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=8,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert result.backend == "triton"
