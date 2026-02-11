from __future__ import annotations

import pytest
import torch

from apex_x.runtime import (
    gather_gate_scatter,
    gather_gate_scatter_reference,
    get_triton_availability,
)
from apex_x.tiles import TilePackTorch, TileUnpackTorch


def _explicit_reference_pipeline(
    *,
    base_map: torch.Tensor,
    heavy_map: torch.Tensor,
    indices: torch.Tensor,
    tile_size: int,
    boundary_proxy: torch.Tensor,
    uncertainty_proxy: torch.Tensor,
    level_priority: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    packer = TilePackTorch()
    unpacker = TileUnpackTorch()

    heavy_packed, meta = packer.pack(heavy_map, indices, tile_size=tile_size, order_mode="hilbert")
    base_packed, _ = packer.pack(base_map, indices, tile_size=tile_size, order_mode="hilbert")
    boundary_packed, _ = packer.pack(
        boundary_proxy, indices, tile_size=tile_size, order_mode="hilbert"
    )
    uncertainty_packed, _ = packer.pack(
        uncertainty_proxy, indices, tile_size=tile_size, order_mode="hilbert"
    )

    boundary_w = torch.nn.functional.softplus(torch.tensor(1.0, dtype=base_map.dtype))
    uncertainty_w = torch.nn.functional.softplus(torch.tensor(1.0, dtype=base_map.dtype))
    alpha_packed = torch.sigmoid(boundary_w * boundary_packed + uncertainty_w * uncertainty_packed)
    fused_packed = base_packed + alpha_packed * (heavy_packed - base_packed)

    merged, pri = unpacker.unpack(
        base_map=base_map,
        packed_out=fused_packed,
        meta=meta,
        level_priority=level_priority,
        overlap_mode="override",
    )
    return merged, pri


def test_gather_gate_scatter_reference_matches_explicit_pipeline() -> None:
    torch.manual_seed(7)
    base = torch.randn((1, 2, 8, 8), dtype=torch.float32)
    heavy = torch.randn((1, 2, 8, 8), dtype=torch.float32)
    idx = torch.tensor([[0, 3]], dtype=torch.int64)
    boundary = torch.rand((1, 1, 8, 8), dtype=torch.float32)
    uncertainty = torch.rand((1, 1, 8, 8), dtype=torch.float32)

    ref = gather_gate_scatter_reference(
        base_map=base,
        heavy_map=heavy,
        indices=idx,
        tile_size=4,
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        level_priority=2,
    )
    explicit_merged, explicit_pri = _explicit_reference_pipeline(
        base_map=base,
        heavy_map=heavy,
        indices=idx,
        tile_size=4,
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        level_priority=2,
    )

    assert ref.backend == "reference"
    assert ref.fallback_reason is None
    assert torch.allclose(ref.merged, explicit_merged, rtol=1e-5, atol=1e-6)
    assert torch.equal(ref.priority_map, explicit_pri)


def test_dispatch_falls_back_to_reference_when_triton_unavailable() -> None:
    torch.manual_seed(11)
    base = torch.randn((1, 1, 8, 8), dtype=torch.float32)
    heavy = torch.randn((1, 1, 8, 8), dtype=torch.float32)
    idx = torch.tensor([[0, 3]], dtype=torch.int64)
    boundary = torch.rand((1, 1, 8, 8), dtype=torch.float32)
    uncertainty = torch.rand((1, 1, 8, 8), dtype=torch.float32)

    out = gather_gate_scatter(
        base_map=base,
        heavy_map=heavy,
        indices=idx,
        tile_size=4,
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        prefer_triton=True,
    )

    availability = get_triton_availability()
    assert out.backend == "reference" or availability.available
    if not availability.available:
        assert out.fallback_reason is not None


def test_dispatch_never_raises_on_legacy_triton_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apex_x.runtime import triton_fused as triton_module

    monkeypatch.setattr(
        triton_module,
        "get_triton_availability",
        lambda: triton_module.TritonAvailability(
            triton_installed=True,
            cuda_available=True,
            cuda_device_count=1,
            reason=None,
        ),
    )

    base = torch.randn((1, 1, 8, 8), dtype=torch.float32)
    heavy = torch.randn((1, 1, 8, 8), dtype=torch.float32)
    idx = torch.tensor([[0]], dtype=torch.int64)
    boundary = torch.rand((1, 1, 8, 8), dtype=torch.float32)
    uncertainty = torch.rand((1, 1, 8, 8), dtype=torch.float32)

    out = gather_gate_scatter(
        base_map=base,
        heavy_map=heavy,
        indices=idx,
        tile_size=4,
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert out.backend == "reference"
    assert out.fallback_reason == "legacy_triton_entrypoint_deprecated_reference_only"
