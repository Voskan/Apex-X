from __future__ import annotations

import torch

import apex_x.model.ff_heavy_path as ff_heavy_path_module
from apex_x.kernels.triton.fused_pack_op_unpack import FusedPackOpUnpackDispatchResult
from apex_x.model import FFHeavyPath


def _constant_film_params(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.full_like(tokens, fill_value=0.2)
    beta = torch.full_like(tokens, fill_value=-0.1)
    return gamma, beta


def test_ff_heavy_path_uses_fused_stage1_when_compatible(
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def _fake_fused_dispatch(
        *,
        feature_map: torch.Tensor,
        indices: torch.Tensor,
        tile_size: int,
        value_scale: float,
        value_bias: float,
        gate_scale: float,
        gate_bias: float,
        require_unique_indices: bool,
        prefer_triton: bool,
        allow_fallback: bool,
        inference_only: bool,
    ) -> FusedPackOpUnpackDispatchResult:
        _ = (
            indices,
            tile_size,
            value_scale,
            value_bias,
            gate_scale,
            gate_bias,
            require_unique_indices,
            prefer_triton,
            allow_fallback,
            inference_only,
        )
        calls["count"] += 1
        return FusedPackOpUnpackDispatchResult(
            merged=torch.full_like(feature_map, fill_value=2.5),
            meta={},
            backend="triton",
            fallback_reason=None,
        )

    monkeypatch.setattr(ff_heavy_path_module, "fused_pack_op_unpack_dispatch", _fake_fused_dispatch)

    path = FFHeavyPath(
        channels=4,
        tile_size=4,
        scan_mode="forward",
        use_refine=False,
        use_fusion_gate=False,
        use_triton_fused_stage1=True,
    ).cpu()
    monkeypatch.setattr(path.film, "film_params", _constant_film_params)

    dense = torch.randn((1, 4, 16, 16), dtype=torch.float32)
    idx = torch.tensor([[0, 3, 4]], dtype=torch.int64)
    path.eval()
    out = path(dense, idx)

    assert calls["count"] == 1
    assert torch.all(out.heavy_features == 2.5)


def test_ff_heavy_path_fused_stage1_matches_decomposed_path_when_constant_film(
    monkeypatch,
) -> None:
    dense = torch.randn((1, 4, 16, 16), dtype=torch.float32)
    idx = torch.tensor([[0, 3, 4]], dtype=torch.int64)

    path_ref = FFHeavyPath(
        channels=4,
        tile_size=4,
        scan_mode="forward",
        use_refine=False,
        use_fusion_gate=False,
        use_triton_fused_stage1=False,
    ).cpu()
    path_fused = FFHeavyPath(
        channels=4,
        tile_size=4,
        scan_mode="forward",
        use_refine=False,
        use_fusion_gate=False,
        use_triton_fused_stage1=True,
    ).cpu()
    path_fused.load_state_dict(path_ref.state_dict(), strict=True)

    monkeypatch.setattr(path_ref.film, "film_params", _constant_film_params)
    monkeypatch.setattr(path_fused.film, "film_params", _constant_film_params)

    path_ref.eval()
    path_fused.eval()

    out_ref = path_ref(dense, idx)
    out_fused = path_fused(dense, idx)

    assert torch.allclose(out_ref.heavy_features, out_fused.heavy_features, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_ref.detail_map, out_fused.detail_map, atol=1e-6, rtol=1e-6)


def test_ff_heavy_path_skips_fused_stage1_when_film_is_not_constant(
    monkeypatch,
) -> None:
    def _should_not_run(**_: object) -> FusedPackOpUnpackDispatchResult:
        raise AssertionError("fused stage-1 dispatch should not run for non-constant FiLM params")

    monkeypatch.setattr(ff_heavy_path_module, "fused_pack_op_unpack_dispatch", _should_not_run)

    path = FFHeavyPath(
        channels=4,
        tile_size=4,
        scan_mode="forward",
        use_refine=False,
        use_fusion_gate=False,
        use_triton_fused_stage1=True,
    ).cpu()
    path.eval()

    dense = torch.randn((1, 4, 16, 16), dtype=torch.float32)
    idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    out = path(dense, idx)
    assert out.heavy_features.shape == dense.shape
