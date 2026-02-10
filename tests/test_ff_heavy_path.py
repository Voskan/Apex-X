from __future__ import annotations

import torch

from apex_x.model import FFHeavyPath


def test_ff_heavy_path_output_shapes_align_with_dense_features_cpu() -> None:
    torch.manual_seed(21)
    path = FFHeavyPath(
        channels=8,
        tile_size=4,
        scan_mode="bidirectional",
        order_mode="hilbert",
        use_refine=True,
        use_fusion_gate=True,
    ).cpu()

    dense = torch.randn(2, 8, 16, 16, dtype=torch.float32)
    idx = torch.tensor([[0, 1, 3], [2, 5, 7]], dtype=torch.int64)
    boundary = torch.rand(2, 1, 8, 8, dtype=torch.float32)
    uncertainty = torch.rand(2, 1, 8, 8, dtype=torch.float32)

    out = path(dense, idx, boundary_proxy=boundary, uncertainty_proxy=uncertainty)

    assert out.heavy_features.shape == dense.shape
    assert out.detail_map.shape == dense.shape
    assert out.alpha.shape == (2, 1, 16, 16)
    assert out.tokens.shape == (2, 3, 8)
    assert out.mixed_tokens.shape == (2, 3, 8)
    assert out.gamma.shape == (2, 3, 8)
    assert out.beta.shape == (2, 3, 8)
    assert out.state.shape == (2, 2, 8)
    assert torch.isfinite(out.heavy_features).all()
    assert torch.isfinite(out.detail_map).all()
    assert torch.isfinite(out.alpha).all()


def test_ff_heavy_path_zero_selected_tiles_produces_zero_detail() -> None:
    path = FFHeavyPath(
        channels=4,
        tile_size=4,
        scan_mode="forward",
        use_refine=True,
        use_fusion_gate=True,
    ).cpu()
    dense = torch.randn(1, 4, 16, 16, dtype=torch.float32)
    idx = torch.empty((1, 0), dtype=torch.int64)

    out = path(dense, idx)

    assert out.heavy_features.shape == dense.shape
    assert out.detail_map.shape == dense.shape
    assert out.tokens.shape == (1, 0, 4)
    assert out.mixed_tokens.shape == (1, 0, 4)
    assert out.gamma.shape == (1, 0, 4)
    assert out.beta.shape == (1, 0, 4)
    assert out.state.shape == (1, 1, 4)
    assert torch.allclose(out.detail_map, torch.zeros_like(dense), atol=1e-6)


def test_ff_heavy_path_deterministic_and_gradient_flow_cpu() -> None:
    torch.manual_seed(8)
    path = FFHeavyPath(
        channels=6,
        tile_size=4,
        scan_mode="bidirectional",
        use_refine=True,
        use_fusion_gate=True,
    ).cpu()
    path.eval()

    dense = torch.randn(1, 6, 16, 16, dtype=torch.float32, requires_grad=True)
    idx = torch.tensor([[0, 3, 5]], dtype=torch.int64)
    boundary = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    uncertainty = torch.rand(1, 1, 16, 16, dtype=torch.float32)

    out1 = path(dense, idx, boundary_proxy=boundary, uncertainty_proxy=uncertainty)
    out2 = path(dense, idx, boundary_proxy=boundary, uncertainty_proxy=uncertainty)

    assert torch.allclose(out1.heavy_features, out2.heavy_features)
    assert torch.allclose(out1.detail_map, out2.detail_map)
    assert torch.allclose(out1.alpha, out2.alpha)

    loss = out1.detail_map.square().mean() + out1.heavy_features.square().mean()
    loss.backward()

    assert dense.grad is not None
    assert torch.isfinite(dense.grad).all()
    assert float(dense.grad.abs().sum().item()) > 0.0

    param_grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in path.parameters()
        if parameter.grad is not None
    )
    assert param_grad_sum > 0.0
