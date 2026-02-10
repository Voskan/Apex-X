from __future__ import annotations

import torch

from apex_x.model import TrackEmbeddingHead


def test_track_embedding_head_shapes_and_unit_norm() -> None:
    torch.manual_seed(3)
    head = TrackEmbeddingHead(in_channels=32, embedding_dim=24, hidden_dim=48).cpu()
    features = torch.randn(2, 32, 16, 16)
    boxes = torch.tensor(
        [
            [[0.1, 0.1, 0.7, 0.7], [0.2, 0.2, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]],
            [[0.1, 0.3, 0.9, 0.9], [0.4, 0.1, 0.6, 0.3], [0.0, 0.0, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )

    out = head(features, boxes, normalized_boxes=True)

    assert out.embeddings.shape == (2, 3, 24)
    assert out.raw_embeddings.shape == (2, 3, 24)
    assert out.pooled_features.shape == (2, 3, 32)
    norms = torch.linalg.vector_norm(out.embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out.embeddings).all()


def test_track_embedding_head_gradient_flow() -> None:
    torch.manual_seed(7)
    head = TrackEmbeddingHead(in_channels=16, embedding_dim=12, hidden_dim=24).cpu()
    features = torch.randn(1, 16, 12, 12, requires_grad=True)
    boxes = torch.tensor([[[1.0, 1.0, 8.0, 8.0], [2.0, 2.0, 10.0, 10.0]]], dtype=torch.float32)

    out = head(features, boxes, image_size=(12, 12), normalized_boxes=False)
    loss = out.embeddings.square().mean() + out.raw_embeddings.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in head.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0.0


def test_track_embedding_head_is_deterministic_for_fixed_input() -> None:
    torch.manual_seed(11)
    head = TrackEmbeddingHead(in_channels=8, embedding_dim=10, hidden_dim=16).cpu()
    features = torch.randn(1, 8, 8, 8)
    boxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [2.0, 2.0, 6.0, 6.0]]], dtype=torch.float32)

    a = head(features, boxes, image_size=(8, 8))
    b = head(features, boxes, image_size=(8, 8))
    assert torch.allclose(a.embeddings, b.embeddings)
    assert torch.allclose(a.raw_embeddings, b.raw_embeddings)
    assert torch.allclose(a.pooled_features, b.pooled_features)
