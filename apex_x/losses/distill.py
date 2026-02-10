from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as f

from .seg_loss import soft_boundary_distance_transform

KLDReduction = Literal["none", "batchmean", "sum", "mean"]


@dataclass(frozen=True)
class DistillationLossOutput:
    """Distillation loss bundle."""

    total_loss: Tensor
    logits_kl_loss: Tensor
    feature_l2_loss: Tensor
    boundary_loss: Tensor
    feature_layers: tuple[str, ...]


def _weighted_mean(per_instance: Tensor, weights: Tensor, *, eps: float = 1e-8) -> Tensor:
    if per_instance.ndim != 2:
        raise ValueError("per_instance must be [B,N]")
    if weights.ndim != 2 or weights.shape != per_instance.shape:
        raise ValueError("weights must match per_instance shape [B,N]")
    denom = weights.sum().clamp(min=eps)
    return (per_instance * weights).sum() / denom


def _normalize_instance_weights(
    instance_weights: Tensor | None,
    *,
    batch_size: int,
    num_instances: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if instance_weights is None:
        return torch.ones((batch_size, num_instances), device=device, dtype=dtype)
    if instance_weights.ndim != 2 or instance_weights.shape != (batch_size, num_instances):
        raise ValueError("instance_weights must be [B,N]")
    weights = instance_weights.to(device=device, dtype=dtype)
    if not torch.isfinite(weights).all():
        raise ValueError("instance_weights must contain finite values")
    if torch.any(weights < 0):
        raise ValueError("instance_weights must be >= 0")
    return weights


def logits_kl_distill(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 1.0,
    class_dim: int = 1,
    reduction: KLDReduction = "batchmean",
) -> Tensor:
    """KL distillation over logits with temperature scaling."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have same shape")
    if student_logits.ndim < 2:
        raise ValueError("logits tensors must be at least 2D")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if reduction not in {"none", "batchmean", "sum", "mean"}:
        raise ValueError("invalid reduction")
    if not torch.isfinite(student_logits).all():
        raise ValueError("student_logits must contain finite values")
    if not torch.isfinite(teacher_logits).all():
        raise ValueError("teacher_logits must contain finite values")

    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits.detach() / temperature
    log_p = f.log_softmax(student_scaled, dim=class_dim)
    q = f.softmax(teacher_scaled, dim=class_dim)

    kl = f.kl_div(log_p, q, reduction=reduction)
    return kl * (temperature * temperature)


def _normalize_feature(x: Tensor) -> Tensor:
    if x.ndim < 2:
        return x
    return f.normalize(x, p=2.0, dim=1, eps=1e-6)


def feature_l2_distill(
    student_features: Mapping[str, Tensor],
    teacher_features: Mapping[str, Tensor],
    *,
    selected_layers: Sequence[str] | None = None,
    layer_weights: Mapping[str, float] | None = None,
    normalize_features: bool = False,
    eps: float = 1e-12,
) -> tuple[Tensor, tuple[str, ...]]:
    """Feature L2 distillation over selected layers."""
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    if not student_features:
        raise ValueError("student_features must not be empty")
    if not teacher_features:
        raise ValueError("teacher_features must not be empty")

    if selected_layers is None:
        layers = sorted(set(student_features) & set(teacher_features))
    else:
        layers = [str(layer) for layer in selected_layers]
    if not layers:
        raise ValueError("no selected layers available for distillation")

    total: Tensor | None = None
    denom = 0.0
    used_layers: list[str] = []
    for layer in layers:
        if layer not in student_features:
            raise ValueError(f"selected layer {layer!r} missing in student_features")
        if layer not in teacher_features:
            raise ValueError(f"selected layer {layer!r} missing in teacher_features")
        student = student_features[layer]
        teacher = teacher_features[layer]
        if student.shape != teacher.shape:
            raise ValueError(f"feature shape mismatch for layer {layer!r}")
        if not torch.isfinite(student).all():
            raise ValueError(f"student feature {layer!r} must be finite")
        if not torch.isfinite(teacher).all():
            raise ValueError(f"teacher feature {layer!r} must be finite")

        s = _normalize_feature(student) if normalize_features else student
        t = _normalize_feature(teacher.detach()) if normalize_features else teacher.detach()
        layer_loss = f.mse_loss(s, t, reduction="mean")

        weight = 1.0
        if layer_weights is not None:
            if layer not in layer_weights:
                raise ValueError(f"layer weight missing for {layer!r}")
            weight = float(layer_weights[layer])
            if weight < 0.0:
                raise ValueError(f"layer weight for {layer!r} must be >= 0")
        if weight == 0.0:
            continue

        used_layers.append(layer)
        scaled = layer_loss * weight
        total = scaled if total is None else total + scaled
        denom += weight

    if total is None or denom <= 0.0:
        sample = next(iter(student_features.values()))
        return sample.new_zeros(()), tuple()

    return total / max(denom, eps), tuple(used_layers)


def _soft_boundary_map(mask_probs: Tensor, *, eps: float = 1e-6) -> Tensor:
    if mask_probs.ndim != 4:
        raise ValueError("mask_probs must be [B,N,H,W]")
    bsz, num_instances, height, width = mask_probs.shape
    flat = mask_probs.reshape(bsz * num_instances, 1, height, width)
    kx = mask_probs.new_tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]).unsqueeze(
        0
    )
    ky = mask_probs.new_tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]).unsqueeze(
        0
    )
    gx = f.conv2d(flat, kx, padding=1)
    gy = f.conv2d(flat, ky, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + eps)
    boundary = grad / (1.0 + grad)
    return boundary.reshape(bsz, num_instances, height, width)


def boundary_distill_loss(
    student_mask_logits: Tensor,
    teacher_mask_logits: Tensor,
    *,
    instance_weights: Tensor | None = None,
    dt_iterations: int = 8,
    dt_temperature: float = 0.25,
) -> Tensor:
    """Boundary distillation using teacher boundary distance-transform weighting."""
    if student_mask_logits.shape != teacher_mask_logits.shape:
        raise ValueError("student and teacher mask logits must have identical shape")
    if student_mask_logits.ndim != 4:
        raise ValueError("mask logits must be [B,N,H,W]")
    if not torch.isfinite(student_mask_logits).all():
        raise ValueError("student_mask_logits must contain finite values")
    if not torch.isfinite(teacher_mask_logits).all():
        raise ValueError("teacher_mask_logits must contain finite values")

    student_probs = torch.sigmoid(student_mask_logits)
    teacher_probs = torch.sigmoid(teacher_mask_logits.detach())

    student_boundary = _soft_boundary_map(student_probs)
    teacher_boundary = _soft_boundary_map(teacher_probs)
    teacher_dt = soft_boundary_distance_transform(
        teacher_boundary,
        iterations=dt_iterations,
        temperature=dt_temperature,
    )

    per_instance = ((student_boundary - teacher_boundary).abs() * (1.0 + teacher_dt)).mean(
        dim=(2, 3)
    )
    weights = _normalize_instance_weights(
        instance_weights,
        batch_size=student_mask_logits.shape[0],
        num_instances=student_mask_logits.shape[1],
        device=student_mask_logits.device,
        dtype=student_mask_logits.dtype,
    )
    return _weighted_mean(per_instance, weights)


def distillation_losses(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_features: Mapping[str, Tensor],
    teacher_features: Mapping[str, Tensor],
    student_mask_logits: Tensor,
    teacher_mask_logits: Tensor,
    temperature: float = 1.0,
    selected_feature_layers: Sequence[str] | None = None,
    feature_layer_weights: Mapping[str, float] | None = None,
    normalize_features: bool = False,
    logits_weight: float = 1.0,
    feature_weight: float = 1.0,
    boundary_weight: float = 1.0,
    dt_iterations: int = 8,
    dt_temperature: float = 0.25,
) -> DistillationLossOutput:
    """Compute combined distillation losses."""
    if logits_weight < 0.0 or feature_weight < 0.0 or boundary_weight < 0.0:
        raise ValueError("logits/feature/boundary weights must be >= 0")

    logits_loss = logits_kl_distill(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )
    feature_loss, layers = feature_l2_distill(
        student_features,
        teacher_features,
        selected_layers=selected_feature_layers,
        layer_weights=feature_layer_weights,
        normalize_features=normalize_features,
    )
    boundary_loss = boundary_distill_loss(
        student_mask_logits,
        teacher_mask_logits,
        dt_iterations=dt_iterations,
        dt_temperature=dt_temperature,
    )

    total = (
        logits_weight * logits_loss
        + feature_weight * feature_loss
        + boundary_weight * boundary_loss
    )
    return DistillationLossOutput(
        total_loss=total,
        logits_kl_loss=logits_loss,
        feature_l2_loss=feature_loss,
        boundary_loss=boundary_loss,
        feature_layers=layers,
    )


__all__ = [
    "DistillationLossOutput",
    "logits_kl_distill",
    "feature_l2_distill",
    "boundary_distill_loss",
    "distillation_losses",
]
