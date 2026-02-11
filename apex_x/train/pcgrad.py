"""PCGrad++ utilities for multi-task gradient conflict handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

DEFAULT_LOSS_GROUP_ORDER: tuple[str, ...] = (
    "det_cls",
    "det_box",
    "seg_mask",
    "seg_boundary",
)


@dataclass(frozen=True, slots=True)
class LossGroup:
    """Named loss group used in PCGrad++ projection."""

    name: str
    loss: Tensor

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LossGroup.name must be non-empty")
        if self.loss.ndim != 0:
            raise ValueError("LossGroup.loss must be a scalar tensor")
        if not torch.isfinite(self.loss):
            raise ValueError("LossGroup.loss must be finite")


@dataclass(frozen=True, slots=True)
class PCGradDiagnostics:
    """Debug metrics for one PCGrad++ application."""

    group_names: tuple[str, ...]
    projected_pairs: int
    conflicting_pairs: int
    conflicting_pairs_after: int
    total_pairs: int
    conflict_rate_before: float
    conflict_rate_after: float
    shared_param_count: int
    head_param_count: int


def group_loss_terms(
    loss_terms: Mapping[str, Tensor],
    *,
    preferred_order: Sequence[str] = DEFAULT_LOSS_GROUP_ORDER,
) -> tuple[LossGroup, ...]:
    """Create deterministic grouped losses for PCGrad++."""
    groups: list[LossGroup] = []
    seen: set[str] = set()

    for name in preferred_order:
        if name in loss_terms:
            groups.append(LossGroup(name=name, loss=loss_terms[name]))
            seen.add(name)

    for name in sorted(loss_terms):
        if name in seen:
            continue
        groups.append(LossGroup(name=name, loss=loss_terms[name]))

    return tuple(groups)


def _collect_grads(
    loss: Tensor,
    params: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> list[Tensor]:
    if not params:
        return []
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    out: list[Tensor] = []
    for grad, param in zip(grads, params, strict=True):
        if grad is None:
            out.append(torch.zeros_like(param))
        else:
            out.append(grad.detach())
    return out


def _dot(grads_a: Sequence[Tensor], grads_b: Sequence[Tensor]) -> Tensor:
    if len(grads_a) != len(grads_b):
        raise ValueError("gradient lists must have equal length")
    total: Tensor | None = None
    for ga, gb in zip(grads_a, grads_b, strict=True):
        term = torch.sum(ga * gb)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("gradient lists must be non-empty")
    return total


def _norm2(grads: Sequence[Tensor], *, eps: float) -> Tensor:
    total: Tensor | None = None
    for grad in grads:
        term = torch.sum(grad * grad)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("gradient list must be non-empty")
    return total + eps


def _count_conflicting_pairs(grad_sets: Sequence[Sequence[Tensor]]) -> int:
    if len(grad_sets) < 2:
        return 0
    conflicts = 0
    for idx, grads_i in enumerate(grad_sets):
        for jdx, grads_j in enumerate(grad_sets):
            if idx == jdx:
                continue
            if float(_dot(grads_i, grads_j).item()) < 0.0:
                conflicts += 1
    return conflicts


def apply_pcgradpp(
    *,
    loss_terms: Mapping[str, Tensor],
    shared_params: Sequence[nn.Parameter],
    head_params: Sequence[nn.Parameter] = (),
    preferred_order: Sequence[str] = DEFAULT_LOSS_GROUP_ORDER,
    eps: float = 1e-12,
) -> PCGradDiagnostics:
    """Project conflicting task gradients for shared params only.

    This function writes `.grad` for both `shared_params` and `head_params`.
    - Shared parameters receive PCGrad++-projected gradient aggregation.
    - Head parameters receive standard total-loss gradients (no projection).
    """
    groups = group_loss_terms(loss_terms, preferred_order=preferred_order)
    if not groups:
        raise ValueError("loss_terms must contain at least one loss")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    shared = [param for param in shared_params if param.requires_grad]
    heads = [param for param in head_params if param.requires_grad]

    for param in shared:
        param.grad = None
    for param in heads:
        param.grad = None

    total_loss = torch.stack([group.loss for group in groups]).sum()
    if heads:
        head_grads = _collect_grads(total_loss, heads, retain_graph=True)
        for param, grad in zip(heads, head_grads, strict=True):
            param.grad = grad.clone()

    if not shared:
        return PCGradDiagnostics(
            group_names=tuple(group.name for group in groups),
            projected_pairs=0,
            conflicting_pairs=0,
            conflicting_pairs_after=0,
            total_pairs=0,
            conflict_rate_before=0.0,
            conflict_rate_after=0.0,
            shared_param_count=0,
            head_param_count=len(heads),
        )

    base_grads: list[list[Tensor]] = []
    for index, group in enumerate(groups):
        grads = _collect_grads(
            group.loss,
            shared,
            retain_graph=index < (len(groups) - 1),
        )
        base_grads.append(grads)

    projected: list[list[Tensor]] = [[grad.clone() for grad in grads] for grads in base_grads]
    total_pairs = len(groups) * max(len(groups) - 1, 0)
    conflicting_pairs_before = _count_conflicting_pairs(base_grads)
    projected_pairs = 0

    for idx, grads_i in enumerate(projected):
        for jdx, grads_j in enumerate(base_grads):
            if idx == jdx:
                continue
            dot_ij = _dot(grads_i, grads_j)
            if float(dot_ij.item()) < 0.0:
                denom = _norm2(grads_j, eps=eps)
                coeff = dot_ij / denom
                for grad_pos in range(len(grads_i)):
                    grads_i[grad_pos] = grads_i[grad_pos] - coeff * grads_j[grad_pos]
                projected_pairs += 1

    conflicting_pairs_after = _count_conflicting_pairs(projected)
    conflict_rate_before = (
        float(conflicting_pairs_before) / float(total_pairs) if total_pairs > 0 else 0.0
    )
    conflict_rate_after = (
        float(conflicting_pairs_after) / float(total_pairs) if total_pairs > 0 else 0.0
    )

    merged_shared: list[Tensor] = []
    for param_pos in range(len(shared)):
        stacked = torch.stack([grads[param_pos] for grads in projected], dim=0)
        merged_shared.append(stacked.mean(dim=0))

    for param, grad in zip(shared, merged_shared, strict=True):
        param.grad = grad.clone()

    return PCGradDiagnostics(
        group_names=tuple(group.name for group in groups),
        projected_pairs=projected_pairs,
        conflicting_pairs=conflicting_pairs_before,
        conflicting_pairs_after=conflicting_pairs_after,
        total_pairs=total_pairs,
        conflict_rate_before=conflict_rate_before,
        conflict_rate_after=conflict_rate_after,
        shared_param_count=len(shared),
        head_param_count=len(heads),
    )


def diagnostics_to_dict(diagnostics: PCGradDiagnostics) -> dict[str, Any]:
    """Serialize diagnostics for logging payloads."""
    return {
        "group_names": list(diagnostics.group_names),
        "projected_pairs": diagnostics.projected_pairs,
        "conflicting_pairs": diagnostics.conflicting_pairs,
        "conflicting_pairs_after": diagnostics.conflicting_pairs_after,
        "total_pairs": diagnostics.total_pairs,
        "conflict_rate_before": diagnostics.conflict_rate_before,
        "conflict_rate_after": diagnostics.conflict_rate_after,
        "shared_param_count": diagnostics.shared_param_count,
        "head_param_count": diagnostics.head_param_count,
    }


__all__ = [
    "DEFAULT_LOSS_GROUP_ORDER",
    "LossGroup",
    "PCGradDiagnostics",
    "group_loss_terms",
    "apply_pcgradpp",
    "diagnostics_to_dict",
]
