from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as f

RegressionLossType = Literal["l1", "mse", "smooth_l1"]


@dataclass(frozen=True)
class OracleDeltaTargets:
    """Oracle delta targets for sampled tiles."""

    sampled_indices: Tensor  # [B,S] int64
    delta_targets: Tensor  # [B,S] float, detached (stop-grad)


@dataclass(frozen=True)
class UtilityOracleLossOutput:
    """Utility supervision losses against oracle delta targets."""

    total_loss: Tensor
    regression_loss: Tensor
    ranking_loss: Tensor
    num_pairs: int


@dataclass(frozen=True)
class OracleDeltaStats:
    count: int
    mean: float
    std: float
    min: float
    max: float
    abs_p95: float
    clipped_count: int
    clipped_ratio: float


def _as_batched_losses(losses: Tensor) -> Tensor:
    if losses.ndim == 1:
        return losses.unsqueeze(0)
    if losses.ndim != 2:
        raise ValueError("loss tensors must be [K] or [B,K]")
    return losses


def _as_batched_indices(indices: Tensor | list[int] | list[list[int]], *, batch: int) -> Tensor:
    idx = indices if isinstance(indices, Tensor) else torch.as_tensor(indices, dtype=torch.int64)

    if idx.ndim == 1:
        idx = idx.unsqueeze(0)
    if idx.ndim != 2:
        raise ValueError("sampled_tile_indices must be [S] or [B,S]")

    if idx.shape[0] == 1 and batch > 1:
        idx = idx.expand(batch, -1)
    if idx.shape[0] != batch:
        raise ValueError("sampled_tile_indices batch size must match losses batch size")
    if idx.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("sampled_tile_indices must be integer typed")
    return idx.to(dtype=torch.int64)


def compute_oracle_delta_targets(
    cheap_distill_loss: Tensor,
    heavy_distill_loss: Tensor,
    sampled_tile_indices: Tensor | list[int] | list[list[int]],
    *,
    clamp_abs: float | None = None,
) -> OracleDeltaTargets:
    """Compute Δ_i = L_distill(cheap, teacher) - L_distill(heavy, teacher) on sampled tiles S."""
    cheap = _as_batched_losses(cheap_distill_loss)
    heavy = _as_batched_losses(heavy_distill_loss)
    if cheap.shape != heavy.shape:
        raise ValueError("cheap_distill_loss and heavy_distill_loss must have same shape")
    if not torch.isfinite(cheap).all():
        raise ValueError("cheap_distill_loss must be finite")
    if not torch.isfinite(heavy).all():
        raise ValueError("heavy_distill_loss must be finite")
    if clamp_abs is not None and clamp_abs <= 0.0:
        raise ValueError("clamp_abs must be > 0 when provided")

    batch, tiles = cheap.shape
    sampled = _as_batched_indices(sampled_tile_indices, batch=batch).to(device=cheap.device)
    if sampled.numel() > 0 and (
        int(sampled.min().item()) < 0 or int(sampled.max().item()) >= tiles
    ):
        raise ValueError("sampled_tile_indices contain out-of-range tile ids")

    delta_full = cheap - heavy
    sampled_delta = torch.gather(delta_full, dim=1, index=sampled)
    if clamp_abs is not None:
        sampled_delta = sampled_delta.clamp(min=-clamp_abs, max=clamp_abs)

    # stop-grad oracle target
    return OracleDeltaTargets(
        sampled_indices=sampled.detach(),
        delta_targets=sampled_delta.detach(),
    )


def utility_regression_loss(
    utility_logits: Tensor,
    targets: OracleDeltaTargets,
    *,
    loss_type: RegressionLossType = "smooth_l1",
) -> Tensor:
    """Regression loss between predicted utility and detached oracle delta targets."""
    utilities = _as_batched_losses(utility_logits)
    batch, tiles = utilities.shape
    if targets.sampled_indices.shape[0] != batch:
        raise ValueError("targets batch size must match utility batch size")
    if targets.sampled_indices.shape != targets.delta_targets.shape:
        raise ValueError("targets sampled_indices and delta_targets must have same shape")
    if targets.sampled_indices.numel() > 0 and (
        int(targets.sampled_indices.min().item()) < 0
        or int(targets.sampled_indices.max().item()) >= tiles
    ):
        raise ValueError("targets sampled indices out of range")

    pred = torch.gather(utilities, dim=1, index=targets.sampled_indices)
    target = targets.delta_targets.detach()
    if loss_type == "l1":
        return f.l1_loss(pred, target, reduction="mean")
    if loss_type == "mse":
        return f.mse_loss(pred, target, reduction="mean")
    if loss_type == "smooth_l1":
        return f.smooth_l1_loss(pred, target, reduction="mean")
    raise ValueError(f"unsupported loss_type {loss_type!r}")


def utility_ranking_loss(
    utility_logits: Tensor,
    targets: OracleDeltaTargets,
    *,
    margin: float = 0.0,
) -> tuple[Tensor, int]:
    """Pairwise ranking loss to preserve oracle utility ordering among sampled tiles."""
    if margin < 0.0:
        raise ValueError("margin must be >= 0")
    utilities = _as_batched_losses(utility_logits)
    batch, tiles = utilities.shape
    if targets.sampled_indices.shape[0] != batch:
        raise ValueError("targets batch size must match utility batch size")
    if targets.sampled_indices.shape != targets.delta_targets.shape:
        raise ValueError("targets sampled_indices and delta_targets must have same shape")
    if targets.sampled_indices.numel() > 0 and (
        int(targets.sampled_indices.min().item()) < 0
        or int(targets.sampled_indices.max().item()) >= tiles
    ):
        raise ValueError("targets sampled indices out of range")

    pred = torch.gather(utilities, dim=1, index=targets.sampled_indices)
    delta = targets.delta_targets.detach()
    sample_count = int(pred.shape[1])
    if sample_count <= 1:
        return pred.new_zeros(()), 0

    losses: list[Tensor] = []
    num_pairs = 0
    for b in range(batch):
        for i in range(sample_count):
            for j in range(i + 1, sample_count):
                diff_target = float(delta[b, i].item() - delta[b, j].item())
                if diff_target == 0.0:
                    continue
                sign = 1.0 if diff_target > 0.0 else -1.0
                # hinge: max(0, margin - sign*(u_i-u_j))
                loss_ij = f.relu(pred.new_tensor(margin) - sign * (pred[b, i] - pred[b, j]))
                losses.append(loss_ij)
                num_pairs += 1

    if not losses:
        return pred.new_zeros(()), 0
    return torch.stack(losses).mean(), num_pairs


def utility_oracle_loss(
    utility_logits: Tensor,
    targets: OracleDeltaTargets,
    *,
    regression_weight: float = 1.0,
    ranking_weight: float = 1.0,
    regression_type: RegressionLossType = "smooth_l1",
    ranking_margin: float = 0.0,
) -> UtilityOracleLossOutput:
    """Combined utility supervision loss (regression + ranking)."""
    if regression_weight < 0.0 or ranking_weight < 0.0:
        raise ValueError("regression_weight and ranking_weight must be >= 0")
    reg = utility_regression_loss(utility_logits, targets, loss_type=regression_type)
    rank, pairs = utility_ranking_loss(utility_logits, targets, margin=ranking_margin)
    total = regression_weight * reg + ranking_weight * rank
    return UtilityOracleLossOutput(
        total_loss=total,
        regression_loss=reg,
        ranking_loss=rank,
        num_pairs=pairs,
    )


def summarize_oracle_delta_targets(
    delta_targets: Tensor,
    *,
    raw_delta_targets: Tensor | None = None,
    clamp_abs: float | None = None,
) -> OracleDeltaStats:
    """Summarize oracle delta-label distribution and clipping diagnostics."""
    values = delta_targets.detach().to(dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return OracleDeltaStats(
            count=0,
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            abs_p95=0.0,
            clipped_count=0,
            clipped_ratio=0.0,
        )

    abs_values = values.abs()
    quantile = torch.quantile(abs_values, q=0.95).item()
    std_value = values.std(unbiased=False).item()

    clipped_count = 0
    if clamp_abs is not None:
        if clamp_abs <= 0.0:
            raise ValueError("clamp_abs must be > 0 when provided")
        raw = (
            values
            if raw_delta_targets is None
            else raw_delta_targets.detach().to(dtype=torch.float32).reshape(-1)
        )
        if raw.shape != values.shape:
            raise ValueError("raw_delta_targets must have the same flattened size as delta_targets")
        clipped_count = int((raw.abs() > float(clamp_abs)).sum().item())

    count = int(values.numel())
    clipped_ratio = float(clipped_count / count) if count > 0 else 0.0
    return OracleDeltaStats(
        count=count,
        mean=float(values.mean().item()),
        std=float(std_value),
        min=float(values.min().item()),
        max=float(values.max().item()),
        abs_p95=float(quantile),
        clipped_count=clipped_count,
        clipped_ratio=clipped_ratio,
    )


__all__ = [
    "RegressionLossType",
    "OracleDeltaTargets",
    "UtilityOracleLossOutput",
    "OracleDeltaStats",
    "compute_oracle_delta_targets",
    "utility_regression_loss",
    "utility_ranking_loss",
    "utility_oracle_loss",
    "summarize_oracle_delta_targets",
]
