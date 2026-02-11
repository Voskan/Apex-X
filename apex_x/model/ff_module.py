from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from apex_x.config import ApexXConfig
from apex_x.routing import (
    BudgetDualController,
    GateMode,
    build_routing_diagnostics,
    deterministic_greedy_selection,
    deterministic_two_stage_selection,
    ste_gate_from_utilities,
)
from apex_x.tiles import OverlapMode, tile_grid_shape

from .ff_heavy_path import FFHeavyPath


def _validate_utility_tensor(name: str, tensor: Tensor, *, batch: int, tiles: int) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be [B,K]")
    if tensor.shape[0] != batch or tensor.shape[1] != tiles:
        raise ValueError(f"{name} shape must be [B,{tiles}]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")


@dataclass(frozen=True)
class FFTrainOutput:
    heavy_features: Tensor  # [B,C,H,W]
    detail_map: Tensor  # [B,C,H,W]
    alpha: Tensor  # [B,1,H,W]
    probabilities: Tensor  # [B,K0]
    gates: Tensor  # [B,K0]
    expected_cost: Tensor  # scalar tensor
    budget_loss: Tensor  # scalar tensor
    selected_l0: list[list[int]]
    diagnostics: dict[str, Any]
    mu: float


@dataclass(frozen=True)
class FFInferOutput:
    heavy_features: Tensor  # [B,C,H,W]
    detail_map: Tensor  # [B,C,H,W]
    alpha: Tensor  # [B,1,H,W]
    selected_l0: list[list[int]]
    selected_l1: list[list[int]]
    l0_kmax_buffers: list[list[int]]
    l1_kmax_buffers: list[list[int]]
    spent_budget_b1: float
    spent_budget_b2: float
    diagnostics: dict[str, Any]


class FFModule(nn.Module):
    """FF routing/execution module for train (STE) and infer (deterministic) modes."""

    def __init__(
        self,
        channels: int,
        config: ApexXConfig | None = None,
        *,
        order_mode: str = "hilbert",
        overlap_mode: OverlapMode = "override",
        scan_mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")

        self.config = config or ApexXConfig()
        self.config.validate()
        self.channels = int(channels)
        use_triton_inference_scan = bool(self.config.runtime.enable_runtime_plugins)
        use_triton_fused_stage1 = bool(self.config.runtime.enable_runtime_plugins)

        self.l0_path = FFHeavyPath(
            channels=self.channels,
            tile_size=self.config.model.tile_size_l0,
            order_mode=order_mode,
            overlap_mode=overlap_mode,
            scan_mode=scan_mode,
            use_triton_inference_scan=use_triton_inference_scan,
            use_triton_fused_stage1=use_triton_fused_stage1,
        )
        self.l1_path = FFHeavyPath(
            channels=self.channels,
            tile_size=self.config.model.tile_size_l1,
            order_mode=order_mode,
            overlap_mode=overlap_mode,
            scan_mode=scan_mode,
            use_triton_inference_scan=use_triton_inference_scan,
            use_triton_fused_stage1=use_triton_fused_stage1,
        )

        self.dual_controller = BudgetDualController(
            budget=self.config.routing.budget_total,
            mu_init=self.config.train.mu_init,
            mu_lr=self.config.train.mu_lr,
            mu_min=self.config.train.mu_min,
            mu_max=self.config.train.mu_max,
            adaptive_lr=self.config.train.dual_adaptive_lr,
            lr_decay=self.config.train.dual_lr_decay,
            delta_clip=self.config.train.dual_delta_clip,
            deadband_ratio=self.config.train.dual_deadband_ratio,
            error_ema_beta=self.config.train.dual_error_ema_beta,
            adaptive_lr_min_scale=self.config.train.dual_lr_min_scale,
            adaptive_lr_max_scale=self.config.train.dual_lr_max_scale,
            logger_name="model.ff_module.dual",
        )
        self.mu_history: list[float] = [self.dual_controller.mu]

    def _grid_for_l0(self, dense_features: Tensor) -> tuple[int, int, int]:
        _, _, height, width = dense_features.shape
        grid_h, grid_w = tile_grid_shape(height, width, self.config.model.tile_size_l0)
        return grid_h, grid_w, grid_h * grid_w

    def _select_from_gate(self, utilities: Tensor, gates: Tensor) -> list[list[int]]:
        bsz, k_tiles = utilities.shape
        selected: list[list[int]] = []
        for b in range(bsz):
            gate_mask = gates[b].detach() >= 0.5
            candidates = torch.nonzero(gate_mask, as_tuple=False).flatten().tolist()
            if not candidates:
                selected.append([])
                continue
            ordered = sorted(candidates, key=lambda idx: (-float(utilities[b, idx].item()), idx))
            selected.append([int(idx) for idx in ordered[: self.config.model.kmax_l0]])
        return selected

    def _run_path_per_sample(
        self,
        path: FFHeavyPath,
        dense_features: Tensor,
        selected_indices: list[list[int]],
        *,
        boundary_proxy: Tensor | None = None,
        uncertainty_proxy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsz, channels, height, width = dense_features.shape
        heavy_out = torch.empty_like(dense_features)
        detail_out = torch.empty_like(dense_features)
        alpha_out = dense_features.new_empty((bsz, 1, height, width))

        for b in range(bsz):
            if selected_indices[b]:
                idx = torch.tensor(
                    selected_indices[b],
                    dtype=torch.int64,
                    device=dense_features.device,
                ).unsqueeze(0)
            else:
                idx = torch.empty((1, 0), dtype=torch.int64, device=dense_features.device)

            boundary_b = None if boundary_proxy is None else boundary_proxy[b : b + 1]
            uncertainty_b = None if uncertainty_proxy is None else uncertainty_proxy[b : b + 1]

            out = path(
                dense_features[b : b + 1],
                idx,
                boundary_proxy=boundary_b,
                uncertainty_proxy=uncertainty_b,
            )
            heavy_out[b : b + 1] = out.heavy_features
            detail_out[b : b + 1] = out.detail_map
            alpha_out[b : b + 1] = out.alpha

        return heavy_out, detail_out, alpha_out

    def forward_train(
        self,
        dense_features: Tensor,
        utility_logits: Tensor,
        *,
        boundary_proxy: Tensor | None = None,
        uncertainty_proxy: Tensor | None = None,
        gate_mode: GateMode = "threshold",
        gate_threshold: float | None = None,
        gate_temperature: float = 1.0,
        update_mu: bool = False,
    ) -> FFTrainOutput:
        if dense_features.ndim != 4:
            raise ValueError("dense_features must be [B,C,H,W]")
        if dense_features.shape[1] != self.channels:
            raise ValueError("dense_features channel dimension does not match module channels")

        bsz = dense_features.shape[0]
        grid_h, grid_w, k_tiles = self._grid_for_l0(dense_features)
        _validate_utility_tensor("utility_logits", utility_logits, batch=bsz, tiles=k_tiles)

        threshold = self.config.routing.theta_on if gate_threshold is None else gate_threshold
        probabilities, gates = ste_gate_from_utilities(
            utility_logits,
            threshold=float(threshold),
            mode=gate_mode,
            temperature=gate_temperature,
        )

        selected_l0 = self._select_from_gate(utility_logits, gates)
        heavy_features, detail_map, alpha = self._run_path_per_sample(
            self.l0_path,
            dense_features,
            selected_l0,
            boundary_proxy=boundary_proxy,
            uncertainty_proxy=uncertainty_proxy,
        )

        expected_cost = self.dual_controller.expected_cost(
            probabilities=probabilities,
            c_heavy=self.config.routing.cost_heavy,
            c_cheap=self.config.routing.cost_cheap,
        )
        if not isinstance(expected_cost, Tensor):
            raise TypeError("expected_cost should be a Tensor for tensor probabilities")

        budget_target = float(self.config.routing.budget_total) * float(bsz)
        budget_loss = self.dual_controller.budget_loss(expected_cost, budget=budget_target)
        if not isinstance(budget_loss, Tensor):
            raise TypeError("budget_loss should be a Tensor for tensor expected_cost")

        if update_mu:
            self.dual_controller.update_mu(
                expected_cost=float(expected_cost.detach().cpu().item()),
                budget=budget_target,
            )
            self.mu_history.append(self.dual_controller.mu)

        selected_count_l0 = sum(len(indices) for indices in selected_l0)
        diagnostics = build_routing_diagnostics(
            utilities_by_level={
                "l0": [float(value) for value in utility_logits.detach().reshape(-1).tolist()],
            },
            selected_counts={"l0": selected_count_l0, "l1": 0, "l2": 0},
            total_counts={"l0": bsz * k_tiles, "l1": 0, "l2": 0},
            budget_used={
                "b1": float(expected_cost.detach().cpu().item()),
                "b2": 0.0,
                "b3": 0.0,
                "total": float(expected_cost.detach().cpu().item()),
            },
            budget_total={
                "b1": float(self.config.routing.budget_b1) * float(bsz),
                "b2": float(self.config.routing.budget_b2) * float(bsz),
                "b3": float(self.config.routing.budget_b3) * float(bsz),
                "total": budget_target,
            },
            mu_history=self.mu_history,
        )
        diagnostics["grid"] = {"l0_h": grid_h, "l0_w": grid_w}

        return FFTrainOutput(
            heavy_features=heavy_features,
            detail_map=detail_map,
            alpha=alpha,
            probabilities=probabilities,
            gates=gates,
            expected_cost=expected_cost,
            budget_loss=budget_loss,
            selected_l0=selected_l0,
            diagnostics=diagnostics,
            mu=self.dual_controller.mu,
        )

    def forward_infer(
        self,
        dense_features: Tensor,
        utilities: Tensor,
        *,
        split_utilities: Tensor | None = None,
        boundary_proxy: Tensor | None = None,
        uncertainty_proxy: Tensor | None = None,
        enable_nesting: bool | None = None,
    ) -> FFInferOutput:
        if dense_features.ndim != 4:
            raise ValueError("dense_features must be [B,C,H,W]")
        if dense_features.shape[1] != self.channels:
            raise ValueError("dense_features channel dimension does not match module channels")

        bsz = dense_features.shape[0]
        grid_h, grid_w, k_tiles = self._grid_for_l0(dense_features)
        _validate_utility_tensor("utilities", utilities, batch=bsz, tiles=k_tiles)

        nesting = self.config.model.effective_nesting_depth() >= 1
        if enable_nesting is not None:
            nesting = nesting and bool(enable_nesting)

        if split_utilities is not None:
            _validate_utility_tensor("split_utilities", split_utilities, batch=bsz, tiles=k_tiles)

        delta_cost = max(
            float(self.config.routing.cost_heavy) - float(self.config.routing.cost_cheap),
            1e-9,
        )
        l0_costs = [delta_cost for _ in range(k_tiles)]

        selected_l0: list[list[int]] = []
        selected_l1: list[list[int]] = []
        l0_kmax_buffers: list[list[int]] = []
        l1_kmax_buffers: list[list[int]] = []
        spent_b1_total = 0.0
        spent_b2_total = 0.0

        for b in range(bsz):
            utility_row = [float(value) for value in utilities[b].detach().cpu().tolist()]
            if nesting and split_utilities is not None:
                split_row = [float(value) for value in split_utilities[b].detach().cpu().tolist()]
                split_overheads = [1.0 for _ in range(k_tiles)]
                two_stage = deterministic_two_stage_selection(
                    l0_utilities=utility_row,
                    l0_delta_costs=l0_costs,
                    split_utilities=split_row,
                    split_overheads=split_overheads,
                    budget_b1=float(self.config.routing.budget_b1),
                    budget_b2=float(self.config.routing.budget_b2),
                    kmax_l0=int(self.config.model.kmax_l0),
                    kmax_l1=int(self.config.model.kmax_l1),
                    l0_grid_h=grid_h,
                    l0_grid_w=grid_w,
                    l1_order_mode="hilbert",
                )
                selected_l0.append([int(idx) for idx in two_stage.l0.selected_indices])
                selected_l1.append([int(idx) for idx in two_stage.l1_ordered_indices])
                l0_kmax_buffers.append([int(idx) for idx in two_stage.l0.kmax_buffer])
                l1_kmax_buffers.append([int(idx) for idx in two_stage.l1_kmax_buffer])
                spent_b1_total += float(two_stage.l0.spent_budget)
                spent_b2_total += float(two_stage.split_spent_budget)
            else:
                l0_result = deterministic_greedy_selection(
                    utilities=utility_row,
                    delta_costs=l0_costs,
                    budget=float(self.config.routing.budget_b1),
                    kmax=int(self.config.model.kmax_l0),
                )
                selected_l0.append([int(idx) for idx in l0_result.selected_indices])
                selected_l1.append([])
                l0_kmax_buffers.append([int(idx) for idx in l0_result.kmax_buffer])
                l1_kmax_buffers.append([-1 for _ in range(int(self.config.model.kmax_l1))])
                spent_b1_total += float(l0_result.spent_budget)

        heavy_l0, _detail_l0, alpha_l0 = self._run_path_per_sample(
            self.l0_path,
            dense_features,
            selected_l0,
            boundary_proxy=boundary_proxy,
            uncertainty_proxy=uncertainty_proxy,
        )

        if nesting and split_utilities is not None and any(selected_l1):
            heavy_features, detail_map, alpha = self._run_path_per_sample(
                self.l1_path,
                heavy_l0,
                selected_l1,
                boundary_proxy=boundary_proxy,
                uncertainty_proxy=uncertainty_proxy,
            )
        else:
            heavy_features = heavy_l0
            detail_map = heavy_l0 - dense_features
            alpha = alpha_l0

        l1_total_tiles = 0
        if nesting and split_utilities is not None:
            l1_h, l1_w = tile_grid_shape(
                dense_features.shape[2],
                dense_features.shape[3],
                self.config.model.tile_size_l1,
            )
            l1_total_tiles = bsz * l1_h * l1_w

        utilities_l1_hist: list[float]
        if split_utilities is None:
            utilities_l1_hist = []
        else:
            utilities_l1_hist = [
                float(value) for value in split_utilities.detach().reshape(-1).tolist()
            ]

        diagnostics = build_routing_diagnostics(
            utilities_by_level={
                "l0": [float(value) for value in utilities.detach().reshape(-1).tolist()],
                "l1": utilities_l1_hist,
            },
            selected_counts={
                "l0": sum(len(indices) for indices in selected_l0),
                "l1": sum(len(indices) for indices in selected_l1),
                "l2": 0,
            },
            total_counts={"l0": bsz * k_tiles, "l1": l1_total_tiles, "l2": 0},
            budget_used={
                "b1": spent_b1_total,
                "b2": spent_b2_total,
                "b3": 0.0,
                "total": spent_b1_total + spent_b2_total,
            },
            budget_total={
                "b1": float(self.config.routing.budget_b1) * float(bsz),
                "b2": float(self.config.routing.budget_b2) * float(bsz),
                "b3": float(self.config.routing.budget_b3) * float(bsz),
                "total": float(self.config.routing.budget_total) * float(bsz),
            },
            mu_history=self.mu_history,
        )
        diagnostics["grid"] = {"l0_h": grid_h, "l0_w": grid_w}
        diagnostics["nesting_enabled"] = bool(nesting and split_utilities is not None)

        return FFInferOutput(
            heavy_features=heavy_features,
            detail_map=detail_map,
            alpha=alpha,
            selected_l0=selected_l0,
            selected_l1=selected_l1,
            l0_kmax_buffers=l0_kmax_buffers,
            l1_kmax_buffers=l1_kmax_buffers,
            spent_budget_b1=spent_b1_total,
            spent_budget_b2=spent_b2_total,
            diagnostics=diagnostics,
        )
