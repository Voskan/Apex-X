"""Staged training flow for Apex-X v4 CPU baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from apex_x.config import ApexXConfig
from apex_x.model import ApexXModel, DetHead, DualPathFPN, PVModule, TeacherModel
from apex_x.routing import (
    BudgetDualController,
    build_routing_diagnostics,
    compute_oracle_delta_targets,
    deterministic_greedy_selection,
    deterministic_two_stage_selection,
    sample_oracle_set,
    ste_gate_from_utilities,
    utility_oracle_loss,
)
from apex_x.runtime import heavy_ops_autocast_context, resolve_precision_policy
from apex_x.tiles import tile_grid_shape
from apex_x.utils import get_logger, log_event, seed_all

from .qat import QuantizationSummary, prepare_int8_ptq, prepare_int8_qat

LOGGER = get_logger(__name__)

StageMetricValue = float | int | bool | str | None


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_id: int
    name: str
    metrics: Mapping[str, StageMetricValue]


@dataclass(frozen=True, slots=True)
class StagedTrainResult:
    stage_results: tuple[StageResult, ...]
    routing_diagnostics: dict[str, Any]
    train_summary: dict[str, Any]
    loss_proxy: float
    final_mu: float


class ApexXTrainer:
    """Implements stage-0..4 trainer flow defined in Apex-X engineering docs."""

    def __init__(self, config: ApexXConfig | None = None, *, num_classes: int = 3) -> None:
        self.config = config or ApexXConfig()
        self.config.validate()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = int(num_classes)
        self.baseline_model = ApexXModel(config=self.config)
        self.teacher = self._build_teacher_model(num_classes=self.num_classes)
        self.teacher.train()

        self.dual_controller = BudgetDualController(
            budget=self.config.routing.budget_total,
            mu_init=self.config.train.mu_init,
            mu_lr=self.config.train.mu_lr,
            mu_min=self.config.train.mu_min,
            mu_max=self.config.train.mu_max,
            logger_name="train.staged.dual",
        )
        self.mu_history: list[float] = [float(self.dual_controller.mu)]
        self.quantization_summary = QuantizationSummary.disabled()
        self._quantization_prepared = False
        self.precision_policy = resolve_precision_policy(self.config)

    def _build_teacher_model(self, *, num_classes: int) -> TeacherModel:
        # Keep the staged trainer CPU-fast by using a compact teacher backbone.
        pv_module = PVModule(
            in_channels=3,
            p3_channels=16,
            p4_channels=24,
            p5_channels=32,
            coarse_level="P4",
        )
        fpn = DualPathFPN(
            pv_p3_channels=16,
            pv_p4_channels=24,
            pv_p5_channels=32,
            ff_channels=16,
            out_channels=16,
        )
        det_head = DetHead(
            in_channels=16,
            num_classes=num_classes,
            hidden_channels=16,
            depth=1,
        )
        return TeacherModel(
            num_classes=num_classes,
            config=self.config,
            pv_module=pv_module,
            fpn=fpn,
            det_head=det_head,
            feature_layers=("P3", "P4"),
            use_ema=True,
            ema_decay=0.99,
            use_ema_for_forward=False,
        )

    def _l0_grid(self) -> tuple[int, int, int]:
        ff_h = self.config.model.input_height // self.config.model.ff_primary_stride
        ff_w = self.config.model.input_width // self.config.model.ff_primary_stride
        grid_h, grid_w = tile_grid_shape(ff_h, ff_w, self.config.model.tile_size_l0)
        return grid_h, grid_w, grid_h * grid_w

    def _stage0_baseline_warmup(self, rng: np.random.RandomState, *, steps: int) -> StageResult:
        det_score_last = 0.0
        selected_total = 0
        for _ in range(steps):
            image = rng.rand(
                1,
                3,
                self.config.model.input_height,
                self.config.model.input_width,
            ).astype(np.float32)
            out = self.baseline_model.forward(image)
            det_score_last = float(out["det"]["scores"][0])
            selected_total += len(out["selected_tiles"])

        mean_selected = selected_total / max(steps, 1)
        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "det_score_last": det_score_last,
            "selected_tiles_mean": float(mean_selected),
        }
        return StageResult(stage_id=0, name="baseline_warmup", metrics=metrics)

    def _stage1_teacher_training(
        self,
        generator: torch.Generator,
        *,
        steps: int,
    ) -> tuple[StageResult, float]:
        optimizer = torch.optim.AdamW(self.teacher.parameters(), lr=1e-3)
        train_h = min(self.config.model.input_height, 64)
        train_w = min(self.config.model.input_width, 64)
        if train_h <= 0 or train_w <= 0:
            raise ValueError("invalid train input shape")

        loss_last = 0.0
        for _ in range(steps):
            image = torch.rand((1, 3, train_h, train_w), generator=generator, dtype=torch.float32)
            with heavy_ops_autocast_context(self.precision_policy):
                out = self.teacher(image, use_ema=False)
                loss = out.logits.square().mean() + 0.05 * out.boundaries.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.teacher.update_ema()
            loss_last = float(loss.detach().item())

        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "teacher_loss_last": loss_last,
            "full_compute_mode": bool(self.teacher.full_compute_mode),
            "qat_wrapped_modules": self.quantization_summary.wrapped_modules,
            "heavy_ops_dtype": self.precision_policy.to_dict()["heavy_ops_dtype"],
            "fp8_enabled": self.precision_policy.fp8_enabled,
        }
        return StageResult(stage_id=1, name="teacher_full_compute", metrics=metrics), loss_last

    def _build_calibration_inputs(
        self,
        generator: torch.Generator,
        *,
        batches: int,
    ) -> list[Tensor]:
        if batches <= 0:
            raise ValueError("batches must be > 0")
        train_h = min(self.config.model.input_height, 64)
        train_w = min(self.config.model.input_width, 64)
        return [
            torch.rand((1, 3, train_h, train_w), generator=generator, dtype=torch.float32)
            for _ in range(batches)
        ]

    def _prepare_quantization(self, generator: torch.Generator) -> None:
        if self._quantization_prepared:
            return
        cfg = self.config
        if cfg.train.qat_enable and cfg.train.qat_int8:
            self.quantization_summary = prepare_int8_qat(self.teacher)
        elif cfg.runtime.precision_profile == "edge":
            calibration_inputs = self._build_calibration_inputs(generator, batches=4)
            self.quantization_summary = prepare_int8_ptq(
                self.teacher,
                calibration_inputs=calibration_inputs,
                forward_fn=lambda module, batch: cast(TeacherModel, module)(
                    batch,
                    use_ema=False,
                ),
            )
        else:
            self.quantization_summary = QuantizationSummary.disabled()
        self._quantization_prepared = True
        log_event(
            LOGGER,
            "quantization_prepare",
            level="DEBUG",
            fields={
                "mode": self.quantization_summary.mode,
                "wrapped_modules": self.quantization_summary.wrapped_modules,
                "calibration_batches": self.quantization_summary.calibration_batches,
                "router_gating_fp16": self.quantization_summary.router_gating_fp16,
                "precision_policy": self.precision_policy.to_dict(),
            },
        )

    def _stage2_oracle_bootstrapping(
        self,
        rng: np.random.RandomState,
        generator: torch.Generator,
        *,
        steps: int,
    ) -> tuple[StageResult, float]:
        _, _, k_tiles = self._l0_grid()
        if k_tiles <= 0:
            raise ValueError("k_tiles must be > 0")

        loss_last = 0.0
        sampled_count_last = 0
        for step in range(steps):
            uncertainty = np.abs(rng.standard_normal(k_tiles)).tolist()
            sampled = sample_oracle_set(
                uncertainty,
                random_fraction=0.2,
                uncertainty_fraction=0.2,
                seed=int(rng.randint(0, 10_000_000)),
            )
            sampled_count_last = len(sampled.indices)
            sampled_idx = torch.as_tensor(sampled.indices, dtype=torch.int64).unsqueeze(0)

            cheap = 0.4 + torch.rand((1, k_tiles), generator=generator, dtype=torch.float32)
            heavy_noise = 0.2 * torch.rand(
                (1, k_tiles),
                generator=generator,
                dtype=torch.float32,
            )
            heavy = (cheap - heavy_noise).clamp(min=0.0)
            oracle_targets = compute_oracle_delta_targets(
                cheap_distill_loss=cheap,
                heavy_distill_loss=heavy,
                sampled_tile_indices=sampled_idx,
                clamp_abs=2.0,
            )

            utility_logits = torch.randn((1, k_tiles), generator=generator, dtype=torch.float32)
            utility_logits.requires_grad_(True)
            oracle_loss = utility_oracle_loss(
                utility_logits=utility_logits,
                targets=oracle_targets,
                regression_weight=1.0,
                ranking_weight=0.25,
                regression_type="smooth_l1",
            )
            oracle_loss.total_loss.backward()
            loss_last = float(oracle_loss.total_loss.detach().item())

            log_event(
                LOGGER,
                "stage2_oracle_step",
                level="DEBUG",
                fields={
                    "step": step,
                    "sampled_tiles": sampled_count_last,
                    "oracle_loss": round(loss_last, 6),
                },
            )

        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "oracle_loss_last": loss_last,
            "sampled_tiles_last": int(sampled_count_last),
        }
        return StageResult(stage_id=2, name="oracle_bootstrap", metrics=metrics), loss_last

    def _stage3_continuous_budgeting(
        self,
        generator: torch.Generator,
        *,
        steps: int,
    ) -> tuple[StageResult, float, list[float]]:
        _, _, k_tiles = self._l0_grid()
        mu_before = float(self.dual_controller.mu)
        expected_cost_last = 0.0
        budget_loss_last = 0.0
        utility_snapshot: list[float] = []

        for _ in range(steps):
            utilities = torch.randn((1, k_tiles), generator=generator, dtype=torch.float32)
            utilities.requires_grad_(True)
            utilities_fp16 = utilities.to(dtype=torch.float16)
            probabilities, _gates = ste_gate_from_utilities(
                utilities_fp16,
                threshold=self.config.routing.theta_on,
                mode="threshold",
            )
            expected_cost = self.dual_controller.expected_cost(
                probabilities=probabilities.to(dtype=torch.float32),
                c_heavy=self.config.routing.cost_heavy,
                c_cheap=self.config.routing.cost_cheap,
            )
            if not isinstance(expected_cost, Tensor):
                raise TypeError("expected_cost must be Tensor for tensor probabilities")
            budget_loss = self.dual_controller.budget_loss(
                expected_cost=expected_cost,
                budget=float(self.config.routing.budget_total),
            )
            if not isinstance(budget_loss, Tensor):
                raise TypeError("budget_loss must be Tensor for tensor expected_cost")

            total_loss = utilities.square().mean() + budget_loss
            total_loss.backward()

            expected_cost_last = float(expected_cost.detach().item())
            budget_loss_last = float(budget_loss.detach().item())
            utility_snapshot = [float(value) for value in utilities.detach().reshape(-1).tolist()]
            self.dual_controller.update_mu(
                expected_cost=expected_cost_last,
                budget=float(self.config.routing.budget_total),
            )
            self.mu_history.append(float(self.dual_controller.mu))

        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "expected_cost_last": expected_cost_last,
            "budget_loss_last": budget_loss_last,
            "mu_before": mu_before,
            "mu_after": float(self.dual_controller.mu),
        }
        return (
            StageResult(stage_id=3, name="continuous_budgeting", metrics=metrics),
            budget_loss_last,
            utility_snapshot,
        )

    def _stage4_deterministic_emulation(
        self,
        utility_snapshot: list[float],
    ) -> tuple[StageResult, dict[str, Any]]:
        grid_h, grid_w, k_tiles = self._l0_grid()
        if len(utility_snapshot) != k_tiles:
            raise ValueError("utility_snapshot length must match tile count")

        delta_cost = max(self.config.routing.cost_heavy - self.config.routing.cost_cheap, 1e-9)
        delta_costs = [float(delta_cost) for _ in range(k_tiles)]

        if self.config.model.effective_nesting_depth() >= 1:
            split_utilities = [max(float(value), 0.0) + 1e-4 for value in utility_snapshot]
            split_overheads = [1.0 for _ in range(k_tiles)]
            two_stage = deterministic_two_stage_selection(
                l0_utilities=utility_snapshot,
                l0_delta_costs=delta_costs,
                split_utilities=split_utilities,
                split_overheads=split_overheads,
                budget_b1=float(self.config.routing.budget_b1),
                budget_b2=float(self.config.routing.budget_b2),
                kmax_l0=int(self.config.model.kmax_l0),
                kmax_l1=int(self.config.model.kmax_l1),
                l0_grid_h=grid_h,
                l0_grid_w=grid_w,
                l1_order_mode="hilbert",
            )
            l0_selected = two_stage.l0.selected_indices
            l1_selected = two_stage.l1_ordered_indices
            spent_b1 = float(two_stage.l0.spent_budget)
            spent_b2 = float(two_stage.split_spent_budget)
        else:
            greedy = deterministic_greedy_selection(
                utilities=utility_snapshot,
                delta_costs=delta_costs,
                budget=float(self.config.routing.budget_b1),
                kmax=int(self.config.model.kmax_l0),
            )
            l0_selected = greedy.selected_indices
            l1_selected = []
            spent_b1 = float(greedy.spent_budget)
            spent_b2 = 0.0

        l1_total = k_tiles * 4 if self.config.model.effective_nesting_depth() >= 1 else 0
        routing_diag = build_routing_diagnostics(
            utilities_by_level={"l0": utility_snapshot, "l1": []},
            selected_counts={"l0": len(l0_selected), "l1": len(l1_selected), "l2": 0},
            total_counts={"l0": k_tiles, "l1": l1_total, "l2": 0},
            budget_used={
                "b1": spent_b1,
                "b2": spent_b2,
                "b3": 0.0,
                "total": spent_b1 + spent_b2,
            },
            budget_total={
                "b1": float(self.config.routing.budget_b1),
                "b2": float(self.config.routing.budget_b2),
                "b3": float(self.config.routing.budget_b3),
                "total": float(self.config.routing.budget_total),
            },
            mu_history=self.mu_history,
        )

        metrics: dict[str, StageMetricValue] = {
            "selected_l0": int(len(l0_selected)),
            "selected_l1": int(len(l1_selected)),
            "spent_b1": spent_b1,
            "spent_b2": spent_b2,
        }
        return (
            StageResult(stage_id=4, name="deterministic_emulation", metrics=metrics),
            routing_diag,
        )

    def run(
        self,
        *,
        steps_per_stage: int = 1,
        seed: int = 0,
        enable_budgeting: bool = True,
    ) -> StagedTrainResult:
        if steps_per_stage <= 0:
            raise ValueError("steps_per_stage must be > 0")

        seed_all(seed=seed, deterministic=self.config.runtime.deterministic)
        rng = np.random.RandomState(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self._prepare_quantization(generator)

        stage0 = self._stage0_baseline_warmup(rng, steps=steps_per_stage)
        stage1, stage1_loss = self._stage1_teacher_training(generator, steps=steps_per_stage)
        stage2, stage2_loss = self._stage2_oracle_bootstrapping(
            rng,
            generator,
            steps=steps_per_stage,
        )
        if enable_budgeting:
            stage3, stage3_loss, utility_snapshot = self._stage3_continuous_budgeting(
                generator,
                steps=steps_per_stage,
            )
        else:
            _, _, k_tiles = self._l0_grid()
            utility_snapshot = [float(v) for v in rng.standard_normal(k_tiles).tolist()]
            stage3 = StageResult(
                stage_id=3,
                name="continuous_budgeting",
                metrics={
                    "steps": int(steps_per_stage),
                    "expected_cost_last": 0.0,
                    "budget_loss_last": 0.0,
                    "mu_before": float(self.dual_controller.mu),
                    "mu_after": float(self.dual_controller.mu),
                    "budgeting_enabled": False,
                },
            )
            stage3_loss = 0.0
        stage4, routing_diag = self._stage4_deterministic_emulation(utility_snapshot)

        train_summary = {
            "routing_diagnostics": routing_diag,
            "training_toggles": {
                "distill_enabled": self.config.train.distill_enabled(),
                "pcgradpp_enabled": self.config.train.pcgradpp_enabled(),
            },
            "quantization": {
                "mode": self.quantization_summary.mode,
                "wrapped_modules": self.quantization_summary.wrapped_modules,
                "calibration_batches": self.quantization_summary.calibration_batches,
                "router_gating_fp16": self.quantization_summary.router_gating_fp16,
            },
            "precision": self.precision_policy.to_dict(),
        }
        loss_proxy = float(stage1_loss + stage2_loss + abs(stage3_loss))
        result = StagedTrainResult(
            stage_results=(stage0, stage1, stage2, stage3, stage4),
            routing_diagnostics=routing_diag,
            train_summary=train_summary,
            loss_proxy=loss_proxy,
            final_mu=float(self.dual_controller.mu),
        )

        log_event(
            LOGGER,
            "staged_training_complete",
            fields={
                "stages": len(result.stage_results),
                "loss_proxy": round(result.loss_proxy, 6),
                "final_mu": round(result.final_mu, 6),
            },
        )
        return result


__all__ = [
    "StageResult",
    "StagedTrainResult",
    "ApexXTrainer",
]
