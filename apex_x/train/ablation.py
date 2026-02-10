"""Ablation grid runner for Apex-X."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from apex_x.config import ApexXConfig
from apex_x.infer import evaluate_fixture_payload, tiny_eval_fixture_payload
from apex_x.infer.tracking import HungarianAssociator

from .trainer import ApexXTrainer

ToggleMode = Literal["on", "off", "both"]

_TOGGLE_NAMES: tuple[str, ...] = (
    "router",
    "budgeting",
    "nesting",
    "ssm",
    "distill",
    "pcgrad",
    "qat",
    "panoptic",
    "tracking",
)


@dataclass(frozen=True, slots=True)
class AblationToggleSet:
    router: bool
    budgeting: bool
    nesting: bool
    ssm: bool
    distill: bool
    pcgrad: bool
    qat: bool
    panoptic: bool
    tracking: bool

    def key(self) -> tuple[bool, ...]:
        return (
            self.router,
            self.budgeting,
            self.nesting,
            self.ssm,
            self.distill,
            self.pcgrad,
            self.qat,
            self.panoptic,
            self.tracking,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "router": int(self.router),
            "budgeting": int(self.budgeting),
            "nesting": int(self.nesting),
            "ssm": int(self.ssm),
            "distill": int(self.distill),
            "pcgrad": int(self.pcgrad),
            "qat": int(self.qat),
            "panoptic": int(self.panoptic),
            "tracking": int(self.tracking),
        }


@dataclass(frozen=True, slots=True)
class AblationRunRecord:
    toggles: AblationToggleSet
    seed: int
    det_map: float
    mask_map: float
    semantic_miou: float
    panoptic_pq: float
    tracking_id_consistency: float
    loss_proxy: float
    selected_ratio_l0: float
    selected_ratio_l1: float
    budget_ratio_total: float
    mu_last: float


@dataclass(frozen=True, slots=True)
class AblationAggregateRecord:
    toggles: AblationToggleSet
    runs: int
    det_map_mean: float
    mask_map_mean: float
    semantic_miou_mean: float
    panoptic_pq_mean: float
    tracking_id_consistency_mean: float
    loss_proxy_mean: float
    selected_ratio_l0_mean: float
    selected_ratio_l1_mean: float
    budget_ratio_total_mean: float
    mu_last_mean: float

    def to_csv_row(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {}
        row.update(self.toggles.as_dict())
        row["runs"] = int(self.runs)
        row["det_map_mean"] = self.det_map_mean
        row["mask_map_mean"] = self.mask_map_mean
        row["semantic_miou_mean"] = self.semantic_miou_mean
        row["panoptic_pq_mean"] = self.panoptic_pq_mean
        row["tracking_id_consistency_mean"] = self.tracking_id_consistency_mean
        row["loss_proxy_mean"] = self.loss_proxy_mean
        row["selected_ratio_l0_mean"] = self.selected_ratio_l0_mean
        row["selected_ratio_l1_mean"] = self.selected_ratio_l1_mean
        row["budget_ratio_total_mean"] = self.budget_ratio_total_mean
        row["mu_last_mean"] = self.mu_last_mean
        return row


def _mode_to_values(mode: str) -> tuple[bool, ...]:
    normalized = mode.strip().lower()
    if normalized == "on":
        return (True,)
    if normalized == "off":
        return (False,)
    if normalized == "both":
        return (False, True)
    raise ValueError(f"invalid toggle mode {mode!r}; expected on/off/both")


def build_ablation_grid(
    *,
    router: ToggleMode,
    budgeting: ToggleMode,
    nesting: ToggleMode,
    ssm: ToggleMode,
    distill: ToggleMode,
    pcgrad: ToggleMode,
    qat: ToggleMode,
    panoptic: ToggleMode,
    tracking: ToggleMode,
    max_experiments: int,
) -> list[AblationToggleSet]:
    if max_experiments <= 0:
        raise ValueError("max_experiments must be > 0")
    grid: list[AblationToggleSet] = []
    for values in product(
        _mode_to_values(router),
        _mode_to_values(budgeting),
        _mode_to_values(nesting),
        _mode_to_values(ssm),
        _mode_to_values(distill),
        _mode_to_values(pcgrad),
        _mode_to_values(qat),
        _mode_to_values(panoptic),
        _mode_to_values(tracking),
    ):
        grid.append(AblationToggleSet(*values))
    grid.sort(key=AblationToggleSet.key)
    return grid[:max_experiments]


def _tracking_proxy_consistency(seed: int) -> float:
    torch.manual_seed(seed)
    associator = HungarianAssociator(
        iou_gate=0.15,
        embedding_distance_gate=0.4,
        iou_weight=0.4,
        embedding_weight=0.6,
        max_age=2,
        memory_bank_size=4,
    )
    scores = torch.tensor([0.9, 0.9], dtype=torch.float32)
    boxes_t0 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]], dtype=torch.float32)
    emb_t0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    out0 = associator.associate(boxes_t0, emb_t0, scores, None, frame_index=0)

    boxes_t1 = torch.tensor([[16.0, 0.0, 26.0, 10.0], [4.0, 0.0, 14.0, 10.0]], dtype=torch.float32)
    emb_t1 = torch.tensor([[0.02, 0.98], [0.98, 0.02]], dtype=torch.float32)
    out1 = associator.associate(boxes_t1, emb_t1, scores, out0.next_state, frame_index=1)

    pairs: dict[int, int] = {}
    for det_idx, track_idx in zip(
        out1.matched_detection_indices.tolist(),
        out1.matched_track_indices.tolist(),
        strict=True,
    ):
        pairs[int(det_idx)] = int(out0.next_state.track_ids[int(track_idx)].item())
    for det_idx, track_id in zip(
        out1.new_detection_indices.tolist(),
        out1.created_track_ids.tolist(),
        strict=True,
    ):
        pairs[int(det_idx)] = int(track_id)

    expected = {0: 1, 1: 0}
    correct = sum(1 for key, value in expected.items() if pairs.get(key) == value)
    return float(correct / len(expected))


def _apply_toggles(base: ApexXConfig, toggles: AblationToggleSet) -> ApexXConfig:
    cfg = ApexXConfig.from_dict(base.to_dict())
    cfg.model.force_dense_routing = not toggles.router
    cfg.model.disable_nesting = not toggles.nesting
    cfg.model.disable_ssm = not toggles.ssm
    cfg.train.disable_distill = not toggles.distill
    cfg.train.disable_pcgradpp = not toggles.pcgrad
    cfg.train.qat_enable = bool(toggles.qat)
    cfg.train.qat_int8 = bool(toggles.qat)
    cfg.train.qat_fp8 = False
    cfg.validate()
    return cfg


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def run_ablation_grid(
    *,
    base_config: ApexXConfig,
    toggles_grid: list[AblationToggleSet],
    seeds: list[int],
    steps_per_stage: int,
) -> tuple[list[AblationRunRecord], list[AblationAggregateRecord]]:
    if not seeds:
        raise ValueError("seeds must not be empty")

    per_seed_records: list[AblationRunRecord] = []
    for toggles in toggles_grid:
        cfg = _apply_toggles(base_config, toggles)
        eval_summary = evaluate_fixture_payload(tiny_eval_fixture_payload())
        for seed in seeds:
            trainer = ApexXTrainer(config=cfg)
            staged = trainer.run(
                steps_per_stage=steps_per_stage,
                seed=seed,
                enable_budgeting=toggles.budgeting,
            )
            diag = staged.routing_diagnostics
            selected_ratio_l0 = _as_float(diag.get("selected_ratios", {}).get("l0"))
            selected_ratio_l1 = _as_float(diag.get("selected_ratios", {}).get("l1"))
            budget_ratio_total = _as_float(
                diag.get("budget_usage", {}).get("total", {}).get("ratio")
            )
            mu_history = diag.get("mu_history", [])
            mu_last_raw = (
                mu_history[-1] if isinstance(mu_history, list) and mu_history else float("nan")
            )
            mu_last = _as_float(mu_last_raw)

            panoptic_pq = eval_summary.panoptic_pq if toggles.panoptic else 0.0
            tracking_score = _tracking_proxy_consistency(seed) if toggles.tracking else 0.0
            per_seed_records.append(
                AblationRunRecord(
                    toggles=toggles,
                    seed=int(seed),
                    det_map=eval_summary.det_map,
                    mask_map=eval_summary.mask_map,
                    semantic_miou=eval_summary.semantic_miou,
                    panoptic_pq=panoptic_pq,
                    tracking_id_consistency=tracking_score,
                    loss_proxy=float(staged.loss_proxy),
                    selected_ratio_l0=selected_ratio_l0,
                    selected_ratio_l1=selected_ratio_l1,
                    budget_ratio_total=budget_ratio_total,
                    mu_last=mu_last,
                ),
            )

    grouped: dict[tuple[bool, ...], list[AblationRunRecord]] = {}
    for record in per_seed_records:
        grouped.setdefault(record.toggles.key(), []).append(record)

    aggregates: list[AblationAggregateRecord] = []
    for key in sorted(grouped):
        group = grouped[key]
        toggles = group[0].toggles
        aggregates.append(
            AblationAggregateRecord(
                toggles=toggles,
                runs=len(group),
                det_map_mean=_mean(rec.det_map for rec in group),
                mask_map_mean=_mean(rec.mask_map for rec in group),
                semantic_miou_mean=_mean(rec.semantic_miou for rec in group),
                panoptic_pq_mean=_mean(rec.panoptic_pq for rec in group),
                tracking_id_consistency_mean=_mean(rec.tracking_id_consistency for rec in group),
                loss_proxy_mean=_mean(rec.loss_proxy for rec in group),
                selected_ratio_l0_mean=_mean(rec.selected_ratio_l0 for rec in group),
                selected_ratio_l1_mean=_mean(rec.selected_ratio_l1 for rec in group),
                budget_ratio_total_mean=_mean(rec.budget_ratio_total for rec in group),
                mu_last_mean=_mean(rec.mu_last for rec in group),
            ),
        )
    return per_seed_records, aggregates


def write_ablation_reports(
    *,
    aggregates: list[AblationAggregateRecord],
    output_csv: str | Path,
    output_markdown: str | Path,
) -> tuple[Path, Path]:
    csv_path = Path(output_csv)
    md_path = Path(output_markdown)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        *list(_TOGGLE_NAMES),
        "runs",
        "det_map_mean",
        "mask_map_mean",
        "semantic_miou_mean",
        "panoptic_pq_mean",
        "tracking_id_consistency_mean",
        "loss_proxy_mean",
        "selected_ratio_l0_mean",
        "selected_ratio_l1_mean",
        "budget_ratio_total_mean",
        "mu_last_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in aggregates:
            writer.writerow(record.to_csv_row())

    lines = [
        "# Apex-X Ablation Report",
        "",
        "| Toggles (router,budgeting,nesting,ssm,distill,pcgrad,qat,panoptic,tracking) | "
        "det mAP | mask mAP | mIoU | PQ | Track | L0 sel | budget ratio | mu |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in aggregates:
        bits = ",".join(str(int(v)) for v in record.toggles.key())
        lines.append(
            f"| `{bits}` | "
            f"{record.det_map_mean:.4f} | "
            f"{record.mask_map_mean:.4f} | "
            f"{record.semantic_miou_mean:.4f} | "
            f"{record.panoptic_pq_mean:.4f} | "
            f"{record.tracking_id_consistency_mean:.4f} | "
            f"{record.selected_ratio_l0_mean:.4f} | "
            f"{record.budget_ratio_total_mean:.4f} | "
            f"{record.mu_last_mean:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


__all__ = [
    "ToggleMode",
    "AblationToggleSet",
    "AblationRunRecord",
    "AblationAggregateRecord",
    "build_ablation_grid",
    "run_ablation_grid",
    "write_ablation_reports",
]
