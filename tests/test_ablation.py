from __future__ import annotations

import csv
from pathlib import Path

from apex_x import ApexXConfig
from apex_x.train import build_ablation_grid, run_ablation_grid, write_ablation_reports


def test_build_ablation_grid_respects_modes_and_limit() -> None:
    grid = build_ablation_grid(
        router="both",
        budgeting="on",
        nesting="off",
        ssm="on",
        distill="off",
        pcgrad="on",
        qat="off",
        panoptic="off",
        tracking="off",
        max_experiments=8,
    )
    assert len(grid) == 2
    assert {entry.router for entry in grid} == {False, True}
    assert all(entry.budgeting for entry in grid)
    assert all(not entry.nesting for entry in grid)


def test_ablation_run_and_reports_smoke(tmp_path: Path) -> None:
    cfg = ApexXConfig()
    grid = build_ablation_grid(
        router="on",
        budgeting="on",
        nesting="on",
        ssm="on",
        distill="on",
        pcgrad="on",
        qat="off",
        panoptic="on",
        tracking="on",
        max_experiments=1,
    )
    _per_seed, aggregates = run_ablation_grid(
        base_config=cfg,
        toggles_grid=grid,
        seeds=[0],
        steps_per_stage=1,
    )
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.runs == 1
    assert 0.0 <= aggregate.det_map_mean <= 1.0
    assert 0.0 <= aggregate.mask_map_mean <= 1.0
    assert 0.0 <= aggregate.semantic_miou_mean <= 1.0
    assert 0.0 <= aggregate.panoptic_pq_mean <= 1.0
    assert 0.0 <= aggregate.tracking_id_consistency_mean <= 1.0

    csv_path, md_path = write_ablation_reports(
        aggregates=aggregates,
        output_csv=tmp_path / "ablation.csv",
        output_markdown=tmp_path / "ablation.md",
    )
    assert csv_path.exists()
    assert md_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert "det_map_mean" in rows[0]
    assert "budget_ratio_total_mean" in rows[0]

    md_text = md_path.read_text(encoding="utf-8")
    assert "Apex-X Ablation Report" in md_text
    assert "det mAP" in md_text
