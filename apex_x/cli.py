from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

import numpy as np
import typer

from apex_x.bench import bench_placeholder
from apex_x.config import ApexXConfig, load_yaml_config
from apex_x.data import dummy_batch
from apex_x.export import NoopExporter
from apex_x.infer import (
    evaluate_fixture_file,
    evaluate_fixture_payload,
    infer_placeholder,
    tiny_eval_fixture_payload,
    write_eval_reports,
)
from apex_x.model import ApexXModel
from apex_x.train import (
    ApexXTrainer,
    ToggleMode,
    build_ablation_grid,
    run_ablation_grid,
    write_ablation_reports,
)
from apex_x.utils import get_logger, log_event, seed_all

app = typer.Typer(help="Apex-X command line interface", no_args_is_help=True)
LOGGER = get_logger(__name__)


def _load_config(config_path: Path, overrides: list[str]) -> ApexXConfig:
    cfg = load_yaml_config(path=config_path, overrides=overrides)
    log_event(
        LOGGER,
        "cli_config_loaded",
        level="DEBUG",
        fields={"path": str(config_path), "override_count": len(overrides)},
    )
    return cfg


def _inference_input(cfg: ApexXConfig) -> np.ndarray:
    return dummy_batch(height=cfg.model.input_height, width=cfg.model.input_width)


def _normalize_toggle_mode(value: str) -> ToggleMode:
    normalized = value.strip().lower()
    if normalized not in {"on", "off", "both"}:
        raise typer.BadParameter(f"invalid toggle mode {value!r}; expected on/off/both")
    return cast(ToggleMode, normalized)


ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to Apex-X YAML config.",
    ),
]

SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override config value via dot path, e.g. --set model.profile=base",
    ),
]


@app.command("train")
def train_cmd(
    config: ConfigOption,
    set_values: SetOption = None,
    steps_per_stage: Annotated[
        int,
        typer.Option(
            "--steps-per-stage",
            min=1,
            help="Number of lightweight optimization iterations per training stage.",
        ),
    ] = 1,
) -> None:
    overrides = set_values or []
    cfg = _load_config(config, overrides)
    seed_all(0, deterministic=cfg.runtime.deterministic)

    trainer = ApexXTrainer(config=cfg)
    result = trainer.run(steps_per_stage=steps_per_stage, seed=0)
    train_summary = result.train_summary
    routing_diag = train_summary.get("routing_diagnostics", {})
    selected_ratio_l0 = (
        float(routing_diag.get("selected_ratios", {}).get("l0", 0.0))
        if isinstance(routing_diag, dict)
        else 0.0
    )
    mu_history = routing_diag.get("mu_history", []) if isinstance(routing_diag, dict) else []
    mu_last = float(mu_history[-1]) if isinstance(mu_history, list) and mu_history else 0.0
    stage_count = len(result.stage_results)

    log_event(
        LOGGER,
        "train_command",
        fields={
            "profile": cfg.model.profile,
            "budget_b1": cfg.routing.budget_b1,
            "stage_count": stage_count,
            "selected_ratio_l0": round(selected_ratio_l0, 6),
            "mu_last": round(mu_last, 6),
            "mu_steps": len(mu_history) if isinstance(mu_history, list) else 0,
        },
    )
    typer.echo(
        "train ok "
        f"profile={cfg.model.profile} "
        f"budget_b1={cfg.routing.budget_b1} "
        f"loss={result.loss_proxy:.4f} "
        f"stage_count={stage_count} "
        f"selected_ratio_l0={selected_ratio_l0:.4f} "
        f"mu_last={mu_last:.4f}"
    )


@app.command("eval")
def eval_cmd(
    config: ConfigOption,
    set_values: SetOption = None,
    fixture: Annotated[
        Path | None,
        typer.Option(
            "--fixture",
            help="Path to evaluation fixture JSON. If omitted, uses built-in tiny fixture.",
        ),
    ] = None,
    report_json: Annotated[
        Path,
        typer.Option(
            "--report-json",
            help="Output path for evaluation JSON report.",
        ),
    ] = Path("artifacts/eval_report.json"),
    report_md: Annotated[
        Path,
        typer.Option(
            "--report-md",
            help="Output path for evaluation markdown report.",
        ),
    ] = Path("artifacts/eval_report.md"),
    panoptic_pq: Annotated[
        bool,
        typer.Option(
            "--panoptic-pq",
            help="Deprecated compatibility flag; PQ is always included in eval report.",
        ),
    ] = False,
) -> None:
    overrides = set_values or []
    cfg = _load_config(config, overrides)
    seed_all(0, deterministic=cfg.runtime.deterministic)

    if fixture is None:
        summary = evaluate_fixture_payload(tiny_eval_fixture_payload())
        fixture_source = "builtin_tiny"
    else:
        summary = evaluate_fixture_file(fixture)
        fixture_source = str(fixture)

    json_path, md_path = write_eval_reports(
        summary,
        json_path=report_json,
        markdown_path=report_md,
    )

    model = ApexXModel(config=cfg)
    out = model.forward(_inference_input(cfg))
    det_score = float(out["det"]["scores"][0])
    log_event(
        LOGGER,
        "eval_command",
        fields={
            "fixture_source": fixture_source,
            "det_score": round(det_score, 6),
            "det_map": round(summary.det_map, 6),
            "mask_map": round(summary.mask_map, 6),
            "semantic_miou": round(summary.semantic_miou, 6),
            "panoptic_pq": round(summary.panoptic_pq, 6),
            "panoptic_source": summary.panoptic_source,
            "report_json": str(json_path),
            "report_md": str(md_path),
            "compat_panoptic_flag": bool(panoptic_pq),
        },
    )
    typer.echo(
        "eval ok "
        f"det_score={det_score:.4f} "
        f"det_map={summary.det_map:.4f} "
        f"mask_map={summary.mask_map:.4f} "
        f"miou={summary.semantic_miou:.4f} "
        f"panoptic_pq={summary.panoptic_pq:.4f} "
        f"report_json={json_path} "
        f"report_md={md_path}"
    )


@app.command("predict")
def predict_cmd(config: ConfigOption, set_values: SetOption = None) -> None:
    overrides = set_values or []
    cfg = _load_config(config, overrides)
    seed_all(0, deterministic=cfg.runtime.deterministic)

    model = ApexXModel(config=cfg)
    out = model.forward(_inference_input(cfg))
    selected = len(out["selected_tiles"])

    infer_diag = infer_placeholder(out)
    b1_ratio = (
        float(infer_diag.get("budget_usage", {}).get("b1", {}).get("ratio", 0.0))
        if isinstance(infer_diag.get("budget_usage", {}).get("b1", {}), dict)
        else 0.0
    )
    log_event(
        LOGGER,
        "predict_command",
        fields={"selected_tiles": selected, "b1_budget_ratio": round(b1_ratio, 6)},
    )
    typer.echo(f"predict ok selected_tiles={selected}")


@app.command("bench")
def bench_cmd(
    config: ConfigOption,
    set_values: SetOption = None,
    iters: Annotated[int, typer.Option("--iters", min=1)] = 20,
) -> None:
    overrides = set_values or []
    cfg = _load_config(config, overrides)
    seed_all(0, deterministic=cfg.runtime.deterministic)

    model = ApexXModel(config=cfg)
    x = _inference_input(cfg)

    timings_ms: list[float] = []
    for _ in range(iters):
        t0 = perf_counter()
        model.forward(x)
        timings_ms.append((perf_counter() - t0) * 1000.0)

    bench_placeholder()
    p50 = float(np.median(np.asarray(timings_ms)))
    p95 = float(np.percentile(np.asarray(timings_ms), 95))

    log_event(LOGGER, "bench_command", fields={"iters": iters, "p50_ms": p50, "p95_ms": p95})
    typer.echo(f"bench ok iters={iters} p50_ms={p50:.4f} p95_ms={p95:.4f}")


@app.command("ablate")
def ablate_cmd(
    config: ConfigOption,
    set_values: SetOption = None,
    name: Annotated[str, typer.Option("--name", help="Ablation experiment name")] = "default",
    router: Annotated[str, typer.Option("--router", help="Toggle mode: on/off/both")] = "both",
    budgeting: Annotated[
        str, typer.Option("--budgeting", help="Toggle mode: on/off/both")
    ] = "both",
    nesting: Annotated[str, typer.Option("--nesting", help="Toggle mode: on/off/both")] = "both",
    ssm: Annotated[str, typer.Option("--ssm", help="Toggle mode: on/off/both")] = "both",
    distill: Annotated[str, typer.Option("--distill", help="Toggle mode: on/off/both")] = "both",
    pcgrad: Annotated[str, typer.Option("--pcgrad", help="Toggle mode: on/off/both")] = "both",
    qat: Annotated[str, typer.Option("--qat", help="Toggle mode: on/off/both")] = "both",
    panoptic: Annotated[
        str, typer.Option("--panoptic", help="Toggle mode: on/off/both")
    ] = "both",
    tracking: Annotated[
        str, typer.Option("--tracking", help="Toggle mode: on/off/both")
    ] = "both",
    seed_values: Annotated[
        list[int] | None,
        typer.Option("--seed", help="Fixed random seed(s). Repeat --seed for multiple values."),
    ] = None,
    steps_per_stage: Annotated[
        int, typer.Option("--steps-per-stage", min=1, help="Trainer stage steps per ablation run.")
    ] = 1,
    max_experiments: Annotated[
        int, typer.Option("--max-experiments", min=1, help="Maximum grid combinations to execute.")
    ] = 32,
    output_csv: Annotated[
        Path,
        typer.Option("--output-csv", help="Output CSV path for aggregated ablation metrics."),
    ] = Path("artifacts/ablation_report.csv"),
    output_md: Annotated[
        Path,
        typer.Option("--output-md", help="Output markdown path for ablation summary."),
    ] = Path("artifacts/ablation_report.md"),
) -> None:
    overrides = set_values or []
    cfg = _load_config(config, overrides)
    seeds = [0, 1] if not seed_values else [int(value) for value in seed_values]
    router_mode = _normalize_toggle_mode(router)
    budgeting_mode = _normalize_toggle_mode(budgeting)
    nesting_mode = _normalize_toggle_mode(nesting)
    ssm_mode = _normalize_toggle_mode(ssm)
    distill_mode = _normalize_toggle_mode(distill)
    pcgrad_mode = _normalize_toggle_mode(pcgrad)
    qat_mode = _normalize_toggle_mode(qat)
    panoptic_mode = _normalize_toggle_mode(panoptic)
    tracking_mode = _normalize_toggle_mode(tracking)

    grid = build_ablation_grid(
        router=router_mode,
        budgeting=budgeting_mode,
        nesting=nesting_mode,
        ssm=ssm_mode,
        distill=distill_mode,
        pcgrad=pcgrad_mode,
        qat=qat_mode,
        panoptic=panoptic_mode,
        tracking=tracking_mode,
        max_experiments=max_experiments,
    )
    _per_seed, aggregates = run_ablation_grid(
        base_config=cfg,
        toggles_grid=grid,
        seeds=seeds,
        steps_per_stage=steps_per_stage,
    )
    csv_path, md_path = write_ablation_reports(
        aggregates=aggregates,
        output_csv=output_csv,
        output_markdown=output_md,
    )

    log_event(
        LOGGER,
        "ablate_command",
        fields={
            "name": name,
            "profile": cfg.model.profile,
            "num_seeds": len(seeds),
            "num_experiments": len(grid),
            "steps_per_stage": steps_per_stage,
            "output_csv": str(csv_path),
            "output_md": str(md_path),
        },
    )
    typer.echo(
        "ablate ok "
        f"name={name} "
        f"profile={cfg.model.profile} "
        f"experiments={len(grid)} "
        f"seeds={len(seeds)} "
        f"output_csv={csv_path} "
        f"output_md={md_path}"
    )


@app.command("export")
def export_cmd(
    config: ConfigOption,
    set_values: SetOption = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file for exported artifact"),
    ] = Path("artifacts/apex_x_export.txt"),
) -> None:
    overrides = set_values or []
    cfg = _load_config(config, overrides)
    model = ApexXModel(config=cfg)

    output.parent.mkdir(parents=True, exist_ok=True)
    exporter = NoopExporter()
    exported_path = exporter.export(model=model, output_path=str(output))

    log_event(LOGGER, "export_command", fields={"output": exported_path})
    typer.echo(f"export ok output={exported_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
