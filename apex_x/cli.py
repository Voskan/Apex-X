from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from apex_x.bench import bench_placeholder
from apex_x.config import ApexXConfig, load_yaml_config
from apex_x.data import dummy_batch
from apex_x.export import ApexXExporter, ShapeMode
from apex_x.infer import (
    evaluate_fixture_file,
    evaluate_fixture_payload,
    evaluate_model_dataset,
    load_eval_dataset_npz,
    run_model_inference,
    tiny_eval_fixture_payload,
    write_eval_reports,
)
from apex_x.model import (
    ApexXModel,
    DetHead,
    DualPathFPN,
    PVModule,
    TeacherModel,
)
from apex_x.runtime import RuntimeCaps, detect_runtime_caps
from apex_x.train import (
    ApexXTrainer,
    ToggleMode,
    build_ablation_grid,
    run_ablation_grid,
    write_ablation_reports,
)
from apex_x.utils import get_logger, log_event, seed_all

LOGGER = get_logger(__name__)

_BACKEND_CANDIDATES = ("cpu", "torch", "triton", "tensorrt")
_FALLBACK_POLICY_CANDIDATES = ("strict", "permissive")
_SHAPE_MODE_CANDIDATES = ("static", "dynamic")
_CLI_IMPLEMENTED_BACKENDS = set(_BACKEND_CANDIDATES)
_BACKEND_FALLBACK_CHAIN: dict[str, tuple[str, ...]] = {
    "cpu": ("cpu",),
    "torch": ("torch", "cpu"),
    "triton": ("triton", "torch", "cpu"),
    "tensorrt": ("tensorrt", "triton", "torch", "cpu"),
}


@dataclass(frozen=True, slots=True)
class BackendDecision:
    requested_backend: str
    selected_backend: str
    fallback_policy: str
    fallback_reason: str | None
    caps: RuntimeCaps


def _normalize_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _BACKEND_CANDIDATES:
        expected = "/".join(_BACKEND_CANDIDATES)
        raise ValueError(f"invalid backend {value!r}; expected {expected}")
    return normalized


def _normalize_fallback_policy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _FALLBACK_POLICY_CANDIDATES:
        expected = "/".join(_FALLBACK_POLICY_CANDIDATES)
        raise ValueError(f"invalid fallback policy {value!r}; expected {expected}")
    return normalized


def _normalize_shape_mode(value: str) -> ShapeMode:
    normalized = value.strip().lower()
    if normalized not in _SHAPE_MODE_CANDIDATES:
        expected = "/".join(_SHAPE_MODE_CANDIDATES)
        raise ValueError(f"invalid shape mode {value!r}; expected {expected}")
    return cast(ShapeMode, normalized)


def _backend_runtime_availability(backend: str, caps: RuntimeCaps) -> tuple[bool, str | None]:
    if backend == "cpu":
        return True, None
    if backend == "torch":
        if caps.cuda.available:
            return True, None
        return False, caps.cuda.reason or "cuda_unavailable"
    if backend == "triton":
        if not caps.cuda.available:
            return False, caps.cuda.reason or "cuda_unavailable"
        if caps.triton.available:
            return True, None
        return False, caps.triton.reason or "triton_unavailable"
    if backend == "tensorrt":
        if not caps.cuda.available:
            return False, caps.cuda.reason or "cuda_unavailable"
        if caps.tensorrt.python_available:
            return True, None
        return False, caps.tensorrt.python_reason or "tensorrt_python_unavailable"
    raise ValueError(f"unsupported backend: {backend}")


def _backend_cli_supported(backend: str) -> tuple[bool, str | None]:
    if backend in _CLI_IMPLEMENTED_BACKENDS:
        return True, None
    return False, "backend_not_implemented_in_cli_path"


def _resolve_backend_decision(
    cfg: ApexXConfig,
    *,
    backend_override: str | None,
    fallback_policy_override: str | None,
) -> BackendDecision:
    requested_backend = (
        _normalize_backend(backend_override)
        if backend_override
        else _normalize_backend(cfg.runtime.backend)
    )
    fallback_policy = (
        _normalize_fallback_policy(fallback_policy_override)
        if fallback_policy_override
        else _normalize_fallback_policy(cfg.runtime.fallback_policy)
    )

    caps = detect_runtime_caps()
    runtime_ok, runtime_reason = _backend_runtime_availability(requested_backend, caps)
    cli_ok, cli_reason = _backend_cli_supported(requested_backend)

    if runtime_ok and cli_ok:
        return BackendDecision(
            requested_backend=requested_backend,
            selected_backend=requested_backend,
            fallback_policy=fallback_policy,
            fallback_reason=None,
            caps=caps,
        )

    fallback_reason = runtime_reason or cli_reason or "backend_unavailable"
    if fallback_policy == "strict":
        raise ValueError(
            f"requested backend '{requested_backend}' is unavailable: {fallback_reason}. "
            "Use --fallback-policy permissive to allow fallback."
        )

    for candidate in _BACKEND_FALLBACK_CHAIN[requested_backend]:
        runtime_candidate_ok, _ = _backend_runtime_availability(candidate, caps)
        cli_candidate_ok, _ = _backend_cli_supported(candidate)
        if runtime_candidate_ok and cli_candidate_ok:
            return BackendDecision(
                requested_backend=requested_backend,
                selected_backend=candidate,
                fallback_policy=fallback_policy,
                fallback_reason=fallback_reason,
                caps=caps,
            )

    raise ValueError(
        f"no available backend for request '{requested_backend}' "
        f"under fallback policy '{fallback_policy}'"
    )


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
        raise ValueError(f"invalid toggle mode {value!r}; expected on/off/both")
    return cast(ToggleMode, normalized)


def train_cmd(
    config: str,
    set_values: list[str] | None = None,
    steps_per_stage: int = 1,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
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
    print(
        "train ok "
        f"profile={cfg.model.profile} "
        f"budget_b1={cfg.routing.budget_b1} "
        f"loss={result.loss_proxy:.4f} "
        f"stage_count={stage_count} "
        f"selected_ratio_l0={selected_ratio_l0:.4f} "
        f"mu_last={mu_last:.4f}"
    )


def eval_cmd(
    config: str,
    set_values: list[str] | None = None,
    backend: str | None = None,
    fallback_policy: str | None = None,
    fixture: str | None = None,
    report_json: str = "artifacts/eval_report.json",
    report_md: str = "artifacts/eval_report.md",
    dataset_npz: str | None = None,
    max_samples: int | None = None,
    panoptic_pq: bool = False,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
    seed_all(0, deterministic=cfg.runtime.deterministic)
    backend_decision = _resolve_backend_decision(
        cfg,
        backend_override=backend,
        fallback_policy_override=fallback_policy,
    )

    fixture_path = Path(fixture) if fixture else None
    report_json_path = Path(report_json)
    report_md_path = Path(report_md)

    if fixture is None:
        summary = evaluate_fixture_payload(tiny_eval_fixture_payload())
        fixture_source = "builtin_tiny"
    else:
        summary = evaluate_fixture_file(fixture_path)
        fixture_source = str(fixture_path)

    json_path, md_path = write_eval_reports(
        summary,
        json_path=report_json_path,
        markdown_path=report_md_path,
    )

    model = ApexXModel(config=cfg)
    inference = run_model_inference(
        model=model,
        input_batch=_inference_input(cfg),
        requested_backend=backend_decision.requested_backend,
        selected_backend=backend_decision.selected_backend,
        fallback_policy=backend_decision.fallback_policy,
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=backend_decision.fallback_reason,
        runtime_caps=backend_decision.caps,
    )
    dataset_eval = None
    if dataset_npz is not None:
        dataset = load_eval_dataset_npz(
            path=Path(dataset_npz),
            expected_height=cfg.model.input_height,
            expected_width=cfg.model.input_width,
        )
        dataset_eval = evaluate_model_dataset(
            model=model,
            images=dataset.images,
            requested_backend=backend_decision.requested_backend,
            selected_backend=backend_decision.selected_backend,
            fallback_policy=backend_decision.fallback_policy,
            precision_profile=cfg.runtime.precision_profile,
            selection_fallback_reason=backend_decision.fallback_reason,
            runtime_caps=backend_decision.caps,
            det_score_target=dataset.det_score_target,
            selected_tiles_target=dataset.selected_tiles_target,
            max_samples=max_samples,
        )

    report_payload = cast(dict[str, object], json.loads(json_path.read_text(encoding="utf-8")))
    report_payload["runtime"] = inference.runtime.to_dict()
    if dataset_eval is not None:
        report_payload["model_eval"] = dataset_eval.to_dict()
    json_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if dataset_eval is not None:
        with md_path.open("a", encoding="utf-8") as f:
            f.write("\n\n## Model Dataset Eval\n\n")
            f.write(f"- source: `{dataset_eval.source}`\n")
            f.write(f"- samples: `{dataset_eval.num_samples}`\n")
            f.write(
                "- det_score: "
                f"`mean={dataset_eval.det_score_mean:.6f}`, "
                f"`std={dataset_eval.det_score_std:.6f}`, "
                f"`min={dataset_eval.det_score_min:.6f}`, "
                f"`max={dataset_eval.det_score_max:.6f}`\n"
            )
            f.write(
                "- selected_tiles: "
                f"`mean={dataset_eval.selected_tiles_mean:.3f}`, "
                f"`p95={dataset_eval.selected_tiles_p95:.3f}`\n"
            )
            if dataset_eval.det_score_target_metrics is not None:
                target = dataset_eval.det_score_target_metrics
                r2_text = (
                    "n/a" if target["r2"] is None else f"{target['r2']:.6f}"
                )
                corr_text = (
                    "n/a"
                    if target["pearson_corr"] is None
                    else f"{target['pearson_corr']:.6f}"
                )
                f.write(
                    "- det_score_target: "
                    f"`mae={target['mae']:.6f}`, "
                    f"`rmse={target['rmse']:.6f}`, "
                    f"`bias={target['bias']:.6f}`, "
                    f"`r2={r2_text}`, "
                    f"`pearson_corr={corr_text}`\n"
                )
            if dataset_eval.selected_tiles_target_metrics is not None:
                tiles_target = dataset_eval.selected_tiles_target_metrics
                f.write(
                    "- selected_tiles_target: "
                    f"`mae={tiles_target['mae']:.6f}`, "
                    f"`rmse={tiles_target['rmse']:.6f}`, "
                    f"`bias={tiles_target['bias']:.6f}`, "
                    f"`exact_match_rate={tiles_target['exact_match_rate']:.6f}`\n"
                )
            f.write(f"- execution_backend: `{dataset_eval.execution_backend}`\n")
            f.write(f"- precision_profile: `{dataset_eval.precision_profile}`\n")

    log_event(
        LOGGER,
        "eval_command",
        fields={
            "fixture_source": fixture_source,
            "runtime": inference.runtime.to_dict(),
            "model_eval": dataset_eval.to_dict() if dataset_eval is not None else None,
            "det_score": round(inference.det_score, 6),
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
    print(
        "eval ok "
        f"backend={inference.runtime.execution_backend} "
        f"requested_backend={inference.runtime.requested_backend} "
        f"selected_backend={inference.runtime.selected_backend} "
        f"fallback_reason={inference.runtime.selection_fallback_reason or 'none'} "
        f"exec_fallback={inference.runtime.execution_fallback_reason or 'none'} "
        f"precision={inference.runtime.precision_profile} "
        f"latency_ms={inference.runtime.latency_ms['total']:.2f} "
        f"det_score={inference.det_score:.4f} "
        f"det_map={summary.det_map:.4f} "
        f"mask_map={summary.mask_map:.4f} "
        f"miou={summary.semantic_miou:.4f} "
        f"panoptic_pq={summary.panoptic_pq:.4f} "
        f"model_eval_samples={dataset_eval.num_samples if dataset_eval is not None else 0} "
        f"report_json={json_path} "
        f"report_md={md_path}"
    )


def predict_cmd(
    config: str,
    set_values: list[str] | None = None,
    backend: str | None = None,
    fallback_policy: str | None = None,
    report_json: str | None = None,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
    seed_all(0, deterministic=cfg.runtime.deterministic)
    backend_decision = _resolve_backend_decision(
        cfg,
        backend_override=backend,
        fallback_policy_override=fallback_policy,
    )

    model = ApexXModel(config=cfg)
    inference = run_model_inference(
        model=model,
        input_batch=_inference_input(cfg),
        requested_backend=backend_decision.requested_backend,
        selected_backend=backend_decision.selected_backend,
        fallback_policy=backend_decision.fallback_policy,
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=backend_decision.fallback_reason,
        runtime_caps=backend_decision.caps,
    )
    selected = inference.selected_tiles

    infer_diag = inference.routing_diagnostics
    b1_ratio = (
        float(infer_diag.get("budget_usage", {}).get("b1", {}).get("ratio", 0.0))
        if isinstance(infer_diag.get("budget_usage", {}).get("b1", {}), dict)
        else 0.0
    )
    log_event(
        LOGGER,
        "predict_command",
        fields={
            "runtime": inference.runtime.to_dict(),
            "selected_tiles": selected,
            "b1_budget_ratio": round(b1_ratio, 6),
        },
    )
    if report_json is not None:
        report_json_path = Path(report_json)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runtime": inference.runtime.to_dict(),
            "selected_tiles": int(selected),
            "det_score": float(inference.det_score),
            "routing_diagnostics": infer_diag,
        }
        report_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "predict ok "
        f"backend={inference.runtime.execution_backend} "
        f"requested_backend={inference.runtime.requested_backend} "
        f"selected_backend={inference.runtime.selected_backend} "
        f"fallback_reason={inference.runtime.selection_fallback_reason or 'none'} "
        f"exec_fallback={inference.runtime.execution_fallback_reason or 'none'} "
        f"precision={inference.runtime.precision_profile} "
        f"latency_ms={inference.runtime.latency_ms['total']:.2f} "
        f"selected_tiles={selected}"
    )


def bench_cmd(
    config: str,
    set_values: list[str] | None = None,
    iters: int = 20,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
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
    print(f"bench ok iters={iters} p50_ms={p50:.4f} p95_ms={p95:.4f}")


def ablate_cmd(
    config: str,
    set_values: list[str] | None = None,
    name: str = "default",
    router: str = "both",
    budgeting: str = "both",
    nesting: str = "both",
    ssm: str = "both",
    distill: str = "both",
    pcgrad: str = "both",
    qat: str = "both",
    panoptic: str = "both",
    tracking: str = "both",
    seed_values: list[int] | None = None,
    steps_per_stage: int = 1,
    max_experiments: int = 32,
    output_csv: str = "artifacts/ablation_report.csv",
    output_md: str = "artifacts/ablation_report.md",
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
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
        output_csv=Path(output_csv),
        output_markdown=Path(output_md),
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
    print(
        "ablate ok "
        f"name={name} "
        f"profile={cfg.model.profile} "
        f"experiments={len(grid)} "
        f"seeds={len(seeds)} "
        f"output_csv={csv_path} "
        f"output_md={md_path}"
    )


def _build_teacher_for_export(config: ApexXConfig, num_classes: int) -> TeacherModel:
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
        config=config,
        pv_module=pv_module,
        fpn=fpn,
        det_head=det_head,
        feature_layers=("P3", "P4"),
        use_ema=True,
        ema_decay=0.99,
        use_ema_for_forward=False,
    )


def export_cmd(
    config: str,
    set_values: list[str] | None = None,
    output: str = "artifacts/apex_x_export_manifest.json",
    shape_mode: str = "static",
    checkpoint: str | None = None,
    num_classes: int = 3,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
    
    # Export TeacherModel (Dense Detector) as it is the trainable/deployable artifact
    model = _build_teacher_for_export(cfg, num_classes=num_classes)
    
    if checkpoint:
        import torch
        ckpt_path = Path(checkpoint)
        log_event(LOGGER, "export_loading_checkpoint", fields={"path": str(ckpt_path)})
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    model.eval()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exporter = ApexXExporter(shape_mode=_normalize_shape_mode(shape_mode))
    exported_path = exporter.export(model=model, output_path=str(output_path))

    log_event(
        LOGGER,
        "export_command",
        fields={"output": exported_path, "shape_mode": _normalize_shape_mode(shape_mode)},
    )
    print(f"export ok output={exported_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Apex-X CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("config", help="Path to config")
    train_parser.add_argument("--set", "-s", action="append", dest="set_values")
    train_parser.add_argument("--steps-per-stage", type=int, default=1)

    # Eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("config", help="Path to config")
    eval_parser.add_argument("--set", "-s", action="append", dest="set_values")
    eval_parser.add_argument("--backend")
    eval_parser.add_argument("--fallback-policy")
    eval_parser.add_argument("--fixture")
    eval_parser.add_argument("--report-json", default="artifacts/eval_report.json")
    eval_parser.add_argument("--report-md", default="artifacts/eval_report.md")
    eval_parser.add_argument("--dataset-npz")
    eval_parser.add_argument("--max-samples", type=int)
    eval_parser.add_argument("--panoptic-pq", action="store_true")

    # Predict
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("config", help="Path to config")
    predict_parser.add_argument("--set", "-s", action="append", dest="set_values")
    predict_parser.add_argument("--backend")
    predict_parser.add_argument("--fallback-policy")
    predict_parser.add_argument("--report-json")

    # Bench
    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("config", help="Path to config")
    bench_parser.add_argument("--set", "-s", action="append", dest="set_values")
    bench_parser.add_argument("--iters", type=int, default=20)

    # Ablate
    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("config", help="Path to config")
    ablate_parser.add_argument("--set", "-s", action="append", dest="set_values")
    ablate_parser.add_argument("--name", default="default")
    ablate_parser.add_argument("--router", default="both")
    ablate_parser.add_argument("--budgeting", default="both")
    ablate_parser.add_argument("--nesting", default="both")
    ablate_parser.add_argument("--ssm", default="both")
    ablate_parser.add_argument("--distill", default="both")
    ablate_parser.add_argument("--pcgrad", default="both")
    ablate_parser.add_argument("--qat", default="both")
    ablate_parser.add_argument("--panoptic", default="both")
    ablate_parser.add_argument("--tracking", default="both")
    ablate_parser.add_argument("--seed", action="append", dest="seed_values", type=int)
    ablate_parser.add_argument("--steps-per-stage", type=int, default=1)
    ablate_parser.add_argument("--max-experiments", type=int, default=32)
    ablate_parser.add_argument("--output-csv", default="artifacts/ablation_report.csv")
    ablate_parser.add_argument("--output-md", default="artifacts/ablation_report.md")

    # Export
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("config", help="Path to config")
    export_parser.add_argument("--set", "-s", action="append", dest="set_values")
    export_parser.add_argument("--output", "-o", default="artifacts/apex_x_export_manifest.json")
    export_parser.add_argument("--shape-mode", default="static")
    export_parser.add_argument("--checkpoint", help="Path to model checkpoint .pt file")
    export_parser.add_argument("--num-classes", type=int, default=3)

    args = parser.parse_args()

    if args.command == "train":
        train_cmd(args.config, args.set_values, args.steps_per_stage)
    elif args.command == "eval":
        eval_cmd(
            args.config,
            args.set_values,
            args.backend,
            args.fallback_policy,
            args.fixture,
            args.report_json,
            args.report_md,
            args.dataset_npz,
            args.max_samples,
            args.panoptic_pq,
        )
    elif args.command == "predict":
        predict_cmd(
            args.config,
            args.set_values,
            args.backend,
            args.fallback_policy,
            args.report_json,
        )
    elif args.command == "bench":
        bench_cmd(args.config, args.set_values, args.iters)
    elif args.command == "ablate":
        ablate_cmd(
            args.config,
            args.set_values,
            args.name,
            args.router,
            args.budgeting,
            args.nesting,
            args.ssm,
            args.distill,
            args.pcgrad,
            args.qat,
            args.panoptic,
            args.tracking,
            args.seed_values,
            args.steps_per_stage,
            args.max_experiments,
            args.output_csv,
            args.output_md,
        )
    elif args.command == "export":
        export_cmd(
            args.config,
            args.set_values,
            args.output,
            args.shape_mode,
            args.checkpoint,
            args.num_classes,
        )


if __name__ == "__main__":
    main()
