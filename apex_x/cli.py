from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from apex_x.config import ApexXConfig, load_yaml_config
from apex_x.data import dummy_batch, run_dataset_preflight, write_dataset_preflight_report
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
from apex_x.model.worldclass_deps import missing_worldclass_dependencies, worldclass_install_hint
from apex_x.runtime import RuntimeCaps, detect_runtime_caps
from apex_x.train import (
    ApexXTrainer,
    ToggleMode,
    build_ablation_grid,
    run_ablation_grid,
    write_ablation_reports,
)
from apex_x.train.checkpoint import extract_model_state_dict, safe_torch_load
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
    seed: int = 0,
    num_classes: int = 3,
    checkpoint_dir: str | None = None,
    resume: str | None = None,
    dataset_path: str | None = None,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
    seed_all(seed, deterministic=cfg.runtime.deterministic)

    trainer = ApexXTrainer(
        config=cfg,
        num_classes=num_classes,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
    )
    if resume:
        trainer.load_training_checkpoint(checkpoint_path=resume, device="cpu")

    result = trainer.run(
        steps_per_stage=steps_per_stage,
        seed=seed,
        dataset_path=dataset_path,
    )
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
            "seed": seed,
            "num_classes": num_classes,
            "dataset_path": dataset_path or "",
        },
    )
    print(
        "train ok "
        f"profile={cfg.model.profile} "
        f"budget_b1={cfg.routing.budget_b1} "
        f"loss={result.loss_proxy:.4f} "
        f"stage_count={stage_count} "
        f"selected_ratio_l0={selected_ratio_l0:.4f} "
        f"mu_last={mu_last:.4f} "
        f"seed={seed}"
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
    tta: bool = False,
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
        use_tta=tta,
    )
    dataset_eval = None
    if dataset_npz is not None:
        dataset = load_eval_dataset_npz(
            path=Path(dataset_npz),
            expected_height=cfg.model.input_height,
            expected_width=cfg.model.input_width,
        )
        # Note: evaluate_model_dataset doesn't easily support TTA yet without refactor.
        # It calls model internally. We'll leave it for now and focus on fixture inference.
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

    # ... (rest of function) ...

def predict_cmd(
    config: str,
    set_values: list[str] | None = None,
    backend: str | None = None,
    fallback_policy: str | None = None,
    report_json: str | None = None,
    tta: bool = False,
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
        use_tta=tta,
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
            "use_tta": tta,
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
            "use_tta": bool(tta),
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
        f"selected_tiles={selected} "
        f"tta={tta}"
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
    strict_load: bool = True,
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)

    # Export TeacherModel (Dense Detector) as it is the trainable/deployable artifact
    model = _build_teacher_for_export(cfg, num_classes=num_classes)

    if checkpoint:
        import torch

        ckpt_path = Path(checkpoint)
        log_event(LOGGER, "export_loading_checkpoint", fields={"path": str(ckpt_path)})
        checkpoint_payload = safe_torch_load(ckpt_path, map_location="cpu")
        state_dict, checkpoint_format = extract_model_state_dict(checkpoint_payload)
        cls_weight = state_dict.get("det_head.cls_pred.weight")
        if isinstance(cls_weight, torch.Tensor) and cls_weight.ndim >= 1:
            inferred_num_classes = int(cls_weight.shape[0])
            if inferred_num_classes > 0 and inferred_num_classes != model.num_classes:
                log_event(
                    LOGGER,
                    "export_checkpoint_num_classes_override",
                    fields={
                        "requested_num_classes": model.num_classes,
                        "checkpoint_num_classes": inferred_num_classes,
                    },
                )
                model = _build_teacher_for_export(cfg, num_classes=inferred_num_classes)
        skipped_shape_mismatch: list[str] = []
        if strict_load:
            try:
                incompatible = model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc}\nUse --no-strict-load to ignore incompatible keys during export."
                ) from exc
        else:
            model_state = model.state_dict()
            filtered_state: dict[str, torch.Tensor] = {}
            for key, value in state_dict.items():
                expected = model_state.get(key)
                if expected is None:
                    continue
                if tuple(expected.shape) != tuple(value.shape):
                    skipped_shape_mismatch.append(key)
                    continue
                filtered_state[key] = value
            incompatible = model.load_state_dict(filtered_state, strict=False)
        log_event(
            LOGGER,
            "export_checkpoint_loaded",
            fields={
                "path": str(ckpt_path),
                "checkpoint_format": checkpoint_format,
                "num_params": len(state_dict),
                "strict_load": bool(strict_load),
                "missing_keys": list(getattr(incompatible, "missing_keys", [])),
                "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
                "skipped_shape_mismatch_keys": skipped_shape_mismatch,
            },
        )

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


def preflight_cmd(profile: str = "worldclass") -> None:
    normalized = str(profile).strip().lower()
    if normalized != "worldclass":
        raise ValueError("preflight profile must be 'worldclass'")
    missing = missing_worldclass_dependencies()
    if missing:
        print(f"preflight fail profile=worldclass missing={','.join(missing)}")
        print(worldclass_install_hint())
        raise SystemExit(1)
    print("preflight ok profile=worldclass")


def dataset_preflight_cmd(
    config: str,
    set_values: list[str] | None = None,
    dataset_path: str | None = None,
    output_json: str = "artifacts/dataset_preflight.json",
) -> None:
    overrides = set_values or []
    cfg = _load_config(Path(config), overrides)
    report = run_dataset_preflight(cfg, dataset_path=dataset_path)
    out_path = write_dataset_preflight_report(report, path=output_json)
    status = "ok" if report.passed else "fail"
    print(
        f"dataset_preflight {status} "
        f"type={report.dataset_type} "
        f"errors={len(report.errors)} "
        f"warnings={len(report.warnings)} "
        f"output={out_path}"
    )
    if not report.passed:
        for err in report.errors[:10]:
            print(f"- {err}")
        raise SystemExit(1)


def evolve_cmd(
    config: str,
    dataset_path: str | None = None,
    gens: int = 10,
    pop: int = 5,
) -> None:
    from apex_x.train.evolve import HyperparameterEvolution, EvoParams
    cfg = _load_config(Path(config), [])
    params = EvoParams(gens=gens, pop_size=pop)
    evo = HyperparameterEvolution(base_config=cfg, dataset_path=dataset_path, params=params)
    evo.run()
    print("evolve ok")


def refine_cmd(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    checkpoint: str,
    device: str,
) -> None:
    from apex_x.tools.refine_sam2 import refine_dataset
    refine_dataset(image_dir, label_dir, output_dir, checkpoint=checkpoint, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apex-X CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("config", help="Path to config")
    train_parser.add_argument("--set", "-s", action="append", dest="set_values")
    train_parser.add_argument("--steps-per-stage", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--num-classes", type=int, default=3)
    train_parser.add_argument("--checkpoint-dir")
    train_parser.add_argument("--resume")
    train_parser.add_argument("--dataset-path")

    # Evolve
    evolve_parser = subparsers.add_parser("evolve")
    evolve_parser.add_argument("config", help="Path to config")
    evolve_parser.add_argument("--dataset-path")
    evolve_parser.add_argument("--gens", type=int, default=10)
    evolve_parser.add_argument("--pop", type=int, default=5)

    # Refine (SAM-2)
    refine_parser = subparsers.add_parser("refine")
    refine_parser.add_argument("--images", required=True, help="Path to image directory")
    refine_parser.add_argument("--labels", required=True, help="Path to bbox label directory")
    refine_parser.add_argument("--output", required=True, help="Path to output directory")
    refine_parser.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt")
    refine_parser.add_argument("--device", default="cuda")

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
    eval_parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation (TTA)")

    # Predict
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("config", help="Path to config")
    predict_parser.add_argument("--set", "-s", action="append", dest="set_values")
    predict_parser.add_argument("--backend")
    predict_parser.add_argument("--fallback-policy")
    predict_parser.add_argument("--report-json")
    predict_parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation (TTA)")

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
    export_parser.add_argument(
        "--strict-load",
        dest="strict_load",
        action="store_true",
        default=True,
        help="Load checkpoint with strict=True",
    )
    export_parser.add_argument(
        "--no-strict-load",
        dest="strict_load",
        action="store_false",
        help="Load checkpoint with strict=False",
    )

    # Preflight
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--profile", default="worldclass")

    dataset_preflight_parser = subparsers.add_parser("dataset-preflight")
    dataset_preflight_parser.add_argument("config", help="Path to config")
    dataset_preflight_parser.add_argument("--set", "-s", action="append", dest="set_values")
    dataset_preflight_parser.add_argument("--dataset-path")
    dataset_preflight_parser.add_argument(
        "--output-json",
        default="artifacts/dataset_preflight.json",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_cmd(
            args.config,
            args.set_values,
            args.steps_per_stage,
            args.seed,
            args.num_classes,
            args.checkpoint_dir,
            args.resume,
            args.dataset_path,
        )
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
    elif args.command == "evolve":
        evolve_cmd(
            config=args.config,
            dataset_path=args.dataset_path,
            gens=args.gens,
            pop=args.pop,
        )
    elif args.command == "refine":
        refine_cmd(
            args.images,
            args.labels,
            args.output,
            args.checkpoint,
            args.device,
        )
    elif args.command == "export":
        export_cmd(
            args.config,
            args.set_values,
            args.output,
            args.shape_mode,
            args.checkpoint,
            args.num_classes,
            args.strict_load,
        )
    elif args.command == "preflight":
        preflight_cmd(args.profile)
    elif args.command == "dataset-preflight":
        dataset_preflight_cmd(
            args.config,
            args.set_values,
            args.dataset_path,
            args.output_json,
        )


if __name__ == "__main__":
    main()
