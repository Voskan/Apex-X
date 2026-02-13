#!/usr/bin/env python3
"""Run a comparable benchmark matrix for Apex-X and optional YOLO baselines."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex_x.config import load_yaml_config
from apex_x.infer import evaluate_model_dataset, load_eval_dataset_npz
from apex_x.model import ApexXModel
from apex_x.runtime import detect_runtime_caps


@dataclass(slots=True)
class ModelBenchmarkResult:
    model: str
    status: str
    latency_ms_mean: float | None
    score_mean: float | None
    detections_mean: float | None
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apex-X SOTA benchmark harness")
    parser.add_argument("--config", default="configs/worldclass.yaml")
    parser.add_argument("--dataset-npz", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--yolo-models", default="yolov8n-seg.pt,yolo11n-seg.pt")
    parser.add_argument("--output-json", default="artifacts/sota_benchmark.json")
    parser.add_argument("--output-md", default="artifacts/sota_benchmark.md")
    return parser.parse_args()


def _to_uint8_image(image_nchw: np.ndarray) -> np.ndarray:
    hwc = np.transpose(image_nchw, (1, 2, 0))
    clipped = np.clip(hwc * 255.0, 0.0, 255.0).astype(np.uint8)
    return clipped


def _benchmark_yolo(
    *,
    images: np.ndarray,
    model_name: str,
    max_samples: int,
) -> ModelBenchmarkResult:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        return ModelBenchmarkResult(
            model=model_name,
            status="skipped",
            latency_ms_mean=None,
            score_mean=None,
            detections_mean=None,
            notes=f"ultralytics_unavailable:{type(exc).__name__}",
        )

    try:
        model = YOLO(model_name)
    except Exception as exc:
        return ModelBenchmarkResult(
            model=model_name,
            status="failed",
            latency_ms_mean=None,
            score_mean=None,
            detections_mean=None,
            notes=f"model_load_failed:{type(exc).__name__}:{exc}",
        )

    latencies: list[float] = []
    scores: list[float] = []
    det_counts: list[int] = []
    n = min(int(max_samples), int(images.shape[0]))

    for idx in range(n):
        image = _to_uint8_image(images[idx])
        started = time.perf_counter()
        results = model.predict(image, verbose=False)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(float(latency_ms))

        if not results:
            det_counts.append(0)
            continue
        pred = results[0]
        boxes = getattr(pred, "boxes", None)
        if boxes is None:
            det_counts.append(0)
            continue
        conf = boxes.conf
        conf_np = conf.detach().cpu().numpy() if conf is not None else np.asarray([], dtype=np.float32)
        det_counts.append(int(conf_np.size))
        if conf_np.size > 0:
            scores.append(float(conf_np.mean()))

    return ModelBenchmarkResult(
        model=model_name,
        status="done",
        latency_ms_mean=float(np.mean(latencies)) if latencies else None,
        score_mean=float(np.mean(scores)) if scores else 0.0,
        detections_mean=float(np.mean(det_counts)) if det_counts else 0.0,
        notes="confidence/latency proxy (not AP without labeled eval set)",
    )


def _markdown(results: list[ModelBenchmarkResult]) -> str:
    lines = [
        "# SOTA Benchmark Report",
        "",
        "All models were run on the same dataset slice and host.",
        "",
        "| Model | Status | Mean Latency (ms) | Mean Score | Mean Detections | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| `{row.model}` | `{row.status}` | "
            f"{'' if row.latency_ms_mean is None else f'{row.latency_ms_mean:.3f}'} | "
            f"{'' if row.score_mean is None else f'{row.score_mean:.5f}'} | "
            f"{'' if row.detections_mean is None else f'{row.detections_mean:.2f}'} | "
            f"{row.notes} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_config(args.config)
    dataset = load_eval_dataset_npz(
        path=args.dataset_npz,
        expected_height=cfg.model.input_height,
        expected_width=cfg.model.input_width,
    )
    images = dataset.images

    model = ApexXModel(config=cfg)
    caps = detect_runtime_caps()
    apex_summary = evaluate_model_dataset(
        model=model,
        images=images,
        requested_backend=cfg.runtime.backend,
        selected_backend=cfg.runtime.backend,
        fallback_policy=cfg.runtime.fallback_policy,
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=caps,
        max_samples=args.max_samples,
    )

    results: list[ModelBenchmarkResult] = [
        ModelBenchmarkResult(
            model="apex_x",
            status="done",
            latency_ms_mean=None,
            score_mean=float(apex_summary.det_score_mean),
            detections_mean=None,
            notes=(
                "primary metric det_score_mean; standard inference"
            ),
        )
    ]

    # Benchmark Apex-X with TTA
    print("Running Apex-X with TTA...")
    apex_tta_summary = evaluate_model_dataset(
        model=model,
        images=images,
        requested_backend=cfg.runtime.backend,
        selected_backend=cfg.runtime.backend,
        fallback_policy=cfg.runtime.fallback_policy,
        precision_profile=cfg.runtime.precision_profile,
        selection_fallback_reason=None,
        runtime_caps=caps,
        max_samples=args.max_samples,
        use_tta=True,
    )
    results.append(
        ModelBenchmarkResult(
            model="apex_x_tta",
            status="done",
            latency_ms_mean=None,
            score_mean=float(apex_tta_summary.det_score_mean),
            detections_mean=None,
            notes=(
                "TTA enabled (scales=[0.8, 1.0, 1.2], flips)"
            ),
        )
    )

    yolo_models = [m.strip() for m in str(args.yolo_models).split(",") if m.strip()]
    for model_name in yolo_models:
        results.append(
            _benchmark_yolo(images=images, model_name=model_name, max_samples=int(args.max_samples))
        )

    payload: dict[str, Any] = {
        "config": args.config,
        "dataset_npz": args.dataset_npz,
        "max_samples": int(args.max_samples),
        "results": [asdict(x) for x in results],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(results), encoding="utf-8")

    print(f"sota_benchmark output_json={out_json} output_md={out_md} models={len(results)}")


if __name__ == "__main__":
    main()
