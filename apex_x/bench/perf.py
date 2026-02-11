from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apex_x import ApexXModel
from apex_x.model import FusionGate
from apex_x.tiles import TilePackTorch, TileUnpackTorch


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return float(ordered[idx])


def _measure_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> list[float]:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iters <= 0:
        raise ValueError("iters must be > 0")

    for _ in range(warmup):
        fn()

    timings_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    return timings_ms


def _infer_benchmark(*, warmup: int, iters: int, seed: int) -> dict[str, float]:
    model = ApexXModel()
    image = np.random.RandomState(seed).rand(1, 3, 128, 128).astype(np.float32)

    timings_ms = _measure_ms(lambda: model.forward(image), warmup=warmup, iters=iters)
    return {
        "infer_p50_ms": float(statistics.median(timings_ms)),
        "infer_p95_ms": _p95(timings_ms),
    }


def _tile_microbench(*, warmup: int, iters: int, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    dtype = torch.float32
    device = torch.device("cpu")
    bsz = 1
    channels = 64
    height = 128
    width = 128
    tile_size = 16

    base = torch.randn((bsz, channels, height, width), dtype=dtype, device=device)
    heavy = torch.randn((bsz, channels, height, width), dtype=dtype, device=device)
    boundary = torch.rand((bsz, 1, height, width), dtype=dtype, device=device)
    uncertainty = torch.rand((bsz, 1, height, width), dtype=dtype, device=device)

    # 8x8 grid for 128/16 tiles. Select a fixed subset for deterministic microbench.
    indices = torch.tensor(
        [[0, 5, 9, 12, 18, 27, 33, 40, 50, 63]],
        dtype=torch.int64,
        device=device,
    )

    packer = TilePackTorch()
    unpacker = TileUnpackTorch()
    gate = FusionGate().to(device=device, dtype=dtype).eval()

    pack_timings_ms = _measure_ms(
        lambda: packer.pack(base, indices, tile_size=tile_size, order_mode="hilbert"),
        warmup=warmup,
        iters=iters,
    )
    packed, meta = packer.pack(heavy, indices, tile_size=tile_size, order_mode="hilbert")

    unpack_timings_ms = _measure_ms(
        lambda: unpacker.unpack(
            base_map=base,
            packed_out=packed,
            meta=meta,
            level_priority=2,
            overlap_mode="override",
        ),
        warmup=warmup,
        iters=iters,
    )

    fusion_timings_ms = _measure_ms(
        lambda: gate(
            base_features=base,
            heavy_features=heavy,
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
        ),
        warmup=warmup,
        iters=iters,
    )

    return {
        "tile_pack_p50_ms": float(statistics.median(pack_timings_ms)),
        "tile_unpack_p50_ms": float(statistics.median(unpack_timings_ms)),
        "fusion_gate_p50_ms": float(statistics.median(fusion_timings_ms)),
    }


def run_cpu_perf_suite(
    *,
    infer_warmup: int = 5,
    infer_iters: int = 30,
    micro_warmup: int = 5,
    micro_iters: int = 50,
    seed: int = 123,
) -> dict[str, Any]:
    infer_metrics = _infer_benchmark(warmup=infer_warmup, iters=infer_iters, seed=seed)
    micro_metrics = _tile_microbench(warmup=micro_warmup, iters=micro_iters, seed=seed)
    metrics = {**infer_metrics, **micro_metrics}

    return {
        "schema_version": 1,
        "suite": "apex_x_cpu_perf_regression",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "config": {
            "infer_warmup": infer_warmup,
            "infer_iters": infer_iters,
            "micro_warmup": micro_warmup,
            "micro_iters": micro_iters,
            "seed": seed,
            "input_shape": [1, 3, 128, 128],
            "tile_shape": [1, 64, 128, 128],
            "tile_size": 16,
        },
        "metrics": metrics,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("json root must be an object")
    return data


def compare_against_baseline(
    *,
    current_report: dict[str, Any],
    baseline_spec: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = baseline_spec.get("metrics", {})
    current_metrics = current_report.get("metrics", {})
    if not isinstance(baseline_metrics, dict) or not isinstance(current_metrics, dict):
        raise ValueError("baseline/current metrics must be objects")

    checks: list[dict[str, Any]] = []
    failed = False
    for metric_name, spec in sorted(baseline_metrics.items()):
        if not isinstance(spec, dict):
            raise ValueError(f"baseline metric spec for {metric_name!r} must be object")
        if "value_ms" not in spec:
            raise ValueError(f"baseline metric spec for {metric_name!r} missing value_ms")
        baseline_value = float(spec["value_ms"])
        tolerance_ratio = float(spec.get("max_regression_ratio", 0.50))
        tolerance_abs_ms = float(spec.get("max_regression_abs_ms", 0.0))
        allowed_max = baseline_value * (1.0 + tolerance_ratio) + tolerance_abs_ms

        current_raw = current_metrics.get(metric_name)
        if current_raw is None:
            failed = True
            checks.append(
                {
                    "metric": metric_name,
                    "status": "missing",
                    "baseline_ms": baseline_value,
                    "allowed_max_ms": allowed_max,
                    "current_ms": None,
                }
            )
            continue
        current_value = float(current_raw)
        status = "pass" if current_value <= allowed_max else "fail"
        if status == "fail":
            failed = True
        checks.append(
            {
                "metric": metric_name,
                "status": status,
                "baseline_ms": baseline_value,
                "allowed_max_ms": allowed_max,
                "current_ms": current_value,
                "regression_ratio": (
                    ((current_value / baseline_value) - 1.0) if baseline_value > 0.0 else None
                ),
            }
        )

    return {
        "suite": current_report.get("suite"),
        "status": "fail" if failed else "pass",
        "checks": checks,
    }


__all__ = [
    "run_cpu_perf_suite",
    "write_json",
    "read_json",
    "compare_against_baseline",
]
