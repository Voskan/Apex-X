from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from apex_x.kernels.triton.fusiongate import fusiongate_dispatch, get_triton_fusiongate_availability
from apex_x.utils.repro import seed_all


@dataclass(frozen=True, slots=True)
class BenchConfig:
    batch: int = 1
    channels: int = 128
    height: int = 128
    width: int = 128
    warmup: int = 10
    iters: int = 50
    seed: int = 123
    dtype: str = "fp16"
    boundary_log_weight: float = 1.0
    uncertainty_log_weight: float = 1.0
    bias: float = 0.0


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered == "fp16":
        return torch.float16
    if lowered == "bf16":
        return torch.bfloat16
    if lowered == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = int(0.95 * (len(ordered) - 1))
    return float(ordered[index])


def _measure_ms(fn: Callable[[], object], warmup: int, iters: int, sync_cuda: bool) -> list[float]:
    for _ in range(warmup):
        fn()
        if sync_cuda:
            torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def run_triton_fusiongate_bench(config: BenchConfig) -> dict[str, Any]:
    dtype = _dtype_from_name(config.dtype)
    availability = get_triton_fusiongate_availability()
    use_cuda = availability.available
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    seed_all(config.seed, deterministic=True)
    base = torch.randn(
        (config.batch, config.channels, config.height, config.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    detail = torch.randn(
        (config.batch, config.channels, config.height, config.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    boundary = torch.rand(
        (config.batch, 1, config.height, config.width),
        dtype=dtype,
        device=device,
    )
    uncertainty = torch.rand(
        (config.batch, 1, config.height, config.width),
        dtype=dtype,
        device=device,
    )

    ref_alpha_timings = _measure_ms(
        lambda: fusiongate_dispatch(
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            boundary_log_weight=config.boundary_log_weight,
            uncertainty_log_weight=config.uncertainty_log_weight,
            bias=config.bias,
            apply_fusion=False,
            prefer_triton=False,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    dispatch_alpha = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        boundary_log_weight=config.boundary_log_weight,
        uncertainty_log_weight=config.uncertainty_log_weight,
        bias=config.bias,
        apply_fusion=False,
        prefer_triton=True,
        allow_fallback=True,
    )

    dispatch_alpha_timings = _measure_ms(
        lambda: fusiongate_dispatch(
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            boundary_log_weight=config.boundary_log_weight,
            uncertainty_log_weight=config.uncertainty_log_weight,
            bias=config.bias,
            apply_fusion=False,
            prefer_triton=True,
            allow_fallback=True,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    ref_fused_timings = _measure_ms(
        lambda: fusiongate_dispatch(
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            base_features=base,
            detail_features=detail,
            boundary_log_weight=config.boundary_log_weight,
            uncertainty_log_weight=config.uncertainty_log_weight,
            bias=config.bias,
            apply_fusion=True,
            prefer_triton=False,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    dispatch_fused = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        base_features=base,
        detail_features=detail,
        boundary_log_weight=config.boundary_log_weight,
        uncertainty_log_weight=config.uncertainty_log_weight,
        bias=config.bias,
        apply_fusion=True,
        prefer_triton=True,
        allow_fallback=True,
    )
    dispatch_fused_timings = _measure_ms(
        lambda: fusiongate_dispatch(
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            base_features=base,
            detail_features=detail,
            boundary_log_weight=config.boundary_log_weight,
            uncertainty_log_weight=config.uncertainty_log_weight,
            bias=config.bias,
            apply_fusion=True,
            prefer_triton=True,
            allow_fallback=True,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    ref_alpha_p50 = float(statistics.median(ref_alpha_timings))
    dispatch_alpha_p50 = float(statistics.median(dispatch_alpha_timings))
    ref_fused_p50 = float(statistics.median(ref_fused_timings))
    dispatch_fused_p50 = float(statistics.median(dispatch_fused_timings))

    return {
        "suite": "triton_fusiongate_microbench",
        "backend_alpha": dispatch_alpha.backend,
        "fallback_alpha": dispatch_alpha.fallback_reason,
        "backend_fused": dispatch_fused.backend,
        "fallback_fused": dispatch_fused.fallback_reason,
        "availability": {
            "triton_installed": availability.triton_installed,
            "cuda_available": availability.cuda_available,
            "cuda_device_count": availability.cuda_device_count,
            "available": availability.available,
            "reason": availability.reason,
        },
        "config": {
            "batch": config.batch,
            "channels": config.channels,
            "height": config.height,
            "width": config.width,
            "warmup": config.warmup,
            "iters": config.iters,
            "seed": config.seed,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "boundary_log_weight": config.boundary_log_weight,
            "uncertainty_log_weight": config.uncertainty_log_weight,
            "bias": config.bias,
        },
        "metrics_ms": {
            "alpha_reference_p50": ref_alpha_p50,
            "alpha_reference_p95": _p95(ref_alpha_timings),
            "alpha_dispatch_p50": dispatch_alpha_p50,
            "alpha_dispatch_p95": _p95(dispatch_alpha_timings),
            "alpha_speedup_ref_over_dispatch": (
                (ref_alpha_p50 / dispatch_alpha_p50) if dispatch_alpha_p50 > 0.0 else None
            ),
            "fused_reference_p50": ref_fused_p50,
            "fused_reference_p95": _p95(ref_fused_timings),
            "fused_dispatch_p50": dispatch_fused_p50,
            "fused_dispatch_p95": _p95(dispatch_fused_timings),
            "fused_speedup_ref_over_dispatch": (
                (ref_fused_p50 / dispatch_fused_p50) if dispatch_fused_p50 > 0.0 else None
            ),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triton FusionGate alpha/fusion microbenchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--boundary-log-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-log-weight", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_triton_fusiongate_bench(
        BenchConfig(
            batch=args.batch,
            channels=args.channels,
            height=args.height,
            width=args.width,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            dtype=args.dtype,
            boundary_log_weight=args.boundary_log_weight,
            uncertainty_log_weight=args.uncertainty_log_weight,
            bias=args.bias,
        )
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
