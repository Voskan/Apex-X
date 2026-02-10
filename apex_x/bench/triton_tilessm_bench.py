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

from apex_x.kernels.triton.tilessm_scan import (
    get_triton_tilessm_availability,
    tilessm_scan_dispatch,
    tilessm_scan_reference,
)
from apex_x.utils.repro import seed_all


@dataclass(frozen=True, slots=True)
class BenchConfig:
    batch: int = 2
    steps: int = 256
    channels: int = 128
    warmup: int = 10
    iters: int = 50
    seed: int = 123
    dtype: str = "fp16"


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


def _random_stable_params(
    channels: int, *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        "decay": (torch.rand((channels,), device=device, dtype=dtype) * 0.8 + 0.1).contiguous(),
        "input_gain": (
            torch.rand((channels,), device=device, dtype=dtype) * 1.5 + 0.05
        ).contiguous(),
        "output_gain": (
            torch.rand((channels,), device=device, dtype=dtype) * 1.5 + 0.05
        ).contiguous(),
        "state_bias": (torch.randn((channels,), device=device, dtype=dtype) * 0.1).contiguous(),
    }


def run_triton_tilessm_bench(config: BenchConfig) -> dict[str, Any]:
    dtype = _dtype_from_name(config.dtype)
    availability = get_triton_tilessm_availability()
    use_cuda = availability.available
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    seed_all(config.seed, deterministic=True)
    tokens = torch.randn(
        (config.batch, config.steps, config.channels),
        dtype=dtype,
        device=device,
    ).contiguous()
    params = _random_stable_params(config.channels, device=device, dtype=dtype)
    decay = params["decay"]
    input_gain = params["input_gain"]
    output_gain = params["output_gain"]
    state_bias = params["state_bias"]
    gate = torch.sigmoid(torch.randn((config.channels,), dtype=dtype, device=device))

    ref_forward = _measure_ms(
        lambda: tilessm_scan_reference(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="forward",
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    ref_backward = _measure_ms(
        lambda: tilessm_scan_reference(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="backward",
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    ref_bidir_avg = _measure_ms(
        lambda: tilessm_scan_reference(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="bidirectional",
            merge_mode="avg",
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    ref_bidir_gated = _measure_ms(
        lambda: tilessm_scan_reference(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="bidirectional",
            merge_mode="gated",
            merge_gate=gate,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    initial = tilessm_scan_dispatch(
        tokens,
        prefer_triton=True,
        allow_fallback=True,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        direction="forward",
    )
    disp_forward = _measure_ms(
        lambda: tilessm_scan_dispatch(
            tokens,
            prefer_triton=True,
            allow_fallback=True,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="forward",
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    disp_backward = _measure_ms(
        lambda: tilessm_scan_dispatch(
            tokens,
            prefer_triton=True,
            allow_fallback=True,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="backward",
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    disp_bidir_avg = _measure_ms(
        lambda: tilessm_scan_dispatch(
            tokens,
            prefer_triton=True,
            allow_fallback=True,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="bidirectional",
            merge_mode="avg",
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    disp_bidir_gated = _measure_ms(
        lambda: tilessm_scan_dispatch(
            tokens,
            prefer_triton=True,
            allow_fallback=True,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            direction="bidirectional",
            merge_mode="gated",
            merge_gate=gate,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    ref_forward_p50 = float(statistics.median(ref_forward))
    disp_forward_p50 = float(statistics.median(disp_forward))
    ref_backward_p50 = float(statistics.median(ref_backward))
    disp_backward_p50 = float(statistics.median(disp_backward))
    ref_bidir_avg_p50 = float(statistics.median(ref_bidir_avg))
    disp_bidir_avg_p50 = float(statistics.median(disp_bidir_avg))
    ref_bidir_gated_p50 = float(statistics.median(ref_bidir_gated))
    disp_bidir_gated_p50 = float(statistics.median(disp_bidir_gated))

    return {
        "suite": "triton_tilessm_microbench",
        "backend": initial.backend,
        "fallback_reason": initial.fallback_reason,
        "availability": {
            "triton_installed": availability.triton_installed,
            "cuda_available": availability.cuda_available,
            "cuda_device_count": availability.cuda_device_count,
            "available": availability.available,
            "reason": availability.reason,
        },
        "config": {
            "batch": config.batch,
            "steps": config.steps,
            "channels": config.channels,
            "warmup": config.warmup,
            "iters": config.iters,
            "seed": config.seed,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
        },
        "metrics_ms": {
            "reference_forward_p50": ref_forward_p50,
            "reference_forward_p95": _p95(ref_forward),
            "dispatch_forward_p50": disp_forward_p50,
            "dispatch_forward_p95": _p95(disp_forward),
            "reference_backward_p50": ref_backward_p50,
            "dispatch_backward_p50": disp_backward_p50,
            "reference_bidirectional_avg_p50": ref_bidir_avg_p50,
            "dispatch_bidirectional_avg_p50": disp_bidir_avg_p50,
            "reference_bidirectional_gated_p50": ref_bidir_gated_p50,
            "dispatch_bidirectional_gated_p50": disp_bidir_gated_p50,
            "speedup_ref_forward_over_dispatch_forward": (
                ref_forward_p50 / disp_forward_p50 if disp_forward_p50 > 0.0 else None
            ),
            "overhead_reference_backward_vs_forward": (
                ref_backward_p50 / ref_forward_p50 if ref_forward_p50 > 0.0 else None
            ),
            "overhead_reference_bidir_avg_vs_forward": (
                ref_bidir_avg_p50 / ref_forward_p50 if ref_forward_p50 > 0.0 else None
            ),
            "overhead_reference_bidir_gated_vs_forward": (
                ref_bidir_gated_p50 / ref_forward_p50 if ref_forward_p50 > 0.0 else None
            ),
            "overhead_dispatch_backward_vs_forward": (
                disp_backward_p50 / disp_forward_p50 if disp_forward_p50 > 0.0 else None
            ),
            "overhead_dispatch_bidir_avg_vs_forward": (
                disp_bidir_avg_p50 / disp_forward_p50 if disp_forward_p50 > 0.0 else None
            ),
            "overhead_dispatch_bidir_gated_vs_forward": (
                disp_bidir_gated_p50 / disp_forward_p50 if disp_forward_p50 > 0.0 else None
            ),
        },
        "throughput_tokens_per_sec": {
            "reference_forward": (
                float(config.batch * config.steps * config.channels) / (ref_forward_p50 / 1000.0)
                if ref_forward_p50 > 0.0
                else None
            ),
            "dispatch_forward": (
                float(config.batch * config.steps * config.channels) / (disp_forward_p50 / 1000.0)
                if disp_forward_p50 > 0.0
                else None
            ),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triton TileSSM scan microbenchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output", type=str, default="")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_triton_tilessm_bench(
        BenchConfig(
            batch=args.batch,
            steps=args.steps,
            channels=args.channels,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            dtype=args.dtype,
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
