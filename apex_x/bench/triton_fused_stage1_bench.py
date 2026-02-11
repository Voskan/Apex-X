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

from apex_x.kernels.triton.fused_pack_op_unpack import (
    fused_pack_op_unpack_dispatch,
    get_triton_fused_stage1_availability,
    separate_pack_op_unpack_reference,
)
from apex_x.kernels.triton.tilepack import tilepack_dispatch
from apex_x.kernels.triton.tileunpack import tileunpack_dispatch
from apex_x.utils.repro import seed_all


@dataclass(frozen=True, slots=True)
class BenchConfig:
    batch: int = 1
    channels: int = 128
    height: int = 128
    width: int = 128
    tile_size: int = 8
    kmax: int = 32
    warmup: int = 10
    iters: int = 50
    seed: int = 123
    dtype: str = "fp16"
    value_scale: float = 1.0
    value_bias: float = 0.0
    gate_scale: float = 1.0
    gate_bias: float = 0.0


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


def _separate_pipeline_dispatch(
    feature: torch.Tensor,
    idx: torch.Tensor,
    *,
    tile_size: int,
    value_scale: float,
    value_bias: float,
    gate_scale: float,
    gate_bias: float,
) -> torch.Tensor:
    packed = tilepack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=tile_size,
        prefer_triton=True,
        allow_fallback=True,
    ).packed
    value = packed * value_scale + value_bias
    gate = packed * gate_scale + gate_bias
    packed_out = value * torch.relu(gate)
    merged = tileunpack_dispatch(
        base_map=feature,
        packed_out=packed_out,
        indices=idx,
        prefer_triton=True,
        allow_fallback=True,
    ).merged
    return merged


def _unique_indices(
    batch: int, kmax: int, max_idx: int, seed: int, device: torch.device
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rows: list[torch.Tensor] = []
    for b in range(batch):
        row = torch.randperm(max_idx, generator=gen, dtype=torch.int64)[:kmax]
        rows.append(torch.sort(row).values)
        gen.manual_seed(seed + b + 1)
    return torch.stack(rows, dim=0).to(dtype=torch.int32, device=device).contiguous()


def run_triton_fused_stage1_bench(config: BenchConfig) -> dict[str, Any]:
    dtype = _dtype_from_name(config.dtype)
    availability = get_triton_fused_stage1_availability()
    use_cuda = availability.available
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    seed_all(config.seed, deterministic=True)
    feature = torch.randn(
        (config.batch, config.channels, config.height, config.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    max_index = (config.height // config.tile_size) * (config.width // config.tile_size)
    idx = _unique_indices(
        batch=config.batch,
        kmax=config.kmax,
        max_idx=max_index,
        seed=config.seed + 11,
        device=device,
    )

    params = {
        "value_scale": config.value_scale,
        "value_bias": config.value_bias,
        "gate_scale": config.gate_scale,
        "gate_bias": config.gate_bias,
    }

    ref_timings = _measure_ms(
        lambda: separate_pack_op_unpack_reference(
            feature,
            idx,
            config.tile_size,
            value_scale=config.value_scale,
            value_bias=config.value_bias,
            gate_scale=config.gate_scale,
            gate_bias=config.gate_bias,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )
    separate_dispatch_timings = _measure_ms(
        lambda: _separate_pipeline_dispatch(
            feature,
            idx,
            tile_size=config.tile_size,
            **params,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    initial = fused_pack_op_unpack_dispatch(
        feature_map=feature,
        indices=idx,
        tile_size=config.tile_size,
        prefer_triton=True,
        allow_fallback=True,
        value_scale=config.value_scale,
        value_bias=config.value_bias,
        gate_scale=config.gate_scale,
        gate_bias=config.gate_bias,
    )
    fused_timings = _measure_ms(
        lambda: fused_pack_op_unpack_dispatch(
            feature_map=feature,
            indices=idx,
            tile_size=config.tile_size,
            prefer_triton=True,
            allow_fallback=True,
            value_scale=config.value_scale,
            value_bias=config.value_bias,
            gate_scale=config.gate_scale,
            gate_bias=config.gate_bias,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    ref_p50 = float(statistics.median(ref_timings))
    separate_p50 = float(statistics.median(separate_dispatch_timings))
    fused_p50 = float(statistics.median(fused_timings))
    return {
        "suite": "triton_fused_stage1_microbench",
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
            "channels": config.channels,
            "height": config.height,
            "width": config.width,
            "tile_size": config.tile_size,
            "kmax": config.kmax,
            "warmup": config.warmup,
            "iters": config.iters,
            "seed": config.seed,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            **params,
        },
        "metrics_ms": {
            "reference_p50": ref_p50,
            "reference_p95": _p95(ref_timings),
            "separate_dispatch_p50": separate_p50,
            "separate_dispatch_p95": _p95(separate_dispatch_timings),
            "fused_dispatch_p50": fused_p50,
            "fused_dispatch_p95": _p95(fused_timings),
            "speedup_ref_over_fused": (ref_p50 / fused_p50) if fused_p50 > 0.0 else None,
            "speedup_separate_over_fused": (separate_p50 / fused_p50 if fused_p50 > 0.0 else None),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triton fused Stage-1 tile pipeline microbenchmark"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--kmax", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--value-bias", type=float, default=0.0)
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_triton_fused_stage1_bench(
        BenchConfig(
            batch=args.batch,
            channels=args.channels,
            height=args.height,
            width=args.width,
            tile_size=args.tile_size,
            kmax=args.kmax,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            dtype=args.dtype,
            value_scale=args.value_scale,
            value_bias=args.value_bias,
            gate_scale=args.gate_scale,
            gate_bias=args.gate_bias,
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
