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
from torch import Tensor

from apex_x.kernels.triton.tileunpack import (
    get_triton_tileunpack_availability,
    tileunpack_dispatch,
    tileunpack_reference,
)
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
    overlap_shift: int = 0
    use_levels: bool = True


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


def _unique_indices(
    batch: int, kmax: int, max_idx: int, *, seed: int, device: torch.device
) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rows: list[torch.Tensor] = []
    for b in range(batch):
        row = torch.randperm(max_idx, generator=gen, dtype=torch.int64)[:kmax]
        rows.append(torch.sort(row).values)
        gen.manual_seed(seed + b + 1)
    return torch.stack(rows, dim=0).to(device=device, dtype=torch.int32).contiguous()


def _meta_from_indices(
    indices: Tensor,
    tile_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    overlap_shift: int = 0,
) -> dict[str, Tensor]:
    grid_h = height // tile_size
    grid_w = width // tile_size
    idx_i64 = indices.to(dtype=torch.int64, device=device)
    origins = torch.stack(
        ((idx_i64 // grid_w) * tile_size, (idx_i64 % grid_w) * tile_size),
        dim=-1,
    )
    if overlap_shift > 0:
        shift = int(overlap_shift)
        kmax = int(indices.shape[1])
        mask = (torch.arange(kmax, device=device).view(1, kmax) % 2) == 1
        origins_y = origins[..., 0]
        origins_x = origins[..., 1]
        origins_y = torch.where(
            mask,
            torch.clamp(origins_y - shift, min=0, max=height - tile_size),
            origins_y,
        )
        origins_x = torch.where(
            mask,
            torch.clamp(origins_x - shift, min=0, max=width - tile_size),
            origins_x,
        )
        origins = torch.stack((origins_y, origins_x), dim=-1)
    return {
        "indices": idx_i64,
        "origins": origins,
        "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=device),
        "grid": torch.tensor([grid_h, grid_w], dtype=torch.int64, device=device),
    }


def run_triton_tileunpack_bench(config: BenchConfig) -> dict[str, Any]:
    dtype = _dtype_from_name(config.dtype)
    availability = get_triton_tileunpack_availability()
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
    packed = torch.randn(
        (config.batch, config.kmax, config.channels, config.tile_size, config.tile_size),
        dtype=dtype,
        device=device,
    ).contiguous()
    max_idx = (config.height // config.tile_size) * (config.width // config.tile_size)
    idx = _unique_indices(
        config.batch, config.kmax, max_idx, seed=config.seed + 17, device=device
    )
    meta = _meta_from_indices(
        idx,
        config.tile_size,
        config.height,
        config.width,
        device=device,
        overlap_shift=config.overlap_shift,
    )
    levels: Tensor | None = None
    if config.use_levels:
        levels = torch.randint(
            0,
            4,
            (config.batch, config.kmax),
            dtype=torch.int32,
            device=device,
        )

    ref_timings = _measure_ms(
        lambda: tileunpack_reference(base, packed, meta=meta, levels=levels),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    dispatch = tileunpack_dispatch(
        base_map=base,
        packed_out=packed,
        meta=meta,
        levels=levels,
        prefer_triton=True,
        allow_fallback=True,
    )
    backend = dispatch.backend
    fallback_reason = dispatch.fallback_reason

    dispatch_timings = _measure_ms(
        lambda: tileunpack_dispatch(
            base_map=base,
            packed_out=packed,
            meta=meta,
            levels=levels,
            prefer_triton=True,
            allow_fallback=True,
        ),
        warmup=config.warmup,
        iters=config.iters,
        sync_cuda=device.type == "cuda",
    )

    ref_p50 = float(statistics.median(ref_timings))
    dispatch_p50 = float(statistics.median(dispatch_timings))
    speedup = (ref_p50 / dispatch_p50) if dispatch_p50 > 0.0 else None

    return {
        "suite": "triton_tileunpack_microbench",
        "backend": backend,
        "fallback_reason": fallback_reason,
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
            "overlap_shift": config.overlap_shift,
            "use_levels": config.use_levels,
        },
        "metrics_ms": {
            "reference_p50": ref_p50,
            "reference_p95": _p95(ref_timings),
            "dispatch_p50": dispatch_p50,
            "dispatch_p95": _p95(dispatch_timings),
            "speedup_ref_over_dispatch": speedup,
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triton TileUnpack microbenchmark")
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
    parser.add_argument("--overlap-shift", type=int, default=0)
    parser.add_argument("--no-levels", action="store_true")
    parser.add_argument("--output", type=str, default="")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_triton_tileunpack_bench(
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
            overlap_shift=args.overlap_shift,
            use_levels=not args.no_levels,
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
