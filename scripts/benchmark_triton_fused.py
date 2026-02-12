from __future__ import annotations

import time
from typing import Any

import torch

from apex_x.runtime import gather_gate_scatter, gather_gate_scatter_reference


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def benchmark_triton_fused(
    *,
    iters: int = 50,
    warmup: int = 10,
    device: str | None = None,
) -> dict[str, Any]:
    if iters <= 0:
        raise ValueError("iters must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    dev = _resolve_device(device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    torch.manual_seed(123)

    base = torch.randn((1, 64, 128, 128), dtype=dtype, device=dev)
    heavy = torch.randn((1, 64, 128, 128), dtype=dtype, device=dev)
    idx = torch.randint(0, (128 // 16) * (128 // 16), (1, 16), device=dev)
    boundary = torch.rand((1, 1, 128, 128), dtype=dtype, device=dev)
    uncertainty = torch.rand((1, 1, 128, 128), dtype=dtype, device=dev)

    def _sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    for _ in range(warmup):
        gather_gate_scatter_reference(
            base_map=base,
            heavy_map=heavy,
            indices=idx,
            tile_size=16,
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            level_priority=2,
        )
        _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        gather_gate_scatter_reference(
            base_map=base,
            heavy_map=heavy,
            indices=idx,
            tile_size=16,
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            level_priority=2,
        )
        _sync()
    ref_ms = ((time.perf_counter() - t0) * 1000.0) / float(iters)

    for _ in range(warmup):
        gather_gate_scatter(
            base_map=base,
            heavy_map=heavy,
            indices=idx,
            tile_size=16,
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            level_priority=2,
            prefer_triton=True,
            allow_fallback=True,
        )
        _sync()

    out = gather_gate_scatter(
        base_map=base,
        heavy_map=heavy,
        indices=idx,
        tile_size=16,
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        level_priority=2,
        prefer_triton=True,
        allow_fallback=True,
    )
    _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = gather_gate_scatter(
            base_map=base,
            heavy_map=heavy,
            indices=idx,
            tile_size=16,
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            level_priority=2,
            prefer_triton=True,
            allow_fallback=True,
        )
        _sync()
    fused_ms = ((time.perf_counter() - t0) * 1000.0) / float(iters)

    speedup = ref_ms / fused_ms if fused_ms > 0.0 else None
    return {
        "device": str(dev),
        "backend": out.backend,
        "fallback_reason": out.fallback_reason,
        "reference_ms": ref_ms,
        "fused_ms": fused_ms,
        "speedup_vs_reference": speedup,
    }


def main() -> None:
    stats = benchmark_triton_fused()
    device = stats["device"]
    backend = stats["backend"]
    ref_ms = float(stats["reference_ms"])
    fused_ms = float(stats["fused_ms"])
    speedup = stats["speedup_vs_reference"]
    fallback_reason = stats["fallback_reason"]

    print(
        "triton_fused_bench "
        f"device={device} backend={backend} "
        f"reference_ms={ref_ms:.4f} fused_ms={fused_ms:.4f} "
        f"speedup={float(speedup):.4f}"
    )
    if fallback_reason:
        print(f"fallback_reason={fallback_reason}")


if __name__ == "__main__":
    main()
