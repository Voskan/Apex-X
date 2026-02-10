from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from apex_x.config import ApexXConfig
from apex_x.kernels.triton.fusiongate import fusiongate_dispatch
from apex_x.kernels.triton.tilepack import tilepack_dispatch, tilepack_reference
from apex_x.kernels.triton.tilessm_scan import tilessm_scan_dispatch, tilessm_scan_reference
from apex_x.kernels.triton.tileunpack import tileunpack_dispatch, tileunpack_reference
from apex_x.model import FFModule
from apex_x.runtime import RuntimeCaps, detect_runtime_caps
from apex_x.tiles import tile_grid_shape
from apex_x.utils.repro import seed_all


@dataclass(frozen=True, slots=True)
class GPUBenchConfig:
    batch: int = 1
    channels: int = 128
    height: int = 128
    width: int = 128
    tile_size: int = 8
    kmax: int = 32
    steps: int = 256
    warmup: int = 10
    iters: int = 50
    seed: int = 123
    dtype: str = "fp16"
    budget_b1: float = 16.0
    budget_b2: float = 8.0
    budget_total: float = 32.0
    trt_engine_path: str = ""
    trt_plugin_lib: str = ""
    trt_input_shapes: tuple[str, ...] = ()


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered == "fp16":
        return torch.float16
    if lowered == "bf16":
        return torch.bfloat16
    if lowered == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype name: {name}")


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return float(ordered[idx])


def _throughput_per_s(work_items: float, p50_ms: float) -> float | None:
    if p50_ms <= 0.0:
        return None
    return float(work_items / (p50_ms / 1000.0))


def _normalize_dtype_for_device(dtype: torch.dtype, *, device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return dtype
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _measure_ms(
    fn: Any,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[list[float], float | None]:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iters <= 0:
        raise ValueError("iters must be > 0")

    sync_cuda = device.type == "cuda"
    if sync_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        fn()
        if sync_cuda:
            torch.cuda.synchronize(device)

    timings_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = None
    if sync_cuda:
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return timings_ms, peak_mb


def _as_metrics_block(
    timings_ms: list[float],
    *,
    work_items: float | None = None,
    work_label: str = "items_per_s",
    peak_mb: float | None = None,
) -> dict[str, Any]:
    p50 = float(statistics.median(timings_ms))
    block: dict[str, Any] = {
        "p50_ms": p50,
        "p95_ms": _p95(timings_ms),
        "peak_memory_mb": peak_mb,
    }
    if work_items is not None:
        block[work_label] = _throughput_per_s(work_items, p50)
    return block


def _unique_indices(
    *,
    batch: int,
    kmax: int,
    max_index: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    if max_index <= 0:
        raise ValueError("max_index must be > 0")
    if kmax <= 0:
        raise ValueError("kmax must be > 0")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rows: list[Tensor] = []
    for b in range(batch):
        row = torch.randperm(max_index, generator=gen, dtype=torch.int64)[:kmax]
        rows.append(torch.sort(row).values)
        gen.manual_seed(seed + b + 1)
    return torch.stack(rows, dim=0).to(device=device, dtype=torch.int32).contiguous()


def _bench_tilepack(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    max_index = (cfg.height // cfg.tile_size) * (cfg.width // cfg.tile_size)
    feature = torch.randn(
        (cfg.batch, cfg.channels, cfg.height, cfg.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    idx = _unique_indices(
        batch=cfg.batch,
        kmax=cfg.kmax,
        max_index=max_index,
        seed=cfg.seed + 1,
        device=device,
    )
    prefer_triton = dtype in {torch.float16, torch.bfloat16}

    ref_timings, ref_peak = _measure_ms(
        lambda: tilepack_reference(feature, idx, cfg.tile_size),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    dispatch_once = tilepack_dispatch(
        feature,
        idx,
        cfg.tile_size,
        prefer_triton=prefer_triton,
        allow_fallback=True,
        inference_only=True,
    )
    disp_timings, disp_peak = _measure_ms(
        lambda: tilepack_dispatch(
            feature,
            idx,
            cfg.tile_size,
            prefer_triton=prefer_triton,
            allow_fallback=True,
            inference_only=True,
        ),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )

    tiles_per_call = float(cfg.batch * cfg.kmax)
    return {
        "backend": dispatch_once.backend,
        "fallback_reason": dispatch_once.fallback_reason,
        "reference": _as_metrics_block(
            ref_timings,
            work_items=tiles_per_call,
            work_label="tiles_per_s",
            peak_mb=ref_peak,
        ),
        "dispatch": _as_metrics_block(
            disp_timings,
            work_items=tiles_per_call,
            work_label="tiles_per_s",
            peak_mb=disp_peak,
        ),
    }


def _bench_tileunpack(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    max_index = (cfg.height // cfg.tile_size) * (cfg.width // cfg.tile_size)
    base = torch.randn(
        (cfg.batch, cfg.channels, cfg.height, cfg.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    packed = torch.randn(
        (cfg.batch, cfg.kmax, cfg.channels, cfg.tile_size, cfg.tile_size),
        dtype=dtype,
        device=device,
    ).contiguous()
    idx = _unique_indices(
        batch=cfg.batch,
        kmax=cfg.kmax,
        max_index=max_index,
        seed=cfg.seed + 2,
        device=device,
    )
    levels = torch.randint(
        low=0,
        high=3,
        size=(cfg.batch, cfg.kmax),
        dtype=torch.int32,
        device=device,
    ).contiguous()
    prefer_triton = dtype in {torch.float16, torch.bfloat16}

    ref_timings, ref_peak = _measure_ms(
        lambda: tileunpack_reference(base, packed, indices=idx, levels=levels),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    dispatch_once = tileunpack_dispatch(
        base,
        packed,
        indices=idx,
        levels=levels,
        prefer_triton=prefer_triton,
        allow_fallback=True,
        inference_only=True,
    )
    disp_timings, disp_peak = _measure_ms(
        lambda: tileunpack_dispatch(
            base,
            packed,
            indices=idx,
            levels=levels,
            prefer_triton=prefer_triton,
            allow_fallback=True,
            inference_only=True,
        ),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )

    tiles_per_call = float(cfg.batch * cfg.kmax)
    return {
        "backend": dispatch_once.backend,
        "fallback_reason": dispatch_once.fallback_reason,
        "reference": _as_metrics_block(
            ref_timings,
            work_items=tiles_per_call,
            work_label="tiles_per_s",
            peak_mb=ref_peak,
        ),
        "dispatch": _as_metrics_block(
            disp_timings,
            work_items=tiles_per_call,
            work_label="tiles_per_s",
            peak_mb=disp_peak,
        ),
    }


def _bench_fusion(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    base = torch.randn(
        (cfg.batch, cfg.channels, cfg.height, cfg.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    detail = torch.randn_like(base)
    boundary = torch.rand((cfg.batch, 1, cfg.height, cfg.width), dtype=dtype, device=device)
    uncertainty = torch.rand((cfg.batch, 1, cfg.height, cfg.width), dtype=dtype, device=device)
    prefer_triton = dtype in {torch.float16, torch.bfloat16}

    ref_timings, ref_peak = _measure_ms(
        lambda: fusiongate_dispatch(
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            base_features=base,
            detail_features=detail,
            apply_fusion=True,
            prefer_triton=False,
            allow_fallback=True,
            inference_only=True,
        ),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    dispatch_once = fusiongate_dispatch(
        boundary_proxy=boundary,
        uncertainty_proxy=uncertainty,
        base_features=base,
        detail_features=detail,
        apply_fusion=True,
        prefer_triton=prefer_triton,
        allow_fallback=True,
        inference_only=True,
    )
    disp_timings, disp_peak = _measure_ms(
        lambda: fusiongate_dispatch(
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            base_features=base,
            detail_features=detail,
            apply_fusion=True,
            prefer_triton=prefer_triton,
            allow_fallback=True,
            inference_only=True,
        ),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )

    elems_per_call = float(cfg.batch * cfg.channels * cfg.height * cfg.width)
    return {
        "backend": dispatch_once.backend,
        "fallback_reason": dispatch_once.fallback_reason,
        "reference": _as_metrics_block(
            ref_timings,
            work_items=elems_per_call,
            work_label="elements_per_s",
            peak_mb=ref_peak,
        ),
        "dispatch": _as_metrics_block(
            disp_timings,
            work_items=elems_per_call,
            work_label="elements_per_s",
            peak_mb=disp_peak,
        ),
    }


def _random_stable_params(
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return {
        "decay": (torch.rand((channels,), generator=gen, device=device, dtype=dtype) * 0.8 + 0.1)
        .contiguous(),
        "input_gain": (
            torch.rand((channels,), generator=gen, device=device, dtype=dtype) * 1.5 + 0.05
        ).contiguous(),
        "output_gain": (
            torch.rand((channels,), generator=gen, device=device, dtype=dtype) * 1.5 + 0.05
        ).contiguous(),
        "state_bias": (
            torch.randn((channels,), generator=gen, device=device, dtype=dtype) * 0.1
        ).contiguous(),
    }


def _bench_tilessm(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    tokens = torch.randn(
        (cfg.batch, cfg.steps, cfg.channels),
        dtype=dtype,
        device=device,
    ).contiguous()
    params = _random_stable_params(
        cfg.channels,
        device=device,
        dtype=dtype,
        seed=cfg.seed + 7,
    )
    prefer_triton = dtype in {torch.float16, torch.bfloat16}

    ref_timings, ref_peak = _measure_ms(
        lambda: tilessm_scan_reference(
            tokens,
            decay=params["decay"],
            input_gain=params["input_gain"],
            output_gain=params["output_gain"],
            state_bias=params["state_bias"],
            direction="forward",
        ),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    dispatch_once = tilessm_scan_dispatch(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="forward",
        prefer_triton=prefer_triton,
        allow_fallback=True,
        inference_only=True,
    )
    disp_timings, disp_peak = _measure_ms(
        lambda: tilessm_scan_dispatch(
            tokens,
            decay=params["decay"],
            input_gain=params["input_gain"],
            output_gain=params["output_gain"],
            state_bias=params["state_bias"],
            direction="forward",
            prefer_triton=prefer_triton,
            allow_fallback=True,
            inference_only=True,
        ),
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )

    tokens_per_call = float(cfg.batch * cfg.steps * cfg.channels)
    trt_section = _bench_tensorrt_tilessm_plugin(
        cfg,
        device=device,
        tokens=tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    return {
        "backend": dispatch_once.backend,
        "fallback_reason": dispatch_once.fallback_reason,
        "torch_reference": _as_metrics_block(
            ref_timings,
            work_items=tokens_per_call,
            work_label="tokens_per_s",
            peak_mb=ref_peak,
        ),
        "triton_dispatch": _as_metrics_block(
            disp_timings,
            work_items=tokens_per_call,
            work_label="tokens_per_s",
            peak_mb=disp_peak,
        ),
        "tensorrt_plugin": trt_section,
    }


def _build_ff_config(
    cfg: GPUBenchConfig,
    *,
    use_triton_fastpath: bool,
) -> ApexXConfig:
    config = ApexXConfig()
    config.model.input_height = cfg.height
    config.model.input_width = cfg.width
    config.model.tile_size_l0 = 16
    config.model.tile_size_l1 = 8
    config.model.tile_size_l2 = 4
    config.model.nesting_depth = 1
    config.model.disable_nesting = False
    config.model.disable_ssm = False
    config.model.force_dense_routing = False
    config.model.kmax_l0 = cfg.kmax
    config.model.kmax_l1 = cfg.kmax * 2
    config.model.kmax_l2 = 0
    config.routing.budget_b1 = cfg.budget_b1
    config.routing.budget_b2 = cfg.budget_b2
    config.routing.budget_b3 = 0.0
    config.routing.budget_total = cfg.budget_total
    config.runtime.enable_runtime_plugins = use_triton_fastpath
    config.runtime.trt_enable = False
    config.runtime.ort_enable = True
    config.validate()
    return config


def _bench_ff_infer_path(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_triton_fastpath: bool,
    input_seed: int,
    state_dict: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    model_cfg = _build_ff_config(cfg, use_triton_fastpath=use_triton_fastpath)
    module = FFModule(channels=cfg.channels, config=model_cfg).to(device=device, dtype=dtype)
    if state_dict is not None:
        module.load_state_dict(state_dict, strict=True)
    module.eval()

    grid_h, grid_w = tile_grid_shape(cfg.height, cfg.width, model_cfg.model.tile_size_l0)
    k_tiles = grid_h * grid_w
    if cfg.kmax > k_tiles:
        raise ValueError(f"kmax={cfg.kmax} must be <= total l0 tiles ({k_tiles})")

    torch.manual_seed(input_seed)
    dense = torch.randn(
        (cfg.batch, cfg.channels, cfg.height, cfg.width),
        dtype=dtype,
        device=device,
    ).contiguous()
    utilities = torch.randn((cfg.batch, k_tiles), dtype=dtype, device=device).contiguous()
    split_utilities = torch.randn((cfg.batch, k_tiles), dtype=dtype, device=device).contiguous()
    boundary = torch.rand((cfg.batch, 1, cfg.height, cfg.width), dtype=dtype, device=device)
    uncertainty = torch.rand((cfg.batch, 1, cfg.height, cfg.width), dtype=dtype, device=device)

    @torch.inference_mode()
    def _run_once() -> Any:
        return module.forward_infer(
            dense_features=dense,
            utilities=utilities,
            split_utilities=split_utilities,
            boundary_proxy=boundary,
            uncertainty_proxy=uncertainty,
            enable_nesting=True,
        )

    first_out = _run_once()
    timings, peak_mb = _measure_ms(
        _run_once,
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    metrics = _as_metrics_block(
        timings,
        work_items=float(cfg.batch),
        work_label="frames_per_s",
        peak_mb=peak_mb,
    )
    selected_l0 = sum(len(row) for row in first_out.selected_l0)
    selected_l1 = sum(len(row) for row in first_out.selected_l1)
    metrics["selected_l0_per_call"] = selected_l0
    metrics["selected_l1_per_call"] = selected_l1
    metrics["budget_b1_spent"] = float(first_out.spent_budget_b1)
    metrics["budget_b2_spent"] = float(first_out.spent_budget_b2)
    return metrics


def _bench_end_to_end_infer(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    seed_all(cfg.seed + 17, deterministic=True)
    eager_cfg = _build_ff_config(cfg, use_triton_fastpath=False)
    eager_module = FFModule(channels=cfg.channels, config=eager_cfg).to(device=device, dtype=dtype)
    eager_state = {key: value.detach().clone() for key, value in eager_module.state_dict().items()}

    eager = _bench_ff_infer_path(
        cfg,
        device=device,
        dtype=dtype,
        use_triton_fastpath=False,
        input_seed=cfg.seed + 101,
        state_dict=eager_state,
    )
    fastpath = _bench_ff_infer_path(
        cfg,
        device=device,
        dtype=dtype,
        use_triton_fastpath=True,
        input_seed=cfg.seed + 101,
        state_dict=eager_state,
    )
    trt_engine = _bench_tensorrt_engine(cfg, device=device)
    return {
        "torch_eager": eager,
        "torch_triton_fastpath": fastpath,
        "tensorrt_engine": trt_engine,
    }


def _resolve_plugin_library_path(explicit_path: str) -> Path | None:
    source = (
        explicit_path.strip()
        if explicit_path
        else os.getenv("APEXX_TRT_PLUGIN_LIB", "").strip()
    )
    if not source:
        return None
    path = Path(source).expanduser().resolve()
    if not path.exists():
        return None
    return path


def _maybe_import_tensorrt() -> Any | None:
    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except Exception:
        return None
    return trt


def _load_plugin_library(path: Path) -> str | None:
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        return f"cdll_load_failed:{exc}"
    return None


def _get_plugin_creator(trt: Any, *, name: str) -> Any | None:
    registry = trt.get_plugin_registry()
    creator = None
    if hasattr(registry, "get_plugin_creator"):
        creator = registry.get_plugin_creator(name, "1", "apexx")
    if creator is not None:
        return creator
    for item in getattr(registry, "plugin_creator_list", []):
        if item.name == name and getattr(item, "plugin_namespace", "") == "apexx":
            return item
    return None


def _build_tilessm_trt_engine(
    trt: Any,
    creator: Any,
    *,
    batch: int,
    steps: int,
    channels: int,
) -> Any:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    tokens = network.add_input("tokens", trt.DataType.HALF, (batch, steps, channels))
    decay = network.add_input("decay", trt.DataType.HALF, (channels,))
    input_gain = network.add_input("input_gain", trt.DataType.HALF, (channels,))
    output_gain = network.add_input("output_gain", trt.DataType.HALF, (channels,))
    state_bias = network.add_input("state_bias", trt.DataType.HALF, (channels,))
    init_state = network.add_input("init_state", trt.DataType.HALF, (batch, channels))

    direction_field = torch.tensor([0], dtype=torch.int32).cpu().numpy()
    clamp_field = torch.tensor([1.0e4], dtype=torch.float32).cpu().numpy()
    fields = [
        trt.PluginField("direction", direction_field, trt.PluginFieldType.INT32),
        trt.PluginField("clamp_value", clamp_field, trt.PluginFieldType.FLOAT32),
    ]
    plugin = creator.create_plugin("tilessm_gpu_bench", trt.PluginFieldCollection(fields))
    if plugin is None:
        raise RuntimeError("failed to create TensorRT TileSSMScan plugin")

    layer = network.add_plugin_v2(
        [tokens, decay, input_gain, output_gain, state_bias, init_state],
        plugin,
    )
    y = layer.get_output(0)
    y.name = "y"
    final_state = layer.get_output(1)
    final_state.name = "final_state"
    network.mark_output(y)
    network.mark_output(final_state)

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    else:
        config.max_workspace_size = 1 << 20  # pragma: no cover

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(serialized)

    engine = builder.build_engine(network, config)  # pragma: no cover
    if engine is None:
        raise RuntimeError("TensorRT build_engine returned None")
    return engine


def _execute_legacy_engine(engine: Any, context: Any, buffers: dict[str, Tensor]) -> None:
    bindings = [0] * int(engine.num_bindings)
    for idx in range(int(engine.num_bindings)):
        name = str(engine.get_binding_name(idx))
        bindings[idx] = int(buffers[name].data_ptr())
    ok = context.execute_v2(bindings)
    if not ok:
        raise RuntimeError("TensorRT execute_v2 returned false")


def _execute_modern_engine(engine: Any, context: Any, buffers: dict[str, Tensor]) -> None:
    for i in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(i))
        context.set_tensor_address(name, int(buffers[name].data_ptr()))
    stream_ptr = int(torch.cuda.current_stream().cuda_stream)
    result = context.execute_async_v3(stream_ptr)
    if isinstance(result, bool) and not result:
        raise RuntimeError("TensorRT execute_async_v3 returned false")


def _bench_tensorrt_tilessm_plugin(
    cfg: GPUBenchConfig,
    *,
    device: torch.device,
    tokens: Tensor,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
) -> dict[str, Any]:
    trt = _maybe_import_tensorrt()
    if trt is None:
        return {"status": "skipped", "reason": "tensorrt_python_unavailable"}

    lib_path = _resolve_plugin_library_path(cfg.trt_plugin_lib)
    if lib_path is None:
        return {"status": "skipped", "reason": "missing_plugin_library"}
    lib_error = _load_plugin_library(lib_path)
    if lib_error is not None:
        return {"status": "skipped", "reason": lib_error}

    creator = _get_plugin_creator(trt, name="TileSSMScan")
    if creator is None:
        return {"status": "skipped", "reason": "tilessm_plugin_creator_not_registered"}

    try:
        engine = _build_tilessm_trt_engine(
            trt,
            creator,
            batch=cfg.batch,
            steps=cfg.steps,
            channels=cfg.channels,
        )
    except Exception as exc:
        return {"status": "skipped", "reason": f"engine_build_failed:{type(exc).__name__}"}

    if engine is None:
        return {"status": "skipped", "reason": "engine_build_returned_none"}

    init_state = torch.zeros((cfg.batch, cfg.channels), device=device, dtype=torch.float16)
    tensors: dict[str, Tensor] = {
        "tokens": tokens.to(dtype=torch.float16).contiguous(),
        "decay": decay.to(dtype=torch.float16).contiguous(),
        "input_gain": input_gain.to(dtype=torch.float16).contiguous(),
        "output_gain": output_gain.to(dtype=torch.float16).contiguous(),
        "state_bias": state_bias.to(dtype=torch.float16).contiguous(),
        "init_state": init_state,
        "y": torch.empty_like(tokens, dtype=torch.float16),
        "final_state": torch.empty((cfg.batch, cfg.channels), device=device, dtype=torch.float16),
    }
    context = engine.create_execution_context()

    def _run() -> None:
        if hasattr(engine, "num_bindings"):
            _execute_legacy_engine(engine, context, tensors)
        else:
            _execute_modern_engine(engine, context, tensors)

    timings, peak_mb = _measure_ms(
        _run,
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    tokens_per_call = float(cfg.batch * cfg.steps * cfg.channels)
    return {
        "status": "ok",
        "plugin_library": str(lib_path),
        "metrics": _as_metrics_block(
            timings,
            work_items=tokens_per_call,
            work_label="tokens_per_s",
            peak_mb=peak_mb,
        ),
    }


def _parse_shape_specs(specs: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    parsed: dict[str, tuple[int, ...]] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"invalid --trt-input-shape value: {raw!r}")
        name, shape_text = raw.split("=", 1)
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError(f"invalid --trt-input-shape value: {raw!r}")
        dims_raw = shape_text.strip().replace("x", " ").split()
        if not dims_raw:
            raise ValueError(f"invalid --trt-input-shape dims: {raw!r}")
        dims: list[int] = []
        for item in dims_raw:
            value = int(item)
            if value <= 0:
                raise ValueError(f"shape dims must be > 0: {raw!r}")
            dims.append(value)
        parsed[cleaned_name] = tuple(dims)
    return parsed


def _trt_dtype_to_torch(trt: Any, trt_dtype: Any) -> torch.dtype | None:
    mapping: dict[Any, torch.dtype] = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    bf16_dtype = getattr(trt.DataType, "BF16", None)
    if bf16_dtype is not None:
        mapping[bf16_dtype] = torch.bfloat16
    return mapping.get(trt_dtype)


def _random_tensor_for_shape(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=gen, dtype=dtype, device=device).contiguous()
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=gen, dtype=torch.int32, device=device).to(
            dtype=torch.bool
        )
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 8, shape, generator=gen, dtype=torch.int32, device=device).to(
            dtype=dtype
        )
    raise ValueError(f"unsupported TensorRT input dtype mapping: {dtype}")


def _resolve_dynamic_shape(
    raw_shape: tuple[int, ...],
    *,
    name: str,
    overrides: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    if all(dim > 0 for dim in raw_shape):
        return raw_shape
    if name in overrides:
        return overrides[name]
    return tuple(1 if dim < 0 else dim for dim in raw_shape)


def _bench_tensorrt_engine(cfg: GPUBenchConfig, *, device: torch.device) -> dict[str, Any]:
    engine_path = cfg.trt_engine_path.strip()
    if not engine_path:
        return {"status": "skipped", "reason": "missing_trt_engine_path"}
    path = Path(engine_path).expanduser().resolve()
    if not path.exists():
        return {"status": "skipped", "reason": f"engine_not_found:{path}"}

    trt = _maybe_import_tensorrt()
    if trt is None:
        return {"status": "skipped", "reason": "tensorrt_python_unavailable"}

    shape_overrides = _parse_shape_specs(cfg.trt_input_shapes)
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine_bytes = path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        return {"status": "skipped", "reason": "engine_deserialize_failed"}
    context = engine.create_execution_context()
    buffers: dict[str, Tensor] = {}

    if hasattr(engine, "num_bindings"):
        input_shapes: dict[int, tuple[int, ...]] = {}
        for idx in range(int(engine.num_bindings)):
            if not bool(engine.binding_is_input(idx)):
                continue
            name = str(engine.get_binding_name(idx))
            raw_shape = tuple(int(dim) for dim in engine.get_binding_shape(idx))
            shape = _resolve_dynamic_shape(raw_shape, name=name, overrides=shape_overrides)
            if hasattr(context, "set_binding_shape"):
                context.set_binding_shape(idx, shape)
            input_shapes[idx] = shape

        seed_base = cfg.seed + 41
        first_batch = 1
        for idx in range(int(engine.num_bindings)):
            name = str(engine.get_binding_name(idx))
            is_input = bool(engine.binding_is_input(idx))
            trt_dtype = engine.get_binding_dtype(idx)
            torch_dtype = _trt_dtype_to_torch(trt, trt_dtype)
            if torch_dtype is None:
                return {"status": "skipped", "reason": f"unsupported_dtype:{trt_dtype}"}
            if is_input:
                shape = input_shapes[idx]
            else:
                shape = tuple(int(dim) for dim in context.get_binding_shape(idx))
                if any(dim <= 0 for dim in shape):
                    return {"status": "skipped", "reason": f"unresolved_output_shape:{name}"}
            if shape and first_batch == 1:
                first_batch = int(shape[0])
            if is_input:
                tensor = _random_tensor_for_shape(
                    shape,
                    dtype=torch_dtype,
                    device=device,
                    seed=seed_base + idx,
                )
            else:
                tensor = torch.empty(shape, dtype=torch_dtype, device=device).contiguous()
            buffers[name] = tensor

        def _run() -> None:
            _execute_legacy_engine(engine, context, buffers)

        timings, peak_mb = _measure_ms(
            _run,
            warmup=cfg.warmup,
            iters=cfg.iters,
            device=device,
        )
        return {
            "status": "ok",
            "engine_path": str(path),
            "mode": "legacy_bindings",
            "metrics": _as_metrics_block(
                timings,
                work_items=float(first_batch),
                work_label="frames_per_s",
                peak_mb=peak_mb,
            ),
        }

    input_names: list[str] = []
    seed_base = cfg.seed + 57
    first_batch = 1
    for i in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(i))
        mode = engine.get_tensor_mode(name)
        is_input = mode == trt.TensorIOMode.INPUT
        raw_shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        if is_input:
            shape = _resolve_dynamic_shape(raw_shape, name=name, overrides=shape_overrides)
            if hasattr(context, "set_input_shape"):
                context.set_input_shape(name, shape)
            input_names.append(name)

    for i in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(i))
        mode = engine.get_tensor_mode(name)
        is_input = mode == trt.TensorIOMode.INPUT
        trt_dtype = engine.get_tensor_dtype(name)
        torch_dtype = _trt_dtype_to_torch(trt, trt_dtype)
        if torch_dtype is None:
            return {"status": "skipped", "reason": f"unsupported_dtype:{trt_dtype}"}

        if is_input:
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
        else:
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
        if any(dim <= 0 for dim in shape):
            return {"status": "skipped", "reason": f"unresolved_tensor_shape:{name}"}
        if shape and first_batch == 1 and name in input_names:
            first_batch = int(shape[0])

        if is_input:
            tensor = _random_tensor_for_shape(
                shape,
                dtype=torch_dtype,
                device=device,
                seed=seed_base + i,
            )
        else:
            tensor = torch.empty(shape, dtype=torch_dtype, device=device).contiguous()
        buffers[name] = tensor

    def _run_modern() -> None:
        _execute_modern_engine(engine, context, buffers)

    timings, peak_mb = _measure_ms(
        _run_modern,
        warmup=cfg.warmup,
        iters=cfg.iters,
        device=device,
    )
    return {
        "status": "ok",
        "engine_path": str(path),
        "mode": "modern_io_tensors",
        "metrics": _as_metrics_block(
            timings,
            work_items=float(first_batch),
            work_label="frames_per_s",
            peak_mb=peak_mb,
        ),
    }


def run_gpu_bench(cfg: GPUBenchConfig) -> dict[str, Any]:
    caps: RuntimeCaps = detect_runtime_caps()
    device = torch.device("cuda" if caps.cuda.available else "cpu")
    dtype = _normalize_dtype_for_device(_dtype_from_name(cfg.dtype), device=device)

    report: dict[str, Any] = {
        "schema_version": 1,
        "suite": "apex_x_gpu_benchmark",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "runtime_caps": caps.to_dict(),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
        },
        "config": {
            "batch": cfg.batch,
            "channels": cfg.channels,
            "height": cfg.height,
            "width": cfg.width,
            "tile_size": cfg.tile_size,
            "kmax": cfg.kmax,
            "steps": cfg.steps,
            "warmup": cfg.warmup,
            "iters": cfg.iters,
            "seed": cfg.seed,
            "budget_b1": cfg.budget_b1,
            "budget_b2": cfg.budget_b2,
            "budget_total": cfg.budget_total,
            "trt_engine_path": cfg.trt_engine_path,
            "trt_plugin_lib": cfg.trt_plugin_lib or os.getenv("APEXX_TRT_PLUGIN_LIB", ""),
            "trt_input_shapes": list(cfg.trt_input_shapes),
        },
    }

    if not caps.cuda.available:
        report["status"] = "skipped"
        report["reason"] = caps.cuda.reason or "cuda_unavailable"
        report["benchmarks"] = {}
        return report

    seed_all(cfg.seed, deterministic=True)
    report["status"] = "ok"
    report["benchmarks"] = {
        "tile_ops": {
            "tilepack": _bench_tilepack(cfg, device=device, dtype=dtype),
            "tileunpack": _bench_tileunpack(cfg, device=device, dtype=dtype),
            "fusion_gate": _bench_fusion(cfg, device=device, dtype=dtype),
        },
        "tilessm": _bench_tilessm(cfg, device=device, dtype=dtype),
        "end_to_end_infer": _bench_end_to_end_infer(cfg, device=device, dtype=dtype),
    }
    return report


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Apex-X GPU Benchmark Report",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- device: `{report.get('environment', {}).get('device', 'unknown')}`",
        f"- dtype: `{report.get('environment', {}).get('dtype', 'unknown')}`",
        f"- timestamp_utc: `{report.get('timestamp_utc', '')}`",
    ]
    if report.get("status") != "ok":
        lines.extend(
            [
                f"- reason: `{report.get('reason', 'unknown')}`",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    tile_ops = report["benchmarks"]["tile_ops"]
    lines.extend(
        [
            "",
            "## Tile Ops",
            "",
            (
                "| op | dispatch backend | ref p50 ms | "
                "dispatch p50 ms | ref p95 ms | dispatch p95 ms |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, name in (
        ("tilepack", "TilePack"),
        ("tileunpack", "TileUnpack"),
        ("fusion_gate", "FusionGate"),
    ):
        row = tile_ops[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row.get("backend", "unknown")),
                    _format_metric(row["reference"]["p50_ms"]),
                    _format_metric(row["dispatch"]["p50_ms"]),
                    _format_metric(row["reference"]["p95_ms"]),
                    _format_metric(row["dispatch"]["p95_ms"]),
                ]
            )
            + " |"
        )

    tilessm = report["benchmarks"]["tilessm"]
    lines.extend(
        [
            "",
            "## TileSSM",
            "",
            "| path | backend | p50 ms | p95 ms | tokens/s |",
            "| --- | --- | ---: | ---: | ---: |",
            "| torch ref | torch | "
            + " | ".join(
                [
                    _format_metric(tilessm["torch_reference"]["p50_ms"]),
                    _format_metric(tilessm["torch_reference"]["p95_ms"]),
                    _format_metric(tilessm["torch_reference"].get("tokens_per_s")),
                ]
            )
            + " |",
            "| dispatch | "
            + " | ".join(
                [
                    str(tilessm.get("backend", "unknown")),
                    _format_metric(tilessm["triton_dispatch"]["p50_ms"]),
                    _format_metric(tilessm["triton_dispatch"]["p95_ms"]),
                    _format_metric(tilessm["triton_dispatch"].get("tokens_per_s")),
                ]
            )
            + " |",
        ]
    )
    trt_plugin = tilessm.get("tensorrt_plugin", {})
    if trt_plugin.get("status") == "ok":
        lines.append(
            "| TensorRT plugin | trt | "
            + " | ".join(
                [
                    _format_metric(trt_plugin["metrics"]["p50_ms"]),
                    _format_metric(trt_plugin["metrics"]["p95_ms"]),
                    _format_metric(trt_plugin["metrics"].get("tokens_per_s")),
                ]
            )
            + " |"
        )
    else:
        lines.append(
            "| TensorRT plugin | skipped | n/a | n/a | n/a |"
        )

    infer = report["benchmarks"]["end_to_end_infer"]
    lines.extend(
        [
            "",
            "## End-to-End Infer",
            "",
            "| path | p50 ms | p95 ms | frames/s | peak memory mb |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in (
        ("torch_eager", "torch eager"),
        ("torch_triton_fastpath", "torch + triton fast-path"),
    ):
        row = infer[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_metric(row["p50_ms"]),
                    _format_metric(row["p95_ms"]),
                    _format_metric(row.get("frames_per_s")),
                    _format_metric(row.get("peak_memory_mb")),
                ]
            )
            + " |"
        )

    trt_engine = infer.get("tensorrt_engine", {})
    if trt_engine.get("status") == "ok":
        metrics = trt_engine["metrics"]
        lines.append(
            "| TensorRT engine | "
            + " | ".join(
                [
                    _format_metric(metrics["p50_ms"]),
                    _format_metric(metrics["p95_ms"]),
                    _format_metric(metrics.get("frames_per_s")),
                    _format_metric(metrics.get("peak_memory_mb")),
                ]
            )
            + " |"
        )
    else:
        lines.append("| TensorRT engine | n/a | n/a | n/a | n/a |")
        lines.append("")
        lines.append(
            f"TensorRT engine benchmark skipped: `{trt_engine.get('reason', 'unknown')}`."
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apex-X GPU benchmark suite")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--kmax", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--budget-b1", type=float, default=16.0)
    parser.add_argument("--budget-b2", type=float, default=8.0)
    parser.add_argument("--budget-total", type=float, default=32.0)
    parser.add_argument("--trt-engine-path", type=str, default="")
    parser.add_argument("--trt-plugin-lib", type=str, default="")
    parser.add_argument(
        "--trt-input-shape",
        action="append",
        default=[],
        help="Dynamic TRT input shape override, e.g. tokens=1x256x128",
    )
    parser.add_argument("--output-json", type=str, default="artifacts/perf_gpu.json")
    parser.add_argument("--output-md", type=str, default="artifacts/perf_gpu.md")
    return parser


def _write_text(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = GPUBenchConfig(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        tile_size=args.tile_size,
        kmax=args.kmax,
        steps=args.steps,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        dtype=args.dtype,
        budget_b1=args.budget_b1,
        budget_b2=args.budget_b2,
        budget_total=args.budget_total,
        trt_engine_path=args.trt_engine_path,
        trt_plugin_lib=args.trt_plugin_lib,
        trt_input_shapes=tuple(args.trt_input_shape),
    )
    report = run_gpu_bench(cfg)
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    md_text = render_markdown_summary(report)
    json_path = _write_text(args.output_json, json_text)
    md_path = _write_text(args.output_md, md_text)
    print(json_text)
    print(f"wrote_json={json_path}")
    print(f"wrote_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
