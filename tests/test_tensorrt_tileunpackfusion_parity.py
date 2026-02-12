from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest
import torch

from apex_x.model.fusion_gate import FusionGate
from apex_x.tiles.torch_ops import TileUnpackTorch

_PLUGIN_LIB_HANDLE: ctypes.CDLL | None = None


def _maybe_import_tensorrt():
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependent
        pytest.skip(f"TensorRT Python is not available: {type(exc).__name__}: {exc}")
    return trt


def _load_plugin_library() -> Path:
    global _PLUGIN_LIB_HANDLE
    lib_env = os.getenv("APEXX_TRT_PLUGIN_LIB", "").strip()
    if not lib_env:
        pytest.skip("APEXX_TRT_PLUGIN_LIB is not set")
    lib_path = Path(lib_env).expanduser().resolve()
    if not lib_path.exists():
        pytest.skip(f"APEXX_TRT_PLUGIN_LIB does not exist: {lib_path}")
    if _PLUGIN_LIB_HANDLE is None:
        _PLUGIN_LIB_HANDLE = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    return lib_path


def _get_creator(trt, *, name: str):
    registry = trt.get_plugin_registry()
    creator = None
    if hasattr(registry, "get_plugin_creator"):
        creator = registry.get_plugin_creator(name, "1", "apexx")
        if creator is None:
            creator = registry.get_plugin_creator(name, "1", "")
    if creator is not None:
        if hasattr(creator, "plugin_namespace"):
            creator.plugin_namespace = ""
        return creator
    for item in registry.plugin_creator_list:
        if item.name == name and getattr(item, "plugin_namespace", "") in {"apexx", ""}:
            if hasattr(item, "plugin_namespace"):
                item.plugin_namespace = ""
            return item
    pytest.skip(f"{name} plugin creator was not registered")


def _build_engine(
    trt,
    creator,
    *,
    batch: int,
    channels: int,
    height: int,
    width: int,
    kmax: int,
    tile_size: int,
):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    base = network.add_input("base", trt.DataType.HALF, (batch, channels, height, width))
    packed = network.add_input(
        "packed",
        trt.DataType.HALF,
        (batch, kmax, channels, tile_size, tile_size),
    )
    idx = network.add_input("idx", trt.DataType.INT32, (batch, kmax))
    levels = network.add_input("levels", trt.DataType.INT32, (batch, kmax))
    alpha = network.add_input("alpha", trt.DataType.HALF, (batch, 1, height, width))

    plugin = creator.create_plugin("tileunpackfusion_pytest", trt.PluginFieldCollection([]))
    if plugin is None:
        pytest.skip("Failed to create TileUnpackFusion plugin instance")

    layer = network.add_plugin_v2([base, packed, idx, levels, alpha], plugin)
    out = layer.get_output(0)
    out.name = "merged"
    network.mark_output(out)

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    else:  # pragma: no cover - TRT API version dependent
        config.max_workspace_size = 1 << 20
    if hasattr(config, "set_flag") and hasattr(trt, "BuilderFlag"):
        config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(serialized)
    return builder.build_engine(network, config)  # pragma: no cover - TRT API version dependent


def _torch_dtype_for_output(engine, trt, name: str) -> torch.dtype:
    if hasattr(engine, "num_bindings"):
        binding_index = int(engine.get_binding_index(name))
        dtype = engine.get_binding_dtype(binding_index)
    else:
        dtype = engine.get_tensor_dtype(name)
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.INT32:
        return torch.int32
    raise RuntimeError(f"unsupported TensorRT dtype for {name}: {dtype}")


def _execute_engine(
    engine,
    *,
    base: torch.Tensor,
    packed: torch.Tensor,
    idx: torch.Tensor,
    levels: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> None:
    context = engine.create_execution_context()
    if hasattr(engine, "num_bindings"):
        bindings = [0] * int(engine.num_bindings)
        bindings[int(engine.get_binding_index("base"))] = int(base.data_ptr())
        bindings[int(engine.get_binding_index("packed"))] = int(packed.data_ptr())
        bindings[int(engine.get_binding_index("idx"))] = int(idx.data_ptr())
        bindings[int(engine.get_binding_index("levels"))] = int(levels.data_ptr())
        bindings[int(engine.get_binding_index("alpha"))] = int(alpha.data_ptr())
        bindings[int(engine.get_binding_index("merged"))] = int(out.data_ptr())
        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        torch.cuda.synchronize()
        return

    context.set_tensor_address("base", int(base.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("packed", int(packed.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("idx", int(idx.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("levels", int(levels.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("alpha", int(alpha.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("merged", int(out.data_ptr()))  # pragma: no cover - API dependent
    stream_ptr = int(torch.cuda.current_stream().cuda_stream)
    context.execute_async_v3(stream_ptr)  # pragma: no cover - API dependent
    torch.cuda.synchronize()


def _meta_from_indices(
    idx: torch.Tensor,
    *,
    tile_size: int,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    grid_h = height // tile_size
    grid_w = width // tile_size
    origins_y = (idx // grid_w) * tile_size
    origins_x = (idx % grid_w) * tile_size
    origins = torch.stack((origins_y, origins_x), dim=-1)
    return {
        "indices": idx.to(dtype=torch.int64),
        "origins": origins.to(dtype=torch.int64),
        "tile_size": torch.tensor(tile_size, dtype=torch.int64, device=idx.device),
        "grid": torch.tensor([grid_h, grid_w], dtype=torch.int64, device=idx.device),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("channels", "height", "width", "tile_size", "kmax"),
    (
        (4, 16, 16, 4, 4),
        (8, 32, 32, 8, 4),
    ),
)
def test_tensorrt_tileunpackfusion_parity_with_torch_reference(
    channels: int,
    height: int,
    width: int,
    tile_size: int,
    kmax: int,
) -> None:
    trt = _maybe_import_tensorrt()
    _load_plugin_library()
    creator = _get_creator(trt, name="TileUnpackFusion")

    batch = 1
    torch.manual_seed(17)

    base = torch.randn((batch, channels, height, width), device="cuda", dtype=torch.float16)
    packed = torch.randn(
        (batch, kmax, channels, tile_size, tile_size),
        device="cuda",
        dtype=torch.float16,
    )
    # Crafted overlaps to exercise level-based priority semantics.
    last_tile = (height // tile_size) * (width // tile_size) - 1
    idx = torch.tensor([[0, 0, 0, last_tile]], device="cuda", dtype=torch.int32)
    levels = torch.tensor([[0, 1, 2, 2]], device="cuda", dtype=torch.int32)

    gate = FusionGate().to(device="cuda")
    with torch.no_grad():
        gate.boundary_log_weight.fill_(0.8)
        gate.uncertainty_log_weight.fill_(0.4)
        gate.bias.fill_(-0.2)
    boundary_proxy = torch.rand((batch, 1, height, width), device="cuda", dtype=torch.float16)
    uncertainty_proxy = torch.rand((batch, 1, height, width), device="cuda", dtype=torch.float16)
    alpha = gate.compute_alpha(boundary_proxy, uncertainty_proxy, like=base).to(dtype=base.dtype)

    unpacked = base.clone()
    priority_map: torch.Tensor | None = None
    idx_i64 = idx.to(dtype=torch.int64)
    levels_i64 = levels.to(dtype=torch.int64)
    unique_levels = sorted({int(v) for v in levels_i64.flatten().tolist()})
    for level in unique_levels:
        select = (levels_i64 == level).squeeze(0)
        packed_level = packed[:, select, ...].contiguous()
        idx_level = idx_i64[:, select].contiguous()
        meta_level = _meta_from_indices(
            idx_level,
            tile_size=tile_size,
            height=height,
            width=width,
        )
        unpacked, priority_map = TileUnpackTorch().unpack(
            base_map=unpacked,
            packed_out=packed_level,
            meta=meta_level,
            level_priority=level,
            priority_map=priority_map,
            overlap_mode="override",
        )
    reference = (base + alpha * (unpacked - base)).contiguous()

    engine = _build_engine(
        trt,
        creator,
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        kmax=kmax,
        tile_size=tile_size,
    )
    if engine is None:
        pytest.skip("TensorRT engine build returned None")

    out = torch.empty(
        base.shape,
        device=base.device,
        dtype=_torch_dtype_for_output(engine, trt, "merged"),
    )
    _execute_engine(engine, base=base, packed=packed, idx=idx, levels=levels, alpha=alpha, out=out)
    assert torch.allclose(out.to(dtype=reference.dtype), reference, atol=2e-3, rtol=2e-3)
