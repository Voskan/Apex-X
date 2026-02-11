from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from apex_x.kernels.triton.tilepack import tilepack_dispatch, tilepack_reference

_PLUGIN_LIB_HANDLE: ctypes.CDLL | None = None


def _maybe_import_tensorrt():
    try:
        import tensorrt as trt
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


def _get_tilepack_creator(trt):
    registry = trt.get_plugin_registry()
    creator = None
    if hasattr(registry, "get_plugin_creator"):
        creator = registry.get_plugin_creator("TilePack", "1", "apexx")
        if creator is None:
            creator = registry.get_plugin_creator("TilePack", "1", "")
    if creator is not None:
        if hasattr(creator, "plugin_namespace"):
            creator.plugin_namespace = ""
        return creator
    for item in registry.plugin_creator_list:
        if item.name == "TilePack" and getattr(item, "plugin_namespace", "") in {"apexx", ""}:
            if hasattr(item, "plugin_namespace"):
                item.plugin_namespace = ""
            return item
    pytest.skip("TilePack plugin creator was not registered (library loaded but creator missing)")


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

    x = network.add_input("x", trt.DataType.HALF, (batch, channels, height, width))
    idx = network.add_input("idx", trt.DataType.INT32, (batch, kmax))
    tile_size_field = np.array([tile_size], dtype=np.int32)
    field = trt.PluginField("tile_size", tile_size_field, trt.PluginFieldType.INT32)
    plugin = creator.create_plugin("tilepack_pytest", trt.PluginFieldCollection([field]))
    if plugin is None:
        pytest.skip("Failed to create TilePack plugin instance from creator")

    layer = network.add_plugin_v2([x, idx], plugin)
    out = layer.get_output(0)
    out.name = "packed"
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


def _execute_engine(engine, *, x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    context = engine.create_execution_context()
    if hasattr(engine, "num_bindings"):
        bindings = [0] * int(engine.num_bindings)
        bindings[int(engine.get_binding_index("x"))] = int(x.data_ptr())
        bindings[int(engine.get_binding_index("idx"))] = int(idx.data_ptr())
        bindings[int(engine.get_binding_index("packed"))] = int(out.data_ptr())
        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        torch.cuda.synchronize()
        return

    # TRT v10+ explicit tensor-address API
    context.set_tensor_address("x", int(x.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("idx", int(idx.data_ptr()))  # pragma: no cover - API dependent
    context.set_tensor_address("packed", int(out.data_ptr()))  # pragma: no cover - API dependent
    stream_ptr = int(torch.cuda.current_stream().cuda_stream)
    context.execute_async_v3(stream_ptr)  # pragma: no cover - API dependent
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("channels", "height", "width", "kmax", "tile_size"),
    (
        (8, 16, 16, 6, 4),
        (16, 32, 32, 8, 8),
    ),
)
def test_tensorrt_tilepack_parity_with_pytorch_reference(
    channels: int,
    height: int,
    width: int,
    kmax: int,
    tile_size: int,
) -> None:
    trt = _maybe_import_tensorrt()
    _load_plugin_library()
    creator = _get_tilepack_creator(trt)

    batch = 1
    torch.manual_seed(123)
    x = torch.randn((batch, channels, height, width), device="cuda", dtype=torch.float16)
    max_tile_index = (height // tile_size) * (width // tile_size) - 1
    idx = torch.linspace(
        0,
        max_tile_index,
        steps=kmax,
        device="cuda",
        dtype=torch.float32,
    ).round().to(dtype=torch.int32).view(1, kmax)

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
        (batch, kmax, channels, tile_size, tile_size),
        device="cuda",
        dtype=_torch_dtype_for_output(engine, trt, "packed"),
    )
    _execute_engine(engine, x=x, idx=idx, out=out)

    ref, _ = tilepack_reference(x, idx, tile_size)
    triton_dispatch = tilepack_dispatch(
        feature_map=x,
        indices=idx,
        tile_size=tile_size,
        prefer_triton=True,
        allow_fallback=False,
    )
    assert triton_dispatch.backend == "triton"
    assert torch.allclose(out.to(dtype=ref.dtype), ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(
        triton_dispatch.packed.to(dtype=ref.dtype),
        ref,
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(
        out.to(dtype=triton_dispatch.packed.dtype),
        triton_dispatch.packed,
        atol=1e-3,
        rtol=1e-3,
    )
