from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from apex_x.kernels.triton.tilessm_scan import tilessm_scan_dispatch, tilessm_scan_reference

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
    steps: int,
    channels: int,
    direction: int,
    clamp_value: float = 1.0e4,
):
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

    direction_field = np.array([direction], dtype=np.int32)
    clamp_field = np.array([clamp_value], dtype=np.float32)
    fields = [
        trt.PluginField("direction", direction_field, trt.PluginFieldType.INT32),
        trt.PluginField("clamp_value", clamp_field, trt.PluginFieldType.FLOAT32),
    ]
    plugin = creator.create_plugin("tilessm_pytest", trt.PluginFieldCollection(fields))
    if plugin is None:
        pytest.skip("Failed to create TileSSMScan plugin instance")

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
    else:  # pragma: no cover - TRT API version dependent
        config.max_workspace_size = 1 << 20
    if hasattr(config, "set_flag") and hasattr(trt, "BuilderFlag"):
        config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(serialized)
    return builder.build_engine(network, config)  # pragma: no cover - TRT API version dependent


def _execute_engine(
    engine,
    *,
    tokens: torch.Tensor,
    decay: torch.Tensor,
    input_gain: torch.Tensor,
    output_gain: torch.Tensor,
    state_bias: torch.Tensor,
    init_state: torch.Tensor,
    y_out: torch.Tensor,
    final_state_out: torch.Tensor,
) -> None:
    context = engine.create_execution_context()
    if hasattr(engine, "num_bindings"):
        bindings = [0] * int(engine.num_bindings)
        bindings[int(engine.get_binding_index("tokens"))] = int(tokens.data_ptr())
        bindings[int(engine.get_binding_index("decay"))] = int(decay.data_ptr())
        bindings[int(engine.get_binding_index("input_gain"))] = int(input_gain.data_ptr())
        bindings[int(engine.get_binding_index("output_gain"))] = int(output_gain.data_ptr())
        bindings[int(engine.get_binding_index("state_bias"))] = int(state_bias.data_ptr())
        bindings[int(engine.get_binding_index("init_state"))] = int(init_state.data_ptr())
        bindings[int(engine.get_binding_index("y"))] = int(y_out.data_ptr())
        bindings[int(engine.get_binding_index("final_state"))] = int(final_state_out.data_ptr())
        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        torch.cuda.synchronize()
        return

    context.set_tensor_address("tokens", int(tokens.data_ptr()))  # pragma: no cover - API dep
    context.set_tensor_address("decay", int(decay.data_ptr()))  # pragma: no cover - API dep
    context.set_tensor_address("input_gain", int(input_gain.data_ptr()))  # pragma: no cover
    context.set_tensor_address("output_gain", int(output_gain.data_ptr()))  # pragma: no cover
    context.set_tensor_address("state_bias", int(state_bias.data_ptr()))  # pragma: no cover
    context.set_tensor_address("init_state", int(init_state.data_ptr()))  # pragma: no cover
    context.set_tensor_address("y", int(y_out.data_ptr()))  # pragma: no cover
    context.set_tensor_address("final_state", int(final_state_out.data_ptr()))  # pragma: no cover
    stream_ptr = int(torch.cuda.current_stream().cuda_stream)
    context.execute_async_v3(stream_ptr)  # pragma: no cover - API dependent
    torch.cuda.synchronize()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("direction", "dir_name"), [(0, "forward"), (1, "backward")])
@pytest.mark.parametrize(
    ("batch", "steps", "channels"),
    (
        (2, 16, 12),
        (1, 32, 8),
    ),
)
def test_tensorrt_tilessm_parity_with_torch_reference(
    direction: int,
    dir_name: str,
    batch: int,
    steps: int,
    channels: int,
) -> None:
    trt = _maybe_import_tensorrt()
    _load_plugin_library()
    creator = _get_creator(trt, name="TileSSMScan")

    torch.manual_seed(42)
    tokens = torch.randn((batch, steps, channels), device="cuda", dtype=torch.float16)
    decay = torch.full((channels,), 0.85, device="cuda", dtype=torch.float16)
    input_gain = torch.linspace(0.4, 0.8, channels, device="cuda", dtype=torch.float16)
    output_gain = torch.linspace(0.6, 1.0, channels, device="cuda", dtype=torch.float16)
    state_bias = torch.linspace(-0.05, 0.05, channels, device="cuda", dtype=torch.float16)
    init_state = torch.randn((batch, channels), device="cuda", dtype=torch.float16) * 0.1

    engine = _build_engine(
        trt,
        creator,
        batch=batch,
        steps=steps,
        channels=channels,
        direction=direction,
    )
    if engine is None:
        pytest.skip("TensorRT engine build returned None")

    y_out = torch.empty(
        tokens.shape,
        device=tokens.device,
        dtype=_torch_dtype_for_output(engine, trt, "y"),
    )
    final_state_out = torch.empty(
        (batch, channels),
        device=tokens.device,
        dtype=_torch_dtype_for_output(engine, trt, "final_state"),
    )
    _execute_engine(
        engine,
        tokens=tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=init_state,
        y_out=y_out,
        final_state_out=final_state_out,
    )

    ref_y, ref_final = tilessm_scan_reference(
        tokens=tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=init_state,
        direction=dir_name,  # type: ignore[arg-type]
    )
    assert torch.allclose(y_out.to(dtype=ref_y.dtype), ref_y, atol=3e-2, rtol=3e-2)
    assert torch.allclose(
        final_state_out.to(dtype=ref_final.dtype),
        ref_final,
        atol=3e-2,
        rtol=3e-2,
    )

    triton_dispatch = tilessm_scan_dispatch(
        tokens=tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=init_state,
        direction=dir_name,  # type: ignore[arg-type]
        prefer_triton=True,
        allow_fallback=False,
    )
    assert triton_dispatch.backend == "triton"
    assert torch.allclose(triton_dispatch.y.to(dtype=ref_y.dtype), ref_y, atol=3e-2, rtol=3e-2)
    assert torch.allclose(
        triton_dispatch.final_state.to(dtype=ref_final.dtype),
        ref_final,
        atol=3e-2,
        rtol=3e-2,
    )
    assert torch.allclose(y_out.to(dtype=triton_dispatch.y.dtype), triton_dispatch.y, atol=3e-2, rtol=3e-2)
    assert torch.allclose(
        final_state_out.to(dtype=triton_dispatch.final_state.dtype),
        triton_dispatch.final_state,
        atol=3e-2,
        rtol=3e-2,
    )
