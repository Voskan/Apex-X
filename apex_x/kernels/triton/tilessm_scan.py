from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor

BackendKind = Literal["reference", "triton"]
ScanDirection = Literal["forward", "backward", "bidirectional"]
BidirectionalMergeMode = Literal["sum", "avg", "gated"]


_tilessm_scan_kernel: Any | None = None
_TRITON_MAX_STEPS = 4096
triton: Any
tl: Any

try:
    triton = __import__("triton")
    tl = __import__("triton.language", fromlist=["language"])
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class TritonTileSSMAvailability:
    triton_installed: bool
    cuda_available: bool
    cuda_device_count: int
    reason: str | None

    @property
    def available(self) -> bool:
        return self.triton_installed and self.cuda_available and self.cuda_device_count > 0


@dataclass(frozen=True, slots=True)
class TileSSMScanDispatchResult:
    y: Tensor
    final_state: Tensor
    backend: BackendKind
    fallback_reason: str | None


def get_triton_tilessm_availability() -> TritonTileSSMAvailability:
    triton_installed = importlib.util.find_spec("triton") is not None
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    reason: str | None = None
    if not triton_installed:
        reason = "triton_not_installed"
    elif _TRITON_IMPORT_ERROR is not None:
        reason = f"triton_import_failed:{_TRITON_IMPORT_ERROR}"
    elif not cuda_available:
        reason = "cuda_unavailable"
    elif device_count <= 0:
        reason = "cuda_device_not_found"

    return TritonTileSSMAvailability(
        triton_installed=triton_installed,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        reason=reason,
    )


def _validate_direction(direction: str) -> ScanDirection:
    if direction in {"forward", "backward", "bidirectional"}:
        return cast(ScanDirection, direction)
    raise ValueError("direction must be one of: forward, backward, bidirectional")


def _validate_merge_mode(merge_mode: str) -> BidirectionalMergeMode:
    if merge_mode in {"sum", "avg", "gated"}:
        return cast(BidirectionalMergeMode, merge_mode)
    raise ValueError("merge_mode must be one of: sum, avg, gated")


def _validate_inputs(
    *,
    tokens: Tensor,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None,
) -> tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B,K,C]")
    if tokens.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("tokens dtype must be float16, bfloat16, or float32")

    batch, _, channels = tokens.shape
    vectors = {
        "decay": decay,
        "input_gain": input_gain,
        "output_gain": output_gain,
        "state_bias": state_bias,
    }
    for name, value in vectors.items():
        if value.ndim != 1 or value.shape[0] != channels:
            raise ValueError(f"{name} must be [C] matching tokens channels")
        if value.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"{name} dtype must be float16, bfloat16, or float32")
    if init_state is not None:
        if init_state.ndim not in {2, 3}:
            raise ValueError("init_state must be [B,C] or [B,2,C]")
        if init_state.ndim == 2 and init_state.shape != (batch, channels):
            raise ValueError("init_state [B,C] must match tokens batch/channels")
        if init_state.ndim == 3 and init_state.shape != (batch, 2, channels):
            raise ValueError("init_state [B,2,C] must match tokens batch/channels")
        if init_state.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("init_state dtype must be float16, bfloat16, or float32")
    return batch, int(tokens.shape[1]), channels


def _sanitize_tokens(tokens: Tensor, clamp_value: float) -> Tensor:
    safe = torch.nan_to_num(
        tokens,
        nan=0.0,
        posinf=abs(float(clamp_value)),
        neginf=-abs(float(clamp_value)),
    )
    limit = abs(float(clamp_value))
    return safe.clamp(-limit, limit)


def _normalize_gate(
    merge_gate: Tensor | None,
    *,
    batch: int,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if merge_gate is None:
        return torch.full((1, 1, channels), 0.5, dtype=dtype, device=device)
    gate = merge_gate.to(dtype=dtype, device=device)
    if gate.ndim == 1:
        if gate.shape[0] != channels:
            raise ValueError("merge_gate [C] must match channels")
        gate = gate.view(1, 1, channels)
    elif gate.ndim == 3:
        if gate.shape[1] != 1 or gate.shape[2] != channels:
            raise ValueError("merge_gate [B,1,C] must match batch/channels")
        if gate.shape[0] != batch:
            raise ValueError("merge_gate batch must match tokens batch")
    else:
        raise ValueError("merge_gate must be [C] or [B,1,C]")
    return gate.clamp(0.0, 1.0)


def _split_init_states(
    init_state: Tensor | None,
    *,
    batch: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor | None, Tensor | None]:
    if init_state is None:
        return None, None
    state = init_state.to(device=device, dtype=dtype).contiguous()
    if state.ndim == 2:
        return state, state
    if state.ndim == 3:
        return state[:, 0, :].contiguous(), state[:, 1, :].contiguous()
    raise ValueError("init_state must be [B,C] or [B,2,C]")


def _merge_bidirectional(
    y_forward: Tensor,
    y_backward: Tensor,
    *,
    merge_mode: BidirectionalMergeMode,
    merge_gate: Tensor | None,
) -> Tensor:
    if y_forward.shape != y_backward.shape:
        raise ValueError("y_forward and y_backward shapes must match")
    if merge_mode == "sum":
        return (y_forward + y_backward).contiguous()
    if merge_mode == "avg":
        return (0.5 * (y_forward + y_backward)).contiguous()
    gate = _normalize_gate(
        merge_gate,
        batch=y_forward.shape[0],
        channels=y_forward.shape[2],
        dtype=y_forward.dtype,
        device=y_forward.device,
    )
    return (gate * y_forward + (1.0 - gate) * y_backward).contiguous()


def _scan_reference_forward(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None,
    clamp_value: float,
) -> tuple[Tensor, Tensor]:
    batch, steps, channels = tokens.shape
    tokens_sanitized = _sanitize_tokens(tokens, clamp_value)
    dtype = tokens_sanitized.dtype
    device = tokens_sanitized.device

    decay_32 = decay.to(device=device, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6).unsqueeze(0)
    one_minus = 1.0 - decay_32
    input_gain_32 = input_gain.to(device=device, dtype=torch.float32).unsqueeze(0)
    output_gain_32 = output_gain.to(device=device, dtype=torch.float32).unsqueeze(0)
    state_bias_32 = state_bias.to(device=device, dtype=torch.float32).unsqueeze(0)

    if init_state is None:
        state_32 = torch.zeros((batch, channels), dtype=torch.float32, device=device)
    else:
        state_32 = init_state.to(device=device, dtype=torch.float32).contiguous()

    outputs = torch.empty_like(tokens_sanitized)
    for step in range(steps):
        token_t = tokens_sanitized[:, step, :].to(dtype=torch.float32)
        driven = input_gain_32 * token_t + state_bias_32
        state_32 = decay_32 * state_32 + one_minus * driven
        outputs[:, step, :] = (output_gain_32 * state_32).to(dtype=dtype)
    return outputs.contiguous(), state_32.to(dtype=dtype).contiguous()


if triton is not None:

    @triton.jit
    def _tilessm_scan_kernel(
        tokens_ptr: Any,  # [B,K,C] contiguous
        init_state_ptr: Any,  # [B,C] contiguous
        decay_ptr: Any,  # [C]
        input_gain_ptr: Any,  # [C]
        output_gain_ptr: Any,  # [C]
        state_bias_ptr: Any,  # [C]
        out_ptr: Any,  # [B,K,C] contiguous
        final_state_ptr: Any,  # [B,C]
        batch: Any,
        channels: Any,
        CLAMP_VALUE: Any,
        STEPS: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        b = pid // channels
        c = pid - b * channels
        if b >= batch:
            return

        decay = tl.load(decay_ptr + c).to(tl.float32)
        decay = tl.minimum(tl.maximum(decay, 1e-6), 1.0 - 1e-6)
        one_minus = 1.0 - decay
        input_gain = tl.load(input_gain_ptr + c).to(tl.float32)
        output_gain = tl.load(output_gain_ptr + c).to(tl.float32)
        bias = tl.load(state_bias_ptr + c).to(tl.float32)

        state = tl.load(init_state_ptr + b * channels + c).to(tl.float32)
        clamp_abs = tl.abs(CLAMP_VALUE)

        for step in range(STEPS):
            offs = (b * STEPS + step) * channels + c
            token = tl.load(tokens_ptr + offs).to(tl.float32)
            token = tl.where(token == token, token, 0.0)
            token = tl.minimum(tl.maximum(token, -clamp_abs), clamp_abs)
            driven = input_gain * token + bias
            state = decay * state + one_minus * driven
            y = output_gain * state
            tl.store(out_ptr + offs, y)

        tl.store(final_state_ptr + b * channels + c, state)

else:
    _tilessm_scan_kernel = None


def _scan_triton_forward_single(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None,
    clamp_value: float,
) -> tuple[Tensor, Tensor]:
    if triton is None or _tilessm_scan_kernel is None:
        raise RuntimeError("Triton is not available")

    batch, steps, channels = tokens.shape
    if tokens.device.type != "cuda":
        raise ValueError("tilessm_scan_triton requires CUDA tokens")
    if steps > _TRITON_MAX_STEPS:
        raise ValueError(
            "tilessm_scan_triton single launch supports "
            f"K<={_TRITON_MAX_STEPS} for compile stability"
        )

    device = tokens.device
    dtype = tokens.dtype
    tokens_c = tokens.contiguous()
    out = torch.empty_like(tokens_c)

    init = (
        torch.zeros((batch, channels), dtype=dtype, device=device)
        if init_state is None
        else init_state.to(device=device, dtype=dtype).contiguous()
    )
    final_state = torch.empty((batch, channels), dtype=dtype, device=device)

    decay_c = decay.to(device=device, dtype=dtype).contiguous()
    input_gain_c = input_gain.to(device=device, dtype=dtype).contiguous()
    output_gain_c = output_gain.to(device=device, dtype=dtype).contiguous()
    state_bias_c = state_bias.to(device=device, dtype=dtype).contiguous()

    grid = (batch * channels,)
    _tilessm_scan_kernel[grid](
        tokens_c,
        init,
        decay_c,
        input_gain_c,
        output_gain_c,
        state_bias_c,
        out,
        final_state,
        batch,
        channels,
        float(clamp_value),
        STEPS=steps,
    )
    return out.contiguous(), final_state.contiguous()


def _scan_triton_forward(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None,
    clamp_value: float,
) -> tuple[Tensor, Tensor]:
    _, steps, _ = tokens.shape
    if steps <= _TRITON_MAX_STEPS:
        return _scan_triton_forward_single(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=init_state,
            clamp_value=clamp_value,
        )

    outputs: list[Tensor] = []
    state_next = init_state
    for start in range(0, steps, _TRITON_MAX_STEPS):
        stop = min(start + _TRITON_MAX_STEPS, steps)
        chunk_tokens = tokens[:, start:stop, :].contiguous()
        chunk_out, state_next = _scan_triton_forward_single(
            chunk_tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=state_next,
            clamp_value=clamp_value,
        )
        outputs.append(chunk_out)

    assert state_next is not None
    merged = torch.cat(outputs, dim=1)
    return merged.contiguous(), state_next.contiguous()


def tilessm_scan_reference(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None = None,
    clamp_value: float = 1e4,
    direction: ScanDirection = "forward",
    merge_mode: BidirectionalMergeMode = "avg",
    merge_gate: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    batch, _, channels = _validate_inputs(
        tokens=tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=init_state,
    )
    direction_v = _validate_direction(direction)
    merge_mode_v = _validate_merge_mode(merge_mode)
    state_forward, state_backward = _split_init_states(
        init_state,
        batch=batch,
        channels=channels,
        device=tokens.device,
        dtype=tokens.dtype,
    )

    if direction_v == "forward":
        return _scan_reference_forward(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=state_forward,
            clamp_value=clamp_value,
        )

    if direction_v == "backward":
        y_rev, state_b = _scan_reference_forward(
            torch.flip(tokens, dims=(1,)),
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=state_backward,
            clamp_value=clamp_value,
        )
        return torch.flip(y_rev, dims=(1,)).contiguous(), state_b

    y_f, state_f = _scan_reference_forward(
        tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=state_forward,
        clamp_value=clamp_value,
    )
    y_b_rev, state_b = _scan_reference_forward(
        torch.flip(tokens, dims=(1,)),
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=state_backward,
        clamp_value=clamp_value,
    )
    y_b = torch.flip(y_b_rev, dims=(1,))
    y = _merge_bidirectional(y_f, y_b, merge_mode=merge_mode_v, merge_gate=merge_gate)
    states = torch.stack((state_f, state_b), dim=1)
    return y, states.contiguous()


def tilessm_scan_triton(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None = None,
    clamp_value: float = 1e4,
    direction: ScanDirection = "forward",
    merge_mode: BidirectionalMergeMode = "avg",
    merge_gate: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    batch, _, channels = _validate_inputs(
        tokens=tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=init_state,
    )
    direction_v = _validate_direction(direction)
    merge_mode_v = _validate_merge_mode(merge_mode)
    state_forward, state_backward = _split_init_states(
        init_state,
        batch=batch,
        channels=channels,
        device=tokens.device,
        dtype=tokens.dtype,
    )

    if direction_v == "forward":
        return _scan_triton_forward(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=state_forward,
            clamp_value=clamp_value,
        )

    if direction_v == "backward":
        y_rev, state_b = _scan_triton_forward(
            torch.flip(tokens, dims=(1,)),
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=state_backward,
            clamp_value=clamp_value,
        )
        return torch.flip(y_rev, dims=(1,)).contiguous(), state_b

    y_f, state_f = _scan_triton_forward(
        tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=state_forward,
        clamp_value=clamp_value,
    )
    y_b_rev, state_b = _scan_triton_forward(
        torch.flip(tokens, dims=(1,)),
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=state_backward,
        clamp_value=clamp_value,
    )
    y_b = torch.flip(y_b_rev, dims=(1,))
    y = _merge_bidirectional(y_f, y_b, merge_mode=merge_mode_v, merge_gate=merge_gate)
    states = torch.stack((state_f, state_b), dim=1)
    return y, states.contiguous()


def tilessm_scan_dispatch(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    init_state: Tensor | None = None,
    clamp_value: float = 1e4,
    direction: ScanDirection = "forward",
    merge_mode: BidirectionalMergeMode = "avg",
    merge_gate: Tensor | None = None,
    prefer_triton: bool = True,
    allow_fallback: bool = True,
    inference_only: bool = True,
) -> TileSSMScanDispatchResult:
    requires_grad = tokens.requires_grad
    for tensor in (decay, input_gain, output_gain, state_bias, init_state, merge_gate):
        if tensor is not None and tensor.requires_grad:
            requires_grad = True
            break

    if inference_only and requires_grad:
        y, final_state = tilessm_scan_reference(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
            init_state=init_state,
            clamp_value=clamp_value,
            direction=direction,
            merge_mode=merge_mode,
            merge_gate=merge_gate,
        )
        return TileSSMScanDispatchResult(
            y=y,
            final_state=final_state,
            backend="reference",
            fallback_reason="autograd_not_supported_for_triton_tilessm_scan",
        )

    availability = get_triton_tilessm_availability()
    if prefer_triton and availability.available:
        try:
            y, final_state = tilessm_scan_triton(
                tokens,
                decay=decay,
                input_gain=input_gain,
                output_gain=output_gain,
                state_bias=state_bias,
                init_state=init_state,
                clamp_value=clamp_value,
                direction=direction,
                merge_mode=merge_mode,
                merge_gate=merge_gate,
            )
            return TileSSMScanDispatchResult(
                y=y,
                final_state=final_state,
                backend="triton",
                fallback_reason=None,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            y, final_state = tilessm_scan_reference(
                tokens,
                decay=decay,
                input_gain=input_gain,
                output_gain=output_gain,
                state_bias=state_bias,
                init_state=init_state,
                clamp_value=clamp_value,
                direction=direction,
                merge_mode=merge_mode,
                merge_gate=merge_gate,
            )
            return TileSSMScanDispatchResult(
                y=y,
                final_state=final_state,
                backend="reference",
                fallback_reason=f"triton_error:{type(exc).__name__}",
            )

    y, final_state = tilessm_scan_reference(
        tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        init_state=init_state,
        clamp_value=clamp_value,
        direction=direction,
        merge_mode=merge_mode,
        merge_gate=merge_gate,
    )
    fallback_reason = None
    if prefer_triton:
        fallback_reason = availability.reason or "triton_path_not_selected"
    return TileSSMScanDispatchResult(
        y=y,
        final_state=final_state,
        backend="reference",
        fallback_reason=fallback_reason,
    )


def scan(
    tokens: Tensor,
    *,
    decay: Tensor,
    input_gain: Tensor,
    output_gain: Tensor,
    state_bias: Tensor,
    direction: ScanDirection = "forward",
    merge_mode: BidirectionalMergeMode = "avg",
    merge_gate: Tensor | None = None,
    prefer_triton: bool = True,
) -> Tensor:
    """Clean directional API: scan(tokens, direction) -> y."""
    result = tilessm_scan_dispatch(
        tokens,
        decay=decay,
        input_gain=input_gain,
        output_gain=output_gain,
        state_bias=state_bias,
        direction=direction,
        merge_mode=merge_mode,
        merge_gate=merge_gate,
        prefer_triton=prefer_triton,
        allow_fallback=True,
        inference_only=True,
    )
    return result.y


__all__ = [
    "BackendKind",
    "ScanDirection",
    "BidirectionalMergeMode",
    "TritonTileSSMAvailability",
    "TileSSMScanDispatchResult",
    "get_triton_tilessm_availability",
    "tilessm_scan_reference",
    "tilessm_scan_triton",
    "tilessm_scan_dispatch",
    "scan",
]
