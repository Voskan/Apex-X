from __future__ import annotations

import pytest
import torch

import apex_x.kernels.triton.tilessm_scan as tilessm_mod
from apex_x.kernels.triton.tilessm_scan import (
    BidirectionalMergeMode,
    get_triton_tilessm_availability,
    scan,
    tilessm_scan_dispatch,
    tilessm_scan_reference,
)
from apex_x.runtime import evaluate_parity_outputs
from apex_x.utils import StableStateSpaceScan
from apex_x.utils.repro import seed_all


def _scan_parameters(
    scan: StableStateSpaceScan, *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        "decay": scan.constrained_decay().to(device=device, dtype=dtype),
        "input_gain": scan.constrained_input_gain().to(device=device, dtype=dtype),
        "output_gain": scan.constrained_output_gain().to(device=device, dtype=dtype),
        "state_bias": scan.state_bias.to(device=device, dtype=dtype),
    }


def _torch_forward_recurrence(
    tokens: torch.Tensor,
    *,
    decay: torch.Tensor,
    input_gain: torch.Tensor,
    output_gain: torch.Tensor,
    state_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps, channels = tokens.shape
    tokens_sanitized = torch.nan_to_num(tokens, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    state = torch.zeros((batch, channels), dtype=torch.float32, device=tokens.device)
    y = torch.empty_like(tokens_sanitized)

    decay_f = (
        decay.to(dtype=torch.float32, device=tokens.device).reshape(1, -1).clamp(1e-6, 1.0 - 1e-6)
    )
    one_minus = 1.0 - decay_f
    in_gain = input_gain.to(dtype=torch.float32, device=tokens.device).reshape(1, -1)
    out_gain = output_gain.to(dtype=torch.float32, device=tokens.device).reshape(1, -1)
    bias = state_bias.to(dtype=torch.float32, device=tokens.device).reshape(1, -1)

    for k in range(steps):
        driven = in_gain * tokens_sanitized[:, k, :].to(dtype=torch.float32) + bias
        state = decay_f * state + one_minus * driven
        y[:, k, :] = (out_gain * state).to(dtype=tokens.dtype)
    return y, state.to(dtype=tokens.dtype)


def test_tilessm_reference_matches_stable_scan_forward() -> None:
    seed_all(41, deterministic=True)
    tokens = torch.randn((2, 19, 8), dtype=torch.float32)
    scan = StableStateSpaceScan(channels=8)
    params = _scan_parameters(scan, device=tokens.device, dtype=tokens.dtype)

    out_ref, state_ref = tilessm_scan_reference(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    out_torch, state_torch = scan(tokens)

    report_out = evaluate_parity_outputs(
        case_name="tilessm-reference-vs-stable-scan-out",
        reference_backend="stable_scan",
        candidate_backend="tilessm_reference",
        reference_output=out_torch,
        candidate_output=out_ref,
    )
    report_state = evaluate_parity_outputs(
        case_name="tilessm-reference-vs-stable-scan-state",
        reference_backend="stable_scan",
        candidate_backend="tilessm_reference",
        reference_output=state_torch,
        candidate_output=state_ref,
    )
    assert report_out.passed is True
    assert report_state.passed is True


def test_tilessm_dispatch_cpu_fallback_matches_reference() -> None:
    seed_all(43, deterministic=True)
    tokens = torch.randn((3, 17, 6), dtype=torch.float32)
    scan = StableStateSpaceScan(channels=6)
    params = _scan_parameters(scan, device=tokens.device, dtype=tokens.dtype)

    dispatch = tilessm_scan_dispatch(
        tokens,
        prefer_triton=True,
        allow_fallback=True,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    expected_y, expected_state = tilessm_scan_reference(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )

    assert dispatch.backend == "reference"
    report_y = evaluate_parity_outputs(
        case_name="tilessm-dispatch-cpu-y",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_dispatch",
        reference_output=expected_y,
        candidate_output=dispatch.y,
    )
    report_state = evaluate_parity_outputs(
        case_name="tilessm-dispatch-cpu-state",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_dispatch",
        reference_output=expected_state,
        candidate_output=dispatch.final_state,
    )
    assert report_y.passed is True
    assert report_state.passed is True


def test_tilessm_dispatch_autograd_fallback_for_inference_only() -> None:
    seed_all(47, deterministic=True)
    tokens = torch.randn((1, 13, 5), dtype=torch.float32, requires_grad=True)
    scan = StableStateSpaceScan(channels=5)
    params = _scan_parameters(scan, device=tokens.device, dtype=tokens.dtype)

    dispatch = tilessm_scan_dispatch(
        tokens,
        inference_only=True,
        prefer_triton=True,
        allow_fallback=True,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    loss = dispatch.y.square().mean() + dispatch.final_state.square().mean()
    loss.backward()

    assert dispatch.backend == "reference"
    assert dispatch.fallback_reason == "autograd_not_supported_for_triton_tilessm_scan"
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_tilessm_availability_object_cpu_safe() -> None:
    availability = get_triton_tilessm_availability()
    assert isinstance(availability.available, bool)
    if not availability.available:
        assert availability.reason is not None


def test_tilessm_backward_direction_parity_vs_torch_manual() -> None:
    seed_all(67, deterministic=True)
    tokens = torch.randn((2, 23, 7), dtype=torch.float32)
    scan_module = StableStateSpaceScan(channels=7)
    params = _scan_parameters(scan_module, device=tokens.device, dtype=tokens.dtype)

    y_dispatch, state_dispatch = tilessm_scan_reference(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="backward",
    )

    y_rev, state_torch = _torch_forward_recurrence(
        torch.flip(tokens, dims=(1,)),
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    y_torch = torch.flip(y_rev, dims=(1,))

    rep_y = evaluate_parity_outputs(
        case_name="tilessm-backward-y",
        reference_backend="torch_manual",
        candidate_backend="tilessm_reference",
        reference_output=y_torch,
        candidate_output=y_dispatch,
    )
    rep_state = evaluate_parity_outputs(
        case_name="tilessm-backward-state",
        reference_backend="torch_manual",
        candidate_backend="tilessm_reference",
        reference_output=state_torch,
        candidate_output=state_dispatch,
    )
    assert rep_y.passed is True
    assert rep_state.passed is True


def test_tilessm_bidirectional_merge_modes_parity_vs_torch_manual() -> None:
    seed_all(71, deterministic=True)
    tokens = torch.randn((2, 15, 6), dtype=torch.float32)
    scan_module = StableStateSpaceScan(channels=6)
    params = _scan_parameters(scan_module, device=tokens.device, dtype=tokens.dtype)
    gate = torch.tensor([0.1, 0.4, 0.7, 0.2, 0.9, 0.5], dtype=tokens.dtype)

    y_f, state_f = _torch_forward_recurrence(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    y_b_rev, state_b = _torch_forward_recurrence(
        torch.flip(tokens, dims=(1,)),
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    y_b = torch.flip(y_b_rev, dims=(1,))
    expected_avg = 0.5 * (y_f + y_b)
    expected_sum = y_f + y_b
    gate_broadcast = gate.view(1, 1, -1)
    expected_gated = gate_broadcast * y_f + (1.0 - gate_broadcast) * y_b
    expected_state = torch.stack((state_f, state_b), dim=1)

    mode_expected: tuple[tuple[BidirectionalMergeMode, torch.Tensor], ...] = (
        ("avg", expected_avg),
        ("sum", expected_sum),
        ("gated", expected_gated),
    )
    for mode, expected in mode_expected:
        y_ref, state_ref = tilessm_scan_reference(
            tokens,
            decay=params["decay"],
            input_gain=params["input_gain"],
            output_gain=params["output_gain"],
            state_bias=params["state_bias"],
            direction="bidirectional",
            merge_mode=mode,
            merge_gate=(gate if mode == "gated" else None),
        )
        rep_y = evaluate_parity_outputs(
            case_name=f"tilessm-bidir-{mode}-y",
            reference_backend="torch_manual",
            candidate_backend="tilessm_reference",
            reference_output=expected,
            candidate_output=y_ref,
        )
        rep_state = evaluate_parity_outputs(
            case_name=f"tilessm-bidir-{mode}-state",
            reference_backend="torch_manual",
            candidate_backend="tilessm_reference",
            reference_output=expected_state,
            candidate_output=state_ref,
        )
        assert rep_y.passed is True
        assert rep_state.passed is True
        assert state_ref.shape == (tokens.shape[0], 2, tokens.shape[2])


def test_clean_scan_api_returns_y_for_direction() -> None:
    seed_all(73, deterministic=True)
    tokens = torch.randn((1, 9, 4), dtype=torch.float32)
    scan_module = StableStateSpaceScan(channels=4)
    params = _scan_parameters(scan_module, device=tokens.device, dtype=tokens.dtype)

    y = scan(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="backward",
        prefer_triton=False,
    )
    assert y.shape == tokens.shape
    assert torch.isfinite(y).all()


def test_tilessm_triton_forward_chunking_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = torch.randn((1, 9000, 2), dtype=torch.float32)
    params = {
        "decay": torch.ones((2,), dtype=torch.float32),
        "input_gain": torch.ones((2,), dtype=torch.float32),
        "output_gain": torch.ones((2,), dtype=torch.float32),
        "state_bias": torch.zeros((2,), dtype=torch.float32),
    }
    launches: list[int] = []

    def _fake_scan_single(
        chunk_tokens: torch.Tensor,
        *,
        decay: torch.Tensor,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        state_bias: torch.Tensor,
        init_state: torch.Tensor | None,
        clamp_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (decay, input_gain, output_gain, state_bias, clamp_value)
        launches.append(int(chunk_tokens.shape[1]))
        state = (
            torch.zeros((chunk_tokens.shape[0], chunk_tokens.shape[2]), dtype=chunk_tokens.dtype)
            if init_state is None
            else init_state
        )
        final_state = state + float(chunk_tokens.shape[1])
        y = torch.full_like(chunk_tokens, fill_value=float(final_state[0, 0].item()))
        return y, final_state

    monkeypatch.setattr(tilessm_mod, "_scan_triton_forward_single", _fake_scan_single)
    out, final_state = tilessm_mod._scan_triton_forward(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        init_state=None,
        clamp_value=1e4,
    )

    assert launches == [4096, 4096, 808]
    assert out.shape == tokens.shape
    assert torch.all(out[:, :4096, :] == 4096.0)
    assert torch.all(out[:, 4096:8192, :] == 8192.0)
    assert torch.all(out[:, 8192:, :] == 9000.0)
    assert torch.all(final_state == 9000.0)


def test_tilessm_triton_forward_single_launch_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = torch.randn((1, 256, 3), dtype=torch.float32)
    params = {
        "decay": torch.ones((3,), dtype=torch.float32),
        "input_gain": torch.ones((3,), dtype=torch.float32),
        "output_gain": torch.ones((3,), dtype=torch.float32),
        "state_bias": torch.zeros((3,), dtype=torch.float32),
    }
    launches: list[int] = []

    def _fake_scan_single(
        chunk_tokens: torch.Tensor,
        *,
        decay: torch.Tensor,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        state_bias: torch.Tensor,
        init_state: torch.Tensor | None,
        clamp_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (decay, input_gain, output_gain, state_bias, init_state, clamp_value)
        launches.append(int(chunk_tokens.shape[1]))
        final_state = torch.full(
            (chunk_tokens.shape[0], chunk_tokens.shape[2]),
            fill_value=float(chunk_tokens.shape[1]),
            dtype=chunk_tokens.dtype,
        )
        return chunk_tokens, final_state

    monkeypatch.setattr(tilessm_mod, "_scan_triton_forward_single", _fake_scan_single)
    out, final_state = tilessm_mod._scan_triton_forward(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        init_state=None,
        clamp_value=1e4,
    )

    assert launches == [256]
    assert torch.equal(out, tokens)
    assert torch.all(final_state == 256.0)
