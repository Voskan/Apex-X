from __future__ import annotations

import pytest
import torch

from apex_x.kernels.triton.tilessm_scan import (
    get_triton_tilessm_availability,
    tilessm_scan_dispatch,
    tilessm_scan_reference,
    tilessm_scan_triton,
)
from apex_x.runtime import ToleranceConfig, evaluate_parity_outputs
from apex_x.utils.repro import seed_all


def _stable_random_params(
    channels: int, *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    decay = torch.rand((channels,), device=device, dtype=dtype) * 0.8 + 0.1
    input_gain = torch.rand((channels,), device=device, dtype=dtype) * 1.5 + 0.05
    output_gain = torch.rand((channels,), device=device, dtype=dtype) * 1.5 + 0.05
    state_bias = torch.randn((channels,), device=device, dtype=dtype) * 0.1
    return {
        "decay": decay,
        "input_gain": input_gain,
        "output_gain": output_gain,
        "state_bias": state_bias,
    }


@pytest.mark.skipif(
    not get_triton_tilessm_availability().available,
    reason="Triton TileSSM scan requires CUDA + Triton",
)
def test_tilessm_triton_parity_fp16() -> None:
    seed_all(53, deterministic=True)
    device = torch.device("cuda")
    tokens = torch.randn((2, 64, 16), device=device, dtype=torch.float16)
    params = _stable_random_params(16, device=device, dtype=torch.float16)

    y_triton, state_triton = tilessm_scan_triton(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    y_ref, state_ref = tilessm_scan_reference(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )

    report_y = evaluate_parity_outputs(
        case_name="tilessm-triton-parity-y",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_triton",
        reference_output=y_ref,
        candidate_output=y_triton,
        tolerances=ToleranceConfig(),
    )
    report_state = evaluate_parity_outputs(
        case_name="tilessm-triton-parity-state",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_triton",
        reference_output=state_ref,
        candidate_output=state_triton,
        tolerances=ToleranceConfig(),
    )
    assert report_y.passed is True
    assert report_state.passed is True


@pytest.mark.skipif(
    not get_triton_tilessm_availability().available,
    reason="Triton TileSSM scan requires CUDA + Triton",
)
def test_tilessm_dispatch_uses_triton_when_available() -> None:
    seed_all(59, deterministic=True)
    device = torch.device("cuda")
    tokens = torch.randn((1, 32, 8), device=device, dtype=torch.float16)
    params = _stable_random_params(8, device=device, dtype=torch.float16)
    dispatch = tilessm_scan_dispatch(
        tokens,
        prefer_triton=True,
        allow_fallback=False,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
    )
    assert dispatch.backend == "triton"


@pytest.mark.skipif(
    not get_triton_tilessm_availability().available,
    reason="Triton TileSSM scan requires CUDA + Triton",
)
def test_tilessm_triton_bidirectional_avg_parity_fp16() -> None:
    seed_all(61, deterministic=True)
    device = torch.device("cuda")
    tokens = torch.randn((1, 40, 10), device=device, dtype=torch.float16)
    params = _stable_random_params(10, device=device, dtype=torch.float16)

    y_triton, state_triton = tilessm_scan_triton(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="bidirectional",
        merge_mode="avg",
    )
    y_ref, state_ref = tilessm_scan_reference(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="bidirectional",
        merge_mode="avg",
    )

    rep_y = evaluate_parity_outputs(
        case_name="tilessm-triton-bidir-avg-y",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_triton",
        reference_output=y_ref,
        candidate_output=y_triton,
        tolerances=ToleranceConfig(),
    )
    rep_state = evaluate_parity_outputs(
        case_name="tilessm-triton-bidir-avg-state",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_triton",
        reference_output=state_ref,
        candidate_output=state_triton,
        tolerances=ToleranceConfig(),
    )
    assert rep_y.passed is True
    assert rep_state.passed is True


@pytest.mark.skipif(
    not get_triton_tilessm_availability().available,
    reason="Triton TileSSM scan requires CUDA + Triton",
)
def test_tilessm_triton_long_sequence_chunked_parity_fp16() -> None:
    seed_all(73, deterministic=True)
    device = torch.device("cuda")
    tokens = torch.randn((1, 5000, 8), device=device, dtype=torch.float16)
    params = _stable_random_params(8, device=device, dtype=torch.float16)

    dispatch = tilessm_scan_dispatch(
        tokens,
        prefer_triton=True,
        allow_fallback=False,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="forward",
    )
    y_ref, state_ref = tilessm_scan_reference(
        tokens,
        decay=params["decay"],
        input_gain=params["input_gain"],
        output_gain=params["output_gain"],
        state_bias=params["state_bias"],
        direction="forward",
    )

    assert dispatch.backend == "triton"
    rep_y = evaluate_parity_outputs(
        case_name="tilessm-triton-long-k-y",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_dispatch",
        reference_output=y_ref,
        candidate_output=dispatch.y,
        tolerances=ToleranceConfig(),
    )
    rep_state = evaluate_parity_outputs(
        case_name="tilessm-triton-long-k-state",
        reference_backend="tilessm_reference",
        candidate_backend="tilessm_dispatch",
        reference_output=state_ref,
        candidate_output=dispatch.final_state,
        tolerances=ToleranceConfig(),
    )
    assert rep_y.passed is True
    assert rep_state.passed is True
