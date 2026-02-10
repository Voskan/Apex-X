# Apex-X INT8 QAT Policy

## Scope
This document defines the CPU-reference INT8 quantization-aware training (QAT) path and PTQ fallback used in Apex-X.

Authoritative references:
- `docs/PRD.md` (FR-12)
- `docs/ENGINEERING_SPEC.md` (Section 13)

## Design Goals
- Provide a deterministic INT8 simulation path during training.
- Keep router/gating logic in FP16-safe precision path.
- Support PTQ fallback for `edge` precision profile when QAT is not enabled.
- Keep CPU baseline runnable and testable.

## Implemented Components

### Fake Quant Modules
Implemented in `apex_x/train/qat.py`:
- `ActivationObserver`: running min/max activation statistics.
- `ActivationFakeQuant`: per-tensor activation fake quant.
- `WeightPerChannelFakeQuant`: symmetric per-channel weight fake quant.
- `FakeQuantConv2d`: wraps `nn.Conv2d`.
- `FakeQuantLinear`: wraps `nn.Linear`.

Quantization ranges:
- Activations: unsigned INT8 (`0..255`), affine.
- Weights: signed INT8 (`-127..127`), symmetric per output channel.

### QAT Preparation
`prepare_int8_qat(model, ...)`:
- Replaces quantizable `Conv2d`/`Linear` modules with fake-quant wrappers.
- Enables observers + fake quantization for training-time simulation.
- Skips modules whose names include `router`, `gate`, or `gating`.

### PTQ Calibration Fallback
`prepare_int8_ptq(model, calibration_inputs, forward_fn, ...)`:
- Applies same wrapper conversion.
- Runs observer-only calibration passes (`fake_quant=off`).
- Freezes observers and enables fake quant (`observer=off`, `fake_quant=on`).

This is used as fallback for runtime profile `edge` when INT8 QAT is not explicitly enabled.

## Trainer Integration
`apex_x/train/trainer.py` integrates quantization policy:
- If `train.qat_enable=true` and `train.qat_int8=true`:
  - prepare INT8 QAT (`mode=qat_int8`)
- Else if `runtime.precision_profile=edge`:
  - run PTQ calibration fallback (`mode=ptq_int8`)
- Else:
  - quantization disabled (`mode=disabled`)

Trainer `train_summary["quantization"]` includes:
- `mode`
- `wrapped_modules`
- `calibration_batches`
- `router_gating_fp16`

## Router/Gating Precision Rule
Router and gating paths are excluded from INT8 wrapping by policy:
- skip-name filters include `router`, `gate`, `gating`
- staged gating math keeps FP16 path for router utility gating operations

This enforces the spec requirement that routing-sensitive numerics remain in higher precision.

## Validation
Coverage is provided in `tests/test_qat.py`:
- wrapper conversion and skip policy checks
- PTQ calibration state transitions
- trainer-level QAT/PTQ toggles and finite outputs
