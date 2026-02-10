# Apex-X Documentation

Apex-X v4 is a dynamic vision compute graph with utility-based routing, continuous budgeting,
and deterministic inference for production-friendly export/runtime workflows.

## Core Documents
- Product Requirements Document: [PRD](PRD.md)
- Engineering Specification: [Engineering Spec](ENGINEERING_SPEC.md)
- Quantization Policy: [QAT / PTQ](QAT.md)
- FP8 Precision Policy: [FP8 Policy](FP8.md)
- Performance Regression: [PERF](PERF.md)
- GPU Performance Benchmark: [PERF_GPU](PERF_GPU.md)
- GPU CI Setup and Security: [CI_GPU](CI_GPU.md)
- Runtime Plugin Specification: [Runtime Plugin Spec](runtime/PLUGIN_SPEC.md)
- Runtime Capability Detection: [Runtime Caps](runtime/CAPS.md)
- Runtime Parity Framework: [Runtime Parity](runtime/PARITY.md)
- TensorRT Runtime Notes: [TensorRT Scaffolding](runtime/TENSORRT.md)
- TensorRT Build Guide: [TensorRT Build/Harness](runtime/TENSORRT_BUILD.md)
- TensorRT INT8 Builder: [Engine Builder + Calibration](runtime/TENSORRT_INT8.md)
- TensorRT Postprocessing: [Decode + NMS](runtime/TENSORRT_POST.md)
- Go Runtime Service: [Go Service](runtime/GO_SERVICE.md)
- Triton Runtime Notes: [Triton Fused Ops](runtime/TRITON.md)
- Triton TilePack: [Triton TilePack Kernel](runtime/TRITON_TILEPACK.md)
- Triton TileUnpack: [Triton TileUnpack Kernel](runtime/TRITON_TILEUNPACK.md)
- Triton FusionGate: [Triton FusionGate Kernel](runtime/TRITON_FUSION.md)
- Triton TileSSM Scan: [Triton TileSSM Baseline](runtime/TRITON_SSM.md)
- Triton Fused Stage-1: [Triton Fused Stage-1](runtime/TRITON_FUSED_STAGE1.md)
- Project Context Memory: [Context](CONTEXT.md)
- Architecture Decisions: [Decisions](DECISIONS.md)
- Active Worklist: [TODO](TODO.md)

## Build Documentation Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[docs]'
mkdocs build --strict
mkdocs serve
```

## Notes
- `mkdocs build --strict` is enforced in CI.
- Authoritative architecture content lives in `docs/PRD.md` and `docs/ENGINEERING_SPEC.md`.
