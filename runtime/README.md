# Runtime

This folder contains runtime-related code and integration points for TensorRT/ORT plugins.

Current state:
- CPU-only reference implementation lives in `apex_x/`.
- Runtime plugin behavior is specified in `docs/runtime/PLUGIN_SPEC.md`.
- TensorRT C++ plugin scaffolding is in `runtime/tensorrt/`.
- Go runtime microservice scaffolding is in `runtime/go/`.

Related docs:
- `docs/runtime/PLUGIN_SPEC.md`
- `docs/runtime/TENSORRT.md`
- `docs/runtime/TRITON.md`
