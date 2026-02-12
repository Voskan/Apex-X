# Changelog

## [0.3.0] - 2026-02-11 "Robust Edge"

### 🚀 Major Features

#### 1. Advanced Computer Vision (Phase 2)
- **Real Satellite Imagery Training**: Added `SatelliteDataset` to support direct training on large-scale GeoTIFFs (up to 4GB+) with automatic sliding-window tiling.
- **Robust Augmentations**: Integrated `albumentations` pipeline with blur, noise, and JPEG compression to make models resilient to low-quality satellite data.
- **High-Capacity Backbones**: Introduced `TimmBackboneAdapter` to allow using State-of-the-Art backbones (EfficientNet, Swin, ConvNeXt) via the `timm` library.
- **Hardware Acceleration**: Enabled custom Triton kernels (`gather_gate_scatter`) for 3x+ speedup on NVIDIA GPUs compared to PyTorch reference implementation.

#### 2. Observability Pipeline (Phase 3)
- **Structured Logging**: Migrated to `structlog` for JSON-formatted logs, enabling seamless integration with ELK/Splunk stacks.
- **Prometheus Metrics**: Exposed a `/metrics` endpoint (port 8000) tracking:
    - End-to-end inference latency (`inference_latency_seconds`)
    - Throughput and Error rates (`inference_requests_total`)
    - GPU memory usage (`gpu_memory_used_bytes`)
- **Distributed Tracing**: Instrumented the service bridge with OpenTelemetry to trace requests across the Go-Python boundary.

#### 3. Service Layer Robustness (Phase 1)
- **Strict Validation**: Refactored `service_bridge.py` to use Pydantic v2 for rigorous input validation and schema enforcement.
- **Health Checks**: Added specific health probe logic to verify backend (ONNX/TensorRT) readiness.

### 🛠 Improvements
- **Environment**: Updated dependency management to support `timm` and `albumentations`.
- **Testing**: Added comprehensive unit and integration tests for new datasets and models (`tests/phase2`).
- **Documentation**: Updated README with "Advanced Usage" and "Observability" guides.

### 🐛 Bug Fixes
- Fixed syntax errors in `triton_fused.py` related to kernel imports.
- Resolved type hinting issues in `SatelliteDataset` and `TransformSample`.
- Fixed various linting issues and unused imports across the codebase.

### ⚠️ Known Issues
- `mypy` check may fail with `NotImplementedError: Cannot serialize TypeGuardedType` on some environments due to an internal bug in the type checker version. This does not affect runtime.
- Metrics scraping from the CLI-based `service_bridge` is ephemeral; for permanent metrics, the service should be run in a daemon mode (future work).
