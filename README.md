# apex-x

Apex-X: A reference repository for dynamic vision compute graphs with utility-based tile routing,
continuous budgeting, deterministic inference budgets, and runtime plugin contracts.

## Authoritative Documents

- Product requirements: `docs/PRD.md`
- Engineering specification: `docs/ENGINEERING_SPEC.md`
- Project memory/context: `docs/CONTEXT.md`
- Release checklist: `docs/release/CHECKLIST.md`
- Migration guide: `docs/release/MIGRATION.md`
- Changelog: `CHANGELOG.md`

## Repository Layout

- `apex_x/`: CPU-only reference implementation
- `tests/`: unit tests for routing, tile ops, and baseline execution
- `docs/`: PRD, engineering spec, decisions, TODO, context
- `docs/runtime/`: runtime/plugin-specific documentation
- `examples/`: runnable examples
- `scripts/`: tooling (perf regression baseline)
- `runtime/`: runtime integrations (Go service + TensorRT scaffolds)

## Quickstart (CPU-only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python examples/run_cpu_baseline.py
python examples/train_stages_smoke.py --config examples/smoke_cpu.yaml --steps-per-stage 1
python scripts/perf_regression.py
```

## Quickstart (GPU)

```bash
# Prerequisite: NVIDIA GPU + CUDA runtime compatible with your PyTorch build.
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install CUDA-enabled PyTorch first (example: CUDA 12.1).
# Pick the wheel index for your CUDA version from https://pytorch.org/get-started/locally/
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install Apex-X + dev tooling
pip install -e .[dev]

# Inspect runtime capabilities (CUDA / Triton / TensorRT / FP8)
python - <<'PY'
import json
from apex_x.runtime import detect_runtime_caps
print(json.dumps(detect_runtime_caps().to_dict(), indent=2))
PY

# GPU smoke benchmark (quick)
python -m apex_x.bench.gpu_bench \
  --warmup 3 \
  --iters 10 \
  --output-json artifacts/perf_gpu_smoke.json \
  --output-md artifacts/perf_gpu_smoke.md

# Full GPU benchmark
python -m apex_x.bench.gpu_bench \
  --output-json artifacts/perf_gpu.json \
  --output-md artifacts/perf_gpu.md

# Optional: compare against committed GPU baseline
python scripts/perf_regression_gpu.py \
  --compare \
  --baseline scripts/perf_baseline_gpu.json \
  --output artifacts/perf_gpu_current.json \
  --summary artifacts/perf_gpu_compare.json

# Optional: enable TensorRT plugin benchmark section
export APEXX_TRT_PLUGIN_LIB=/abs/path/to/libapexx_trt_plugins.so
python -m apex_x.bench.gpu_bench
```

## Developer Commands

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# Lint + format checks
ruff check .
black --check .

# Type checking
mypy

# Tests
pytest

# Staged trainer CLI
apex-x train --config tests/fixtures/apex_x_config.yaml --steps-per-stage 1

# Staged trainer smoke script
python examples/train_stages_smoke.py --config examples/smoke_cpu.yaml --steps-per-stage 1

# Optional local perf smoke
python scripts/perf_regression.py

# Perf regression check against committed baseline
python scripts/perf_regression.py \
  --compare \
  --baseline scripts/perf_baseline_cpu.json \
  --output artifacts/perf_current.json \
  --summary artifacts/perf_compare.json

# Generate release evidence draft (JSON + Markdown)
python scripts/release_attestation.py \
  --runtime-target cpu \
  --artifact-path performance_report=artifacts/perf_compare.json \
  --output-json artifacts/release/release_attestation_local.json \
  --output-md artifacts/release/release_attestation_local.md

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Development Workflow

1. Read `docs/PRD.md` and `docs/ENGINEERING_SPEC.md`.
2. Implement small, test-backed changes.
3. Run `ruff check .`, `black --check .`, `mypy`, and `pytest`.
4. Update `docs/DECISIONS.md` and `docs/CONTEXT.md` for architecture or milestone changes.

## Runtime and Export

Runtime/plugin behavior and export contracts are defined in:

- `docs/ENGINEERING_SPEC.md`
- `docs/runtime/PLUGIN_SPEC.md`
- `docs/runtime/TENSORRT.md`

Runtime scaffolds:

- TensorRT C++ stubs: `runtime/tensorrt/`
- Go HTTP microservice: `runtime/go/`

Quick smoke for Go runtime:

```bash
cd runtime/go
go test ./...
# optional real ORT inference bridge (without native Go ORT bindings)
export APEXX_ORT_BRIDGE_CMD="python -m apex_x.runtime.service_bridge"
go run ./cmd/apexx-runtime -addr :8080 -adapter onnxruntime
```

## Documentation Site

Build and preview docs locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[docs]'
mkdocs build --strict
mkdocs serve
```

Main docs entrypoint:

- `docs/index.md`
