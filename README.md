# apex-x

Apex-X areference repository for dynamic vision compute graphs with utility-based tile routing,
continuous budgeting, deterministic inference budgets, and runtime plugin contracts.

## Authoritative Documents

- Product requirements: `docs/PRD.md`
- Engineering specification: `docs/ENGINEERING_SPEC.md`
- Project memory/context: `docs/CONTEXT.md`

## Repository Layout

- `apex_x/`: CPU-only reference implementation
- `tests/`: unit tests for routing, tile ops, and baseline execution
- `docs/`: PRD, engineering spec, decisions, TODO, context
- `docs/runtime/`: runtime/plugin-specific documentation
- `examples/`: runnable examples
- `scripts/`: tooling (perf regression baseline)
- `runtime/`: runtime integration placeholders

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
