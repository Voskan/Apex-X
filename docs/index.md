# Apex-X Documentation

Primary documentation entrypoints:

- Project overview and runnable quickstart: `README.md`
- Training flow and checkpoint policy: `docs/TRAINING_GUIDE.md`
- Runtime and deployment notes: `docs/runtime/`
- Benchmark protocol/docs: `docs/benchmarks.md`

## Core docs

1. `docs/ENGINEERING_SPEC.md`
2. `docs/algorithms.md`
3. `docs/DECISIONS.md`
4. `docs/release/CHECKLIST.md`

## CLI reference (current)

- `python -m apex_x.cli train ...`
- `python -m apex_x.cli eval ...`
- `python -m apex_x.cli predict ...`
- `python -m apex_x.cli bench ...`
- `python -m apex_x.cli ablate ...`
- `python -m apex_x.cli export ...`
- `python -m apex_x.cli dataset-preflight ...`

## Validation rule for docs

User-facing claims should be backed by reproducible artifacts:
- config used
- commit hash
- checkpoint lineage
- evaluation reports
