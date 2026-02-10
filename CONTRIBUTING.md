# Contributing to Apex-X

## Ground Rules
- Read `docs/PRD.md` and `docs/ENGINEERING_SPEC.md` before proposing architecture changes.
- Keep behavior deterministic for inference policy and tile ordering.
- Maintain CPU baseline operability.

## Development Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Branch and PR Workflow
1. Create a focused branch.
2. Add or update tests for any behavior change.
3. Run local checks:
```bash
ruff check .
pytest
```
4. Open a PR with:
- problem statement
- design summary
- risk assessment
- before/after behavior

## Documentation Requirements
Any change to routing, budgets, tile contracts, or runtime behavior must update:
- `docs/PRD.md`
- `docs/ENGINEERING_SPEC.md`
- `docs/DECISIONS.md` (if architectural)
- `docs/CONTEXT.md` (current status + next steps)

Required policy:
- Every significant PR must update `docs/CONTEXT.md` with what changed and next steps.
- Any architectural or convention decision change must update `docs/DECISIONS.md` in the same PR.

## Commit Guidance
- Use clear commit messages.
- Keep PRs small enough for focused review.
- Do not mix unrelated refactors with feature changes.
