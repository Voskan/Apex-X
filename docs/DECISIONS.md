# Architectural Decisions

## ADR-0001: Authoritative Documentation Split
- Date: 2026-02-07
- Decision: Keep requirements in `docs/PRD.md` and implementation contract in `docs/ENGINEERING_SPEC.md`.
- Rationale: Separates product scope from engineering detail while preserving traceability.
- Consequence: Any architecture change must update both files.

## ADR-0002: CPU-Only Reference Baseline First
- Date: 2026-02-07
- Decision: Start with a pure Python + NumPy CPU baseline before runtime kernels.
- Rationale: Fast iteration, deterministic debugging, and testable contracts independent of GPU/runtime stack.
- Consequence: Runtime parity work follows once behavior is stabilized.

## ADR-0003: Deterministic Greedy Budgeting for Inference
- Date: 2026-02-07
- Decision: Use utility-per-cost greedy selection with fixed `Kmax` buffers.
- Rationale: Deterministic, simple, and export/runtime friendly.
- Consequence: Potential global-optimality gap vs exact knapsack is accepted for speed and determinism.

## ADR-0004: Ordering Modes Required in Contract
- Date: 2026-02-07
- Decision: Support both Hilbert and multi-direction ordering modes.
- Rationale: Preserves geometry for sequence mixing and supports runtime experimentation.
- Consequence: Ordering determinism tests are mandatory.

## ADR-0005: Naming Conventions Baseline
- Date: 2026-02-07
- Decision:
  - Python modules/files use `snake_case`.
  - Classes use `PascalCase`.
  - Protocol interfaces use `*Protocol` suffix (for example `RouterProtocol`).
  - Config sections use stable top-level keys: `model`, `routing`, `train`, `data`, `runtime`.
- Rationale: Consistent naming lowers integration friction and improves discoverability across modules.
- Consequence: New public interfaces and new files must follow these naming rules.

## ADR-0006: Tensor Shape Conventions
- Date: 2026-02-07
- Decision:
  - Image/features default to channel-first `NCHW` (`[B,C,H,W]`).
  - Detection boxes use `[B,N,4]` with `cx,cy,w,h`.
  - Packed tile tensors use `[B,K,C,t,t]`.
  - Tile index tensors use `[B,K]` and rely on fixed `Kmax` contracts.
- Rationale: Stable tensor contracts are required for deterministic runtime behavior and export parity.
- Consequence: Any shape-contract change must update `docs/PRD.md`, `docs/ENGINEERING_SPEC.md`, and tests.

## ADR-0007: Determinism Rules
- Date: 2026-02-07
- Decision:
  - Inference tile selection uses deterministic greedy utility-per-cost with fixed `Kmax`.
  - Tile ordering remains deterministic (Hilbert or multi-direction scan).
  - Reproducibility utilities (`seed_all`, deterministic toggles) are part of baseline workflow.
  - Export/runtime path must avoid Python-side inference control flow.
- Rationale: Determinism is a hard product constraint for regression tracking and deployment confidence.
- Consequence: Any nondeterministic path requires explicit rationale and targeted determinism tests.
