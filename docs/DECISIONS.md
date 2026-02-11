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

## ADR-0008: Runtime Selection vs Execution Traceability
- Date: 2026-02-10
- Decision:
  - Runtime metadata must distinguish:
    - requested backend
    - selected backend
    - actual execution backend
    - selection and execution fallback reasons
    - latency breakdown (`total`, `backend_execute`, `backend_preflight`)
  - TensorRT CLI inference must execute real serialized engines when runtime artifacts are provided, instead of placeholder stubs.
- Rationale: Backend fallback chains can otherwise hide real execution behavior and invalidate performance/correctness evidence.
- Consequence:
  - Runner/CLI reports are required to expose full backend selection and execution trace fields.
  - TensorRT runtime path must enforce explicit engine artifact contract and deterministic strict/permissive behavior.
  - Go runtime service `/predict` payloads must expose the same runtime metadata key schema.
  - Go runtime service must map queue saturation to HTTP `429` and request timeout to HTTP `504`.
  - Go runtime service should support optional shadow-canary parity telemetry for safe rollout validation.

## ADR-0009: Eval Dataset Optional Target Contract
- Date: 2026-02-10
- Decision:
  - `eval --dataset-npz` supports optional scalar target keys:
    - `det_score_target` (compat alias `det_scores_target`)
    - `selected_tiles_target` (compat alias `selected_tiles_targets`)
  - When `det_score_target` exists, `model_eval` report includes deterministic regression metrics
    (`mae`, `rmse`, `bias`, `r2`, `pearson_corr`) in addition to aggregate model outputs.
  - When `selected_tiles_target` exists, `model_eval` report includes
    (`mae`, `rmse`, `bias`, `exact_match_rate`).
- Rationale: This adds lightweight, repeatable quality tracking on environments
  where full box/mask ground-truth evaluation is not yet wired end-to-end.
- Consequence:
  - Dataset shape/length validation is strict and fails early on mismatch.
  - Future full GT parity work can extend this contract without breaking current reports.

## ADR-0010: Go Runtime Bridge Backends and Error Taxonomy
- Date: 2026-02-11
- Decision:
  - Go runtime adapters may execute real ORT/TRT inference through an external bridge command:
    - `APEXX_ORT_BRIDGE_CMD`
    - `APEXX_TRT_BRIDGE_CMD`
  - ORT/TRT adapters must fail closed when bridge/native execution is unavailable
    (no synthetic score fallback in production path).
  - Bridge contract is JSON stdin/stdout and implemented by `apex_x/runtime/service_bridge.py`.
  - Backend errors are classified and mapped explicitly:
    - backend unavailable -> HTTP `503`
    - backend inference/protocol failures -> HTTP `502`
  - Canary mode payload capture must be policy-driven and optional:
    - `off|mismatch|error|all`
    - JSONL sink with max-size guard
- Rationale: Native Go bindings are environment-dependent; bridge mode enables real backend execution
  while preserving deterministic HTTP failure semantics.
- Consequence:
  - Service behavior remains SLA-safe under backend faults with stable status-code policy.
  - Canary overhead and timeout/overflow rates are enforced by CI gate test thresholds.
  - Native CGO TensorRT execution is still tracked separately until full implementation and GPU validation.

## ADR-0011: Budget-Aware Temporal Hysteresis Metrics
- Date: 2026-02-11
- Decision:
  - Temporal hysteresis rollout supports optional per-frame active cap (`max_active`) to preserve
    budget constraints during state carryover.
  - Clipping priority is deterministic:
    1. keep previously-active tiles
    2. higher utility first
    3. lower tile id tie-break
  - Temporal quality gates expose explicit metrics:
    - `tile_flip_rate`
    - `temporal_consistency`
    - `mean_active_ratio`
- Rationale: Hysteresis reduces flicker but can temporarily retain extra tiles; budget-aware clipping
  keeps anti-flicker behavior without violating frame-level active limits.
- Consequence:
  - Routing APIs now include budget-aware hysteresis update and sequence stability summaries.
  - Sequence-level tests are required to prove reduced flip churn under budget caps.

## ADR-0012: Recursive Quadtree Split Policy for L2
- Date: 2026-02-11
- Decision:
  - Inference budgeting supports deterministic recursive split selection to depth 2:
    - `L0` selection under `B1`
    - `L0 -> L1` split parent selection under `B2`
    - `L1 -> L2` split parent selection under `B3`
  - Parent ordering at each split stage uses:
    1. split score (`S/O_split`) descending
    2. tile id ascending tie-break
  - Child expansion remains capacity-constrained by `Kmax_L1` and `Kmax_L2`.
- Rationale: Quality-sensitive regions often require second-level refinement; deterministic recursion
  keeps behavior reproducible while honoring explicit multi-level budget constraints.
- Consequence:
  - Routing layer now exposes a three-stage deterministic selection API.
  - Tests must cover `B1/B2/B3` constraints, tie-break determinism, and `Kmax` limits.

## ADR-0013: Adaptive Dual-Budget Controller Stabilizers
- Date: 2026-02-11
- Decision:
  - Dual budget update supports optional adaptive controls:
    - step decay (`mu_lr / (1 + decay * step)`)
    - EMA-based scale modulation with bounded range (`lr_min_scale..lr_max_scale`)
    - normalized-error deadband (`|error/B| <= deadband_ratio`)
    - delta clipping (`[-delta_clip, +delta_clip]`)
  - Dual variable remains projected into fixed bounds `[mu_min, mu_max]`.
- Rationale: Constant-rate dual ascent can oscillate near target budgets and overreact to noisy cost traces.
  Adaptive scaling plus deadband/clipping improves convergence stability without changing base objective.
- Consequence:
  - Training config now exposes explicit dual schedule/stabilizer parameters.
  - Stage-3 trainer metrics include dual controller dynamics (`effective_lr`, `error_ema`, update count).
  - Convergence tests must cover over-budget, under-budget, deadband, and clip-capped behavior.

## ADR-0014: PCGrad++ Shared-Trunk Scope with Conflict-Rate Telemetry
- Date: 2026-02-11
- Decision:
  - PCGrad++ projection is applied only to shared trunk parameters.
  - Head-specific parameters always use standard total-loss gradients (no projection).
  - Diagnostics must include conflict metrics before and after projection:
    - `conflicting_pairs`
    - `conflicting_pairs_after`
    - `total_pairs`
    - `conflict_rate_before`
    - `conflict_rate_after`
- Rationale: Restricting projection to shared trunk prevents unintended task-head updates, while
  before/after metrics quantify whether projection actually reduces gradient conflict pressure.
- Consequence:
  - Trainer reports now include PCGrad diagnostics payloads suitable for CI/report tracking.
  - Tests must prove head gradients remain unchanged and conflict-rate reduction is non-regressive.

## ADR-0015: FP8 Requested-vs-Effective Telemetry Contract
- Date: 2026-02-11
- Decision:
  - FP8-capable flows must expose explicit requested-vs-effective precision telemetry:
    - `requested_dtype`
    - `effective_dtype`
    - `fp8_requested`
    - `fp8_enabled`
    - `fp8_fallback_reason`
  - FP8 fallback reasons must use canonical runtime reason-codes.
  - GPU benchmark and regression wrappers accept explicit FP8 request mode (`--dtype fp8`).
- Rationale: Operational readiness requires proving whether FP8 was actually enabled or silently
  downgraded; requested-vs-effective telemetry prevents false confidence in perf claims.
- Consequence:
  - FP8 rollout evidence is now testable on non-CUDA hosts (fallback telemetry) and measurable on
    supported GPUs via explicit FP8 benchmark artifacts.

## ADR-0016: Oracle Sampling Triad and Delta-Label Diagnostics
- Date: 2026-02-11
- Decision:
  - Oracle tile subset sampling combines three components:
    - random subset
    - uncertainty-weighted subset
    - long-tail subset (deterministic top scores from remaining tiles)
  - Oracle delta labels are clipped by configured `clamp_abs`, and trainer reports:
    - sample composition counts
    - delta distribution stats
    - clipping ratio
- Rationale: Mixing random + uncertain + long-tail examples improves utility-head coverage while
  clipping/diagnostics prevents silent instability from outlier labels.
- Consequence:
  - `sample_oracle_set(...)` now supports long-tail selection inputs.
  - Stage-2 trainer metrics/logs expose explicit oracle distribution + clipping diagnostics.

## ADR-0017: TensorRT INT8 Sensitive-Layer FP16 Enforcement
- Date: 2026-02-11
- Decision:
  - In TensorRT INT8 builds, sensitive layers (matched by configurable keywords, default
    `router` and `kan`) are forced to FP16 precision constraints.
  - Precision constraint enforcement is strict by default:
    - build fails if matched layer cannot be constrained via TensorRT APIs.
  - Build results expose per-layer precision evidence (`layer_precision_status`).
- Rationale: Routing/decision layers are numerically sensitive; silently quantizing them to INT8
  can destabilize gating behavior and degrade quality.
- Consequence:
  - `TensorRTEngineBuildConfig` includes strict precision-constraint policy.
  - Tests validate keyword-based enforcement, strict failure behavior, and emitted precision report.

## ADR-0018: TensorRT Plugin Creator Contract Validation
- Date: 2026-02-11
- Decision:
  - TensorRT builder validates plugin creator contracts before engine build:
    - creator presence in registry
    - version match
    - namespace match
    - plugin field-signature metadata coverage
  - Strict plugin validation remains default and fails build on required-plugin mismatches.
  - Optional plugin mismatches are reported as warnings when strict mode is disabled.
- Rationale: Plugin name-only checks are insufficient for production safety; ABI/contract drifts can
  silently break builds or runtime behavior.
- Consequence:
  - `TensorRTEngineBuildConfig` supports explicit `PluginContract` overrides.
  - `PluginStatus` now carries detailed mismatch diagnostics for actionable failures.
  - Tests cover version/namespace/field-signature mismatch behaviors without requiring CUDA runtime.

## ADR-0019: TensorRT INT8 Calibration Cache Governance
- Date: 2026-02-11
- Decision:
  - TensorRT INT8 calibration cache reuse is governed by deterministic cache keys that include:
    - model/export identity hash
    - plugin contract metadata (version + namespace)
    - precision profile
    - calibration dataset version
  - Calibration dataset version supports:
    - explicit configuration (`calibration_dataset_version`)
    - deterministic auto-digest from calibration batches
  - Calibrator cache files use structured metadata when key governance is active;
    stale/mismatched keys are invalidated automatically.
  - Legacy raw cache blobs are accepted only when key governance is disabled.
- Rationale: INT8 accuracy/perf stability requires preventing stale calibration reuse after model,
  plugin, precision-profile, or dataset changes.
- Consequence:
  - `EngineBuildResult` now exposes `calibration_cache_key` and `calibration_dataset_version`.
  - Non-CUDA tests validate deterministic key behavior and stale-cache invalidation rules.

## ADR-0020: TensorRT Parity Harness Matrix + Sweep Contract
- Date: 2026-02-11
- Decision:
  - Parity validation for TensorRT readiness is executed as backend-pair matrix checks:
    - reference vs triton
    - reference vs tensorrt
    - triton vs tensorrt
  - Matrix checks are composed into profile-aware sweep runs covering shape and precision cases.
  - CPU-safe parity harness tests are mandatory to keep matrix/sweep semantics stable even when
    CUDA/TensorRT runtime is unavailable on CI host.
- Rationale: Production parity readiness needs a repeatable structure, not ad-hoc pair checks.
  Matrix+sweep contract ensures consistent coverage expansion from CPU-safe CI to deployment GPUs.
- Consequence:
  - `apex_x/runtime/parity.py` now exposes matrix+sweep APIs and aggregate sweep reporting.
  - CUDA-host execution remains required to close final parity evidence for Triton/TensorRT runtime.

## ADR-0021: Triton TileSSM Long-Sequence Chunked Execution
- Date: 2026-02-11
- Decision:
  - Triton TileSSM forward scan no longer hard-fails for long sequences beyond single-launch
    specialization bounds.
  - For `K` above the per-launch limit, runtime executes chunked launches and streams recurrent
    state between chunks.
- Rationale: Large-scene workloads should remain on accelerated path where possible; hard
  sequence caps force unnecessary fallback and create avoidable latency cliffs.
- Consequence:
  - `tilessm_scan_triton(...)` now supports long-sequence chunking semantics.
  - CPU-safe tests validate chunk partitioning and state carry-over contract.
  - Deployment CUDA benchmarks remain required for final perf evidence.

## ADR-0022: Triton TileUnpack Blend Dispatch Parity Contract
- Date: 2026-02-11
- Decision:
  - `tileunpack_dispatch(...)` no longer treats `overlap_mode=\"blend\"` as a forced
    reference-only branch.
  - Blend overlap semantics remain ordered and parity-equivalent to reference behavior.
- Rationale: Forced fallback branches in supported overlap modes prevent full accelerated runtime
  readiness and obscure backend capability expectations.
- Consequence:
  - Blend dispatch path now uses `tileunpack_triton(...)` entrypoint when accelerated path is
    selected.
  - CPU/GPU overlap tests were expanded to validate blend dispatch contract and parity behavior.

## ADR-0023: FF Heavy-Path Stage-1 Fused Selector Gating
- Date: 2026-02-11
- Decision:
  - FF heavy-path inference can use Stage-1 fused dispatch only when strict compatibility
    predicates are satisfied:
    - eval mode
    - identity refine block
    - effectively constant FiLM parameters
    - unique selected tile indices
  - non-compatible cases deterministically use decomposed `pack -> FiLM -> unpack` path.
- Rationale: Stage-1 fused kernel offers speed benefits but is not mathematically equivalent to
  the full decomposed path for arbitrary FiLM/refine dynamics; gating preserves correctness.
- Consequence:
  - runtime selector now chooses fused path only for compatible inference cases.
  - unit tests cover activation, parity in compatible mode, and fallback behavior.

## ADR-0024: Triton Autotune Registry and Benchmark Telemetry
- Date: 2026-02-11
- Decision:
  - Triton kernels with autotune candidates publish per-op/per-shape-bucket selected config
    telemetry into a shared in-process registry.
  - Registry entries are keyed by `op_name + shape_bucket` and track:
    - selected launch config metadata
    - selection source (`triton_best_config`, `heuristic`, `registry_cache`)
    - cache counters (`launches`, `cache_hits`, `cache_misses`)
  - GPU benchmark report includes registry telemetry in both JSON and Markdown outputs.
- Rationale: Performance readiness needs explicit visibility into which launch configs are used
  per workload shape; without this, regressions can hide behind implicit runtime autotune behavior.
- Consequence:
  - Triton tile/fusion/fused-stage1 kernels now emit autotune registry events.
  - `gpu_bench` now emits `triton_autotune.summary` and `triton_autotune.entries`.
  - CPU-safe tests validate registry cache accounting and report formatting contract.

## ADR-0025: Unified Perf Regression Wrappers Including TensorRT Shape Sweep
- Date: 2026-02-11
- Decision:
  - CPU, GPU, and TensorRT perf regression wrappers use one shared pass/fail formula based on
    baseline ratio + absolute tolerance.
  - TensorRT shape-sweep regression is tracked with a dedicated wrapper:
    - `scripts/perf_regression_trt.py`
    - baseline: `scripts/perf_baseline_trt.json`
  - GPU CI/weekly workflows execute TRT compare + trend only when `TRT_ENGINE_PATH` is provided.
- Rationale: Performance governance must be consistent across runtime tiers; TRT coverage should not
  rely on ad-hoc checks or manual inspection.
- Consequence:
  - TRT shape-sweep now has baseline-template, compare, and trend artifact modes aligned with CPU/GPU.
  - Deployment runner validation remains required for final threshold tuning and mandatory gate policy.
