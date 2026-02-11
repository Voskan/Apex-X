# Apex-X v4 PRD

## 1. Document Status
- Project: `Apex-X v4`
- Purpose: Authoritative product requirements for the Apex-X open-source repository
- This document is normative for scope, behavior, and acceptance criteria
- Implementation details are specified in `docs/ENGINEERING_SPEC.md`

## 2. Product Goal
Apex-X v4 is a dual-stream, utility-routed vision architecture that preserves quality while enforcing explicit compute budgets.

Primary outcomes:
- high-quality detection and segmentation under fixed latency/compute budgets
- deterministic inference behavior suitable for production runtime plugins
- clean export path to TensorRT/ORT with dynamic selection handled via `Kmax` buffers and plugins

## 3. Scope
### 3.1 In Scope
- Tasks: DET, INST-SEG, SEM-SEG, optional POSE/TRACK/PANO
- Dual stream architecture: PV (always-on) + FF (sparse high-res tiles)
- Utility-based routing with oracle supervision
- Continuous-budget training and deterministic-budget inference
- Quadtree nesting (`L0/L1/L2`) with split policy
- Tile graph pack/unpack contracts and deterministic ordering
- Temporal hysteresis for anti-flicker
- Runtime plugin contracts for TensorRT/ORT
- QAT policy for INT8/FP8 deployment paths

### 3.2 Out of Scope (for baseline release)
- Final production GPU kernels
- Multi-node distributed training recipes
- End-to-end tracking benchmarks

## 4. Terms
- `PV`: Peripheral Vision stream (low-res, dense)
- `FF`: Foveal Focus stream (high-res, sparse tiles)
- `L0/L1/L2`: tile nesting levels (`L0` coarse, `L1/L2` finer)
- `Kmax`: fixed upper bound on active tile buffer length
- `B, B1, B2`: total and per-level budgets
- `U_i`: utility score for tile `i`
- `S_i`: split utility for tile `i`
- `O_split`: split overhead cost term

## 5. Functional Requirements
### FR-1 Dual Stream
System shall compute PV densely and FF sparsely.

### FR-2 Utility Router
System shall score tiles with utility `U_i` and optional split utility `S_i`.

### FR-3 Oracle Supervision
System shall support oracle delta utility labels:
\[
\Delta_i = L_{distill}(y^{(i=0)}, y_T) - L_{distill}(y^{(i=1)}, y_T)
\]
where `i=0` means heavy path disabled on tile `i`, `i=1` means enabled.

### FR-4 Continuous Budgeting (Training)
System shall optimize routing probabilities via:
\[
p_i = \sigma(U_i),\quad g_i = STE(p_i)
\]
Expected cost:
\[
\mathbb{E}[C] = \sum_i \left(p_i C_h + (1-p_i) C_c\right)
\]
Constrained objective:
\[
L = L_{main} + \mu(\mathbb{E}[C] - B),\quad \mu \ge 0
\]
Dual update concept:
- if `E[C] > B`, increase `mu`
- if `E[C] < B`, decrease `mu`
- if adaptive schedule is enabled, effective update rate must be:
  - step-decayed
  - scaled by EMA of normalized budget error
  - bounded by configured min/max scale
- optional deadband near target and delta clipping must be supported to avoid oscillation.

### FR-5 Deterministic Inference Budgeting
System shall select FF tiles deterministically by utility-per-cost:
\[
score_i = \frac{U_i}{\Delta C_i}
\]
sorted descending with selection until budget or `Kmax` is reached.

### FR-6 Quadtree Nesting
System shall support two-level and optional three-level nesting:
- `L0` under budget `B1`
- `L1` split under budget `B2`
- optional `L2` split under budget `B3`
Split score:
\[
score^{split}_i = \frac{S_i}{O_{split}}
\]
Split parent ordering must be deterministic (score desc, tile-id asc tie-break).

### FR-7 Tile Tensor Contracts
System shall support deterministic tile gather/scatter:
- Pack: `[B,C,Hf,Wf] + idx[B,K] -> P[B,K,C,t,t] + meta`
- Unpack: `F_base + P_out + meta -> F_merged`

### FR-8 Deterministic Tile Ordering
System shall support:
- Hilbert ordering
- Multi-direction scan ordering (`LR`, `RL`, `UD`, `DU`)

### FR-9 Tile-SSM and Fusion Gate
System shall include a tile-sequence mixer (Mamba-like placeholder acceptable in baseline) and fusion gate behavior.

### FR-10 Temporal Stability
System shall support hysteresis:
\[
z_i(t)=\mathbf{1}[U_i(t)>\theta_{on}\;\lor\;(U_i(t)>\theta_{off}\land z_i(t-1)=1)]
\]
with `theta_on > theta_off`.

For deployment routing, system shall support budget-aware hysteresis carryover:
- temporal state reuse must not violate per-frame active-tile budget
- when clipping is required, selection must be deterministic

Temporal quality reporting shall include:
- tile flip rate
- temporal consistency (`1 - flip_rate`)

### FR-11 Runtime Plugin Contracts
System shall define plugin interfaces for TensorRT/ORT:
- `TilePack`
- `TileSSMScan`
- `TileUnpackFusion`
- optional `MaskedConv`
- fused `DecodeNMS`
TensorRT build path shall validate plugin creator contract metadata at build time:
- presence
- version
- namespace
- expected plugin field signature

### FR-12 QAT/Precision Policy
System shall define deployment profiles:
- Quality: FP16-heavy
- Balanced: FP16/FP8 mixed
- Edge: INT8 (router FP16)
For INT8 deployment builds, sensitive routing layers (router/KAN-like) must be enforced to FP16
and reported in per-layer precision diagnostics.
For INT8 calibration-cache reuse, cache governance must be deterministic and bind to:
- model/export identity hash
- plugin version/namespace contract metadata
- precision profile
- calibration dataset version (explicit version or deterministic dataset digest)
INT4 allowed only with guaranteed kernels and quality gate.

### FR-13 Perf Regression Requirements
System shall include repeatable latency and memory regressions with thresholds and fail criteria.
System shall apply one regression compare formula across CPU, GPU, and TensorRT shape-sweep reports.
System shall include backend parity sweep harnesses that validate:
- reference vs triton
- reference vs tensorrt
- triton vs tensorrt
across representative shape and precision cases under profile-specific tolerances.
System shall support optional dataset-level target regression diagnostics in eval reports
when benchmark/eval datasets provide explicit scalar targets
(`det_score_target`, `selected_tiles_target`).

### FR-14 Runtime Capability Transparency Contract
System shall expose deterministic runtime capability diagnostics for backend selection:
- canonical capability matrix for `cpu`, `torch_cuda`, `triton`, `tensorrt`
- required vs optional capabilities per backend
- stable reason-code catalog for non-available paths
- strict vs permissive fallback behavior described in runtime docs
- runtime report fields include:
  - `requested_backend`
  - `selected_backend`
  - `execution_backend`
  - `selection_fallback_reason`
  - `execution_fallback_reason`
  - `requested_dtype`
  - `effective_dtype`
  - `fp8_requested`
  - `fp8_enabled`
  - `fp8_fallback_reason`
  - `latency_ms.total`
  - `latency_ms.backend_execute`
  - `latency_ms.backend_preflight`
These runtime telemetry fields must be exposed consistently in CLI reports and service API responses.
Service runtime shall classify overload/timeout failures deterministically:
- queue saturation -> HTTP `429`
- predict timeout -> HTTP `504`
Service runtime shall classify backend execution failures deterministically:
- backend unavailable -> HTTP `503`
- backend inference/protocol failures -> HTTP `502`
Service runtime shall support optional canary shadow-compare mode with mismatch telemetry counters.
Canary payload capture (when enabled) shall use explicit policy modes (`off|mismatch|error|all`)
and bounded storage settings.
Triton TileSSM runtime path shall support long-sequence execution using chunked scan launches
with deterministic recurrent state carry-over between chunks.
Triton TileUnpack dispatch shall support both overlap modes (`override`, `blend`) without
hardcoded forced reference fallback branches when accelerated path is selected.
Stage-1 Triton fused route in FF heavy-path inference shall be compatibility-gated and must
fall back deterministically to decomposed path when compatibility predicates are not satisfied.
Triton GPU benchmark reporting shall expose per-op shape-bucket autotune telemetry with cached
selected launch configuration metadata and cache hit/miss counters.

## 6. Non-Functional Requirements
- Determinism: same input + same config -> same active tile set
- Exportability: avoid Python-side runtime control flow
- CPU baseline runnable from clean environment
- Testability: correctness, stability, performance checks

## 7. Pseudo-Code Requirements
### 7.1 Utility Oracle Labeling
```text
for each minibatch:
  sample oracle tile subset S (random + high teacher uncertainty + long-tail)
  for i in S:
    y0 = forward_with_tile(i, enabled=0)
    y1 = forward_with_tile(i, enabled=1)
    Delta_i = L_distill(y0, y_teacher) - L_distill(y1, y_teacher)
    Delta_i = clamp(Delta_i, -tau, tau)
  train router utility head to predict Delta_i (L1 or ranking loss)
  log oracle diagnostics:
    sample composition + delta distribution + clipping ratio
```

### 7.2 Continuous Budget Training
```text
U = router(features)
p = sigmoid(U)
g = STE(p)
y = model_forward_with_gate(g)
L_main = task_losses(y, target)
E_cost = sum_i(p_i * C_h + (1-p_i) * C_c)
L = L_main + mu * (E_cost - B)
backprop(L)
e = (E_cost - B) / B
ema_e = beta*ema_e + (1-beta)*e
eta_eff = (eta_mu / (1 + decay*step)) * clip(1 + abs(ema_e), lr_min, lr_max)
delta = eta_eff * (E_cost - B)
if abs(e) <= deadband_ratio:
  delta = 0
delta = clip(delta, -delta_clip, delta_clip)  # optional
mu = clip(mu + delta, mu_min, mu_max)
```

### 7.3 Deterministic Inference Budgeting
```text
U = router(features)
for each tile i:
  score_i = U_i / DeltaC_i
order = argsort(score, descending=True)
active = []
spent = 0
for i in order:
  if len(active) == Kmax: break
  if spent + DeltaC_i <= B:
    active.append(i)
    spent += DeltaC_i
return active
```

### 7.4 Quadtree L0/L1/L2 Policy
```text
L0 = select_by_budget(U_L0, cost_L0, B1, Kmax_L0)
candidates = []
for i in L0:
  split_score_i = S_i / O_split_i
  candidates.append((i, split_score_i))
L1 = top_by_budget(candidates, B2, Kmax_L1)
if nesting_depth == 2:
  L2 = repeat_split_for_L1_under_B3
```

### 7.5 TilePack/TileUnpack and Ordering
```text
idx_ordered = order_idx(idx, mode=hilbert|multi_direction)
P, meta = TilePack(F, idx_ordered, t)
P_mix = TileSSM_and_local_refine(P)
F_merged = TileUnpackFusion(F_base, P_mix, meta, gate)
```

## 8. Architecture Requirements
- PV stream: always-dense low-res features
- FF stream: sparse high-res packed tiles
- FPN split: low-res dense path + high-res sparse path
- Tile-SSM + local refinement in FF heavy blocks
- DET/SEG heads consume merged features

## 9. Loss Requirements
- DET: focal/quality focal + IoU-family box loss (optional DFL)
- INST-SEG: BCE + Dice + boundary loss
- Multi-task conflict handling: PCGrad++ on shared trunk only
- PCGrad++ diagnostics must report conflict metrics before/after projection:
  - `conflicting_pairs`, `conflicting_pairs_after`
  - `total_pairs`, `conflict_rate_before`, `conflict_rate_after`

PCGrad projection when cosine similarity < 0:
\[
g_a \leftarrow g_a - \frac{g_a \cdot g_b}{\lVert g_b \rVert^2} g_b
\]

## 10. Acceptance Criteria
Release is accepted only if all are true:
- deterministic tile selection and ordering pass tests
- pack/unpack semantics pass identity and overlap tests
- training budget control converges around target budget
- CPU baseline runs via `examples/run_cpu_baseline.py`
- CI passes lint and tests
- runtime plugin contracts and precision profiles are documented
- runtime capability matrix and reason-code contract are documented and test-backed
- parity tolerance profiles (`quality`, `balanced`, `edge`) are documented and test-backed
- PCGrad++ projection is restricted to shared-trunk params and conflict-rate metrics are visible in training reports

## 11. Dependencies and Interfaces
- reference implementation: Python + NumPy (CPU-only baseline)
- runtime targets: TensorRT and ONNX Runtime via plugin contracts

## 12. Deliverables in This Repository
- Source: `apex_x/`
- Tests: `tests/`
- Authoritative docs: `docs/PRD.md`, `docs/ENGINEERING_SPEC.md`
- Runtime docs: `docs/runtime/PLUGIN_SPEC.md`
- Project memory: `docs/CONTEXT.md`

## 13. Change Control
Any architectural change to routing, budgets, tile contracts, ordering, plugin behavior, or precision policy must update:
- `docs/PRD.md`
- `docs/ENGINEERING_SPEC.md`
- `docs/DECISIONS.md`
- `docs/CONTEXT.md`
