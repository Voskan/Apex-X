# Apex-X v4 Engineering Specification

## 1. Authority and Scope
This document is the engineering contract for Apex-X v4 implementation.
`docs/PRD.md` defines product requirements; this file defines implementation details, equations, interfaces, and test criteria.

## 2. System Overview
Apex-X v4 is a dynamic compute graph with:
- dense low-resolution PV stream
- sparse high-resolution FF stream
- tile utility router
- budget controllers for training and inference
- tile pack/ssm/unpack path for heavy compute

## 3. Tensor and Shape Contracts
### 3.1 Inputs
- Image: `x in R^{B x 3 x H x W}`
- Optional temporal state:
  - previous tile mask `z(t-1)`
  - previous utilities `U(t-1)`
  - previous SSM state `s(t-1)`

### 3.2 Core Feature Maps
- PV feature map (`stride 16 or 32`): `F_pv`
- FF primary feature map (`stride 8`): `F_ff8`
- FF refine feature map (`stride 4`, optional): `F_ff4`

### 3.3 Tile Pack/Unpack
Pack:
- Inputs:
  - `F: [B,C,Hf,Wf]`
  - `idx: [B,K]`
  - fixed tile size `t`
- Outputs:
  - `P: [B,K,C,t,t]`
  - `meta = {origins, ordered_idx, grid_shape, tile_size}`

Unpack:
- Inputs:
  - `F_base: [B,C,Hf,Wf]`
  - `P_out: [B,K,C,t,t]`
  - `meta`
  - optional gate and priority masks
- Output:
  - `F_merged: [B,C,Hf,Wf]`

Overlap semantics:
- default: higher nesting level overrides lower (`L2 > L1 > L0`)
- optional weighted blend if explicitly configured

## 4. Router Specification
### 4.1 Router Inputs per Tile
For tile `i`, aggregate PV statistics over mapped PV region:
- objectness `o_hat`
- uncertainty `u_hat`
- boundary proxy `b_hat`
- variance/energy `v_hat`
- optional motion `m_hat`

Feature vector:
\[
x_i \in \mathbb{R}^d,\; d\approx 12\ldots24
\]

### 4.2 Router Outputs
- utility `U_i`
- split utility `S_i`
- optional temporal keep score `T_i`

### 4.3 Utility Oracle (DeltaLoss)
For oracle subset `S`:
\[
\Delta_i = L_{distill}(y^{(i=0)}, y_T) - L_{distill}(y^{(i=1)}, y_T)
\]
Train router utility head against `Delta_i` with `L1` or ranking loss.
Clamp outliers:
\[
\Delta_i \leftarrow clamp(\Delta_i, -\tau, \tau)
\]

Pseudo-code:
```text
S = random_subset(tiles, rate=0.1..0.2)
    U uncertainty_weighted_subset
    U long_tail_subset
for i in S:
  loss_off = L_distill(run(tile_i=off), y_teacher)
  loss_on  = L_distill(run(tile_i=on),  y_teacher)
  Delta_i = clamp(loss_off - loss_on, -tau, tau)
L_util = regression_or_ranking(U_i, Delta_i)
```
Required training diagnostics per step/epoch:
- sampling composition (`random`, `uncertainty`, `long_tail`)
- delta label distribution (`mean`, `std`, `abs_p95`, `min`, `max`)
- clipping diagnostics (`clipped_count`, `clipped_ratio`)

## 5. Continuous Budgeting (Training)
### 5.1 Gating
\[
p_i = \sigma(U_i),\quad g_i = STE(p_i)
\]
Forward uses `g_i` as hard gate; backward uses straight-through gradient.

### 5.2 Expected Cost
\[
\mathbb{E}[C] = \sum_i \left(p_i C_h + (1-p_i) C_c\right)
\]
where `C_h` and `C_c` are heavy and cheap path costs.

### 5.3 Dual Optimization
Objective:
\[
L = L_{main} + \mu(\mathbb{E}[C]-B),\quad \mu\ge0
\]
Projected dual ascent with optional adaptive schedule:
\[
\eta_{eff}(t)=\frac{\eta_\mu}{1+\lambda t}\cdot
\text{clip}(1+|\text{EMA}(e_t)|,\;s_{min},\;s_{max}),\quad
e_t=\frac{\mathbb{E}[C]-B}{B}
\]
\[
\Delta\mu=\eta_{eff}(t)\cdot(\mathbb{E}[C]-B),\quad
\mu \leftarrow \text{clip}(\mu+\Delta\mu,\;[\mu_{min},\mu_{max}])
\]
Optional stabilizers:
- deadband: skip updates when `|e_t| <= deadband_ratio`
- delta clip: clip `Delta mu` into `[-delta_clip, delta_clip]`

Pseudo-code:
```text
U = Router(x)
p = sigmoid(U)
g = STE(p)
y = Forward(g)
L_main = L_det + lambda_seg*L_seg + ...
E = sum(p_i*C_h + (1-p_i)*C_c)
L = L_main + mu*(E - B)
opt_theta.step(grad(L, theta))
e = (E - B) / B
ema_e = beta*ema_e + (1-beta)*e
eta_eff = (eta_mu / (1 + decay*step)) * clip(1 + abs(ema_e), lr_min, lr_max)
delta = eta_eff * (E - B)
if abs(e) <= deadband_ratio:
  delta = 0
delta = clip(delta, -delta_clip, delta_clip)  # optional
mu = clip(mu + delta, mu_min, mu_max)
```

## 6. Deterministic Inference Budgeting
### 6.1 Greedy Utility-per-Cost
Given `U_i`, `DeltaC_i`, budget `B`, and fixed `Kmax`:
\[
score_i = \frac{U_i}{\Delta C_i}
\]
Sort descending; accept while both constraints hold:
- `spent + DeltaC_i <= B`
- `|active| < Kmax`

### 6.2 Kmax Buffer Contract
- runtime buffers are allocated to `Kmax`
- actual selected tile count is `K <= Kmax`
- unused entries are ignored via valid-count metadata

### 6.3 Temporal Hysteresis with Budget-Aware Carryover
Temporal anti-flicker rule:
\[
z_i(t)=\mathbf{1}[U_i(t)>\theta_{on}\;\lor\;(U_i(t)>\theta_{off}\land z_i(t-1)=1)]
\]
with `theta_on > theta_off`.

Budget-aware enforcement:
- hysteresis state update is computed first
- if active count exceeds per-frame limit (`max_active`), deterministic clipping is applied:
  1. keep tiles that were already active in previous frame
  2. then sort by utility descending
  3. tie-break by tile id ascending

This preserves temporal state reuse while preventing budget overflow in frame-level routing masks.

Temporal stability metrics:
- `tile_flip_rate`:
\[
\frac{\text{total state transitions}}{(T-1)\cdot N_{tiles}}
\]
- `temporal_consistency = 1 - tile_flip_rate`
- `mean_active_ratio`: average active fraction across all frames

Pseudo-code:
```text
scores = [U_i / DeltaC_i for i in tiles]
order = argsort(scores, descending=True)
active, spent = [], 0
for i in order:
  if len(active) >= Kmax: break
  if spent + DeltaC_i <= B:
    active.append(i)
    spent += DeltaC_i
write_to_kmax_buffer(active)
```

## 7. Quadtree Nesting L0/L1/L2
### 7.1 Tile Hierarchy
- `L0`: coarse tiles (`t0`)
- `L1`: `t1=t0/2`
- `L2`: `t2=t1/2` (optional)

### 7.2 Budget Split
- `B1`: heavy compute for selected `L0`
- `B2`: split budget for finer tiles
- optional `B3` for `L2`

### 7.3 Split Utility
\[
score_i^{split} = \frac{S_i}{O_{split,i}}
\]
Select split candidates under `B2`.

Recursive depth-2 contract:
- stage `L0 -> L1` uses `B2`
- stage `L1 -> L2` uses `B3`
- both stages enforce deterministic parent ranking:
  1. split score descending
  2. tile id ascending tie-break
- each accepted parent expands to exactly 4 children
- expansion is capacity-constrained by `Kmax_L1` / `Kmax_L2`

Pseudo-code:
```text
L0 = select_tiles(U_L0, cost_L0, B1, Kmax_L0)
split_candidates = []
for i in L0:
  split_candidates.append((i, S_i / O_split_i))
L1 = select_tiles(split_candidates, split_cost, B2, Kmax_L1)
if nesting_depth >= 2:
  L2 = recursive_split(L1, B3, Kmax_L2)
```

## 8. Ordering and Sequence Geometry
### 8.1 Required Modes
- Hilbert ordering for geometry-preserving locality
- Multi-direction scan aggregation:
  - `L->R`
  - `R->L`
  - `U->D`
  - `D->U`

### 8.2 Determinism Rule
Ordering function must be pure and deterministic for the same input tile set and grid shape.

## 9. Tile-SSM and Local Refine
### 9.1 Tokenization
Mode A (tile token):
\[
v_i = Pool(P_i),\quad v_i\in\mathbb{R}^C
\]
Sequence `v_1..v_K` processed by Tile-SSM.

Mode B (sub-patch token):
- split each tile into `2x2` or `4x4` patches
- use extended token sequence for fine detail

### 9.2 Mamba-like Placeholder Contract
Tile-SSM module must provide:
- streaming scan over ordered token sequence
- optional persistent recurrent state
- bounded temporary memory

### 9.3 Fusion Gate
Heavy output and base feature merge:
\[
F_{merged} = F_{base} + g_{fuse} \odot (F_{heavy} - F_{base})
\]
where `g_fuse in [0,1]` may be scalar, channel-wise, or spatial.

## 10. Losses and Multi-task Gradient Handling
### 10.1 Task Losses
- DET: focal/quality focal + IoU-family box loss (+ optional DFL)
- INST-SEG: BCE + Dice + boundary loss
- Optional task heads define their own loss terms

### 10.2 PCGrad++ for Shared Trunk
For gradient pair `(g_a, g_b)` on shared params, if `cos(g_a, g_b) < 0`:
\[
g_a \leftarrow g_a - \frac{g_a\cdot g_b}{||g_b||^2}g_b
\]
Do not project head-specific gradients.
Diagnostics contract:
- compute conflict pairs before projection over ordered task-group pairs
- compute conflict pairs after projection over the projected gradient set
- expose:
  - `conflicting_pairs`
  - `conflicting_pairs_after`
  - `total_pairs`
  - `conflict_rate_before`
  - `conflict_rate_after`
- trainer logs must include the above metrics when PCGrad++ is enabled.

## 11. Temporal Hysteresis and State Reuse
### 11.1 State
- `z(t-1)`: previous active tiles
- `U(t-1)`: previous utility
- `s(t-1)`: previous SSM state

### 11.2 Update Rule
Hysteresis:
\[
z_i(t)=\mathbf{1}[U_i(t)>\theta_{on} \lor (U_i(t)>\theta_{off} \land z_i(t-1)=1)]
\]
`theta_on > theta_off` to prevent flicker.

Stability loss:
\[
L_{stab}=\sum_i |p_i(t)-p_i(t-1)|
\]
Optional spatial TV term may be added.

## 12. Runtime Plugin Specifications
Detailed runtime notes also live in `docs/runtime/PLUGIN_SPEC.md`.

### 12.1 TilePack Plugin
Inputs: `F, idx`
Outputs: `P, meta`
Requirements:
- FP16 and FP8 data support
- deterministic ordering
- bounded workspace

### 12.2 TileSSMScan Plugin
Inputs: packed tokens/tiles + optional prior state
Outputs: mixed outputs + next state
Requirements:
- streaming scan
- multi-direction mode
- no unbounded temporary allocations

### 12.3 TileUnpackFusion Plugin
Inputs: `F_base, P_out, meta, optional gate`
Outputs: merged feature map
Requirements:
- residual + gate semantics
- overlap priority support

### 12.4 Optional Plugins
- `MaskedConv`
- fused DET decode + NMS

### 12.5 Build-Time Plugin Contract Checks
TensorRT builder must validate plugin creator contracts before engine build:
- required creators are present in registry
- creator version matches expected contract version
- creator namespace matches expected contract namespace
- creator field-signature metadata covers expected fields for each plugin

Strict mode is default and must fail fast with actionable mismatch diagnostics.

## 13. QAT and Precision Policy
### 13.1 Profiles
- Quality: FP16-heavy path, larger tile budget
- Balanced: FP16/FP8 mixed
- Edge: INT8 with router in FP16

### 13.2 QAT Rules
- keep router logits and normalization in higher precision during QAT
- calibrate heavy branches per-level
- validate parity vs FP16 baseline before enabling INT8 by default

### 13.3 Acceptance Threshold
- mAP/mIoU degradation must stay within agreed profile thresholds

### 13.4 FP8 Telemetry Contract
When FP8 is requested, runtime/benchmark reports must expose:
- `requested_dtype`
- `effective_dtype`
- `fp8_requested`
- `fp8_enabled`
- `fp8_fallback_reason`

`fp8_fallback_reason` must be a canonical reason-code from the runtime capability catalog.

### 13.5 INT8 Sensitive-Layer Precision Contract
In TensorRT INT8 builds:
- layers matching sensitive keywords (`router`, `kan` by default) must be constrained to FP16.
- strict mode must fail build when a matched layer cannot be constrained via TensorRT precision APIs.
- build results must expose per-layer precision constraint evidence:
  - `layer_name`
  - `matched_keyword`
  - `precision_applied`
  - `output_constraints_applied`

### 13.6 INT8 Calibration Cache Governance Contract
In TensorRT INT8 builds with calibration cache enabled:
- calibration cache keys must be deterministic and include:
  - model/export identity hash
  - plugin contract metadata (version + namespace)
  - precision profile
  - calibration dataset version
- calibration dataset version may be:
  - explicitly configured (`calibration_dataset_version`)
  - auto-derived from calibration batch digest
- cache governance rules:
  - unchanged key must reuse existing cache
  - changed key must invalidate cache automatically
  - legacy raw cache blobs are accepted only when key governance is disabled
- build results must expose:
  - `calibration_cache_key`
  - `calibration_dataset_version`

## 14. Performance Regression Testing
### 14.1 Required Metrics
- latency `p50/p95`
- memory peak
- selected tile count distribution
- budget adherence (`spent <= B`)

### 14.2 Regression Policy
A change fails perf gate if:
- `p95` latency worsens over threshold
- memory peak exceeds threshold
- budget overshoot rate is non-zero in deterministic mode

### 14.3 Baseline Tooling
- `scripts/perf_regression.py` provides CPU reference timing
- runtime-specific harnesses must follow same report schema

## 15. Export Contracts (TRT/ORT)
- dynamic tile count only via fixed `Kmax` buffers + valid count
- fixed tile size and fixed max nesting depth in model profile
- inference graph must not depend on Python control flow

## 16. CPU Baseline Requirements
Reference implementation must:
- run on CPU-only environment
- implement routing, deterministic budgeting, pack/unpack, and placeholder Tile-SSM
- include tests for core contracts

## 17. Validation Matrix
### 17.1 Correctness
- pack/unpack identity
- overlap priority correctness
- deterministic ordering
- deterministic selection

### 17.2 Training Stability
- non-collapsing router probabilities under budget
- distillation gap trend improves over epochs

### 17.3 Export Parity
- bounded quality drop between PyTorch and TRT/ORT profiles

### 17.4 Eval Dataset Contract (CLI Model Eval Path)
For `apex_x cli eval --dataset-npz`:
- required:
  - `.npy` or `.npz` input with `images` tensor shape `[N,3,H,W]`
- optional:
  - `.npz` key `det_score_target` (compat alias `det_scores_target`) with shape `[N]` or `[N,1]`
  - `.npz` key `selected_tiles_target` (compat alias `selected_tiles_targets`) with shape `[N]` or `[N,1]`
- report contract:
  - `model_eval.det_score` aggregate stats (`mean/std/min/max`)
  - `model_eval.selected_tiles` aggregate stats (`mean/p95`)
  - when `det_score_target` is provided, `model_eval.det_score_target` includes:
    - `mae`
    - `rmse`
    - `bias`
    - `r2` (nullable when target variance is zero)
    - `pearson_corr` (nullable when variance is zero)
  - when `selected_tiles_target` is provided, `model_eval.selected_tiles_target` includes:
    - `mae`
    - `rmse`
    - `bias`
    - `exact_match_rate`

## 18. Runtime Capability and Parity Contracts
### 18.1 Backend Capability Matrix
Canonical backend matrix:
- `cpu`:
  - required: CPU torch execution
  - optional: none
- `torch_cuda`:
  - required: `cuda.available = true`
  - optional: none
- `triton`:
  - required: `cuda.available = true`, `triton.available = true`
  - optional: Triton version metadata
  - TileSSM long-sequence contract:
    - when sequence length exceeds single-launch bound, runtime must use chunked scan launches
      with recurrent state carry-over across chunks
  - TileUnpack overlap-mode contract:
    - dispatch supports `override` and `blend` modes
    - blend mode must not rely on hardcoded forced-reference dispatch branches
  - Stage-1 fused selector contract:
    - FF heavy-path inference may use fused stage-1 route only under explicit compatibility gates
    - non-compatible cases must fall back to decomposed path with deterministic behavior
- `tensorrt`:
  - required: `cuda.available = true`, `tensorrt.python_available = true`
  - optional: local header availability for build workflows

Precision overlays:
- TensorRT INT8 requires:
  - TensorRT Python module
  - CUDA
  - `BuilderFlag.INT8`
- FP8 requires:
  - FP8 torch dtype exposure
  - CUDA
  - compute capability policy gate (`sm90+`)

### 18.2 Reason-Code Contract
Runtime capability probes must emit deterministic reason codes only from the catalog:
- CUDA: `cuda_unavailable`, `cuda_device_not_found`, `cuda_device_index_out_of_range`, `cuda_query_failed`
- Triton: `triton_not_installed`
- TensorRT Python: `tensorrt_python_not_installed`, `tensorrt_python_import_failed`
- TensorRT INT8: `tensorrt_python_unavailable`, `cuda_required_for_tensorrt_int8`, `tensorrt_int8_builder_flag_missing`
- FP8: `torch_build_missing_fp8_dtype`, `fp8_requires_cuda`, `cuda_compute_capability_unknown`, `compute_capability_below_sm90`

Hardware metadata (for example compute capability values) must be exposed in dedicated fields, not dynamic reason strings.

### 18.3 TensorRT CLI Runtime Execution Contract
For CLI/inference-runner TensorRT execution:
- required runtime artifact:
  - `APEXX_TRT_ENGINE_PATH` (serialized engine file)
- optional artifacts and controls:
  - `APEXX_EXPORT_MANIFEST_PATH` for manifest/ONNX hash preflight
  - `APEXX_TRT_VERIFY_MANIFEST_HASH` (`true` by default)
  - `APEXX_TRT_PLUGIN_LIB` (`os.pathsep`-separated plugin `.so/.dylib` paths)
  - `APEXX_TRT_INPUT_NAME` for explicit input tensor binding when engine has multiple inputs
  - `APEXX_TRT_EXTRA_INPUTS_NPZ` for named auxiliary inputs (`.npz`)
  - `APEXX_TRT_PRIMARY_OUTPUT_NAME` for explicit primary output mapping into CLI result schema
  - `APEXX_TRT_DET_BOXES_NAME`, `APEXX_TRT_DET_SCORES_NAME`,
    `APEXX_TRT_DET_CLASS_IDS_NAME`, `APEXX_TRT_DET_VALID_NAME`
    for explicit DET output mapping
- strict/permissive policy:
  - strict mode must fail on preflight/runtime execution errors
  - permissive mode must fall back deterministically and emit execution fallback reason
- runtime telemetry:
  - `runtime.latency_ms.total` for full inference-call wall time
  - `runtime.latency_ms.backend_execute` for backend execution segment
  - `runtime.latency_ms.backend_preflight` for backend preflight/handshake segment
  - schema applies to both Python CLI reports and Go service `/predict` response payloads
- Go runtime bridge execution contract:
  - ORT bridge env controls:
    - `APEXX_ORT_BRIDGE_CMD`
  - TRT bridge env controls:
    - `APEXX_TRT_BRIDGE_CMD`
  - bridge payload protocol:
    - request JSON stdin: `{backend, artifact_path, requests[]}`
    - response JSON stdout: `{results[], error?}`
  - adapters must fail closed when bridge/native backend execution is unavailable
    (no synthetic score fallback in production path)
- service error mapping (Go runtime):
  - queue saturation must return HTTP `429`
  - predict timeout must return HTTP `504`
  - backend unavailable must return HTTP `503`
  - backend inference/protocol failures must return HTTP `502`
- service canary mode (Go runtime):
  - optional secondary adapter compares sampled requests asynchronously
  - mismatch telemetry counters must be exported in service metrics
  - capture policy controls:
    - `canary-capture-policy` / `APEXX_CANARY_CAPTURE_POLICY` (`off|mismatch|error|all`)
    - `canary-capture-path` / `APEXX_CANARY_CAPTURE_PATH` (JSONL sink)
    - `canary-capture-max-bytes` / `APEXX_CANARY_CAPTURE_MAX_BYTES` (size guard)
  - CI SLA gate:
    - `TestCanaryLoadGateThresholds` validates timeout/queue-overflow rates and canary overhead thresholds

### 18.4 Parity Tolerance Profiles
Parity tolerance presets must be versioned and testable:
- `quality`
- `balanced`
- `edge`

Each preset defines:
- op-level tolerances
- end-to-end tolerances
- mismatch ratio limit

Profile selection must be explicit in parity tests.

TensorRT parity harness requirements:
- backend pair matrix must cover:
  - `reference` vs `triton`
  - `reference` vs `tensorrt`
  - `triton` vs `tensorrt`
- sweep dimensions must include:
  - shape cases (at least small/medium representative inputs)
  - precision cases (at least FP16 + FP32 candidate envelopes on CPU-safe tests;
    INT8/FP8 where runtime support exists)

### 18.5 Triton Autotune Registry Contract
Triton perf runs must expose deterministic autotune telemetry:
- registry key:
  - `op_name + shape_bucket`
- registry value:
  - `selected_config` (`BLOCK_*`, `num_warps`, `num_stages`, when available)
  - `selection_source` (`triton_best_config`, `heuristic`, `registry_cache`)
  - cache counters (`launches`, `cache_hits`, `cache_misses`)
- benchmark report requirements:
  - JSON includes `triton_autotune.summary` and `triton_autotune.entries`
  - Markdown includes a readable autotune registry section for regression review
  - `summary` must include `cache_entries` and `cache_hit_rate`

### 18.6 Unified Perf Regression Policy (CPU/GPU/TRT)
Perf regression wrappers must share one pass/fail formula:
- fail when:
  - `current_ms > baseline_ms * (1 + max_regression_ratio) + max_regression_abs_ms`

Regression wrappers:
- CPU: `scripts/perf_regression.py`
- GPU: `scripts/perf_regression_gpu.py`
- TensorRT shape-sweep: `scripts/perf_regression_trt.py`

All wrappers must support:
- baseline template generation mode
- compare mode with explicit baseline
- normalized trend artifact output with shared schema keys:
  - `schema_version`
  - `suite`
  - `timestamp_utc`
  - `overall_status`
  - `total_metrics`
  - `failed_metrics`
  - `metrics[]`

## 19. File Ownership Map
- `apex_x/`: reference implementation
- `tests/`: correctness checks
- `docs/runtime/`: runtime plugin details
- `scripts/`: repeatable perf checks

## 20. Change Management
Any changes to equations, contracts, ordering, or precision policy require synchronized updates across:
- `docs/PRD.md`
- `docs/ENGINEERING_SPEC.md`
- `docs/DECISIONS.md`
- `docs/CONTEXT.md`
