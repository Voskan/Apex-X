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
S = random_subset(tiles, rate=0.1..0.2) U high_uncertainty_tiles
for i in S:
  loss_off = L_distill(run(tile_i=off), y_teacher)
  loss_on  = L_distill(run(tile_i=on),  y_teacher)
  Delta_i = clamp(loss_off - loss_on, -tau, tau)
L_util = regression_or_ranking(U_i, Delta_i)
```

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
Projected dual ascent:
\[
\mu \leftarrow max(0, \mu + \eta_\mu (\mathbb{E}[C]-B))
\]

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
mu = max(0, mu + eta_mu*(E - B))
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

## 18. File Ownership Map
- `apex_x/`: reference implementation
- `tests/`: correctness checks
- `docs/runtime/`: runtime plugin details
- `scripts/`: repeatable perf checks

## 19. Change Management
Any changes to equations, contracts, ordering, or precision policy require synchronized updates across:
- `docs/PRD.md`
- `docs/ENGINEERING_SPEC.md`
- `docs/DECISIONS.md`
- `docs/CONTEXT.md`
