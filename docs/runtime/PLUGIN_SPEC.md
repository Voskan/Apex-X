# Runtime Plugin Specification (TensorRT / ORT)

## 1. Purpose
Defines runtime plugin contracts for Apex-X v4 sparse tile execution.

## 2. TilePack
Inputs:
- feature tensor `F[B,C,Hf,Wf]`
- indices `idx[B,K]`

Outputs:
- packed tensor `P[B,K,C,t,t]`
- metadata `meta`

Requirements:
- deterministic index ordering support
- FP16/FP8 tensors
- no hidden reordering outside declared `order_idx` mode

## 3. TileSSMScan
Inputs:
- `tokens[B,K,C]` (tile-token sequence)
- `decay[C]`, `input_gain[C]`, `output_gain[C]`, `state_bias[C]`
- optional `init_state[B,C]`
- optional direction flag (forward/backward; implementation-dependent via plugin field)

Outputs:
- mixed output `y[B,K,C]`
- next recurrent state `state_next[B,C]`

Requirements:
- streaming scan semantics
- numerically stable recurrence constraints:
  - decay clamped to `(1e-6, 1-1e-6)`
  - tokens sanitized/clamped before state update
- forward direction required; backward direction optional but preferred
- bounded temporary memory

## 4. TileUnpackFusion
Inputs:
- `F_base[B,C,H,W]` (dense base feature map)
- `P_out[B,K,C,t,t]` (packed tile outputs)
- `idx[B,K]` (tile indices on the tile grid)
- `levels[B,K]` (nesting priority tags, e.g. `L0=0`, `L1=1`, `L2=2`)
- optional `alpha[B,1,H,W]` gate map for fusion

Outputs:
- merged feature map `F_merged[B,C,H,W]`

Requirements:
- deterministic overlap priority (`L2 > L1 > L0` by default)
- tie-break determinism within same level (stable per-tile order)
- supports:
  - overwrite mode: winner tile pixel overwrites base
  - fusion mode: `F = F_base + alpha * (F_winner - F_base)` when `alpha` is provided

## 5. Decode + NMS
Provide fused decode and batched NMS path for DET head.

Recommended runtime contract:
- inputs:
  - `cls_logits[B,N,C]`
  - `box_reg[B,N,4]` (`l,t,r,b` logits/distances)
  - `quality[B,N]`
  - decode metadata (`centers`, `strides`, or equivalent)
- outputs:
  - `boxes[B,M,4]`, `scores[B,M]`, `class_ids[B,M]`, `valid_counts[B]`
- semantics:
  - deterministic tie-breaks for equal scores
  - class-wise NMS
  - bounded `pre_nms_topk` and `max_detections`

## 6. Precision Profiles
- Quality: FP16-heavy
- Balanced: FP16/FP8
- Edge: INT8 (router and sensitive norms remain FP16)

## 7. Validation
Each plugin release must pass:
- shape and determinism tests
- numerical parity checks vs reference CPU implementation
- latency/memory regression gates
