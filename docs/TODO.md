# Apex-X TODO

## Near-Term
- Implement full DET head outputs on multi-scale features (`P3..P7`) in CPU baseline API.
- Implement instance segmentation prototype assembly + tile boundary refine path.
- Add quadtree split selection implementation (`B1/B2`, `S_i/O_split`) for `L1/L2`.
- Add temporal state object and hysteresis unit tests with sequence-level assertions.

## Runtime and Export
- Add ONNX export contract checks for fixed `Kmax` and tile-size constants.
- Add plugin I/O compliance test vectors for `TilePack`, `TileSSMScan`, `TileUnpackFusion`.
- Add TensorRT/ORT parity harness skeleton.
- Implement full Triton fused kernels for:
  - pack + pre-norm gather path
  - tile scan / Tile-SSM streaming path
  - unpack + gated fusion scatter path
- Implement full TensorRT plugin stack (non-placeholder):
  - `TilePack`
  - `TileSSMScan`
  - `TileUnpackFusion`
  - fused decode + NMS path
- Implement ONNX Runtime custom-op adapter path for sparse tile execution.
- Add profile-based runtime selection (`Quality/Balanced/Edge`) with strict regression gates.
- Add end-to-end TRT/ORT vs reference parity tests (accuracy + deterministic tile-set parity).

## Training System
- Add utility-oracle sampling utilities and distillation delta label pipeline.
- Add continuous-budget trainer loop with dual variable scheduling.
- Add PCGrad++ projection hook for shared trunk parameters.

## Quality Gates
- Add regression thresholds file for p50/p95 latency and memory.
- Add repeatability tests for deterministic ordering modes.
- Add stricter CI checks for docs-contract drift.
