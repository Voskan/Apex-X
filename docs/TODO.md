# Apex-X Full Readiness Backlog

Last updated: 2026-02-11

This file is the execution backlog for taking Apex-X from CPU-first reference + partial GPU acceleration
to a full production-ready GPU project with strict correctness, performance, and reproducibility gates.

The target is "100% ready" under a concrete engineering definition, not marketing wording.

## 1. Definition of "100% Ready"

All items below must be true:

1. End-to-end inference (`predict`, `eval`, service runtime) works on:
   - `cpu` reference
   - `torch` CUDA path
   - `triton` accelerated path
   - `tensorrt` engine path
2. No critical runtime path depends on placeholder/no-op stubs.
3. CPU vs GPU vs TRT parity is validated on golden sets with explicit tolerances.
4. Performance gates are enforced in CI:
   - latency (`p50`, `p95`)
   - throughput
   - peak memory
   - compile/build success on target environments
5. Precision policy is operational:
   - FP16 stable
   - INT8 calibrated path stable
   - FP8 policy and fallback behavior validated
6. Go runtime service executes real inference backends (not synthetic scoring stubs).
7. Release checklist is pass/fail with reproducible artifacts and rollback playbook.

## 2. Current Snapshot (Context for Prioritization)

- CPU baseline is stable and well tested.
- Triton kernel paths exist for major tile ops and TileSSM, with fallback semantics.
- TensorRT builder/calibration and plugin code paths exist, but full integration and validation
  must be tightened end-to-end.
- GPU CI exists but is optional; not yet a mandatory merge gate for GPU-critical changes.
- Some CLI/service/export paths still need stronger production-grade backend wiring.

## 3. Execution Rules

1. Every task must include:
   - concrete code changes
   - tests
   - documentation updates
   - measurable acceptance criteria
2. No task is marked done without:
   - green tests
   - parity evidence
   - benchmark evidence (when performance-related)
3. Any shape-contract or runtime-contract change must update:
   - `docs/PRD.md`
   - `docs/ENGINEERING_SPEC.md`
   - `docs/DECISIONS.md`
   - `docs/CONTEXT.md`

Status legend for tasks below:
- `Status: [x]` completed and validated
- `Status: [~]` partially implemented
- `Status: [ ]` not started

## 4. Critical Path (Order)

1. Phase P0: freeze contracts and observability baseline (completed 2026-02-10)
2. Phase P1: end-to-end backend selection and execution
3. Phase P2: Triton parity + performance completion
4. Phase P3: TensorRT production completion
5. Phase P4: training/math excellence track
6. Phase P5: Go service production integration
7. Phase P6: CI/CD hard gates and release readiness

Recently completed and removed from active queue:
- `P2-04` Remove legacy fused Triton stub dependency (2026-02-10)
- `P5-01` Replace synthetic adapter scoring with real backend inference calls (2026-02-11)
- `P5-02` Service-level batching and timeout SLA enforcement (2026-02-11)
- `P5-03` Canary parity mode in service runtime (2026-02-11)
- `P6-03` Release checklist CI evidence attestation wiring (2026-02-11)
- `X-03` Backward compatibility migration notes + changelog (2026-02-11)
- `P4-06` Temporal hysteresis and state reuse quality gates (2026-02-11)
- `X-01` Runtime documentation consistency sweep (2026-02-11)
- `P4-05` Quadtree split policy completion with budget-aware recursion (2026-02-11)
- `P4-03` Deterministic inference budgeting stress tests (2026-02-11)
- `P4-02` Continuous budget controller convergence guarantees (2026-02-11)
- `P4-04` PCGrad++ completeness on shared trunk in multi-task training (2026-02-11)
- `P4-01` Utility-oracle data pipeline hardening (2026-02-11)
- `P1-02` Replace export no-op with real export pipeline (2026-02-11)
- `P1-03` End-to-end `predict` path on GPU backends (2026-02-11)
- `P1-04` End-to-end `eval` path with real model execution (2026-02-11)
- `P3-04` FP16/INT8 mixed precision policy enforcement for sensitive layers (2026-02-11)
- `P3-01` End-to-end plugin registration and contract validation at build time (2026-02-11)
- `P3-03` INT8 production calibration flow and cache governance (2026-02-11)
- Device validation: Triton GPU parity suite on CUDA host (2026-02-11)
- Device validation: GPU perf baseline artifacts on CUDA host (2026-02-11)
- Device validation: Go ORT bridge with real ONNX model on host (2026-02-11)

Device/deployment-blocked validation queue:
- Blocking snapshot (2026-02-11):
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1` (`NVIDIA GeForce RTX 2070 SUPER`, `sm75`)
  - `triton` Python module available (`3.5.1`)
  - `tensorrt` Python module available (`10.15.1.29`)
  - `onnxruntime` Python module available (`1.24.1`)
  - `cmake` CLI available (`4.2.1`)
  - TensorRT plugin shared library is still unavailable on this host (missing TensorRT headers + CUDA compiler toolchain)
  - GitHub API branch-protection checks require authenticated `gh` (not installed on host)
- `Status: [ ]` Run TensorRT engine shape sweep on deployment GPU and attach artifacts:
  - local run completed with synthetic dynamic engine:
    - `artifacts/models/trt_bench_dynamic.engine`
    - `artifacts/perf_trt_shape_sweep.json`
    - `artifacts/perf_trt_shape_sweep.md`
  - blocker: final deployment rerun is still required with production deployment engine.
  - `python -m apex_x.bench.trt_engine_sweep --trt-engine-path <engine> --shape-case "input=1x3x128x128" --shape-case "input=1x3x256x256" --output-json artifacts/perf_trt_shape_sweep.json --output-md artifacts/perf_trt_shape_sweep.md`
- `Status: [ ]` Run TensorRT regression compare/trend wrapper on deployment GPU and archive artifacts:
  - local run completed and compare currently passes on synthetic engine:
    - `artifacts/perf_trt_current.json`
    - `artifacts/perf_trt_compare.json`
    - `artifacts/perf_trt_trend.json`
  - blocker: baseline tuning with final deployment engine is still required.
  - `python scripts/perf_regression_trt.py --compare --baseline scripts/perf_baseline_trt.json --output artifacts/perf_trt_current.json --summary artifacts/perf_trt_compare.json --trend-output artifacts/perf_trt_trend.json --trt-engine-path <engine> --shape-case "input=1x3x128x128" --shape-case "input=1x3x256x256"`
- `Status: [ ]` Run TensorRT plugin parity tests on CUDA host:
  - blocker: tests are skipped while `APEXX_TRT_PLUGIN_LIB` is unset; shared plugin `.so` is not built on this host.
  - local evidence: `artifacts/trt_plugin_parity_pytest.log`
  - `python -m pytest -q tests/test_tensorrt_tilepack_parity.py tests/test_tensorrt_tileunpackfusion_parity.py tests/test_tensorrt_tilessm_parity.py tests/test_tensorrt_nms_decode_parity.py`
- `Status: [ ]` Run TensorRT plugin C++ shape/serialization tests after native build:
  - local configure/build completed (`artifacts/trt_cmake_configure.log`, `artifacts/trt_cmake_build.log`),
    but plugin shared library and native tests are skipped because TensorRT headers and CUDA compiler are absent.
  - `ctest` currently reports no discovered tests (`artifacts/trt_ctest.log`).
  - `cmake -S runtime/tensorrt -B runtime/tensorrt/build -DAPEXX_ENABLE_TRT=ON`
  - `cmake --build runtime/tensorrt/build -j`
  - `ctest --test-dir runtime/tensorrt/build --output-on-failure`
- `Status: [ ]` Capture FP8 benchmark evidence on supported GPU (`sm90+`) for P4-07 closure:
  - blocker: host GPU is `sm75`; FP8 request falls back with `compute_capability_below_sm90`
  - `python -m apex_x.bench.gpu_bench --dtype fp8 --warmup 10 --iters 50 --output-json artifacts/perf_gpu_fp8.json --output-md artifacts/perf_gpu_fp8.md`
  - optional regression wrapper:
    - `python scripts/perf_regression_gpu.py --dtype fp8 --output artifacts/perf_gpu_fp8_current.json --compare --baseline scripts/perf_baseline_gpu.json --summary artifacts/perf_gpu_fp8_compare.json`
- `Status: [ ]` Validate GPU mandatory PR gate on GitHub protected branch settings:
  - blocker: cannot query/modify protected-branch settings without authenticated GitHub CLI/API access
  - Required status check: `GPU Perf Regression / gpu-perf-regression`
  - `gh api repos/Voskan/Apex-X/branches/main/protection --jq '.required_status_checks.checks[].context'`
  - Open a GPU-critical PR and confirm merge is blocked until GPU workflow passes
- `Status: [ ]` Run weekly GPU trend workflow on deployment runner and archive artifacts:
  - blocker: requires GitHub Actions dispatch on self-hosted GPU runner with repo variable management
  - Enable repository variable `APEXX_ENABLE_GPU_WEEKLY=true`
  - `gh variable set APEXX_ENABLE_GPU_WEEKLY --body true`
  - `gh workflow run perf_trend_weekly.yml -f run_gpu=true -f trt_engine_path=<engine>`
  - `gh run watch <run-id>`
  - `gh run download <run-id> -n perf-trend-gpu-weekly -D artifacts/weekly_gpu`
  - Verify `.github/workflows/perf_trend_weekly.yml` uploads:
    - `artifacts/perf_gpu_current_weekly.json`
    - `artifacts/perf_gpu_compare_weekly.json`
    - `artifacts/perf_gpu_trend_weekly.json`
- `Status: [ ]` Validate Go runtime TRT bridge/native path with deployment engine on CUDA+TensorRT host:
  - local bridge validation completed with real TensorRT engine artifact:
    - engine: `artifacts/models/trt_bench_dynamic.engine`
    - Go tests: `artifacts/go_trt_bridge_test_real_engine.log`
    - bridge probe: `artifacts/service_bridge_trt_real_engine.json`
  - blocker: final deployment rerun is still required with production deployment engine.
  - `cd runtime/go && APEXX_TRT_ENGINE_PATH=<engine.plan> APEXX_TRT_BRIDGE_CMD="python -m apex_x.runtime.service_bridge" CGO_ENABLED=1 go test -tags tensorrt ./...`

---

## Phase P1 - End-to-End Runtime Path Completion

---

## Phase P2 - Triton Completion (Correctness + Speed)

### P2-01. Close feature gaps in Triton tile unpack path
Status: [~]

Why:
- Fallback-only branches break expectations for full GPU mode.

Implementation:
- Implement missing behavior branches in Triton path where currently forced to reference fallback
  (for example blend-related branch if configured).
- Keep deterministic overlap semantics identical to reference.

Progress (2026-02-11):
- Removed forced reference-only dispatch branch for `overlap_mode=\"blend\"`.
- Added ordered blend composition path in `tileunpack_triton(...)` with parity-equivalent
  overlap semantics.
- Extended overlap tests:
  - CPU dispatch contract no longer expects hardcoded blend fallback reason.
  - GPU overlap suite now includes blend parity case.
- Remaining gap:
  - optimize blend path with dedicated Triton kernels and attach CUDA perf evidence.

Files:
- `apex_x/kernels/triton/tileunpack.py`
- `tests/test_triton_tileunpack*.py`
- `docs/runtime/TRITON_TILEUNPACK.md`

Acceptance:
- Triton path covers all configured operation modes in spec.
- No silent fallback for supported modes.

Validation:
- `python -m pytest -q tests/test_triton_tileunpack*`

### P2-02. Stabilize TileSSM Triton limits and long-sequence behavior
Status: [~]

Why:
- Sequence-length limitations and edge cases can invalidate large-scene workloads.

Implementation:
- Review and improve compile/runtime limits.
- Add chunked/streamed path when sequence length exceeds kernel-friendly bound.

Progress (2026-02-11):
- Added long-sequence chunked forward execution in Triton TileSSM path:
  - `K > 4096` now executes as streamed chunk launches with state carry-over.
  - preserves forward/backward/bidirectional composition semantics while avoiding
    oversized single-launch kernels.
- Added CPU-safe unit tests for chunking contract:
  - `tests/test_triton_tilessm_parity_dispatch.py`
  - validates multi-chunk and single-launch paths for `_scan_triton_forward(...)`.
- Remaining gap:
  - run CUDA Triton parity/perf tests for large-`K` scenarios and capture deployment artifacts.

Files:
- `apex_x/kernels/triton/tilessm_scan.py`
- `tests/test_triton_tilessm_*`
- `docs/runtime/TRITON_SSM.md`

Acceptance:
- Large `K` scenarios pass correctness tests with documented fallback/streaming policy.

Validation:
- `python -m pytest -q tests/test_triton_tilessm_*`

### P2-03. Promote fused stage-1 path to default accelerated route
Status: [~]

Why:
- Stage-1 fused kernel exists and should be first-class for real speedups.

Implementation:
- Integrate `fused_pack_op_unpack` path into runtime selector where compatible.
- Ensure deterministic output parity against decomposed path.

Progress (2026-02-11):
- Added compatibility-gated Stage-1 fused selector in `FFHeavyPath` inference path:
  - uses `fused_pack_op_unpack_dispatch(...)` when strict compatibility predicates pass
  - falls back deterministically to decomposed `pack -> FiLM -> unpack` otherwise.
- Integrated runtime plugin wiring:
  - `FFModule` now forwards runtime plugin enablement into `FFHeavyPath` fused-stage selector.
- Added unit coverage:
  - `tests/test_ff_heavy_path_fused_stage1.py`
  - validates selector activation, decomposed parity, and fallback on non-constant FiLM params.
- Remaining gap:
  - capture CUDA benchmark evidence showing measurable speedup against decomposed route
    under compatible selector scenarios.

Files:
- `apex_x/kernels/triton/fused_pack_op_unpack.py`
- `apex_x/model/ff_heavy_path.py`
- `tests/test_triton_fused_stage1_*`
- `docs/runtime/TRITON_FUSED_STAGE1.md`

Acceptance:
- Measurable speedup over decomposed path on target GPUs.
- Parity checks pass under profile tolerances.

Validation:
- `python -m pytest -q tests/test_triton_fused_stage1_*`
- `python -m apex_x.bench.gpu_bench --warmup 10 --iters 50`

### P2-05. Triton perf autotune and kernel configuration registry
Status: [x]

Why:
- "Best result" requires shape-aware tuning, not static launch settings.

Implementation:
- Add autotune registry per op/shape bucket.
- Cache selected configs and expose telemetry.

Progress (2026-02-11):
- Added Triton autotune registry module:
  - `apex_x/kernels/triton/autotune_registry.py`
  - tracks per-op/per-shape-bucket selected launch config, selection source, and cache hit/miss counts
- Wired registry telemetry into Triton kernels with autotune configs:
  - TilePack (`_tilepack_kernel`)
  - TileUnpack priority/scatter (`_tileunpack_priority_kernel`, `_tileunpack_scatter_kernel`)
  - FusionGate alpha/fuse (`_fusiongate_alpha_kernel`, `_fusiongate_fuse_kernel`)
  - Fused stage-1 pack/op/unpack (`_fused_pack_op_unpack_kernel`)
- Extended GPU benchmark report output:
  - JSON now includes `triton_autotune` block with `summary` + `entries`
  - Markdown summary now includes `Triton Autotune Registry` table
- Added CPU-safe contract tests:
  - `tests/test_triton_autotune_registry.py`
- Captured CUDA benchmark evidence with autotune telemetry and p50/p95 comparisons:
  - `artifacts/perf_gpu.json`
  - `artifacts/perf_gpu.md`

Files:
- `apex_x/kernels/triton/*.py`
- `apex_x/bench/gpu_bench.py`
- `docs/PERF_GPU.md`

Acceptance:
- Benchmark report captures chosen config and resulting p50/p95 improvements.

Validation:
- `python -m pytest -q tests/test_triton_autotune_registry.py tests/test_gpu_bench_fp8.py`
- `python -m apex_x.bench.gpu_bench --output-json artifacts/perf_gpu.json --output-md artifacts/perf_gpu.md`

---

## Phase P3 - TensorRT Production Completion

### P3-02. Shape inference, serialization, and dynamic-shape coverage for TRT plugins
Status: [~]

Why:
- Dynamic shape failures are a common production outage source.

Implementation:
- Complete shape function coverage for all required plugins.
- Add serialization/deserialization roundtrip tests.

Files:
- `runtime/tensorrt/src/*.cpp`
- `runtime/tensorrt/plugins/*`
- `tests/test_tensorrt_plugins*.py`
- `docs/runtime/PLUGIN_SPEC.md`

Acceptance:
- Plugins pass shape + serialization tests across defined profile ranges.

Validation:
- `ctest --test-dir runtime/tensorrt/build`
- `python -m pytest -q tests/test_tensorrt_plugins*`

### P3-05. TensorRT end-to-end parity harness against reference and Triton
Status: [~]

Why:
- Need evidence that TRT path is fast and correct.

Implementation:
- Build parity suite:
  - reference vs triton
  - reference vs tensorrt
  - triton vs tensorrt
- Include shape sweep and precision sweep.

Progress (2026-02-11):
- Added backend matrix parity APIs in `apex_x/runtime/parity.py`:
  - `ParityMatrixCase`
  - `run_parity_matrix_case(...)`
  - `run_parity_sweep(...)`
  - `ParitySweepReport`
- Added CPU-safe sweep tests for TRT parity harness contract:
  - `tests/test_trt_parity_harness.py`
  - validates pair matrix, shape sweep, and precision sweep execution.
- Remaining gap:
  - execute real CUDA-backed parity suites for Triton and TensorRT engines and attach artifacts
    from deployment environment.

Files:
- `apex_x/runtime/parity.py`
- `tests/test_trt_parity*.py`
- `docs/runtime/PARITY.md`

Acceptance:
- All required cases pass tolerance and mismatch-ratio thresholds.

Validation:
- `python -m pytest -q tests/test_trt_parity*`

---

## Phase P4 - Math and Model Quality Track ("Best Math" Execution)

### P4-07. FP8 policy from declaration to measurable runtime value
Status: [~]

Why:
- FP8 policy exists; readiness requires operational and measured gain.

Implementation:
- Integrate FP8 execution where supported for heavy ops.
- Add explicit fallback telemetry and parity/perf tests.

Progress (2026-02-11):
- Precision fallback reasons were normalized to canonical runtime reason-codes in:
  - `apex_x/runtime/precision.py`
- GPU bench now supports explicit FP8 request mode with operational telemetry:
  - `--dtype fp8` in `apex_x/bench/gpu_bench.py`
  - report fields:
    - `requested_dtype`
    - `effective_dtype`
    - `fp8_requested`
    - `fp8_enabled`
    - `fp8_fallback_reason`
- GPU perf regression wrapper now accepts FP8 request mode:
  - `scripts/perf_regression_gpu.py --dtype fp8`
- Added FP8 telemetry tests:
  - `tests/test_gpu_bench_fp8.py`
  - expanded `tests/test_precision_policy.py`
- Remaining gap:
  - run FP8 benchmark on supported deployment GPU (sm90+) and attach measurable perf evidence.

Files:
- `apex_x/runtime/precision.py`
- `apex_x/model/ff_heavy_path.py`
- `docs/FP8.md`
- `tests/test_precision_policy.py`

Acceptance:
- On supported GPUs, FP8 path runs and demonstrates measurable speed/memory advantage
  within agreed quality envelope.

Validation:
- `python -m pytest -q tests/test_precision_policy.py`
- `python -m apex_x.bench.gpu_bench --output-json artifacts/perf_gpu_fp8.json`

---

## Phase P6 - CI/CD Hard Gates and Release

### P6-01. Make GPU regression workflow mandatory for GPU-critical changes
Status: [~]

Why:
- Optional GPU checks allow silent performance/correctness regressions.

Implementation:
- Enforce required status checks for files touching:
  - `apex_x/kernels/`
  - `apex_x/runtime/`
  - `runtime/tensorrt/`
- Keep controlled self-hosted runner policy.

Progress (2026-02-11):
- `.github/workflows/perf_gpu.yml` now triggers on `pull_request` when GPU-critical paths change.
- Trusted-branch policy is enforced:
  - same-repo PRs run `gpu-perf-regression` on self-hosted GPU runner
  - fork PRs are blocked by `blocked-untrusted-pr` (no untrusted code on self-hosted GPU)
- Contract test coverage now protects workflow policy from regressions:
  - `tests/test_gpu_ci_workflow_contract.py`
  - validates PR path triggers, trust guard, and self-hosted GPU runner labels.
- Remaining gap:
  - repository branch protection must require `GPU Perf Regression / gpu-perf-regression` status check.

Files:
- `.github/workflows/perf_gpu.yml`
- `.github/workflows/ci.yml`
- `docs/CI_GPU.md`

Acceptance:
- PRs touching GPU-critical paths cannot merge without GPU checks.

Validation:
- Workflow dry-run on protected branch policy.

### P6-02. Unified perf regression policy (CPU + GPU + TRT)
Status: [~]

Why:
- Need one rulebook for perf pass/fail.

Implementation:
- Normalize baseline schema and thresholds across CPU/GPU suites.
- Add trend artifact generation for weekly comparison.

Progress (2026-02-11):
- Added normalized trend artifact output to both regression runners:
  - `scripts/perf_regression.py --trend-output ...`
  - `scripts/perf_regression_gpu.py --trend-output ...`
- Added TensorRT shape-sweep regression wrapper aligned with the same compare/trend policy:
  - `scripts/perf_regression_trt.py`
  - baseline spec: `scripts/perf_baseline_trt.json`
  - normalized trend artifact: `artifacts/perf_trt_trend*.json`
- Added committed TensorRT baseline template file for compare-mode wiring:
  - `scripts/perf_baseline_trt.json` (template, metrics populated on deployment TensorRT runner)
- Local TensorRT compare/trend smoke run is now passing with a real CUDA engine artifact:
  - `artifacts/models/trt_bench_dynamic.engine`
  - `artifacts/perf_trt_current.json`
  - `artifacts/perf_trt_compare.json`
  - `artifacts/perf_trt_trend.json`
- CI jobs now publish trend artifacts:
  - CPU: `artifacts/perf_trend_cpu_ci.json` in `.github/workflows/ci.yml`
  - GPU: `artifacts/perf_gpu_trend_ci.json` in `.github/workflows/perf_gpu.yml`
- GPU and weekly workflows now run optional TensorRT compare/trend when `TRT_ENGINE_PATH` is provided:
  - CI: `artifacts/perf_trt_current_ci.json`, `artifacts/perf_trt_compare_ci.json`, `artifacts/perf_trt_trend_ci.json`
  - weekly: `artifacts/perf_trt_current_weekly.json`, `artifacts/perf_trt_compare_weekly.json`, `artifacts/perf_trt_trend_weekly.json`
- Added weekly trend workflow:
  - `.github/workflows/perf_trend_weekly.yml`
  - CPU weekly artifact set is always generated
  - GPU weekly artifact set is generated on self-hosted GPU when enabled (`APEXX_ENABLE_GPU_WEEKLY=true`)
- Remaining gap:
  - validate GPU compare + weekly GPU trend jobs on deployment CUDA runner.
  - tune `scripts/perf_baseline_gpu.json` and `scripts/perf_baseline_trt.json` against deployment hardware;
    local `--dtype fp8` compare run currently fails baseline limits on this `sm75` host.

Files:
- `scripts/perf_regression.py`
- `scripts/perf_regression_gpu.py`
- `scripts/perf_baseline_cpu.json`
- `scripts/perf_baseline_gpu.json`
- `docs/PERF.md`
- `docs/PERF_GPU.md`

Acceptance:
- Same regression formula and reporting format across suites.

Validation:
- `python scripts/perf_regression.py --compare --baseline scripts/perf_baseline_cpu.json`
- `python scripts/perf_regression_gpu.py --compare --baseline scripts/perf_baseline_gpu.json`
- `python scripts/perf_regression_trt.py --compare --baseline scripts/perf_baseline_trt.json --trt-engine-path <engine>`

## 5. Cross-Cutting Work Packages

### X-02. Telemetry schema normalization
Status: [x]

Scope:
- Ensure CLI, Python runtime, and Go service emit aligned telemetry keys:
  - backend
  - precision profile
  - fallback reason
  - selected tile stats
  - latency breakdown

Progress (2026-02-10):
- Python runner telemetry now emits explicit latency breakdown in runtime metadata:
  - `runtime.latency_ms.total`
  - `runtime.latency_ms.backend_execute`
  - `runtime.latency_ms.backend_preflight`
- CLI `predict`/`eval` JSON reports include the same runtime latency fields.
- Test coverage added for runtime latency schema in:
  - `tests/test_infer_runner.py`
  - `tests/test_cli.py`
- Go runtime `/predict` response now emits aligned runtime telemetry keys:
  - `requested_backend`, `selected_backend`, `execution_backend`
  - `fallback_policy`, `precision_profile`
  - `selection_fallback_reason`, `execution_fallback_reason`
  - `latency_ms.total`, `latency_ms.backend_execute`, `latency_ms.backend_preflight`
- Go runtime telemetry schema is covered by service tests:
  - `runtime/go/internal/service/batcher_test.go`
  - `runtime/go/internal/service/http_test.go`
  - `runtime/go/internal/service/integration_test.go`

Acceptance:
- Telemetry fields are documented once and validated in tests.

## 6. Evidence Requirements for "Best Result / Best Math"

To claim top-tier quality credibly, the repository must provide:

1. Reproducible benchmark protocol:
   - fixed datasets/splits
   - fixed preprocessing
   - fixed seed policy
2. Baseline comparison set:
   - strong public baselines under equal constraints
3. Full reporting:
   - quality metrics
   - latency/throughput
   - memory and cost
4. Ablation report:
   - router/oracle
   - budgeting
   - quadtree depth
   - precision modes
5. Public artifact package:
   - configs
   - logs
   - checkpoints/engines
   - scripts for rerun

Without this evidence package, claims should be phrased as "improved within tested setup", not global best.

---

## 7. Exit Criteria by Phase

Phase exit is allowed only when all phase tasks are done and validated.

- P0 exit:
  - contracts/tolerances/golden replay are locked
- P1 exit:
  - CLI/eval/export real backend path complete
- P2 exit:
  - Triton feature parity + perf gates pass
- P3 exit:
  - TensorRT plugin/build/parity/int8 gates pass
- P4 exit:
  - math quality track shows reproducible improvements under budget
- P5 exit:
  - Go service uses real inference with SLA + telemetry
- P6 exit:
  - CI hard gates and release checklist operational

Project can be called "full production-ready GPU support" only after all exits pass.
