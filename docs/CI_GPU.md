# GPU CI Perf Regression

## Goal
Run GPU performance regression checks for Apex-X while keeping default CI safe and cheap on public runners.

This workflow is:
- `.github/workflows/perf_gpu.yml`

CPU perf regression remains in:
- `.github/workflows/ci.yml` (`perf-regression` job)

## Trigger Model
- Pull request (`pull_request`, mandatory for GPU-critical paths):
  - triggers when a PR changes:
    - `apex_x/kernels/**`
    - `apex_x/runtime/**`
    - `runtime/tensorrt/**`
  - PRs from forks are blocked by policy in this workflow (no untrusted code on self-hosted GPU)
  - PRs from trusted branches in this repository execute GPU regression on self-hosted runner
- Manual (`workflow_dispatch`):
  - default is `run_mode=skip` (safe default)
  - set `run_mode=self-hosted-gpu` to execute benchmark on self-hosted GPU
- Scheduled (`schedule`):
  - runs only when repository variable `APEXX_ENABLE_GPU_NIGHTLY` is set to `true`
  - intended for trusted environments with self-hosted GPU runner capacity

## Runner Requirements (Self-Hosted)
Recommended labels:
- `self-hosted`
- `linux`
- `x64`
- `gpu`

Minimum environment:
- NVIDIA driver + CUDA runtime compatible with installed PyTorch
- Python 3.11+ toolchain
- enough free GPU memory for fixed benchmark profile

## Baseline and Regression Policy
Stored baseline:
- `scripts/perf_baseline_gpu.json`

Regression runner:
- `scripts/perf_regression_gpu.py`
- `scripts/perf_regression_trt.py` (optional TRT shape-sweep wrapper when engine path is provided)

Compare command used in workflow:
```bash
python scripts/perf_regression_gpu.py \
  --compare \
  --baseline scripts/perf_baseline_gpu.json \
  --output artifacts/perf_gpu_current_ci.json \
  --summary artifacts/perf_gpu_compare_ci.json \
  --trend-output artifacts/perf_gpu_trend_ci.json
```

Fail behavior:
- workflow fails when any tracked metric exceeds:
  - `allowed_max = baseline_value * (1 + max_regression_ratio) + max_regression_abs_ms`
- workflow also fails when benchmark status is not `ok` (for example CUDA unavailable)

Trend artifact:
- `artifacts/perf_gpu_trend_ci.json` contains normalized metric deltas for weekly/per-run tracking.
- release evidence draft is auto-generated:
  - `artifacts/release/release_attestation_gpu_ci.json`
  - `artifacts/release/release_attestation_gpu_ci.md`

## Baseline Maintenance
Regenerate template on the target GPU runner:
```bash
python scripts/perf_regression_gpu.py \
  --emit-baseline-template \
  --baseline scripts/perf_baseline_gpu.json
```

Then tighten tolerances per runner stability:
- microbench metrics: lower absolute thresholds
- end-to-end metrics: slightly wider p95 threshold

## Optional TensorRT Inputs
Workflow dispatch supports:
- `trt_engine_path`: optional `.engine` for TRT end-to-end bench
- `trt_plugin_lib`: optional plugin shared library path

If omitted, TRT sections are skipped and reported as such.

Optional TRT wrapper command (same compare formula + normalized trend artifact):
```bash
python scripts/perf_regression_trt.py \
  --compare \
  --baseline scripts/perf_baseline_trt.json \
  --output artifacts/perf_trt_current_ci.json \
  --summary artifacts/perf_trt_compare_ci.json \
  --trend-output artifacts/perf_trt_trend_ci.json \
  --trt-engine-path /abs/path/to/apex_x.engine
```

## Required Check Setup
To enforce merge blocking for GPU-critical changes, set branch protection to require:
- `GPU Perf Regression / gpu-perf-regression`

With this policy, GPU-critical PRs cannot merge without passing GPU regression.

## Security Notes
Self-hosted GPU runners execute repository code and are high-trust resources.

Recommendations:
- Keep fork PRs blocked from self-hosted GPU execution (workflow has explicit guard job).
- Do not use `pull_request_target` for GPU perf execution.
- Use dedicated, isolated self-hosted GPU runners for CI only.
- Avoid exposing production secrets to GPU perf workflow.
- Keep repository permissions minimal (`contents: read` is used by default).
- Pin and review third-party actions before changing workflow dependencies.
