# GPU CI Perf Regression

## Goal
Run GPU performance regression checks for Apex-X while keeping default CI safe and cheap on public runners.

This workflow is:
- `.github/workflows/perf_gpu.yml`

CPU perf regression remains in:
- `.github/workflows/ci.yml` (`perf-regression` job)

## Trigger Model
- Manual (`workflow_dispatch`):
  - default is `run_mode=skip` (safe default)
  - set `run_mode=self-hosted-gpu` to execute benchmark on self-hosted GPU
- Nightly (`schedule`):
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

Compare command used in workflow:
```bash
python scripts/perf_regression_gpu.py \
  --compare \
  --baseline scripts/perf_baseline_gpu.json \
  --output artifacts/perf_gpu_current_ci.json \
  --summary artifacts/perf_gpu_compare_ci.json
```

Fail behavior:
- workflow fails when any tracked metric exceeds:
  - `allowed_max = baseline_value * (1 + max_regression_ratio) + max_regression_abs_ms`
- workflow also fails when benchmark status is not `ok` (for example CUDA unavailable)

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

## Security Notes
Self-hosted GPU runners execute repository code and are high-trust resources.

Recommendations:
- Do not run this workflow for untrusted fork PRs.
- Keep this workflow off `pull_request`/`pull_request_target` triggers.
- Use dedicated, isolated self-hosted GPU runners for CI only.
- Avoid exposing production secrets to GPU perf workflow.
- Keep repository permissions minimal (`contents: read` is used by default).
- Pin and review third-party actions before changing workflow dependencies.
