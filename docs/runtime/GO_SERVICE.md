# Go Runtime Service (`runtime/go`)

## Overview
`runtime/go` provides an HTTP microservice wrapper for Apex-X runtime adapters with:
- `/health`, `/predict`, `/metrics`
- dynamic batching (`max_batch_size`, `batch_window`)
- per-request budget profile (`quality`, `balanced`, `edge`)
- adapter loading paths:
  - ONNX Runtime model loader path with optional Python bridge execution
  - TensorRT engine loader path via CGO wrapper (when built with `-tags tensorrt`)
  - optional TensorRT Python bridge execution path (`APEXX_TRT_BRIDGE_CMD`)

## Endpoints

### `GET /health`
Returns service health and adapter identity.

Example response:
```json
{
  "status": "ok",
  "adapter": "onnxruntime-cpu-baseline"
}
```

### `POST /predict`
Request:
```json
{
  "request_id": "demo-1",
  "budget_profile": "balanced",
  "input": [0.1, 0.2, 0.3]
}
```

Response:
```json
{
  "request_id": "demo-1",
  "budget_profile": "balanced",
  "selected_tiles": 32,
  "scores": [0.2],
  "backend": "onnxruntime-cpu-baseline",
  "runtime": {
    "requested_backend": "onnxruntime-cpu-baseline",
    "selected_backend": "onnxruntime-cpu-baseline",
    "execution_backend": "onnxruntime-cpu-baseline",
    "fallback_policy": "strict",
    "precision_profile": "balanced",
    "selection_fallback_reason": null,
    "execution_fallback_reason": null,
    "latency_ms": {
      "total": 1.27,
      "backend_execute": 0.74,
      "backend_preflight": 0.53
    }
  }
}
```

Runtime payload follows the same key schema used in Python CLI reports:
- `requested_backend`, `selected_backend`, `execution_backend`
- `fallback_policy`, `precision_profile`
- `selection_fallback_reason`, `execution_fallback_reason`
- `latency_ms.total`, `latency_ms.backend_execute`, `latency_ms.backend_preflight`

Optional request metadata passthrough keys:
- `requested_backend`
- `fallback_policy` (`strict` or `permissive`)
- `selection_fallback_reason`
- `execution_fallback_reason`

Service error policy:
- `429 Too Many Requests` when request queue is saturated
- `504 Gateway Timeout` when per-request predict timeout is exceeded
- `503 Service Unavailable` when backend runtime is unavailable (for example missing bridge executable)
- `502 Bad Gateway` when backend bridge/runtime returns execution or protocol failures

Canary parity mode:
- Optional shadow backend compare in background (`canary_adapter`)
- Primary response path stays on primary backend; canary result is telemetry-only
- Sampling policy via `canary_sample_rate`
- Optional payload capture policy with JSONL sink:
  - `off` (default)
  - `mismatch` (capture only mismatch events)
  - `error` (capture canary adapter/runtime errors)
  - `all` (capture match + mismatch + error events)
- Prometheus metrics:
  - `apexx_canary_samples_total`
  - `apexx_canary_compares_total`
  - `apexx_canary_mismatches_total`
  - `apexx_canary_errors_total`
  - `apexx_canary_mismatch_ratio`

### `GET /metrics`
Prometheus-style text output includes:
- request counters and latency
- batch counters and batch-size stats
- queue wait and inference timing
- batch error counters

## Dynamic Batching
Batcher controls:
- `max_batch_size`
- `batch_window_ms`
- `queue_size`

Each request keeps its own budget profile and is forwarded as part of a batch to the adapter.

Recorded metrics:
- batch sizes (`avg`, `max`)
- queue time (`avg`, `max`)
- inference time (`avg`, `max`)
- errors (`requests_failed`, `batch_errors`)

## Canary SLA Gate (CI)
Go runtime includes a load-gate integration test for SLA enforcement under canary mode:

```bash
cd runtime/go
go test ./internal/service -run TestCanaryLoadGateThresholds -count=1 -v
```

Gate thresholds are configurable via environment variables:
- `APEXX_GO_CANARY_GATE_REQUESTS`
- `APEXX_GO_CANARY_GATE_CONCURRENCY`
- `APEXX_GO_CANARY_GATE_PRIMARY_DELAY_MS`
- `APEXX_GO_CANARY_GATE_CANARY_DELAY_MS`
- `APEXX_GO_CANARY_GATE_MAX_OVERHEAD_RATIO`
- `APEXX_GO_CANARY_GATE_MAX_OVERHEAD_ABS_MS`
- `APEXX_GO_CANARY_GATE_MAX_TIMEOUT_RATE`
- `APEXX_GO_CANARY_GATE_MAX_QUEUE_OVERFLOW_RATE`

## Run: CPU Mode (ONNX Baseline Loader)
From repository root:
```bash
cd runtime/go
go test ./...
mkdir -p models
# Put a non-empty ONNX file at models/apex-x.onnx
go run ./cmd/apexx-runtime \
  -addr :8080 \
  -adapter onnxruntime \
  -model-path models/apex-x.onnx \
  -predict-timeout-ms 100 \
  -canary-adapter onnxruntime \
  -canary-model-path models/apex-x.onnx \
  -canary-sample-rate 0.1 \
  -canary-score-abs-tol 0.001 \
  -canary-timeout-ms 150 \
  -canary-capture-policy mismatch \
  -canary-capture-path artifacts/canary_capture.jsonl \
  -canary-capture-max-bytes 10485760 \
  -batch-window-ms 8 \
  -max-batch-size 8 \
  -queue-size 256 \
  -default-budget-profile balanced \
  -log-format json \
  -log-level info
```

For real ONNXRuntime execution on hosts without native Go ORT bindings, configure Python bridge:
```bash
export APEXX_ORT_BRIDGE_CMD="python -m apex_x.runtime.service_bridge"
```

## Run: TensorRT Mode (CGO Loader Path)
TensorRT mode requires build tags and CGO:
```bash
cd runtime/go
CGO_ENABLED=1 go run -tags tensorrt ./cmd/apexx-runtime \
  -addr :8080 \
  -adapter tensorrt \
  -engine-path models/apex-x.plan \
  -predict-timeout-ms 100 \
  -canary-adapter onnxruntime \
  -canary-model-path models/apex-x.onnx \
  -canary-sample-rate 0.1 \
  -canary-capture-policy mismatch \
  -canary-capture-path artifacts/canary_capture.jsonl \
  -batch-window-ms 8 \
  -max-batch-size 8 \
  -queue-size 256 \
  -default-budget-profile balanced
```

Current TensorRT status:
- engine file loader/validation via CGO: implemented
- optional Python bridge inference execution path: implemented (`APEXX_TRT_BRIDGE_CMD`)
- native TensorRT CGO execution path: scaffold/pending

ORT/TRT adapters are fail-closed: if bridge/native backend execution is unavailable,
`/predict` returns backend error status (`503`/`502`) instead of synthetic scores.

## Environment Variables
Path overrides:
- `APEXX_ORT_MODEL_PATH`: fallback ONNX model path if `-model-path` is empty
- `APEXX_TRT_ENGINE_PATH`: fallback TensorRT engine path if `-engine-path` is empty
- `APEXX_ORT_BRIDGE_CMD`: command for ORT bridge execution (example: `python -m apex_x.runtime.service_bridge`)
- `APEXX_TRT_BRIDGE_CMD`: command for TensorRT bridge execution
- `APEXX_PREDICT_TIMEOUT_MS`: fallback for `-predict-timeout-ms`
- `APEXX_CANARY_ADAPTER`: fallback for `-canary-adapter`
- `APEXX_CANARY_MODEL_PATH`: fallback for `-canary-model-path`
- `APEXX_CANARY_ENGINE_PATH`: fallback for `-canary-engine-path`
- `APEXX_CANARY_SAMPLE_RATE`: fallback for `-canary-sample-rate`
- `APEXX_CANARY_SCORE_ABS_TOL`: fallback for `-canary-score-abs-tol`
- `APEXX_CANARY_TIMEOUT_MS`: fallback for `-canary-timeout-ms`
- `APEXX_CANARY_CAPTURE_POLICY`: fallback for `-canary-capture-policy`
- `APEXX_CANARY_CAPTURE_PATH`: fallback for `-canary-capture-path`
- `APEXX_CANARY_CAPTURE_MAX_BYTES`: fallback for `-canary-capture-max-bytes`

Logging:
- `APEXX_LOG_FORMAT`: `json` | `text` | `discard`
- `APEXX_LOG_LEVEL`: `debug` | `info` | `warn` | `error`

Telemetry hook toggle:
- `APEXX_ENABLE_OTEL_HOOKS`: `true/false`
  - enables service telemetry hook extension point (no-op hook implementation by default)

## Example `curl`
Health:
```bash
curl -s http://localhost:8080/health
```

Predict:
```bash
curl -s http://localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"request_id":"demo-1","budget_profile":"quality","input":[0.1,0.2,0.3]}'
```

Metrics:
```bash
curl -s http://localhost:8080/metrics
```
