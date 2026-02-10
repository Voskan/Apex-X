# Go Runtime Service (`runtime/go`)

## Overview
`runtime/go` provides an HTTP microservice wrapper for Apex-X runtime adapters with:
- `/health`, `/predict`, `/metrics`
- dynamic batching (`max_batch_size`, `batch_window`)
- per-request budget profile (`quality`, `balanced`, `edge`)
- adapter loading paths:
  - ONNX Runtime CPU baseline loader path
  - TensorRT engine loader path via CGO wrapper (when built with `-tags tensorrt`)

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
  "backend": "onnxruntime-cpu-baseline"
}
```

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
  -batch-window-ms 8 \
  -max-batch-size 8 \
  -queue-size 256 \
  -default-budget-profile balanced \
  -log-format json \
  -log-level info
```

## Run: TensorRT Mode (CGO Loader Path)
TensorRT mode requires build tags and CGO:
```bash
cd runtime/go
CGO_ENABLED=1 go run -tags tensorrt ./cmd/apexx-runtime \
  -addr :8080 \
  -adapter tensorrt \
  -engine-path models/apex-x.plan \
  -batch-window-ms 8 \
  -max-batch-size 8 \
  -queue-size 256 \
  -default-budget-profile balanced
```

Current TensorRT status:
- engine file loader/validation via CGO: implemented
- full TensorRT inference execution path: scaffold/pending

## Environment Variables
Path overrides:
- `APEXX_ORT_MODEL_PATH`: fallback ONNX model path if `-model-path` is empty
- `APEXX_TRT_ENGINE_PATH`: fallback TensorRT engine path if `-engine-path` is empty

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
