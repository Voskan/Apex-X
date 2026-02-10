# Apex-X Go Runtime Service

This service is a lightweight runtime wrapper for Apex-X inference backends.

## Features
- HTTP endpoints:
  - `POST /predict`
  - `GET /health`
  - `GET /metrics` (Prometheus-style text)
- short-window batching queue for `/predict`
- per-request budget profile (`quality`, `balanced`, `edge`)
- adapter layer:
  - ONNX Runtime CPU baseline adapter scaffold
  - TensorRT adapter scaffold via CGO build tags

## Run Locally
```bash
cd runtime/go
go test ./...
mkdir -p models
# Place a real ONNX model at models/apex-x.onnx (non-empty file).
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

## API

### `GET /health`
Response:
```json
{"status":"ok","adapter":"onnxruntime-cpu-stub"}
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
  "backend": "onnxruntime-cpu-stub"
}
```

### `GET /metrics`
Example:
```text
apexx_requests_total 12
apexx_requests_failed_total 0
apexx_batches_total 4
apexx_inflight 0
apexx_request_latency_ms_avg 1.243210
```

## TensorRT Adapter Scaffold (CGO)
The TensorRT adapter now includes CGO engine loading validation.

- Default builds (no tags) return a clear "unavailable" error.
- Enable scaffold compilation with:
```bash
go test -tags tensorrt ./...
```
This requires `CGO_ENABLED=1`.

Current status:
- interface + build-tag wiring in place
- engine-file loader via CGO is implemented
- concrete TensorRT inference execution path remains pending

See full service runtime docs:
- `docs/runtime/GO_SERVICE.md`

## Docker
```bash
cd runtime/go
docker build -t apexx-runtime-go .
docker run --rm -p 8080:8080 apexx-runtime-go
```

Or with compose:
```bash
cd runtime/go
docker compose up --build
```
