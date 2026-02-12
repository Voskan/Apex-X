from __future__ import annotations

from typing import Any, cast
from warnings import filterwarnings

# Suppress prometheus_client warnings about existing metrics registration
filterwarnings("ignore", module="prometheus_client")

try:
    from prometheus_client import (
        REGISTRY as PROM_REGISTRY,
        Counter as PROM_COUNTER,
        Gauge as PROM_GAUGE,
        Histogram as PROM_HISTOGRAM,
        start_http_server as PROM_START_HTTP_SERVER,
    )
    Counter = PROM_COUNTER
    Gauge = PROM_GAUGE
    Histogram = PROM_HISTOGRAM
    REGISTRY = PROM_REGISTRY
    start_http_server = PROM_START_HTTP_SERVER
except ImportError:
    # Dummy implementation for environments without prometheus_client
    class DummyMetric:
        def inc(self, amount: float = 1.0) -> None:
            _ = amount

        def set(self, value: float) -> None:
            _ = value

        def observe(self, value: float) -> None:
            _ = value

        def labels(self, **kwargs: object) -> "DummyMetric":
            _ = kwargs
            return self

    def _dummy_metric_factory(*_args: object, **_kwargs: object) -> DummyMetric:
        return DummyMetric()

    def _dummy_start_http_server(_port: int) -> None:
        return None

    Counter = cast(Any, _dummy_metric_factory)
    Gauge = cast(Any, _dummy_metric_factory)
    Histogram = cast(Any, _dummy_metric_factory)
    REGISTRY = cast(Any, None)
    start_http_server = cast(Any, _dummy_start_http_server)


# Application Constants
NAMESPACE = "apex_x"

# --- Metrics Definitions ---

# Latency
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency",
    ["model_version", "precision"],
    namespace=NAMESPACE,
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

ROUTER_LATENCY_SECONDS = Histogram(
    "router_latency_seconds",
    "Time spent in the routing decision logic",
    ["strategy"],
    namespace=NAMESPACE,
)

# Throughput / Reliability
INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["status"], # success, error, timeout
    namespace=NAMESPACE,
)

# Resource Usage
GPU_MEMORY_USED_BYTES = Gauge(
    "gpu_memory_used_bytes",
    "Current GPU memory usage",
    ["device"],
    namespace=NAMESPACE,
)

# Business Logic / Model Behavior
ACTIVE_TILES_COUNT = Histogram(
    "active_tiles_count",
    "Number of high-res tiles selected by the router",
    ["strategy"],
    namespace=NAMESPACE,
    buckets=[0, 1, 5, 10, 20, 50, 100], 
)

BUDGET_USAGE_RATIO = Histogram(
    "budget_usage_ratio",
    "Ratio of actual cost vs budget",
    ["budget_type"], # latency, compute
    namespace=NAMESPACE,
)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus HTTP server."""
    try:
        start_http_server(port)
    except Exception as e:
        print(f"Failed to start metrics server on port {port}: {e}")
