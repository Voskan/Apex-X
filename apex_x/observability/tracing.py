from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.propagate import extract
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    trace = None  # type: ignore[assignment]
    extract = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    ConsoleSpanExporter = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

# Configure OTEL
_CONFIGURED = False


class _NoopSpan:
    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(
        self,
        _name: str,
        context: Any = None,
        attributes: dict[str, str | int | float] | None = None,
    ) -> Generator[_NoopSpan, None, None]:
        _ = context
        _ = attributes
        yield _NoopSpan()


def configure_tracing(service_name: str = "apex_x_python") -> None:
    """Configure OpenTelemetry tracing."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if not _OTEL_AVAILABLE:
        _CONFIGURED = True
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    # In production, we'd use OTLPSpanExporter, but for now Console or NoOp
    if os.environ.get("APEX_X_TRACE_EXPORT", "false").lower() == "true":
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer(name: str = "apex_x") -> Any:
    """Get a tracer instance."""
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    if not _CONFIGURED:
        configure_tracing()
    return trace.get_tracer(name)


@contextmanager
def start_span(
    name: str,
    context_carrier: dict[str, str] | None = None,
    attributes: dict[str, str | int | float] | None = None,
) -> Generator[Any, None, None]:
    """Start a new span, optionally extracting parent context."""
    tracer = get_tracer()

    if _OTEL_AVAILABLE and extract is not None and context_carrier:
        ctx = extract(context_carrier)
    else:
        ctx = None

    with tracer.start_as_current_span(name, context=ctx, attributes=attributes) as span:
        yield span
