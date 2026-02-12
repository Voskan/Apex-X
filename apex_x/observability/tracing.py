from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Configure OTEL
_CONFIGURED = False

def configure_tracing(service_name: str = "apex_x_python") -> None:
    """Configure OpenTelemetry tracing."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    # In production, we'd use OTLPSpanExporter, but for now Console or NoOp
    if os.environ.get("APEX_X_TRACE_EXPORT", "false").lower() == "true":
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer(name: str = "apex_x") -> trace.Tracer:
    """Get a tracer instance."""
    if not _CONFIGURED:
        configure_tracing()
    return trace.get_tracer(name)


@contextmanager
def start_span(
    name: str,
    context_carrier: dict[str, str] | None = None,
    attributes: dict[str, str | int | float] | None = None,
) -> Generator[trace.Span, None, None]:
    """Start a new span, optionally extracting parent context."""
    tracer = get_tracer()
    
    ctx = extract(context_carrier) if context_carrier else None
    
    with tracer.start_as_current_span(name, context=ctx, attributes=attributes) as span:
        yield span
