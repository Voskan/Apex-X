from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

import structlog

_BASE_LOGGER_NAME = "apex_x"
_CONFIGURED = False


def _configure_structlog(level_name: str, force_json: bool = False) -> None:
    """Configure structlog processors and formatter."""
    
    # Determine if we should output JSON or human-readable
    # Default to JSON for production safety, unless in TTY or explicitly requested human
    if force_json or os.environ.get("APEX_X_LOG_FORMAT", "").lower() == "json":
        renderer = structlog.processors.JSONRenderer()
    elif sys.stderr.isatty() and os.environ.get("APEX_X_LOG_FORMAT", "").lower() != "json":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        renderer,
    ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Standard library logging interop
    # Remove existing handlers to avoid duplication
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
            
    handler = logging.StreamHandler(sys.stderr)
    # Use structlog to format the standard library log records
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level_name.upper())


def configure_logging(level: int | str = "INFO", force: bool = False) -> Any:
    """Configure the shared apex_x logger with structlog."""
    global _CONFIGURED

    if isinstance(level, int):
        level_name = logging.getLevelName(level)
    else:
        level_name = level

    if _CONFIGURED and not force:
        return structlog.get_logger(_BASE_LOGGER_NAME)

    _configure_structlog(str(level_name))
    _CONFIGURED = True
    return structlog.get_logger(_BASE_LOGGER_NAME)


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger."""
    configure_logging()
    if name is None:
        return structlog.get_logger(_BASE_LOGGER_NAME)
    # Structlog doesn't use dot notation for hierarchy in the same way as logging
    # but we can bind the name.
    return structlog.get_logger(name)


def log_event(
    logger: Any,
    event: str,
    *,
    level: int | str = "INFO",
    fields: Mapping[str, Any] | None = None,
) -> None:
    """Emit structured key/value event logs."""
    # Maps 'log_event' to structlog calls
    # logger should be a structlog BoundLogger
    
    payload = fields or {}
    
    if isinstance(level, int):
        level_name = logging.getLevelName(level).lower()
    else:
        level_name = str(level).lower()
        
    log_method = getattr(logger, level_name, logger.info)
    log_method(event, **payload)
