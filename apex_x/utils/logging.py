from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from rich.logging import RichHandler

_BASE_LOGGER_NAME = "apex_x"
_CONFIGURED = False


def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if normalized not in logging._nameToLevel:  # noqa: SLF001
        raise ValueError(f"Unknown log level: {level}")
    return int(logging._nameToLevel[normalized])  # noqa: SLF001


def configure_logging(level: int | str = "INFO", force: bool = False) -> logging.Logger:
    """Configure the shared apex_x logger with rich output."""
    global _CONFIGURED

    logger = logging.getLogger(_BASE_LOGGER_NAME)
    if _CONFIGURED and not force:
        logger.setLevel(_normalize_level(level))
        return logger

    logger.handlers.clear()
    handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a consistent logger namespace for all Apex-X modules."""
    configure_logging()
    if name is None or name == _BASE_LOGGER_NAME:
        return logging.getLogger(_BASE_LOGGER_NAME)
    if name.startswith(f"{_BASE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int | str = "INFO",
    fields: Mapping[str, Any] | None = None,
) -> None:
    """Emit structured key/value event logs through a shared rich logger."""
    payload: dict[str, Any] = {"event": event}
    if fields:
        payload.update(fields)

    message = " ".join(f"{key}={payload[key]!r}" for key in sorted(payload))
    logger.log(_normalize_level(level), message)
