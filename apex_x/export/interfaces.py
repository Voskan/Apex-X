from __future__ import annotations

from typing import Any, Protocol


class Exporter(Protocol):
    """Public export interface for backend/runtime serialization."""

    def export(self, model: Any, output_path: str) -> str:
        """Export model artifacts and return destination path."""
        ...
