from __future__ import annotations

from typing import Any

from .interfaces import Exporter


class NoopExporter(Exporter):
    """Reference exporter that records a placeholder artifact."""

    def export(self, model: Any, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Apex-X export placeholder\n")
            f.write(f"model={model.__class__.__name__}\n")
        return output_path
