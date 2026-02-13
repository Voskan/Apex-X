"""Dependency preflight helpers for worldclass (TeacherModelV3) paths."""

from __future__ import annotations

import importlib.util

WORLDCLASS_DEPENDENCIES: tuple[str, ...] = (
    "transformers",
    "timm",
    "peft",
    "safetensors",
)


def missing_worldclass_dependencies(
    *,
    required: tuple[str, ...] = WORLDCLASS_DEPENDENCIES,
) -> list[str]:
    """Return missing optional dependency names for worldclass model paths."""
    missing: list[str] = []
    for package in required:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def worldclass_install_hint() -> str:
    """Return installation hint for missing worldclass dependencies."""
    return (
        "Install worldclass dependencies with either "
        "\"pip install 'apex-x[worldclass]'\" "
        "or "
        "\"pip install transformers timm peft safetensors\"."
    )


def ensure_worldclass_dependencies(
    *,
    context: str,
    required: tuple[str, ...] = WORLDCLASS_DEPENDENCIES,
) -> None:
    """Raise ImportError with actionable hint when deps are missing."""
    missing = missing_worldclass_dependencies(required=required)
    if not missing:
        return
    missing_text = ", ".join(missing)
    raise ImportError(
        f"{context} requires optional dependencies: {missing_text}. "
        f"{worldclass_install_hint()}"
    )


__all__ = [
    "WORLDCLASS_DEPENDENCIES",
    "missing_worldclass_dependencies",
    "worldclass_install_hint",
    "ensure_worldclass_dependencies",
]
