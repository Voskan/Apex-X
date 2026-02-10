from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any

import numpy as np

try:
    torch: Any = import_module("torch")
except Exception:  # pragma: no cover - optional dependency in minimal envs
    torch = None


_DETERMINISTIC_ENABLED = False


def seed_all(seed: int, deterministic: bool | None = None) -> dict[str, str | bool | None]:
    """Seed Python, NumPy, and torch RNGs for reproducible runs.

    Notes:
    - On CPU, this is typically sufficient for deterministic behavior.
    - On CUDA, deterministic algorithms may reduce performance and some ops may still
      be non-deterministic depending on backend kernels.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")

    random.seed(seed)
    np.random.seed(seed)

    # Runtime-only note: setting PYTHONHASHSEED here does not retroactively
    # change hash randomization for the current process start.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic is not None:
        set_deterministic_mode(deterministic)

    state = get_determinism_state()
    state["seed"] = str(seed)
    return state


def set_deterministic_mode(enabled: bool, warn_only: bool = False) -> dict[str, str | bool | None]:
    """Toggle deterministic execution mode.

    For CUDA determinism, this also toggles CUBLAS workspace config and cudnn
    benchmark/deterministic knobs (when torch is available).
    """
    global _DETERMINISTIC_ENABLED
    _DETERMINISTIC_ENABLED = enabled

    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    if torch is not None:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = enabled
            torch.backends.cudnn.benchmark = not enabled

    return get_determinism_state()


def get_determinism_state() -> dict[str, str | bool | None]:
    """Return current determinism-related runtime state."""
    state: dict[str, str | bool | None] = {
        "deterministic_enabled": _DETERMINISTIC_ENABLED,
        "has_torch": torch is not None,
        "cuda_available": None,
        "torch_deterministic_algorithms": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    if torch is not None:
        state["cuda_available"] = torch.cuda.is_available()
        state["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        if torch.backends.cudnn.is_available():
            state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)

    return state


def reproducibility_notes() -> str:
    """Human-readable reproducibility notes for CPU vs CUDA."""
    return (
        "CPU: seed_all() is usually enough for deterministic behavior.\\n"
        "CUDA: enable deterministic mode, disable cudnn benchmark, and set CUBLAS workspace.\\n"
        "CUDA determinism can reduce speed and some kernels may still be non-deterministic."
    )


@contextmanager
def deterministic_mode(
    enabled: bool = True,
    warn_only: bool = False,
) -> Iterator[dict[str, str | bool | None]]:
    """Context manager that toggles deterministic mode and restores prior state."""
    previous = get_determinism_state()
    set_deterministic_mode(enabled=enabled, warn_only=warn_only)
    try:
        yield get_determinism_state()
    finally:
        prev_enabled = bool(previous.get("deterministic_enabled"))
        set_deterministic_mode(enabled=prev_enabled, warn_only=warn_only)
