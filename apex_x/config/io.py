from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from apex_x.utils.logging import get_logger, log_event

from .schema import ApexXConfig

LOGGER = get_logger(__name__)


def load_yaml_config(path: str | Path, overrides: Sequence[str] | None = None) -> ApexXConfig:
    """Load config from YAML and apply CLI-style dot-path overrides."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError("Top-level YAML config must be a mapping")

    cfg = ApexXConfig.from_dict(payload)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    log_event(
        LOGGER,
        "config_loaded",
        level="DEBUG",
        fields={"path": str(config_path), "override_count": len(overrides or ())},
    )
    return cfg


def apply_overrides(cfg: ApexXConfig, overrides: Sequence[str]) -> ApexXConfig:
    """Apply dot-path overrides, e.g. model.profile=base routing.budget_b1=20."""
    data = deepcopy(cfg.to_dict())
    for override in overrides:
        path, value = _parse_override(override)
        _set_nested_value(data, path.split("."), value)

    return ApexXConfig.from_dict(data)


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must contain '=': {raw}")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key is empty: {raw}")

    value = yaml.safe_load(raw_value)
    return key, value


def _set_nested_value(data: dict[str, Any], path: list[str], value: Any) -> None:
    cursor: dict[str, Any] = data
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            full_key = ".".join(path)
            raise KeyError(f"Unknown override path: {full_key}")
        cursor = child

    leaf = path[-1]
    if leaf not in cursor:
        full_key = ".".join(path)
        raise KeyError(f"Unknown override path: {full_key}")
    cursor[leaf] = value
