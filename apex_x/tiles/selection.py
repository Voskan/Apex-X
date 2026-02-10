from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _normalize_level(level: str) -> str:
    normalized = level.strip().lower()
    if normalized not in {"l0", "l1", "l2"}:
        raise ValueError("level must be one of: l0, l1, l2")
    return normalized


def _normalize_indices(values: list[int], field_name: str) -> list[int]:
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError(f"{field_name} must contain non-negative indices")
        out.append(ivalue)
    return out


def _normalize_budgets_used(budgets_used: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in budgets_used.items():
        v = float(value)
        if not math.isfinite(v) or v < 0.0:
            raise ValueError("budgets_used values must be finite and >= 0")
        out[str(name)] = v
    return out


def _to_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} values must be integer-like")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("-"):
            sign = -1
            stripped = stripped[1:]
        else:
            sign = 1
        if stripped.isdigit():
            return sign * int(stripped)
    raise ValueError(f"{field_name} values must be integer-like")


def _to_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} values must be numeric")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{field_name} values must be numeric") from exc
    raise ValueError(f"{field_name} values must be numeric")


def _to_json_value(value: object) -> JsonValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _to_json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            out[str(key)] = _to_json_value(item)
        return out
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


@dataclass
class TileSelection:
    """Selection state for one quadtree level."""

    level: str
    indices: list[int]
    ordered_indices: list[int]
    meta: dict[str, JsonValue] = field(default_factory=dict)
    budgets_used: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.level = _normalize_level(self.level)
        self.indices = _normalize_indices(self.indices, "indices")
        self.ordered_indices = _normalize_indices(self.ordered_indices, "ordered_indices")
        if Counter(self.ordered_indices) != Counter(self.indices):
            raise ValueError("ordered_indices must be a permutation of indices")
        meta_json = _to_json_value(self.meta)
        if not isinstance(meta_json, dict):
            raise ValueError("meta must serialize to a JSON object")
        self.meta = meta_json
        self.budgets_used = _normalize_budgets_used(self.budgets_used)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "level": self.level,
            "indices": _to_json_value(self.indices),
            "ordered_indices": _to_json_value(self.ordered_indices),
            "meta": self.meta,
            "budgets_used": _to_json_value(self.budgets_used),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> TileSelection:
        level_value = payload.get("level")
        if not isinstance(level_value, str):
            raise ValueError("level must be a string")

        raw_indices = payload.get("indices")
        if not isinstance(raw_indices, list):
            raise ValueError("indices must be a list")
        indices = [_to_int(v, "indices") for v in raw_indices]

        raw_ordered = payload.get("ordered_indices")
        if not isinstance(raw_ordered, list):
            raise ValueError("ordered_indices must be a list")
        ordered_indices = [_to_int(v, "ordered_indices") for v in raw_ordered]

        raw_meta = payload.get("meta", {})
        if not isinstance(raw_meta, dict):
            raise ValueError("meta must be an object")
        meta = {str(k): _to_json_value(v) for k, v in raw_meta.items()}

        raw_budgets = payload.get("budgets_used", {})
        if not isinstance(raw_budgets, dict):
            raise ValueError("budgets_used must be an object")
        budgets_used = {str(k): _to_float(v, "budgets_used") for k, v in raw_budgets.items()}

        return cls(
            level=level_value,
            indices=indices,
            ordered_indices=ordered_indices,
            meta=meta,
            budgets_used=budgets_used,
        )

    def save_json(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> TileSelection:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("tile selection JSON must be an object")
        return cls.from_dict(payload)


@dataclass
class TileSelectionTrace:
    """Multi-level selection trace for debugging and ablations."""

    selections: list[TileSelection]
    run_meta: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selections:
            raise ValueError("selections must not be empty")
        run_meta_json = _to_json_value(self.run_meta)
        if not isinstance(run_meta_json, dict):
            raise ValueError("run_meta must serialize to a JSON object")
        self.run_meta = run_meta_json

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "selections": [selection.to_dict() for selection in self.selections],
            "run_meta": self.run_meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> TileSelectionTrace:
        raw_selections = payload.get("selections")
        if not isinstance(raw_selections, list):
            raise ValueError("selections must be a list")
        selections: list[TileSelection] = []
        for item in raw_selections:
            if not isinstance(item, dict):
                raise ValueError("each selection item must be an object")
            item_payload = {str(k): _to_json_value(v) for k, v in item.items()}
            selections.append(TileSelection.from_dict(item_payload))

        raw_run_meta = payload.get("run_meta", {})
        if not isinstance(raw_run_meta, dict):
            raise ValueError("run_meta must be an object")
        run_meta = {str(k): _to_json_value(v) for k, v in raw_run_meta.items()}
        return cls(selections=selections, run_meta=run_meta)

    def save_json(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> TileSelectionTrace:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("tile selection trace JSON must be an object")
        return cls.from_dict(payload)

    def for_level(self, level: str) -> TileSelection | None:
        normalized = _normalize_level(level)
        for selection in self.selections:
            if selection.level == normalized:
                return selection
        return None
