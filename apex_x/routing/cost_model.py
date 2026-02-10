from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _normalize_level(level: str) -> str:
    normalized = level.strip().lower()
    if normalized not in {"l0", "l1", "l2"}:
        raise ValueError("level must be one of: l0, l1, l2")
    return normalized


def _validate_non_negative_count(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


@dataclass(slots=True)
class LevelCost:
    """Per-level cost terms in cost units (or empirical ms surrogate)."""

    c_cheap: float
    c_heavy: float
    pack_overhead: float = 0.0
    unpack_overhead: float = 0.0
    split_overhead: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.c_cheap) or self.c_cheap < 0.0:
            raise ValueError("c_cheap must be finite and >= 0")
        if not math.isfinite(self.c_heavy) or self.c_heavy <= self.c_cheap:
            raise ValueError("c_heavy must be finite and > c_cheap")
        for name, value in (
            ("pack_overhead", self.pack_overhead),
            ("unpack_overhead", self.unpack_overhead),
            ("split_overhead", self.split_overhead),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

    def io_overhead(self) -> float:
        return self.pack_overhead + self.unpack_overhead

    def to_dict(self) -> dict[str, float]:
        return {
            "c_cheap": self.c_cheap,
            "c_heavy": self.c_heavy,
            "pack_overhead": self.pack_overhead,
            "unpack_overhead": self.unpack_overhead,
            "split_overhead": self.split_overhead,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LevelCost:
        return cls(
            c_cheap=float(data["c_cheap"]),
            c_heavy=float(data["c_heavy"]),
            pack_overhead=float(data.get("pack_overhead", 0.0)),
            unpack_overhead=float(data.get("unpack_overhead", 0.0)),
            split_overhead=float(data.get("split_overhead", 0.0)),
        )


@dataclass(slots=True)
class CalibrationRecord:
    level: str
    measured_timings: dict[str, float]
    blend: float
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "measured_timings": self.measured_timings,
            "blend": self.blend,
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        measured_raw = dict(data.get("measured_timings", {}))
        measured = {str(k): float(v) for k, v in measured_raw.items()}
        return cls(
            level=_normalize_level(str(data["level"])),
            measured_timings=measured,
            blend=float(data["blend"]),
            applied=bool(data["applied"]),
        )


@dataclass(slots=True)
class StaticCostModel:
    """Reference cost model with per-level terms and calibration history."""

    levels: dict[str, LevelCost] = field(default_factory=dict)
    calibration_history: list[CalibrationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.levels:
            self.levels = {
                "l0": LevelCost(c_cheap=0.2, c_heavy=1.0, split_overhead=1.0),
                "l1": LevelCost(c_cheap=0.2, c_heavy=1.0, split_overhead=1.0),
                "l2": LevelCost(c_cheap=0.2, c_heavy=1.0, split_overhead=1.0),
            }
        normalized: dict[str, LevelCost] = {}
        for level, cost in self.levels.items():
            key = _normalize_level(level)
            cost.validate()
            normalized[key] = cost
        for required in ("l0", "l1", "l2"):
            if required not in normalized:
                raise ValueError("levels must include l0, l1, and l2")
        self.levels = normalized

    def _level_cost(self, level: str) -> LevelCost:
        return self.levels[_normalize_level(level)]

    def cheap_cost(self, level: str, num_tiles: int = 1) -> float:
        _validate_non_negative_count(num_tiles, "num_tiles")
        cost = self._level_cost(level)
        return float(num_tiles) * cost.c_cheap

    def heavy_cost(
        self,
        level: str,
        num_tiles: int = 1,
        include_pack_unpack: bool = True,
    ) -> float:
        _validate_non_negative_count(num_tiles, "num_tiles")
        cost = self._level_cost(level)
        per_tile = cost.c_heavy + (cost.io_overhead() if include_pack_unpack else 0.0)
        return float(num_tiles) * per_tile

    def delta_cost(
        self,
        level: str,
        num_tiles: int = 1,
        include_pack_unpack: bool = True,
    ) -> float:
        return self.heavy_cost(level, num_tiles, include_pack_unpack) - self.cheap_cost(
            level,
            num_tiles,
        )

    def split_overhead(self, level: str, num_splits: int = 1) -> float:
        _validate_non_negative_count(num_splits, "num_splits")
        cost = self._level_cost(level)
        return float(num_splits) * cost.split_overhead

    def expected_level_cost(
        self,
        level: str,
        probabilities: list[float],
        include_pack_unpack: bool = True,
    ) -> float:
        cost = self._level_cost(level)
        heavy = cost.c_heavy + (cost.io_overhead() if include_pack_unpack else 0.0)
        total = 0.0
        for p in probabilities:
            if p < 0.0 or p > 1.0:
                raise ValueError("probabilities must be within [0, 1]")
            total += p * heavy + (1.0 - p) * cost.c_cheap
        return total

    def total_cost(
        self,
        heavy_tiles_by_level: dict[str, int],
        cheap_tiles_by_level: dict[str, int] | None = None,
        splits_by_level: dict[str, int] | None = None,
        include_pack_unpack: bool = True,
    ) -> float:
        cheap_tiles_by_level = cheap_tiles_by_level or {}
        splits_by_level = splits_by_level or {}
        total = 0.0
        levels = {"l0", "l1", "l2"}
        levels.update(_normalize_level(level) for level in heavy_tiles_by_level)
        levels.update(_normalize_level(level) for level in cheap_tiles_by_level)
        levels.update(_normalize_level(level) for level in splits_by_level)

        for level in sorted(levels):
            total += self.heavy_cost(
                level,
                num_tiles=int(heavy_tiles_by_level.get(level, 0)),
                include_pack_unpack=include_pack_unpack,
            )
            total += self.cheap_cost(level, num_tiles=int(cheap_tiles_by_level.get(level, 0)))
            total += self.split_overhead(level, num_splits=int(splits_by_level.get(level, 0)))
        return total

    def apply_empirical_calibration(
        self,
        level: str,
        measured_timings: dict[str, float],
        blend: float = 1.0,
        apply: bool = True,
    ) -> dict[str, float]:
        if blend < 0.0 or blend > 1.0:
            raise ValueError("blend must be within [0, 1]")

        normalized_level = _normalize_level(level)
        cost = self._level_cost(normalized_level)

        allowed = {"c_cheap", "c_heavy", "pack_overhead", "unpack_overhead", "split_overhead"}
        measured: dict[str, float] = {}
        for key, value in measured_timings.items():
            if key not in allowed:
                raise ValueError(f"unsupported calibration key: {key}")
            fval = float(value)
            if not math.isfinite(fval) or fval < 0.0:
                raise ValueError("calibration values must be finite and >= 0")
            measured[key] = fval

        if apply:
            for key, target in measured.items():
                current = float(getattr(cost, key))
                updated = (1.0 - blend) * current + blend * target
                setattr(cost, key, updated)
            cost.validate()

        record = CalibrationRecord(
            level=normalized_level,
            measured_timings=measured,
            blend=blend,
            applied=apply,
        )
        self.calibration_history.append(record)
        return record.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": {level: self.levels[level].to_dict() for level in ("l0", "l1", "l2")},
            "calibration_history": [record.to_dict() for record in self.calibration_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaticCostModel:
        raw_levels = dict(data.get("levels", {}))
        levels: dict[str, LevelCost] = {}
        for level, raw in raw_levels.items():
            if not isinstance(raw, dict):
                raise ValueError("levels entries must be objects")
            levels[_normalize_level(str(level))] = LevelCost.from_dict(raw)

        raw_history = data.get("calibration_history", [])
        if not isinstance(raw_history, list):
            raise ValueError("calibration_history must be a list")
        history: list[CalibrationRecord] = []
        for item in raw_history:
            if not isinstance(item, dict):
                raise ValueError("calibration_history entries must be objects")
            history.append(CalibrationRecord.from_dict(item))

        return cls(levels=levels, calibration_history=history)

    def save_json(self, path: str | Path) -> Path:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path_obj

    @classmethod
    def load_json(cls, path: str | Path) -> StaticCostModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("cost model JSON must be an object")
        return cls.from_dict(payload)
