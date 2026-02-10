from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class RouterProtocol(Protocol):
    """Public routing interface for utility prediction."""

    def predict_utilities(self, tile_signals: Sequence[float]) -> list[float]:
        """Return utility per tile in deterministic order."""
        ...


@runtime_checkable
class BudgetControllerProtocol(Protocol):
    """Public budget selection interface."""

    def select(
        self,
        utilities: Sequence[float],
        costs: Sequence[float],
        budget: float,
        kmax: int,
    ) -> tuple[list[int], float]:
        """Return selected tile indices and spent budget."""
        ...


@runtime_checkable
class CostModelProtocol(Protocol):
    """Public cost-model interface for routing/budget accounting."""

    def cheap_cost(self, level: str, num_tiles: int = 1) -> float:
        ...

    def heavy_cost(
        self,
        level: str,
        num_tiles: int = 1,
        include_pack_unpack: bool = True,
    ) -> float:
        ...

    def delta_cost(
        self,
        level: str,
        num_tiles: int = 1,
        include_pack_unpack: bool = True,
    ) -> float:
        ...

    def split_overhead(self, level: str, num_splits: int = 1) -> float:
        ...

    def apply_empirical_calibration(
        self,
        level: str,
        measured_timings: Mapping[str, float],
        blend: float = 1.0,
        apply: bool = True,
    ) -> Mapping[str, float]:
        ...

    def to_dict(self) -> dict[str, object]:
        ...


# Backward-compatible aliases used by existing modules.
Router = RouterProtocol
BudgetController = BudgetControllerProtocol
CostModel = CostModelProtocol
