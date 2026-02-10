from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

TileMeta = dict[str, np.ndarray]


@runtime_checkable
class TilePackerProtocol(Protocol):
    """Public tile packing interface."""

    def pack(
        self,
        feature_map: np.ndarray,
        indices: np.ndarray,
        tile_size: int,
        order_mode: str = "hilbert",
    ) -> tuple[np.ndarray, TileMeta]:
        """Gather selected tiles into contiguous packed tensor."""
        ...


@runtime_checkable
class TileUnpackerProtocol(Protocol):
    """Public tile unpacking interface."""

    def unpack(
        self,
        base_map: np.ndarray,
        packed_out: np.ndarray,
        meta: TileMeta,
        level_priority: int = 1,
        priority_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scatter packed tiles back into base map."""
        ...


@runtime_checkable
class TileCodecProtocol(TilePackerProtocol, TileUnpackerProtocol, Protocol):
    """Combined tile interface used by model forward path."""


# Backward-compatible aliases used by existing modules.
TilePack = TilePackerProtocol
TileUnpack = TileUnpackerProtocol
TileCodec = TileCodecProtocol
