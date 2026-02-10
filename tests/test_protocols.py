from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from apex_x import ApexXConfig
from apex_x.routing import (
    BudgetControllerProtocol,
    GreedyBudgetController,
    IdentityRouter,
    RouterProtocol,
)
from apex_x.runtime import NullRuntimeAdapter, RuntimeAdapterProtocol
from apex_x.tiles import NumpyTileCodec, TilePackerProtocol


def _use_router(router: RouterProtocol) -> list[float]:
    return router.predict_utilities([0.1, 0.2, 0.3])


def _use_budget_controller(controller: BudgetControllerProtocol) -> tuple[list[int], float]:
    return controller.select(utilities=[0.5, 0.1], costs=[1.0, 1.0], budget=1.0, kmax=1)


def _use_runtime(adapter: RuntimeAdapterProtocol, cfg: ApexXConfig) -> Mapping[str, Any]:
    image = np.zeros((1, 3, cfg.model.input_height, cfg.model.input_width), dtype=np.float32)
    adapter.initialize(cfg)
    out = adapter.run(image)
    adapter.shutdown()
    return out


def test_router_and_budget_protocol_conformance() -> None:
    router = IdentityRouter()
    controller = GreedyBudgetController()

    assert isinstance(router, RouterProtocol)
    assert isinstance(controller, BudgetControllerProtocol)
    assert _use_router(router) == [0.1, 0.2, 0.3]
    selected, spent = _use_budget_controller(controller)
    assert selected == [0]
    assert spent == 1.0


def test_tile_packer_protocol_conformance() -> None:
    codec = NumpyTileCodec()
    assert isinstance(codec, TilePackerProtocol)

    feature = np.zeros((1, 3, 8, 8), dtype=np.float32)
    indices = np.asarray([[0]], dtype=np.int64)
    packed, _ = codec.pack(feature, indices, tile_size=4)
    assert packed.shape == (1, 1, 3, 4, 4)


def test_runtime_adapter_protocol_conformance() -> None:
    cfg = ApexXConfig()
    adapter = NullRuntimeAdapter()
    assert isinstance(adapter, RuntimeAdapterProtocol)

    out = _use_runtime(adapter, cfg)
    assert "det" in out
