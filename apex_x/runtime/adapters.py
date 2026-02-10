from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from apex_x.config import ApexXConfig
from apex_x.model import ApexXModel

from .interfaces import RuntimeAdapterProtocol


class NullRuntimeAdapter(RuntimeAdapterProtocol):
    """Reference runtime adapter that delegates to ApexXModel on CPU."""

    def __init__(self) -> None:
        self._model: ApexXModel | None = None

    def initialize(self, config: ApexXConfig) -> None:
        self._model = ApexXModel(config=config)

    def run(self, image: np.ndarray) -> Mapping[str, Any]:
        if self._model is None:
            raise RuntimeError("Runtime adapter is not initialized")
        return self._model.forward(image)

    def shutdown(self) -> None:
        self._model = None
