from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from apex_x.config import ApexXConfig


@runtime_checkable
class RuntimeAdapterProtocol(Protocol):
    """Runtime adapter protocol for deployment backends (TRT/ORT/CPU reference)."""

    def initialize(self, config: ApexXConfig) -> None:
        """Prepare runtime adapter state for inference."""
        ...

    def run(self, image: np.ndarray) -> Mapping[str, Any]:
        """Run inference on one batch/frame and return model outputs."""
        ...

    def shutdown(self) -> None:
        """Release runtime adapter resources."""
        ...
