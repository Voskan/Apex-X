from __future__ import annotations

import numpy as np
import pytest

from apex_x import ApexXModel
from apex_x.utils import (
    deterministic_mode,
    get_determinism_state,
    seed_all,
    set_deterministic_mode,
)


def test_seed_all_numpy_is_deterministic() -> None:
    seed_all(123, deterministic=True)
    a = np.random.rand(8)

    seed_all(123, deterministic=True)
    b = np.random.rand(8)

    assert np.allclose(a, b)


def test_seed_all_torch_is_deterministic_if_available() -> None:
    torch = pytest.importorskip("torch")

    seed_all(456, deterministic=True)
    a = torch.rand(6)

    seed_all(456, deterministic=True)
    b = torch.rand(6)

    assert torch.allclose(a, b)


def test_model_forward_deterministic_with_fixed_seed() -> None:
    seed_all(7, deterministic=True)
    image_a = np.random.rand(1, 3, 128, 128).astype(np.float32)
    model_a = ApexXModel()
    out_a = model_a.forward(image_a)

    seed_all(7, deterministic=True)
    image_b = np.random.rand(1, 3, 128, 128).astype(np.float32)
    model_b = ApexXModel()
    out_b = model_b.forward(image_b)

    assert np.array_equal(image_a, image_b)
    assert out_a["selected_tiles"] == out_b["selected_tiles"]
    assert np.allclose(out_a["merged"], out_b["merged"])


def test_deterministic_mode_toggle_and_context_restore() -> None:
    set_deterministic_mode(False)
    assert get_determinism_state()["deterministic_enabled"] is False

    with deterministic_mode(True) as state:
        assert state["deterministic_enabled"] is True
        assert get_determinism_state()["deterministic_enabled"] is True

    assert get_determinism_state()["deterministic_enabled"] is False
