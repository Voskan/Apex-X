from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex_x import ApexXModel
from apex_x.utils import (
    build_replay_manifest,
    deterministic_mode,
    get_determinism_state,
    hash_json_sha256,
    seed_all,
    set_deterministic_mode,
    stable_json_dumps,
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


def test_hash_json_sha256_is_order_stable() -> None:
    payload_a = {"b": [1, 2, 3], "a": {"x": 1, "y": 2}}
    payload_b = {"a": {"y": 2, "x": 1}, "b": [1, 2, 3]}

    assert stable_json_dumps(payload_a) == stable_json_dumps(payload_b)
    assert hash_json_sha256(payload_a) == hash_json_sha256(payload_b)


def test_build_replay_manifest_contains_seed_config_and_artifact_hashes(tmp_path: Path) -> None:
    artifact_path = tmp_path / "trace.json"
    artifact_path.write_text('{"selected":[0,1,2]}\n', encoding="utf-8")

    config = {"profile": "quality", "kmax": 32}
    manifest_a = build_replay_manifest(
        seed=11,
        config=config,
        artifact_paths={"trace": artifact_path},
    )
    manifest_b = build_replay_manifest(
        seed=11,
        config=config,
        artifact_paths={"trace": artifact_path},
    )

    assert manifest_a["seed"] == 11
    assert manifest_a["config_sha256"] == hash_json_sha256(config)
    assert "trace" in manifest_a["artifact_hashes"]
    assert "trace" in manifest_a["artifact_sha256"]
    assert (
        manifest_a["artifact_hashes"]["trace"]["sha256"]
        == manifest_b["artifact_hashes"]["trace"]["sha256"]
    )
    assert manifest_a["manifest_sha256"] == manifest_b["manifest_sha256"]
