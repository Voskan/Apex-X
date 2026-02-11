from __future__ import annotations

from pathlib import Path

import numpy as np

from apex_x.runtime.tensorrt import (
    PluginStatus,
    build_calibration_cache_key,
    build_calibration_dataset_digest,
)
from apex_x.runtime.tensorrt.calibrator import CalibratorConfig, _EntropyCalibratorBase


def _loader_once() -> list[np.ndarray]:
    return [np.ones((1, 8), dtype=np.float32)]


def _plugin_statuses(version: str) -> tuple[PluginStatus, ...]:
    return (
        PluginStatus(
            name="TilePack",
            required=True,
            found=True,
            expected_version="1",
            discovered_version=version,
            version_match=version == "1",
            expected_namespace="apexx",
            discovered_namespace="apexx",
            namespace_match=True,
            expected_fields=("tile_size",),
            discovered_fields=("tile_size",),
            missing_fields=(),
            field_signature_match=True,
        ),
    )


def test_build_calibration_cache_key_changes_with_precision_and_plugin_version() -> None:
    base = build_calibration_cache_key(
        onnx_sha256="a" * 64,
        plugin_statuses=_plugin_statuses("1"),
        precision_profile="edge",
        calibration_dataset_version="dataset-v1",
    )
    changed_profile = build_calibration_cache_key(
        onnx_sha256="a" * 64,
        plugin_statuses=_plugin_statuses("1"),
        precision_profile="balanced",
        calibration_dataset_version="dataset-v1",
    )
    changed_plugin = build_calibration_cache_key(
        onnx_sha256="a" * 64,
        plugin_statuses=_plugin_statuses("2"),
        precision_profile="edge",
        calibration_dataset_version="dataset-v1",
    )
    changed_dataset = build_calibration_cache_key(
        onnx_sha256="a" * 64,
        plugin_statuses=_plugin_statuses("1"),
        precision_profile="edge",
        calibration_dataset_version="dataset-v2",
    )

    assert base != changed_profile
    assert base != changed_plugin
    assert base != changed_dataset


def test_calibration_dataset_digest_tracks_data_changes() -> None:
    dataset_a = [np.zeros((1, 4), dtype=np.float32)]
    dataset_b = [np.ones((1, 4), dtype=np.float32)]

    digest_a = build_calibration_dataset_digest(dataset_a)
    digest_a_repeat = build_calibration_dataset_digest(dataset_a)
    digest_b = build_calibration_dataset_digest(dataset_b)

    assert digest_a == digest_a_repeat
    assert digest_a != digest_b


def test_calibration_cache_roundtrip_reuses_when_key_matches(tmp_path: Path) -> None:
    cache_path = tmp_path / "calibration.cache"
    key = "cache-key-v1"

    writer = _EntropyCalibratorBase(
        loader=_loader_once(),
        config=CalibratorConfig(
            input_names=("x",),
            cache_path=cache_path,
            cache_key=key,
            device="cpu",
        ),
    )
    writer.write_calibration_cache(b"payload")

    reader = _EntropyCalibratorBase(
        loader=_loader_once(),
        config=CalibratorConfig(
            input_names=("x",),
            cache_path=cache_path,
            cache_key=key,
            device="cpu",
        ),
    )
    assert reader.read_calibration_cache() == b"payload"


def test_calibration_cache_invalidates_on_key_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "calibration.cache"

    writer = _EntropyCalibratorBase(
        loader=_loader_once(),
        config=CalibratorConfig(
            input_names=("x",),
            cache_path=cache_path,
            cache_key="key-A",
            device="cpu",
        ),
    )
    writer.write_calibration_cache(b"payload")

    mismatched = _EntropyCalibratorBase(
        loader=_loader_once(),
        config=CalibratorConfig(
            input_names=("x",),
            cache_path=cache_path,
            cache_key="key-B",
            device="cpu",
        ),
    )
    assert mismatched.read_calibration_cache() is None


def test_legacy_cache_blob_is_ignored_when_key_is_required(tmp_path: Path) -> None:
    cache_path = tmp_path / "legacy.cache"
    cache_path.write_bytes(b"legacy-raw-cache")

    reader = _EntropyCalibratorBase(
        loader=_loader_once(),
        config=CalibratorConfig(
            input_names=("x",),
            cache_path=cache_path,
            cache_key="required-key",
            device="cpu",
        ),
    )
    assert reader.read_calibration_cache() is None

    reader_legacy_ok = _EntropyCalibratorBase(
        loader=_loader_once(),
        config=CalibratorConfig(
            input_names=("x",),
            cache_path=cache_path,
            cache_key=None,
            device="cpu",
        ),
    )
    assert reader_legacy_ok.read_calibration_cache() == b"legacy-raw-cache"
