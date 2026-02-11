from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apex_x.runtime.tensorrt import (
    EngineBuildResult,
    TensorRTEngineBuildConfig,
    TensorRTEngineBuilder,
    load_export_manifest,
)
from apex_x.utils import hash_file_sha256


def _write_manifest(path: Path, onnx_path: Path, onnx_sha: str | None) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "format": "onnx",
        "shape_mode": "static",
        "profile": "small",
        "artifacts": {
            "onnx_path": str(onnx_path),
            "onnx_sha256": onnx_sha,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_export_manifest_validates_hash(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, onnx_path, hash_file_sha256(onnx_path))

    loaded = load_export_manifest(manifest_path=manifest, verify_onnx_hash=True)
    assert loaded.manifest_path == manifest.resolve()
    assert loaded.onnx_path == onnx_path.resolve()
    assert loaded.onnx_sha256 is not None
    assert loaded.shape_mode == "static"
    assert loaded.profile == "small"


def test_load_export_manifest_hash_mismatch_fails(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, onnx_path, "0" * 64)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_export_manifest(manifest_path=manifest, verify_onnx_hash=True)


def test_build_from_export_manifest_passes_onnx_to_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, onnx_path, hash_file_sha256(onnx_path))

    engine_path = tmp_path / "model.engine"
    builder = TensorRTEngineBuilder()
    observed: dict[str, Path] = {}

    def _fake_build_from_onnx(
        *,
        onnx_path: str | Path,
        engine_path: str | Path,
        build: TensorRTEngineBuildConfig | None = None,
        calibration_batches: Any = None,
    ) -> EngineBuildResult:
        observed["onnx_path"] = Path(onnx_path).resolve()
        out = Path(engine_path).resolve()
        out.write_bytes(b"fake-engine")
        return EngineBuildResult(
            engine_path=out,
            used_fp16=False,
            used_int8=False,
            plugin_status=(),
            calibration_cache_path=None,
        )

    monkeypatch.setattr(builder, "build_from_onnx", _fake_build_from_onnx)
    result = builder.build_from_export_manifest(
        manifest_path=manifest,
        engine_path=engine_path,
        verify_onnx_hash=True,
    )
    assert observed["onnx_path"] == onnx_path.resolve()
    assert result.engine_path == engine_path.resolve()
    assert result.engine_path.exists()
