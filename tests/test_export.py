from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_x.config import ApexXConfig
from apex_x.export import ApexXExporter
from apex_x.model import ApexXModel


def _build_model() -> ApexXModel:
    return ApexXModel(config=ApexXConfig())


def test_export_writes_manifest_and_onnx(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    model = _build_model()
    out = tmp_path / "manifest.json"

    exporter = ApexXExporter(shape_mode="static")
    exported = exporter.export(model=model, output_path=str(out))

    assert Path(exported) == out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "onnx"
    assert payload["shape_mode"] == "static"
    assert payload["contracts"]["kmax"]["l0"] == model.config.model.kmax_l0
    assert payload["input_shape"] == [
        1,
        3,
        model.config.model.input_height,
        model.config.model.input_width,
    ]

    onnx_path = Path(payload["artifacts"]["onnx_path"])
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0
    assert len(payload["artifacts"]["onnx_sha256"]) == 64


def test_export_dynamic_shape_mode(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    model = _build_model()
    out = tmp_path / "manifest.json"

    exporter = ApexXExporter(shape_mode="dynamic")
    exporter.export(model=model, output_path=str(out))

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["shape_mode"] == "dynamic"
    assert payload["input_shape"] == ["batch", 3, "height", "width"]


def test_export_rejects_invalid_kmax_contract(tmp_path: Path) -> None:
    cfg = ApexXConfig()
    cfg.model.kmax_l0 = 0
    model = ApexXModel(config=cfg)

    exporter = ApexXExporter(shape_mode="static")
    with pytest.raises(ValueError, match="kmax_l0"):
        exporter.export(model=model, output_path=str(tmp_path / "manifest.json"))


def test_export_rejects_unsupported_format(tmp_path: Path) -> None:
    cfg = ApexXConfig()
    cfg.runtime.export_format = "torchscript"
    model = ApexXModel(config=cfg)

    exporter = ApexXExporter(shape_mode="static")
    with pytest.raises(ValueError, match="unsupported runtime.export_format"):
        exporter.export(model=model, output_path=str(tmp_path / "manifest.json"))
