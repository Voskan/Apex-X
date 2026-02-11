from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

from apex_x.config import ApexXConfig
from apex_x.model import TeacherModel

from .interfaces import Exporter

ShapeMode = Literal["static", "dynamic"]


@dataclass(frozen=True, slots=True)
class ExportPaths:
    manifest_path: Path
    onnx_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_paths(output_path: str) -> ExportPaths:
    target = Path(output_path).expanduser()
    if target.suffix:
        manifest_path = target
        stem = target.stem or "apex_x_export"
        out_dir = target.parent
    else:
        out_dir = target
        manifest_path = out_dir / "export_manifest.json"
        stem = "apex_x_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{stem}.onnx"
    return ExportPaths(manifest_path=manifest_path, onnx_path=onnx_path)


def _resolve_config(model: Any) -> ApexXConfig:
    cfg = getattr(model, "config", None)
    if not isinstance(cfg, ApexXConfig):
        raise TypeError("model must expose ApexXConfig at model.config")
    return cfg


def _validate_export_contract(cfg: ApexXConfig) -> dict[str, int]:
    model_cfg = cfg.model
    depth = model_cfg.effective_nesting_depth()
    ff_h = model_cfg.input_height // model_cfg.ff_primary_stride
    ff_w = model_cfg.input_width // model_cfg.ff_primary_stride

    if model_cfg.kmax_l0 <= 0:
        raise ValueError("export contract violation: model.kmax_l0 must be > 0")
    if depth >= 1 and model_cfg.kmax_l1 <= 0:
        raise ValueError(
            "export contract violation: model.kmax_l1 must be > 0 for nesting depth >=1"
        )
    if depth < 1 and model_cfg.kmax_l1 != 0:
        raise ValueError(
            "export contract violation: model.kmax_l1 must be 0 for nesting depth ==0"
        )
    if depth >= 2 and model_cfg.kmax_l2 <= 0:
        raise ValueError(
            "export contract violation: model.kmax_l2 must be > 0 for nesting depth >=2"
        )
    if depth < 2 and model_cfg.kmax_l2 != 0:
        raise ValueError("export contract violation: model.kmax_l2 must be 0 for nesting depth <2")
    if ff_h % model_cfg.tile_size_l0 != 0 or ff_w % model_cfg.tile_size_l0 != 0:
        raise ValueError("export contract violation: FF map must be divisible by tile_size_l0")
    if depth >= 1 and (ff_h % model_cfg.tile_size_l1 != 0 or ff_w % model_cfg.tile_size_l1 != 0):
        raise ValueError("export contract violation: FF map must be divisible by tile_size_l1")
    if depth >= 2 and (ff_h % model_cfg.tile_size_l2 != 0 or ff_w % model_cfg.tile_size_l2 != 0):
        raise ValueError("export contract violation: FF map must be divisible by tile_size_l2")

    return {"ff_height": ff_h, "ff_width": ff_w, "nesting_depth": depth}


def _export_onnx_model(
    model: Any,
    dummy_input: Any,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, Any] | None = None,
) -> None:
    import torch

    # Ensure model is in eval mode
    if hasattr(model, "eval"):
        model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )


def _shape_spec(cfg: ApexXConfig, shape_mode: ShapeMode) -> list[int | str]:
    if shape_mode == "dynamic":
        return ["batch", 3, "height", "width"]
    return [1, 3, cfg.model.input_height, cfg.model.input_width]


class TeacherExportWrapper(torch.nn.Module):
    def __init__(self, model: TeacherModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out.logits, out.boundaries



class ApexXExporter(Exporter):
    """Export Apex-X runtime bundle with manifest and ONNX graph artifact."""

    def __init__(self, *, shape_mode: ShapeMode = "static") -> None:
        if shape_mode not in {"static", "dynamic"}:
            raise ValueError("shape_mode must be 'static' or 'dynamic'")
        self._shape_mode = shape_mode

    def export(self, model: Any, output_path: str) -> str:
        cfg = _resolve_config(model)
        derived = _validate_export_contract(cfg)
        if cfg.runtime.export_format != "onnx":
            raise ValueError(
                "unsupported runtime.export_format="
                f"{cfg.runtime.export_format!r}; only 'onnx' is supported"
            )

        paths = _resolve_paths(output_path)
        
        # Prepare dummy input for tracing
        import torch
        dummy_input = torch.randn(
            1, 3, cfg.model.input_height, cfg.model.input_width
        )
        
        dynamic_axes = None
        if self._shape_mode == "dynamic":
            dynamic_axes = {
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            }

        input_names = ["input"]
        output_names = ["output"]
        model_to_export = model

        if isinstance(model, TeacherModel):
            import torch
            model_to_export = TeacherExportWrapper(model)
            output_names = ["logits", "boundaries"]

        _export_onnx_model(
            model=model_to_export,
            dummy_input=dummy_input,
            output_path=paths.onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        
        # We need to calculate input shape for manifest
        input_shape = _shape_spec(cfg, self._shape_mode)

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "format": "onnx",
            "shape_mode": self._shape_mode,
            "model_class": model.__class__.__name__,
            "profile": cfg.model.profile,
            "runtime": {
                "backend": cfg.runtime.backend,
                "precision_profile": cfg.runtime.precision_profile,
            },
            "input_shape": input_shape,
            "contracts": {
                "nesting_depth": derived["nesting_depth"],
                "kmax": {
                    "l0": cfg.model.kmax_l0,
                    "l1": cfg.model.kmax_l1,
                    "l2": cfg.model.kmax_l2,
                },
                "tile_size": {
                    "l0": cfg.model.tile_size_l0,
                    "l1": cfg.model.tile_size_l1,
                    "l2": cfg.model.tile_size_l2,
                },
                "ff_map": {
                    "height": derived["ff_height"],
                    "width": derived["ff_width"],
                },
            },
            "artifacts": {
                "onnx_path": str(paths.onnx_path.resolve()),
                "onnx_sha256": _sha256_file(paths.onnx_path),
            },
        }
        paths.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return str(paths.manifest_path)


class NoopExporter(ApexXExporter):
    """Backward-compatible alias for legacy code paths."""
