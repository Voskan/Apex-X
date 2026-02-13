"""Dataset contract preflight checks for training paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from apex_x.config import ApexXConfig
from apex_x.data.coco import load_coco_dataset, segmentation_to_mask
from apex_x.data.satellite import SatelliteDataset

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(slots=True)
class DatasetPreflightReport:
    passed: bool
    dataset_type: str
    resolved_dataset_root: str
    checks: list[str]
    errors: list[str]
    warnings: list[str]
    stats: dict[str, int | float | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "dataset_type": str(self.dataset_type),
            "resolved_dataset_root": str(self.resolved_dataset_root),
            "checks": list(self.checks),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
        }


def infer_dataset_type(config: ApexXConfig, dataset_path: str | Path | None) -> str:
    configured = str(config.data.dataset_type).strip().lower()
    if configured != "auto":
        return configured
    if config.data.coco_train_images and config.data.coco_train_annotations:
        return "coco"
    root_text = str(dataset_path) if dataset_path is not None else str(config.data.dataset_root)
    if root_text.strip():
        root = Path(root_text).expanduser()
        if (root / "data.yaml").exists():
            return "yolo"
        return "satellite"
    return "synthetic"


def _resolve_dataset_root(config: ApexXConfig, dataset_path: str | Path | None) -> str:
    if dataset_path is None:
        return str(config.data.dataset_root).strip()
    return str(dataset_path).strip()


def _validate_yolo_polygon(parts: list[float], *, num_classes: int) -> str | None:
    if len(parts) < 7:
        return "line has fewer than 3 polygon points"
    if (len(parts) - 1) % 2 != 0:
        return "line has odd number of coordinates"
    class_id = int(parts[0])
    if class_id < 0 or class_id >= max(1, num_classes):
        return f"class id {class_id} out of bounds [0, {max(0, num_classes - 1)}]"
    coords = np.asarray(parts[1:], dtype=np.float64)
    if not np.isfinite(coords).all():
        return "line has non-finite coordinates"
    if np.any(coords < 0.0) or np.any(coords > 1.0):
        return "coordinates must be normalized in [0,1]"
    points = coords.reshape(-1, 2)
    if points.shape[0] < 3:
        return "polygon has fewer than 3 points"
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area <= 1e-12:
        return "polygon area is degenerate"
    return None


def _run_coco_preflight(
    config: ApexXConfig,
    *,
    checks: list[str],
    errors: list[str],
    warnings: list[str],
    stats: dict[str, int | float | str],
) -> None:
    image_root = Path(config.data.coco_train_images).expanduser()
    ann_path = Path(config.data.coco_train_annotations).expanduser()
    if not image_root.exists():
        errors.append(f"COCO train image directory not found: {image_root}")
        return
    if not ann_path.exists():
        errors.append(f"COCO train annotation file not found: {ann_path}")
        return
    checks.append("coco_paths_exist")

    try:
        dataset = load_coco_dataset(ann_path, strict=True, use_cache=False)
    except Exception as exc:
        errors.append(f"COCO annotation parse failed: {exc}")
        return
    checks.append("coco_annotations_parse")

    stats["coco_image_count"] = int(dataset.image_count)
    stats["coco_annotation_count"] = int(dataset.annotation_count)
    stats["coco_category_count"] = int(dataset.category_count)

    if dataset.image_count <= 0:
        errors.append("COCO annotation contains zero images")
    if dataset.category_count <= 0:
        errors.append("COCO annotation contains zero categories")

    sample_images = list(dataset.images_by_id.values())[:32]
    missing_image_files = 0
    for image in sample_images:
        path = image_root / image.file_name
        if not path.exists():
            missing_image_files += 1
    stats["coco_missing_image_files_in_sample"] = int(missing_image_files)
    if missing_image_files > 0:
        errors.append(f"COCO sample contains {missing_image_files} missing image files")
    checks.append("coco_image_files_sample")

    seg_checked = 0
    seg_invalid = 0
    for annotation in dataset.annotations[:64]:
        if annotation.segmentation is None:
            continue
        image = dataset.images_by_id.get(annotation.image_id)
        if image is None:
            seg_invalid += 1
            continue
        try:
            mask = segmentation_to_mask(
                annotation.segmentation,
                image_height=int(image.height),
                image_width=int(image.width),
            )
            if mask.shape != (int(image.height), int(image.width)):
                seg_invalid += 1
        except Exception:
            seg_invalid += 1
        seg_checked += 1
    stats["coco_segmentation_checked"] = int(seg_checked)
    stats["coco_segmentation_invalid"] = int(seg_invalid)
    if seg_invalid > 0:
        errors.append(f"COCO segmentation decode failed for {seg_invalid} sampled annotations")
    if seg_checked == 0:
        warnings.append("COCO sampled annotations had no segmentation payloads")
    checks.append("coco_segmentation_sample")


def _run_yolo_preflight(
    root: Path,
    *,
    checks: list[str],
    errors: list[str],
    warnings: list[str],
    stats: dict[str, int | float | str],
) -> None:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        errors.append(f"YOLO data.yaml not found: {yaml_path}")
        return
    checks.append("yolo_data_yaml_exists")

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except Exception as exc:
        errors.append(f"YOLO data.yaml parse failed: {exc}")
        return

    names = payload.get("names", [])
    class_count = len(names) if isinstance(names, (dict, list)) else 0
    if class_count <= 0:
        errors.append("YOLO names/classes are missing in data.yaml")
        return
    stats["yolo_class_count"] = int(class_count)
    checks.append("yolo_class_names")

    train_split = payload.get("train", "train/images")
    train_images_dir = Path(train_split)
    if not train_images_dir.is_absolute():
        train_images_dir = root / train_images_dir
    train_labels_dir = train_images_dir.parent / "labels"
    if not train_images_dir.exists():
        errors.append(f"YOLO train images directory not found: {train_images_dir}")
        return
    if not train_labels_dir.exists():
        errors.append(f"YOLO train labels directory not found: {train_labels_dir}")
        return
    checks.append("yolo_train_paths_exist")

    image_files = sorted(
        [path for path in train_images_dir.iterdir() if path.suffix.lower() in _IMAGE_EXTENSIONS]
    )
    stats["yolo_train_image_count"] = int(len(image_files))
    if not image_files:
        errors.append("YOLO train images directory is empty")
        return
    checks.append("yolo_train_images_non_empty")

    sample_images = image_files[:64]
    parsed_annotations = 0
    invalid_annotations = 0
    for image_file in sample_images:
        label_path = train_labels_dir / f"{image_file.stem}.txt"
        if not label_path.exists():
            warnings.append(f"Missing label for image: {image_file.name}")
            continue
        for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            text = line.strip()
            if not text:
                continue
            try:
                parts = [float(value) for value in text.split()]
            except Exception:
                invalid_annotations += 1
                continue
            error = _validate_yolo_polygon(parts, num_classes=class_count)
            if error is not None:
                invalid_annotations += 1
                warnings.append(f"{label_path.name}:{line_idx + 1}: {error}")
            parsed_annotations += 1
    stats["yolo_annotations_checked"] = int(parsed_annotations)
    stats["yolo_annotations_invalid"] = int(invalid_annotations)
    if parsed_annotations <= 0:
        errors.append("No YOLO annotations were parsed from sampled label files")
    elif invalid_annotations > 0:
        errors.append(f"YOLO sampled annotations contain {invalid_annotations} invalid records")
    checks.append("yolo_annotation_contract_sample")


def _run_satellite_preflight(
    root: Path,
    *,
    checks: list[str],
    errors: list[str],
    stats: dict[str, int | float | str],
) -> None:
    if not root.exists():
        errors.append(f"Satellite dataset root not found: {root}")
        return
    checks.append("satellite_root_exists")

    image_files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not image_files:
        errors.append(f"No image files found in satellite dataset root: {root}")
        return
    stats["satellite_image_file_count"] = int(len(image_files))
    checks.append("satellite_image_files_non_empty")

    try:
        dataset = SatelliteDataset(
            root_dir=root,
            tile_size=256,
            stride=128,
            require_mask=True,
            limit_tiles=16,
        )
    except Exception as exc:
        errors.append(f"SatelliteDataset initialization failed: {exc}")
        return

    if len(dataset) <= 0:
        errors.append("SatelliteDataset produced zero tiles")
        return
    stats["satellite_tile_count_sample"] = int(len(dataset))
    checks.append("satellite_dataset_tiles")


def run_dataset_preflight(
    config: ApexXConfig,
    *,
    dataset_path: str | Path | None = None,
) -> DatasetPreflightReport:
    dataset_type = infer_dataset_type(config, dataset_path)
    dataset_root = _resolve_dataset_root(config, dataset_path)

    checks: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int | float | str] = {}

    if dataset_type == "synthetic":
        if config.train.allow_synthetic_fallback:
            warnings.append(
                "Dataset resolved to synthetic mode and allow_synthetic_fallback=true. "
                "This mode is intended only for smoke/debug."
            )
            checks.append("synthetic_mode_allowed")
        else:
            errors.append(
                "Dataset resolved to synthetic mode while allow_synthetic_fallback=false."
            )
    elif dataset_type == "coco":
        _run_coco_preflight(
            config,
            checks=checks,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    elif dataset_type == "yolo":
        root = Path(dataset_root).expanduser()
        _run_yolo_preflight(
            root,
            checks=checks,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    elif dataset_type == "satellite":
        root = Path(dataset_root).expanduser()
        _run_satellite_preflight(
            root,
            checks=checks,
            errors=errors,
            stats=stats,
        )
    else:
        errors.append(f"Unsupported dataset_type for preflight: {dataset_type}")

    passed = len(errors) == 0
    return DatasetPreflightReport(
        passed=passed,
        dataset_type=dataset_type,
        resolved_dataset_root=dataset_root,
        checks=checks,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def write_dataset_preflight_report(
    report: DatasetPreflightReport,
    *,
    path: str | Path,
) -> Path:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


__all__ = [
    "DatasetPreflightReport",
    "infer_dataset_type",
    "run_dataset_preflight",
    "write_dataset_preflight_report",
]
