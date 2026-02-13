from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from apex_x.config import ApexXConfig
from apex_x.data.preflight import run_dataset_preflight


def _write_tiny_yolo_dataset(tmp_path: Path, *, label_line: str) -> Path:
    root = tmp_path / "yolo_ds"
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "train: train/images\n"
        "val: train/images\n"
        "names:\n"
        "  0: roof\n",
        encoding="utf-8",
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(image).save(images_dir / "sample.png")
    (labels_dir / "sample.txt").write_text(label_line + "\n", encoding="utf-8")
    return root


def test_dataset_preflight_synthetic_policy_respected() -> None:
    cfg = ApexXConfig()
    cfg.train.allow_synthetic_fallback = False
    report_fail = run_dataset_preflight(cfg)
    assert report_fail.passed is False
    assert report_fail.dataset_type == "synthetic"
    assert any("allow_synthetic_fallback=false" in msg for msg in report_fail.errors)

    cfg.train.allow_synthetic_fallback = True
    report_pass = run_dataset_preflight(cfg)
    assert report_pass.passed is True
    assert report_pass.dataset_type == "synthetic"


def test_dataset_preflight_yolo_rejects_invalid_class_bounds(tmp_path: Path) -> None:
    root = _write_tiny_yolo_dataset(
        tmp_path,
        label_line="3 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8",
    )
    cfg = ApexXConfig()
    cfg.train.allow_synthetic_fallback = False
    cfg.data.dataset_type = "yolo"
    cfg.data.dataset_root = str(root)
    report = run_dataset_preflight(cfg)
    assert report.passed is False
    assert any("invalid records" in msg for msg in report.errors)


def test_dataset_preflight_yolo_accepts_valid_contract(tmp_path: Path) -> None:
    root = _write_tiny_yolo_dataset(
        tmp_path,
        label_line="0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8",
    )
    cfg = ApexXConfig()
    cfg.train.allow_synthetic_fallback = False
    cfg.data.dataset_type = "yolo"
    cfg.data.dataset_root = str(root)
    report = run_dataset_preflight(cfg)
    assert report.passed is True
    assert report.dataset_type == "yolo"
    assert report.stats.get("yolo_annotations_invalid", 1) == 0
